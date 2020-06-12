""" run decoding of X-ext (+ abs)"""
import logging
logging.basicConfig(level=logging.ERROR)
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time

from cytoolz import identity

import torch
from torch.utils.data import DataLoader

from data.batcher import tokenize, preproc

from decoding import Abstractor, Extractor, DecodeDataset, DecodeDatasetEntity, ExtractorEntity, BeamAbstractor, AbsDecodeDataset, AbsDecodeDatasetGAT
from decoding import BeamAbstractorGAT
from decoding import make_html_safe
from nltk import sent_tokenize
from torch import multiprocessing as mp
from cytoolz import identity, concat, curry
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
import pickle


MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor

@curry
def tokenize_keepcase(max_len, texts):
    return [t.strip().split()[:max_len] for t in texts]

def decode(save_path, abs_dir, split, batch_size, max_len, cuda, min_len):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        raise Exception('abs directory none!')
    else:
        #abstractor = Abstractor(abs_dir, max_len, cuda)
        abstractor = BeamAbstractor(abs_dir, max_len, cuda, min_len, reverse=args.reverse)

    bert = abstractor._bert
    if bert:
        tokenizer = abstractor._tokenizer
    if bert:
        import logging
        logging.basicConfig(level=logging.ERROR)

    # if args.docgraph or args.paragraph:
    #     abstractor = BeamAbstractorGAT(abs_dir, max_len, cuda, min_len, reverse=args.reverse)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = AbsDecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    os.makedirs(save_path)
    # prepare save paths and logs
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = 5  # greedy decoding only
    beam_size = 5
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)
    os.makedirs(join(save_path, 'output'))

    # Decoding
    i = 0
    length = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            if bert:
                tokenized_article_batch = map(tokenize_keepcase(args.max_input), raw_article_batch)
            else:
                tokenized_article_batch = map(tokenize(args.max_input), raw_article_batch)
            ext_arts = []
            ext_inds = []
            beam_inds = []
            pre_abs = list(tokenized_article_batch)
            pre_abs = [article[0] for article in pre_abs]
            for j in range(len(pre_abs)):
                beam_inds += [(len(beam_inds), 1)]
            all_beams = abstractor(pre_abs, beam_size, diverse=1.0)
            dec_outs = rerank_mp(all_beams, beam_inds)

            for dec_out in dec_outs:
                if bert:
                    text = ''.join(' '.join(dec_out).split(' '))
                    dec_out = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                          errors=tokenizer.errors)
                    dec_out = [dec_out]


                dec_out = sent_tokenize(' '.join(dec_out))
                ext = [sent.split(' ') for sent in dec_out]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += ext
            dec_outs = ext_arts

            assert i == batch_size * i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j + n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100, timedelta(seconds=int(time() - start))
                ), end='')
                length += len(decoded_sents)
        print('average summary length:', length / i)

def decodeGAT(save_path, abs_dir, split, batch_size, max_len, cuda, min_len, docgraph):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        raise Exception('abs directory none!')
    else:
        #abstractor = Abstractor(abs_dir, max_len, cuda)
        abstractor = BeamAbstractorGAT(abs_dir, max_len, cuda, min_len, reverse=args.reverse, docgraph=docgraph)


    bert = abstractor._bert
    if bert:
        tokenizer = abstractor._tokenizer
        print('use bert')
        logging.basicConfig(level=logging.ERROR)

    # setup loader
    def coll(batch):
        articles, nodes, edges, subgraphs, paras = list(filter(bool, list(zip(*batch))))
        return (articles, nodes, edges, subgraphs, paras)
    dataset = AbsDecodeDatasetGAT(split, docgraph)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    os.makedirs(save_path)
    # prepare save paths and logs
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = 5  # greedy decoding only
    beam_size = 5
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)
    os.makedirs(join(save_path, 'output'))

    # Decoding
    i = 0
    length = 0
    with torch.no_grad():
        for i_debug, raw_batch in enumerate(loader):
            raw_article_batch, nodes, edges, paras, subgraphs = raw_batch
            raw_sents_batch = [[' '.join(article)] for article in raw_article_batch]
            if bert:
                tokenized_article_batch = map(tokenize_keepcase(args.max_input), raw_sents_batch)
            else:
                tokenized_article_batch = map(tokenize(args.max_input), raw_sents_batch)
            # tokenized_article_batch = map(tokenize(args.max_input), raw_sents_batch)
            ext_arts = []
            ext_inds = []
            beam_inds = []
            pre_abs = list(tokenized_article_batch)
            pre_abs = [article[0] for article in pre_abs]
            for j in range(len(pre_abs)):
                beam_inds += [(len(beam_inds), 1)]
            all_beams = abstractor((pre_abs, nodes, edges, paras, subgraphs, raw_article_batch, args.max_input), beam_size, diverse=1.0)
            dec_outs = rerank_mp(all_beams, beam_inds)

            for dec_out in dec_outs:
                if bert:
                    text = ''.join(' '.join(dec_out).split(' '))
                    dec_out = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                          errors=tokenizer.errors)
                    dec_out = [dec_out]

                dec_out = sent_tokenize(' '.join(dec_out))
                ext = [sent.split(' ') for sent in dec_out]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += ext
            dec_outs = ext_arts

            assert i == batch_size * i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j + n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100, timedelta(seconds=int(time() - start))
                ), end='')
                length += len(decoded_sents)
        print('average summary length:', length / i)


_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)


def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def length_wu(cur_len, alpha=0.):
    """GNMT length re-ranking score.
    See "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return ((5 + cur_len) / 6.0) ** alpha

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    # all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    # # repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    # lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    try:
        lp = sum(h.logprob for h in hyps) / sum(length_wu(len(h.sequence)+1, alpha=0.9) for h in hyps)
    except ZeroDivisionError:
        lp = -1e5
    return lp


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model')
    parser.add_argument('--reverse', action='store_true', help='if true then abstractor is trained with rl')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--docgraph', action='store_true', help='if model contains gat encoder docgraph')
    parser.add_argument('--paragraph', action='store_true', help='if model contains gat encoder paragraph')
    parser.add_argument('--max_input', type=int, default=1024, help='maximum input length')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--max_dec_word', type=int, action='store', default=100,
                        help='maximun words to be decoded for the abstractor')
    parser.add_argument('--min_dec_word', type=int, action='store', default=0,
                        help='maximun words to be decoded for the abstractor')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')


    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    data_split = 'test' if args.test else 'val'
    print(args)
    if args.docgraph or args.paragraph:
        assert not all([args.docgraph, args.paragraph])
        decodeGAT(args.path, args.abs_dir,
               data_split, args.batch, args.max_dec_word, args.cuda, args.min_dec_word, args.docgraph)
    else:
        decode(args.path, args.abs_dir,
           data_split, args.batch, args.max_dec_word, args.cuda, args.min_dec_word)
