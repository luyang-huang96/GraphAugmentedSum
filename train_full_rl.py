""" full training (train rnn-ext + abs + RL) """
from training import BasicTrainer
import argparse
import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle


from toolz.sandbox.core import unzip
from cytoolz import identity, curry, concat

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic, SelfCritic, SelfCriticEntity
from model.extract import PtrExtractSumm, PtrExtractSummGAT, PtrExtractSummSubgraph


from rl import get_grad_fn
from rl import A2CPipeline, SCPipeline
from decoding import load_best_ckpt
from decoding import Abstractor, ArticleBatcher, ArticleBatcherGraph
from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ

from pytorch_transformers import BertTokenizer, BertModel, BertConfig
from data.RLbatcher import build_batchers_graph, build_batchers_graph_bert
from model.rl_ext import SCExtractorRLGraph, SelfCriticGraph


MAX_ABS_LEN = 100
BERT_MAX_LEN = 512

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents

class RLDataset_entity(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key='filtered_rule1_input_mention_cluster'):
        super().__init__(split, DATA_DIR)
        self.key = key
        print('using key: ', key)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        input_clusters = js_data[self.key]
        return art_sents, abs_sents, input_clusters


def load_ext_net(ext_dir):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] in ['ml_rnn_extractor', "ml_gat_extractor", "ml_subgraph_gat_extractor"]
    net_name = ext_meta['net']
    ext_ckpt = load_best_ckpt(ext_dir)
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    if ext_meta['net'] == 'ml_rnn_extractor':
        ext = PtrExtractSumm(**ext_args)
    elif ext_meta['net'] == "ml_gat_extractor":
        ext = PtrExtractSummGAT(**ext_args)
    elif ext_meta['net'] == "ml_subgraph_gat_extractor":
        ext = PtrExtractSummSubgraph(**ext_args)
    else:
        raise Exception('not implemented')
    ext.load_state_dict(ext_ckpt)
    return ext, vocab


def configure_net(abs_dir, ext_dir, cuda, sc, tv, rl_dir=''):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir)
    if sc:
        agent = SelfCritic(extractor,
                           ArticleBatcher(agent_vocab, cuda),
                           time_variant=tv
        )
    else:
        agent = ActorCritic(extractor._sent_enc,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, cuda))

    if rl_dir != '':
        ckpt = load_best_ckpt(rl_dir, reverse=True)
        agent.load_state_dict(ckpt)

    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))
    print('agent:', agent)

    return agent, agent_vocab, abstractor, net_args

def configure_net_graph(abs_dir, ext_dir, cuda, docgraph=True, paragraph=False):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    assert not all([docgraph, paragraph])
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir)


    agent = SelfCriticGraph(extractor,
                            ArticleBatcherGraph(agent_vocab, cuda),
                            cuda,
                            docgraph,
                            paragraph
        )


    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args

def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    train_params['gamma']          = gamma
    train_params['reward']         = reward
    train_params['stop_coeff']     = stop_coeff
    train_params['stop_reward']    = stop_reward

    return train_params

def build_batchers(batch_size):
    def coll(batch):
        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts
        art_batch, abs_batch = unzip(batch)
        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents
    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader

def build_batchers_bert(batch_size, bert_sent, bert_stride, max_len):
    config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking',
                                        output_hidden_states=True,
                                        output_attentions=False)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

    @curry
    def coll(tokenizer, batch):
        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts
        @curry
        def prepro(tokenizer, d, max_len=512):
            """ make sure data is not empty"""
            source_sents, extracts = d
            tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
            tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
            tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
            word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
            truncated_word_num = []
            total_count = 0
            for num in word_num:
                if total_count + num < max_len:
                    truncated_word_num.append(num)
                else:
                    truncated_word_num.append(512 - total_count)
                    break
                total_count += num
            tokenized_sents = list(concat(tokenized_sents))[:max_len]
            tokenized_sents = tokenizer.convert_tokens_to_ids(tokenized_sents)
            abs_sents = tokenize(None, extracts)
            art_sents = tokenize(None, source_sents)

            return (art_sents, tokenized_sents, truncated_word_num), abs_sents
        art_batch, abs_batch = unzip(batch)
        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents, abs_sents = list(zip(*list(map(prepro(tokenizer), zip(art_batch, abs_batch)))))
        return art_sents, abs_sents

    @curry
    def coll_sent(tokenizer, batch):
        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts
        @curry
        def prepro(tokenizer, d, max_len=150, max_sent_len=60):
            """ make sure data is not empty"""
            source_sents, extracts = d
            tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
            tokenized_sents = tokenized_sents[:max_sent_len]
            tokenized_sents = [['[CLS]'] + tokenized_sent[:max_len - 1] for tokenized_sent in tokenized_sents]
            tokenized_sents = [tokenizer.convert_tokens_to_ids(tokenized_sent) for tokenized_sent in tokenized_sents]
            word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
            tokenized_sents = [tokenizer.convert_tokens_to_ids(tokenized_sent) for tokenized_sent in tokenized_sents]
            abs_sents = tokenize(None, extracts)
            art_sents = tokenize(None, source_sents)

            return (art_sents, tokenized_sents, word_num), abs_sents
        art_batch, abs_batch = unzip(batch)
        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents, abs_sents = list(zip(*list(map(prepro(tokenizer), zip(art_batch, abs_batch)))))
        return art_sents, abs_sents

    @curry
    def coll_stride(tokenizer, batch, max_len=1024, stride=256):
        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts
        @curry
        def prepro(tokenizer, d, max_len=1024, stride=256):
            """ make sure data is not empty"""
            source_sents, extracts = d
            tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
            tokenized_sents = [['[CLS]'] + tokenized_sent for tokenized_sent in tokenized_sents]
            tokenized_sents = [tokenizer.convert_tokens_to_ids(tokenized_sent) for tokenized_sent in tokenized_sents]
            word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
            truncated_word_num = []
            total_count = 0
            for num in word_num:
                if total_count + num < max_len:
                    truncated_word_num.append(num)
                else:
                    truncated_word_num.append(max_len - total_count)
                    break
                total_count += num
            tokenized_sents = list(concat(tokenized_sents))[:max_len]
            tokenized_sents_lists = [tokenized_sents[:BERT_MAX_LEN]]
            length = len(tokenized_sents) - BERT_MAX_LEN
            i = 1
            while length > 0:
                tokenized_sents_lists.append(tokenized_sents[(i * BERT_MAX_LEN - stride) :((i + 1) * BERT_MAX_LEN - stride)])
                i += 1
                length -= (BERT_MAX_LEN - stride)
            abs_sents = tokenize(None, extracts)
            art_sents = tokenize(None, source_sents)

            return (art_sents, tokenized_sents_lists, truncated_word_num), abs_sents
        art_batch, abs_batch = unzip(batch)
        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents, abs_sents = list(zip(*list(map(prepro(tokenizer, max_len=max_len, stride=stride), zip(art_batch, abs_batch)))))
        return art_sents, abs_sents

    if bert_sent:
        loader = DataLoader(
            RLDataset('train'), batch_size=batch_size,
            shuffle=True, num_workers=4,
            collate_fn=coll_sent(tokenizer)
        )
        val_loader = DataLoader(
            RLDataset('val'), batch_size=batch_size,
            shuffle=False, num_workers=4,
            collate_fn=coll_sent(tokenizer)
        )
    elif bert_stride > 0:
        print('stride size:', bert_stride)
        loader = DataLoader(
            RLDataset('train'), batch_size=batch_size,
            shuffle=True, num_workers=4,
            collate_fn=coll_stride(tokenizer, max_len=max_len, stride=bert_stride)
        )
        val_loader = DataLoader(
            RLDataset('val'), batch_size=batch_size,
            shuffle=False, num_workers=4,
            collate_fn=coll_stride(tokenizer, max_len=max_len, stride=bert_stride)
        )
    else:
        loader = DataLoader(
            RLDataset('train'), batch_size=batch_size,
            shuffle=True, num_workers=4,
            collate_fn=coll(tokenizer)
        )
        val_loader = DataLoader(
            RLDataset('val'), batch_size=batch_size,
            shuffle=False, num_workers=4,
            collate_fn=coll(tokenizer)
        )
    return cycle(loader), val_loader


def train(args):
    if not exists(args.path):
        os.makedirs(args.path)

    # make net
    if args.docgraph or args.paragraph:
        agent, agent_vocab, abstractor, net_args = configure_net_graph(
            args.abs_dir, args.ext_dir, args.cuda, args.docgraph, args.paragraph)
    else:
        agent, agent_vocab, abstractor, net_args = configure_net(
            args.abs_dir, args.ext_dir, args.cuda, True, False, args.rl_dir)

    if args.bert_stride > 0:
        assert args.bert_stride == agent._bert_stride
    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )

    if args.docgraph or args.paragraph:
        if args.bert:
            train_batcher, val_batcher = build_batchers_graph_bert(args.batch, args.key, args.adj_type, args.max_bert_word, args.docgraph, args.paragraph)
        else:
            train_batcher, val_batcher = build_batchers_graph(args.batch, args.key, args.adj_type, args.gold_key, args.docgraph, args.paragraph)
    elif args.bert:
        train_batcher, val_batcher = build_batchers_bert(args.batch, args.bert_sent, args.bert_stride, args.max_bert_word)
    else:
        train_batcher, val_batcher = build_batchers(args.batch)
    # TODO different reward
    if args.reward == 'rouge-l':
        reward_fn = compute_rouge_l
    elif args.reward == 'rouge-1':
        reward_fn = compute_rouge_n(n=1)
    elif args.reward == 'rouge-2':
        reward_fn = compute_rouge_n(n=2)
    elif args.reward == 'rouge-l-s':
        reward_fn = compute_rouge_l_summ
    else:
        raise Exception('Not prepared reward')
    stop_reward_fn = compute_rouge_n(n=1)

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {}
        abs_ckpt['state_dict'] = load_best_ckpt(args.abs_dir, reverse=True)
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'))
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f)
        # save configuration
    meta = {}
    meta['net']           = 'rnn-ext_abs_rl'
    meta['net_args']      = net_args
    meta['train_params']  = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=1e-5,
                                  patience=args.lr_p)

    if args.docgraph or args.paragraph:
        entity = True
    else:
        entity = False
    pipeline = SCPipeline(meta['net'], agent, abstractor,
                               train_batcher, val_batcher,
                               optimizer, grad_fn,
                               reward_fn, entity, args.bert)


    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--docgraph', action='store_true', help='docgraph model')
    parser.add_argument('--paragraph', action='store_true', help='paragraph model')
    parser.add_argument('--bert', action='store_true', help='use bert')
    parser.add_argument('--bert_stride', type=int, default=0, action='store', help='deal with longer sequence larger than maximum BERT length')
    parser.add_argument('--max_bert_word', type=int, action='store', default=1024,
                        help='maximum words fed to bert. recommend 1024/2048, only works when bert stride larger than 0')
    parser.add_argument('--bert_sent', action='store_true', help='use bert on sentence level!')
    parser.add_argument('--sent_enc', action='store', type=str, default='cnn', help='sentence encpder type: cnn or mean(for bert)')

    parser.add_argument('--rl_dir', action='store', default='',
                        help='for continute training, give old checkpoint path')

    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')
    parser.add_argument('--key', type=str, default='nodes_pruned2', help='use which cluster type')

    # training options
    parser.add_argument('--reward', action='store', default='rouge-1',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=2,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--gold_key', action='store', default='InSalientSent', type=str,
                        help='attention type')
    parser.add_argument('--adj_type', action='store', default='edge_as_node', type=str,
                        help='concat_triple, edge_up, edge_down, no_edge, edge_as_node')


    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for che    ckpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    torch.cuda.set_device(args.gpu_id)

    train(args)
