""" train the abstractor"""
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').disabled = True
logging.basicConfig(level=logging.ERROR)
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose, concat

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.copy_summ_multiencoder import CopySummGAT, CopySummParagraph
from model.util import sequence_loss
from training import get_basic_grad_fn, rl_validate
from training import AbsSelfCriticalPipeline, BasicTrainer

from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn, coll_fn_graph, coll_fn_graph_rl
from data.batcher import convert_batch_copy_rl, batchify_fn_copy_rl
from data.batcher import prepro_fn_copy_bert, convert_batch_copy_rl_bert, batchify_fn_copy_rl_bert
from data.batcher import BucketedGenerater
from data.abs_batcher import prepro_graph, convert_batch_graph_rl, batchify_fn_graph_rl
from data.abs_batcher import prepro_graph_bert, convert_batch_graph_rl_bert, batchify_fn_graph_rl_bert

from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding
import re
from toolz.sandbox import unzip
from reward.clozereward import cloze_reward
import pickle


# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset_paulus(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents = (
            js_data['article'], js_data['abstract'])
        art_sents = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        return art_sents, abs_sents

class Dataset_RLgraph(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, key, reward_data_dir=None):
        super().__init__(split, DATA_DIR)
        self.node_key = key
        self.edge_key = key.replace('nodes', 'edges')
        if reward_data_dir is not None:
            self._reward_data_dir = join(reward_data_dir, split)
        else:
            self._reward_data_dir = None

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        if self._reward_data_dir is not None:
            with open(join(self._reward_data_dir, '{}.json'.format(i))) as f:
                question_data = json.load(f)
            try:
                questions = question_data['questions']
            except KeyError:
                questions = []
        art_sents, abs_sents, nodes, edges, subgraphs, paras = (
            js_data['article'], js_data['abstract'], js_data[self.node_key], js_data[self.edge_key], js_data['subgraphs'], js_data['paragraph_merged'])
        #art_sents = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        if self._reward_data_dir is not None:
            return art_sents, abs_sents, nodes, edges, subgraphs, paras, questions
        else:
            return art_sents, abs_sents, nodes, edges, subgraphs, paras



class MatchDataset_all2all(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)

        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted_combine'])
        extracts = sorted(extracts)
        matched_arts = [art_sents[i] for i in extracts]
        matched_arts = [' '.join(matched_arts)]
        abs_sents = [' '.join(abs_sents)]

        return matched_arts, abs_sents

def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=lambda storage, loc: storage
    )['state_dict']
    return ckpt

def configure_net(abs_dir):
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    net_args = abs_meta['net_args']

    abs_ckpt = load_best_ckpt(abs_dir)
    net = CopySumm(**net_args)
    net.load_state_dict(abs_ckpt)
    return net, net_args

def configure_net_graph(abs_dir, docgraph, paragraph):
    assert not all([docgraph, paragraph])
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    net_args = abs_meta['net_args']

    abs_ckpt = load_best_ckpt(abs_dir)
    if docgraph:
        net = CopySummGAT(**net_args)
    else:
        net = CopySummParagraph(**net_args)
    net.load_state_dict(abs_ckpt)
    return net, net_args

def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam', 'adagrad']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    if opt == 'adagrad':
        opt_kwargs['initial_accumulator_value'] = 0.1
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params

def build_batchers(word2id, cuda, debug):
    prepro = prepro_fn(args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy_rl(PAD, START, END, cuda=cuda),
        convert_batch_copy_rl(UNK, word2id)
    )


    train_loader = DataLoader(
        MatchDataset_paulus('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        MatchDataset_paulus('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)

    return train_batcher, val_batcher

def build_batchers_bert(tokenizer, cuda, debug):
    prepro = prepro_fn_copy_bert(tokenizer, args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy_rl_bert(tokenizer, cuda=cuda, min_len=args.min_abs),
        convert_batch_copy_rl_bert(tokenizer, args.max_art)
    )

    train_loader = DataLoader(
        MatchDataset_paulus('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        MatchDataset_paulus('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                        single_run=True, fork=not debug)

    return train_batcher, val_batcher



def build_batchers_graph(word2id, cuda, debug, key, adj_type, docgraph, reward_data_dir):
    prepro = prepro_graph(args.max_art, args.max_abs, adj_type, docgraph=docgraph, reward_data_dir=reward_data_dir)
    def sort_key(sample):
        src, target, nodes = sample[0], sample[1], sample[3]
        return (len(target), len(src), len(nodes))
    batchify = compose(
        batchify_fn_graph_rl(PAD, START, END, cuda=cuda, adj_type=adj_type, docgraph=docgraph, reward_data_dir=reward_data_dir),
        convert_batch_graph_rl(UNK, word2id, docgraph=docgraph, reward_data_dir=reward_data_dir)
    )
    if reward_data_dir is not None:
        if docgraph:
            _coll_fn = coll_fn_graph_rl
        else:
            _coll_fn = coll_fn_graph_rl(max_node=400)
    else:
        _coll_fn = coll_fn_graph


    train_loader = DataLoader(
        Dataset_RLgraph('train', key, reward_data_dir=reward_data_dir), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        Dataset_RLgraph('val', key, reward_data_dir=reward_data_dir), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                        single_run=True, fork=not debug)

    return train_batcher, val_batcher

def build_batchers_graph_bert(tokenizer, cuda, debug, key, adj_type, docgraph, reward_data_dir):
    #prepro = prepro_graph(args.max_art, args.max_abs, adj_type, docgraph=docgraph, reward_data_dir=reward_data_dir)
    with open(os.path.join(DATA_DIR, 'roberta-base-align.pkl'), 'rb') as f:
        align = pickle.load(f)

    prepro = prepro_graph_bert(tokenizer, align, args.max_art, args.max_abs, adj_type,
                                docgraph=docgraph, reward_data_dir=reward_data_dir)
    def sort_key(sample):
        src, target, nodes = sample[0], sample[1], sample[3]
        return (len(src), len(target), len(nodes))
    batchify = compose(
        batchify_fn_graph_rl_bert(tokenizer, cuda=cuda, adj_type=adj_type, docgraph=docgraph, reward_data_dir=reward_data_dir),
        convert_batch_graph_rl_bert(tokenizer, args.max_art, docgraph=docgraph, reward_data_dir=reward_data_dir)
    )
    if reward_data_dir is not None:
        if docgraph:
            _coll_fn = coll_fn_graph_rl(max_node=400)
        else:
            _coll_fn = coll_fn_graph_rl(max_node=800)
    else:
        _coll_fn = coll_fn_graph


    train_loader = DataLoader(
        Dataset_RLgraph('train', key, reward_data_dir=reward_data_dir), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=8 if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        Dataset_RLgraph('val', key, reward_data_dir=reward_data_dir), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=8 if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                        single_run=True, fork=not debug)

    return train_batcher, val_batcher


def main(args):
    # create data batcher, vocabulary
    # batcher
    word2id = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))

    # reward func
    if args.reward_model_dir is not None:
        assert args.reward_data_dir is not None
        reward_func = cloze_reward(args.reward_model_dir, args.cloze_device)
        reward_weight = args.reward_weight
    else:
        reward_func = None
        reward_weight = 0.

    # make net
    if args.docgraph or args.paragraph:
        net, net_args = configure_net_graph(args.abs_dir, args.docgraph, args.paragraph)
    else:
        net, net_args = configure_net(args.abs_dir)

    bert = net._bert
    if bert:
        print('model use bert')
        import logging
        print('disable')
        logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
        logging.getLogger('transformers.tokenization_utils').disabled = True

    if args.docgraph or args.paragraph:
        if bert:
            tokenizer = net._bert_model._tokenizer
            train_batcher, val_batcher = build_batchers_graph_bert(tokenizer,
                                                              args.cuda, args.debug, args.key, net._adj_type,
                                                              args.docgraph, args.reward_data_dir)
        else:
            train_batcher, val_batcher = build_batchers_graph(word2id,
                                                    args.cuda, args.debug, args.key, net._adj_type, args.docgraph, args.reward_data_dir)
    else:
        if bert:
            tokenizer = net._bert_model._tokenizer
            train_batcher, val_batcher = build_batchers_bert(tokenizer,
                                                        args.cuda, args.debug)
        else:
            train_batcher, val_batcher = build_batchers(word2id,
                                                args.cuda, args.debug)


    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'base_abstractor'
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    local_coh_fun = None



    # prepare trainer
    if args.cuda:
        net = net.cuda()


    multigpu = False

    val_fn = rl_validate(net, reward_func=reward_func, reward_coef=reward_weight, bert=bert)
    grad_fn = get_basic_grad_fn(net, args.clip)

    optimizer = optim.AdamW(net.parameters(), **train_params['optimizer'][1])



    #optimizer = optim.Adagrad(net.parameters(), **train_params['optimizer'][1])


    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)
    print('rouge weights:', [args.r1, args.r2, args.rl])
    pipeline = AbsSelfCriticalPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn, optimizer, grad_fn, reward_func, reward_weight, local_coh_fun, 0.,
                                       accumulate_g_step=args.accumulate_g_step, weights=[args.r1, args.r2, args.rl], bert=bert, multigpu=multigpu, ml_loss=args.ml_loss)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler, val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description='training of the abstractor (RL)'
    )
    parser.add_argument('--path', required=True, help='root of saving the model')

    parser.add_argument('--abs_dir', required=True, help='root of the abs model')
    parser.add_argument('--reward_model_dir', help='root of the reward model')
    parser.add_argument('--reward_data_dir', help='root of the reward data')
    parser.add_argument('--reward_weight', type=float, default=0.05, help='root of the reward data')
    parser.add_argument('--r1', type=float, action='store', default=1/3)
    parser.add_argument('--r2', type=float, action='store', default=1/3)
    parser.add_argument('--rl', type=float, action='store', default=1/3)
    parser.add_argument('--ml_loss', action='store_true')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=1024,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=150,
                        help='maximun words in a single abstract sentence')
    parser.add_argument('--min_abs', type=int, action='store', default=0,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=2,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=0.2,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=30,
                        help='the training batch size')
    parser.add_argument('--accumulate_g_step', type=int, action='store', default=1,
                        help='accumulate gradient')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=6000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=8,
                        help='patience for early stopping')

    # graph info
    parser.add_argument('--docgraph', action='store_true', help='uses gat encoder')
    parser.add_argument('--paragraph', action='store_true', help='encode topic flow')


    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    parser.add_argument('--cloze_gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    #args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    torch.cuda.set_device(args.gpu_id)
    args.cloze_device = 'cuda:' + str(args.cloze_gpu_id)


    args.n_gpu = 1

    if args.docgraph:
        args.key = 'nodes_pruned2'
    else:
        args.key = 'nodes'


    print(args)
    main(args)
