""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from utils import PAD, UNK, START, END
import pickle

BERT_MAX_LEN = 512
MAX_FREQ = 100

# Batching functions
def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets

@curry
def coll_fn_graph(data):
    source_lists, target_lists, nodes, edges, subgraphs, paras = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, source_lists))
    #sources = source_lists
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets, nodes, edges, subgraphs, paras

@curry
def coll_fn_graph_rl(data, max_node=200):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, targets, nodes, questions = d[0], d[1], d[2], d[6]
        return (source_sents and targets) and len(nodes) < max_node and questions
    data = list(filter(is_good_data, data))
    source_lists, target_lists, nodes, edges, subgraphs, paras, questions = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, source_lists))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets, nodes, edges, subgraphs, paras, questions

@curry
def tokenize(max_len, texts):
    return [t.strip().lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def prepro_fn_copy_bert(tokenizer, max_src_len, max_tgt_len, batch):
    assert max_src_len in [512, 768, 1024, 1536, 2048]
    sources, targets = batch
    sources = [[tokenizer.bos_token] + tokenizer.tokenize(source)[:max_src_len-2] + [tokenizer.eos_token] for source in sources]
    targets = [tokenizer.tokenize(target)[:max_tgt_len] for target in targets]
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy_bert(tokenizer, max_src_len, batch):
    stride = 256
    word2id = tokenizer.encoder
    unk = word2id[tokenizer._unk_token]
    sources, targets = map(list, unzip(batch))
    src_length = [len(src) for src in sources]
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    if max_src_len > BERT_MAX_LEN:
        new_sources = []
        for source in sources:
            if len(source) < BERT_MAX_LEN:
                new_sources.append(source)
            else:
                new_sources.append(source[:BERT_MAX_LEN])
                length = len(source) - BERT_MAX_LEN
                i = 1
                while length > 0:
                    new_sources.append(source[i * stride:i * stride + BERT_MAX_LEN])
                    i += 1
                    length -= (BERT_MAX_LEN - stride)
        sources = new_sources
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = [sources, list(zip(src_exts, tar_ins, targets, src_length))]
    return batch

@curry
def batchify_fn_copy_bert(tokenizer, data, cuda=True):
    start = tokenizer.encoder[tokenizer._bos_token]
    end = tokenizer.encoder[tokenizer._eos_token]
    pad = tokenizer.encoder[tokenizer._pad_token]
    sources, ext_srcs, tar_ins, targets, src_lens = (data[0], ) + tuple(map(list, unzip(data[1])))

    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]
    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)
    ext_vsize = ext_src.max().item() + 1
    if ext_vsize < len(tokenizer.encoder):
        ext_vsize = len(tokenizer.encoder)
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


def get_bert_align_dict(filename='preprocessing/bertalign-base.pkl'):
    with open(filename, 'rb') as f:
        bert_dict = pickle.load(f)
    return bert_dict

@curry
def subgraph_make_adj_edge_in(subgraphs, cuda=True):
    subgraph_triples, subgraph_node_lists = subgraphs
    adjs = []
    for triples, node_lists in zip(subgraph_triples, subgraph_node_lists):
        if len(node_lists) != 0:
            adj = make_adj_edge_in(triples, len(node_lists), len(triples), cuda=cuda)
        else:
            adj = []
        adjs.append(adj)
    return adjs

@curry
def subgraph_make_adj(subgraphs, cuda=True):
    subgraph_triples, subgraph_node_lists = subgraphs
    adjs = []
    for triples, node_lists in zip(subgraph_triples, subgraph_node_lists):
        if len(node_lists) != 0:
            adj = make_adj(triples, len(node_lists), len(node_lists), cuda=cuda)
        else:
            adj = []
        adjs.append(adj)
    return adjs

def create_word_freq_in_para_feat(paras, tokenized_sents, tokenized_article):
    sent_align_para = []
    last_idx = 0
    for sent in range(len(tokenized_sents)):
        flag = False
        for _idx, para in enumerate(paras):
            if sent in para:
                sent_align_para.append(_idx)
                last_idx = _idx
                flag = True
                break
        if not flag:
            sent_align_para.append(last_idx)
    word_count = {}
    for word in tokenized_article:
        try:
            word_count[word] += 1
        except KeyError:
            word_count[word] = 1
    word_inpara_count = {}
    for word in list(set(tokenized_article)):
        count = 0
        for sent in tokenized_sents:
            if word in sent:
                count += 1
        word_inpara_count[word] = count

    # sent_freq_feat = [[word_count[word] for word in sent] for sent in tokenized_sents]
    article_freq_feat = [word_count[word] if word_count[word] < MAX_FREQ-1 else MAX_FREQ-1 for word in tokenized_article]
    article_inpara_freq_feat = [word_inpara_count[word] if word_inpara_count[word] < MAX_FREQ-1 else MAX_FREQ-1 for word in tokenized_article]
    sent_freq_feat = [[word_count[word] if word_count[word] < MAX_FREQ-1 else MAX_FREQ-1 for word in sent] for sent in tokenized_sents]
    sent_inpara_freq_feat = [[word_inpara_count[word] if word_inpara_count[word] < MAX_FREQ-1 else MAX_FREQ-1 for word in sent] for sent in tokenized_sents]

    return article_freq_feat, article_inpara_freq_feat, sent_freq_feat, sent_inpara_freq_feat

def create_sent_node_align(nodeinsents, sent_len):
    sent_node_aligns = [[] for _ in range(sent_len)]
    for _nid, nodeinsent in enumerate(nodeinsents):
        for _sid in nodeinsent:
            if _sid < sent_len:
                sent_node_aligns[_sid].append(_nid)
    sent_node_aligns = [list(set(sent_node_align)) if len(sent_node_align)>0 else [len(nodeinsents)] for sent_node_align in sent_node_aligns]
    sent_node_aligns.append([len(nodeinsents)])
    return sent_node_aligns

@curry
def preproc(tokenized_sents, clusters):
    pad = 0
    split_token = '<split>'
    cluster_words = []
    cluster_wpos = []
    cluster_spos = []
    for cluster in clusters:
        scluster_word = []
        scluster_wpos = []
        scluster_spos = []
        for mention in cluster:
            if len(mention['text'].strip().split(' ')) == len(
                    list(range(mention['position'][3] + 1, mention['position'][4] + 1))):
                scluster_word += mention['text'].lower().strip().split(' ')
                scluster_wpos += list(range(mention['position'][3] + 1, mention['position'][4] + 1))
                scluster_spos += [mention['position'][0] + 1 for _ in
                                  range(len(mention['text'].strip().split(' ')))]
                scluster_word.append(split_token)
                scluster_wpos.append(pad)
                scluster_spos.append(pad)
            else:
                sent_num = mention['position'][0]
                word_start = mention['position'][3]
                word_end = mention['position'][4]
                # if word_end > 99:
                #     word_end = 99
                scluster_word += tokenized_sents[sent_num][word_start:word_end]
                scluster_wpos += list(range(word_start, word_end))
                scluster_spos += [mention['position'][0] + 1 for _ in
                                  range(word_start + 1, word_end + 1)]
                scluster_word.append(split_token)
                scluster_wpos.append(pad)
                scluster_spos.append(pad)
        if scluster_word != []:
            scluster_word.pop()
            scluster_wpos.pop()
            scluster_spos.pop()
            cluster_words.append(scluster_word)
            cluster_wpos.append(scluster_wpos)
            cluster_spos.append(scluster_spos)
            if len(scluster_word) != len(scluster_wpos):
                continue
                # print(scluster_word)
                # print(scluster_wpos)
                # print('cluster:', cluster)
            assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)
    clusters = (cluster_words, cluster_wpos, cluster_spos)

    return clusters


@curry
def convert_batch(unk, word2id, batch):
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy(unk, word2id, batch):
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return batch

def make_adj_triple(triples, dim1, dim2, cuda=True):
    adj = torch.zeros(dim1, dim2).cuda() if cuda else torch.zeros(dim1, dim2)
    for i,j,k in triples:
        adj[i, j] = 1
        adj[k, j] = 1

    return adj

def make_adj_edge_in(triples, dim, edge_num, cuda=True):
    adj = torch.zeros(dim, dim).cuda() if cuda else torch.zeros(dim, dim)
    node_num = dim - edge_num
    for i,j,k in triples:
        adj[i, k] = 1
        adj[k, i] = 1
        adj[i, node_num+j] = 1
        adj[k, node_num+j] = 1
        adj[node_num+j, i] = 1
        adj[node_num+j, k] = 1
    for i in range(dim):
        adj[i, i] = 1
    return adj

def make_adj_edge_in_bidirectional(triples, dim, edge_num, cuda=True):
    adj_in = torch.zeros(dim, dim).cuda() if cuda else torch.zeros(dim, dim)
    adj_out = torch.zeros(dim, dim).cuda() if cuda else torch.zeros(dim, dim)
    node_num = dim - edge_num
    for i,j,k in triples:
        adj_in[i, k] = 1
        adj_out[k, i] = 1
        adj_in[i, node_num+j] = 1
        adj_out[k, node_num+j] = 1
        adj_in[node_num+j, i] = 1
        adj_out[node_num+j, k] = 1
    for i in range(dim):
        adj_in[i, i] = 1
        adj_out[i, i] = 1

    return (adj_in, adj_out)

def make_adj(triples, dim1, dim2, cuda=True):
    assert dim1 == dim2
    adj = torch.zeros(dim1, dim2).cuda() if cuda else torch.zeros(dim1, dim2)
    for i,j,k in triples:
        adj[i, k] = 1
        adj[k, i] = 1
    for i in range(dim1):
        adj[i, i] = 1
    return adj

def make_adj_bidirectional(triples, dim1, dim2, cuda=True):
    assert dim1 == dim2
    adj_in = torch.zeros(dim1, dim2).cuda() if cuda else torch.zeros(dim1, dim2)
    adj_out = torch.zeros(dim1, dim2).cuda() if cuda else torch.zeros(dim1, dim2)
    for i,j,k in triples:
        adj_in[i, k] = 1
        adj_out[k, i] = 1
    for i in range(dim1):
        adj_in[i, i] = 1
        adj_out[i, i] = 1
    return (adj_in, adj_out)

def normalize_adjs(adjs):
    d = adjs.sum(1, keepdim=True)
    d[d == 0] = 1e-8
    adjs = adjs / d
    return adjs


@curry
def pad_batch_tensorize(inputs, pad, cuda=True, max_num=0):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    try:
        max_len = max(len(ids) for ids in inputs)
    except ValueError:
        # print('inputs:', inputs)
        # print('batch_size:', batch_size)
        if inputs == []:
            max_len = 1
            batch_size = 1
    if max_len < max_num:
        max_len = max_num
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

@curry
def pad_batch_tensorize_3d(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len_1 = max([len(_input) for _input in inputs])
    max_len_2 = max([len(_in) for _input in inputs for _in in _input])
    tensor_shape = (batch_size, max_len_1, max_len_2)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        for j, _in in enumerate(ids):
            tensor[i, j, :len(_in)] = tensor_type(_in)
    return tensor

@curry
def pad_batch_tensorize_adjs(adjs):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    #tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if len(adjs[0]) == 2:
        batch_size = len(adjs)
        max_len_1 = max([adj_in.size(0) for adj_in, adj_out in adjs])
        max_len_2 = max([adj_in.size(1) for adj_in, adj_out in adjs])
        tensor = torch.zeros(batch_size, max_len_1, max_len_2, 2).to(adjs[0][0].device)
        #tensor.fill_(pad)
        for i, (adj_in, adj_out) in enumerate(adjs):
            size1 = adj_in.size(0)
            size2 = adj_in.size(1)
            tensor[i, :size1, :size2, 0] = adj_in
            tensor[i, :size1, :size2, 1] = adj_out
    elif len(adjs[0]) == 1:
        batch_size = len(adjs)
        max_len_1 = max([adj_in[0].size(0) for adj_in in adjs])
        max_len_2 = max([adj_in[0].size(1) for adj_in in adjs])
        tensor = torch.zeros(batch_size, max_len_1, max_len_2).to(adjs[0][0].device)
        # tensor.fill_(pad)
        for i, adj_in in enumerate(adjs):
            size1 = adj_in[0].size(0)
            size2 = adj_in[0].size(1)
            tensor[i, :size1, :size2] = adj_in[0]
    else:
        raise Exception('dimension error')
    return tensor.view(batch_size, max_len_1, -1).contiguous()



@curry
def double_pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize double pad and turn it to one hot vector

    :param inputs: List of List of size B containing torch tensors of shape [[T, ...],]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    #tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batch_size = len(inputs)
    max_sent_num = max([len(labels) for labels in inputs])
    max_side_num = max([labels[-1][0] for labels in inputs]) + 1
    tensor_shape = (batch_size, max_sent_num, max_side_num)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    if pad < 0:
        for batch_id, labels in enumerate(inputs):
            for sent_id, label in enumerate(labels):
                tensor[batch_id, sent_id, :] = 0
                for label_id in label:
                    tensor[batch_id, sent_id, label_id] = 1
    else:
        for batch_id, labels in enumerate(inputs):
            for sent_id, label in enumerate(labels):
                for label_id in label:
                    tensor[batch_id, sent_id, label_id] = 1
    return tensor

@curry
def batchify_fn(pad, start, end, data, cuda=True):
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    tar_ins = [[start] + tgt for tgt in targets]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def convert_batch_copy_rl(unk, word2id, batch):
    raw_sources, raw_targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in raw_sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, raw_sources)
    sources = conver2id(unk, word2id, raw_sources)
    tar_ins = conver2id(unk, word2id, raw_targets)
    targets = conver2id(unk, ext_word2id, raw_targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return (batch, ext_word2id, raw_sources, raw_targets)

@curry
def batchify_fn_copy_rl(pad, start, end, data, cuda=True):
    batch, ext_word2id, raw_articles, raw_targets = data
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(batch)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    extend_vsize = len(ext_word2id)
    ext_id2word = {_id:_word for _word, _id in ext_word2id.items()}
    #print('ext_size:', ext_vsize, extend_vsize)
    fw_args = (source, src_lens, ext_src, extend_vsize,
               START, END, UNK, 100)
    loss_args = (raw_articles, ext_id2word, raw_targets)
    return fw_args, loss_args

@curry
def convert_batch_copy_rl_bert(tokenizer, max_src_len, batch):
    stride = 256
    word2id = tokenizer.encoder
    unk = word2id[tokenizer._unk_token]
    raw_sources, raw_targets = map(list, unzip(batch))
    src_length = [len(src) for src in raw_sources]
    ext_word2id = dict(word2id)
    for source in raw_sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, raw_sources)
    if max_src_len > BERT_MAX_LEN:
        new_sources = []
        for source in raw_sources:
            if len(source) < BERT_MAX_LEN:
                new_sources.append(source)
            else:
                new_sources.append(source[:BERT_MAX_LEN])
                length = len(source) - BERT_MAX_LEN
                i = 1
                while length > 0:
                    new_sources.append(source[i * stride:i * stride + BERT_MAX_LEN])
                    i += 1
                    length -= (BERT_MAX_LEN - stride)
        sources = new_sources
    else:
        sources = raw_sources
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, raw_targets)
    targets = conver2id(unk, ext_word2id, raw_targets)
    batch = [sources, list(zip(src_exts, tar_ins, targets, src_length))]
    return (batch, ext_word2id, raw_sources, raw_targets)

@curry
def batchify_fn_copy_rl_bert(tokenizer, data, cuda=True, min_len=0):
    start = tokenizer.encoder[tokenizer._bos_token]
    end = tokenizer.encoder[tokenizer._eos_token]
    pad = tokenizer.encoder[tokenizer._pad_token]
    unk = tokenizer.encoder[tokenizer._unk_token]
    batch, ext_word2id, raw_articles, raw_targets = data
    sources, ext_srcs, tar_ins, targets, src_lens = (batch[0],) + tuple(map(list, unzip(batch[1])))

    #src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)
    questions = None

    extend_vsize = len(ext_word2id)
    ext_id2word = {_id:_word for _word, _id in ext_word2id.items()}
    #print('ext_size:', ext_vsize, extend_vsize)
    fw_args = (source, src_lens, ext_src, extend_vsize,
               start, end, unk, 150, min_len, tar_ins)
    loss_args = (raw_articles, ext_id2word, raw_targets, questions, targets)
    return fw_args, loss_args

def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                print('length loader:', len(self._loader))
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i), end=' ')

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
