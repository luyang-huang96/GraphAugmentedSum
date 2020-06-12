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

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d[0], d[1]
        word_num = len(' '.join(source_sents).split(' '))
        return source_sents and extracts and word_num > 5
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch


@curry
def tokenize(max_len, texts):
    return [t.strip().lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

def get_bert_align_dict(filename='preprocessing/bertalign-base.pkl'):
    with open(filename, 'rb') as f:
        bert_dict = pickle.load(f)
    return bert_dict

@curry
def prepro_fn_extract_gat_classfcation(tokenizer, align, batch, max_len=1024, stride=256, node_max_len=30):
    assert max_len in [512, 1024, 1536, 2048]
    def prepro_one(sample):
        source_sents, extracts, nodes, edges = sample
        original_order = ' '.join(source_sents).lower().split(' ')
        order_match = {}
        count = 0
        node_target = []
        for i, word in enumerate(original_order):
            order_match[i] = list(range(count, count + align[word]))
            count += align[word]
        tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
        # tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
        # tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
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
        cleaned_extracts = list(filter(lambda e: e < len(truncated_word_num),
                                       extracts))
        tokenized_sents_lists = [tokenized_sents[:BERT_MAX_LEN]]
        length = len(tokenized_sents) - BERT_MAX_LEN
        i = 1
        while length > 0:
            tokenized_sents_lists.append(tokenized_sents[i*BERT_MAX_LEN-stride:(i+1)*BERT_MAX_LEN-stride])
            i += 1
            length -= (BERT_MAX_LEN - stride)


        # find out of range and useless nodes
        other_nodes = set()
        oor_nodes = [] # out of range nodes will not included in the graph
        for _id, content in nodes.items():
            words = [_ for mention in content['content:'] for pos in mention['word_pos'] if pos != -1 and order_match.__contains__(pos) for _ in
                     order_match[pos]]
            words = [word for word in words if word < max_len]
            words = words[:node_max_len]
            if len(words) != 0:
                other_nodes.add(_id)
            else:
                oor_nodes.append(_id)

        activated_nodes = set()
        for _id, content in edges.items():
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']
                new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in order_match[word] if _ < max_len]
                new_words = new_words[:node_max_len]
                if len(new_words) > 0:
                    activated_nodes.add(content['content']['arg1'])
                    activated_nodes.add(content['content']['arg2'])
        oor_nodes.extend(list(other_nodes - activated_nodes))




        # process nodes
        sorted_nodes = sorted(nodes.items(), key=lambda x:int(x[0].split('_')[1]))
        nodewords = []
        id2node = {}
        ii = 0
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:
                words = [_ for mention in content['content:'] for pos in mention['word_pos'] if pos != -1 and order_match.__contains__(pos) for _ in order_match[pos]]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                if len(words) != 0:
                    nodewords.append(words)
                    id2node[_id] = ii
                    ii += 1
                    node_target.append(content['summary_worthy'])
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            #print('warning! no nodes in this sample')
            nodewords = [[0],[2]]
            node_target = [0, 0]
        nodelength = [len(words) for words in nodewords]

        # process edges

        triples = []
        relations = []
        sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))

        ii = 0
        for _id, content in sorted_edges:
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']
                new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in order_match[word] if _ < max_len]
                new_words = new_words[:node_max_len]
                if len(new_words) > 0:
                    node1 = id2node[content['content']['arg1']]
                    node2 = id2node[content['content']['arg2']]
                    edge = int(_id.split('_')[1])
                    triples.append([node1, ii, node2])
                    ii += 1
                    relations.append(new_words)
        if len(relations) == 0:
            #print('warning! no edges in this sample')
            relations = [[1]]
            triples = [[0, 0, 1]]
        rlength = [len(words) for words in relations]
        assert len(node_target) == len(nodewords)
        return tokenized_sents_lists, (cleaned_extracts, truncated_word_num), (nodewords, nodelength, node_target), (relations, rlength, triples)
    batch = list(map(prepro_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_gat_classfcation(tokenizer, batch):
    def convert_one(sample):
        tokenized_sents_lists, (extracts, word_num), (nodes, nlength, node_target), (relations, rlength, triples) = sample
        id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(word_num))
        return id_sents, extracts, word_num, nodes, nlength, node_target, relations, rlength, triples
    batch = list(map(convert_one, batch))
    return batch

def make_adj(triples, dim1, dim2, cuda=True):
    adj = torch.zeros(dim1, dim2).cuda() if cuda else torch.zeros(dim1, dim2)
    for i,j,k in triples:
        adj[i, j] = 1
        adj[k, j] = 1
    return adj


@curry
def batchify_fn_extract_ptr_gat_classfcation(pad, data, cuda=True):
    source_lists, targets, word_nums, nodes, nlength, node_target, relations, rlength, triples = tuple(map(list, unzip(data)))
    adjs = [make_adj(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]

    src_nums = list(map(len, word_nums))
    word_nums = list(word_nums)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    source_lists = [source for source_list in source_lists for source in source_list]
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()


    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    node_target = pad_batch_tensorize(node_target, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )


    fw_args = (src_nums, tar_in, (sources, word_nums), (_nodes, nmask, node_num), (_relations, rmask, triples, adjs))
    loss_args = (node_target, )
    return fw_args, loss_args

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
def batchify_fn_extract_ptr_gat_multitask(pad, data, cuda=True):
    source_lists, targets, word_nums, nodes, nlength, node_target, relations, rlength, triples = tuple(map(list, unzip(data)))
    adjs = [make_adj(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]

    src_nums = list(map(len, word_nums))
    word_nums = list(word_nums)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    source_lists = [source for source_list in source_lists for source in source_list]
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()


    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    node_target = pad_batch_tensorize(node_target, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )


    fw_args = (src_nums, tar_in, (sources, word_nums), (_nodes, nmask, node_num), (_relations, rmask, triples, adjs))
    loss_args = (target, node_target)
    return fw_args, loss_args