import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from utils import PAD, UNK, START, END
import pickle
from data.batcher import pad_batch_tensorize, pad_batch_tensorize_3d
from data.batcher import make_adj_triple, make_adj, make_adj_edge_in
from data.RLbatcher import process_nodes, process_subgraphs, process_nodes_bert
from data.ExtractBatcher import subgraph_make_adj_edge_in, subgraph_make_adj

MAX_FREQ = 100
BERT_MAX_LEN = 512


@curry
def tokenize(max_len, texts):
    return [t.strip().lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def coll_fn_gat(data, max_node_num=200):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, targets, nodes = d[0], d[1], d[2]
        word_num = len(' '.join(source_sents).split(' '))
        target_num = len(' '.join(targets).split(' '))
        return source_sents and targets and word_num > 5 and len(d[2]) < max_node_num and target_num > 4
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

@curry
def create_word_freq_in_para_feat(paras, tokenized_sents, max_src_len):
    para_num = len(paras)
    if para_num == 0:
        #print('paras:', paras)
        return [1 for word in list(concat(tokenized_sents))][:max_src_len]

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


    tokenized_article = list(concat(tokenized_sents))
    tokenized_paras = [[] for _ in range(para_num)]
    for _sid, sent in enumerate(tokenized_sents):
        tokenized_paras[sent_align_para[_sid]].extend(sent)
    word_inpara_count = {}
    for word in list(set(tokenized_article)):
        count = 0
        for para in tokenized_paras:
            if word in para:
                count += 1
        word_inpara_count[word] = count
    article_inpara_freq_feat = [word_inpara_count[word] if word_inpara_count[word] < MAX_FREQ-1 else MAX_FREQ-1 for word in
                         tokenized_article][:max_src_len]

    return article_inpara_freq_feat


def make_node_lists(subgraphs, paras, oor_nodes, id2edge, id2node, max_sent):
    node_lists = []
    edge_lists = []
    triples = []
    for _sgid, subgraph in enumerate(subgraphs):
        try:
            paraid = paras[_sgid][0]
        except:
            paraid = 0
        if type(paraid) != type(max_sent):
            paraid = 0
        if paraid > max_sent - 1:
            continue
        if subgraph == []:
            node_lists.append([])
            triples.append([])
            edge_lists.append([])
        else:
            node_list = set()
            triple = []
            edge_list = []
            eidx = []
            for _triple in subgraph:
                if _triple[0] not in oor_nodes and _triple[2] not in oor_nodes and id2edge.__contains__(_triple[1]):
                    node_list.add(id2node[_triple[0]])
                    node_list.add(id2node[_triple[2]])
                    eidx.append(_triple[1])
            node_list = list(sorted(node_list))
            for _triple in subgraph:
                if _triple[0] not in oor_nodes and _triple[2] not in oor_nodes and id2edge.__contains__(_triple[1]):
                    idx1 = node_list.index(id2node[_triple[0]])
                    idx2 = node_list.index(id2node[_triple[2]])
                    _idxe = id2edge[_triple[1]]
                    idxe_in_subgraph = eidx.index(_triple[1])
                    edge_list.append(_idxe)
                    triple.append([idx1, idxe_in_subgraph, idx2])
            triples.append(triple)
            node_lists.append(node_list)
            edge_lists.append(edge_list)

    if len(node_lists) == 0:
        node_lists.append([])
        triples.append([])
        edge_lists.append([])

    return node_lists, edge_lists, triples

@curry
def count_max_sent(source_sent, max_source_num):
    count = 0
    max_sent = len(source_sent)
    for _sid, sent in enumerate(source_sent):
        count += len(sent)
        if count > max_source_num:
            max_sent = _sid + 1
            break
    return max_sent

@curry
def prepro_fn_gat(max_src_len, max_tgt_len, batch, node_max_len=30, key='summary_worthy', adj_type='edge_as_node', docgraph=True):
    # sources, targets, nodes, edges = batch
    # sources = tokenize(max_src_len, sources)
    # targets = tokenize(max_tgt_len, targets)
    # batch = list(zip(sources, targets))
    def prepro_one(sample):
        source, target, nodes, edges, subgraphs, paras = sample
        source_sent = [sent.strip().lower().split() for sent in source]
        source = ' '.join(source).strip().lower().split()[:max_src_len]
        target = ' '.join(target).strip().lower().split()[:max_tgt_len]

        word_freq_feat = create_word_freq_in_para_feat(paras, source_sent, max_src_len)
        assert len(source) == len(word_freq_feat)


        max_len = max_src_len
        # find out of range and useless nodes
        other_nodes = set()
        oor_nodes = []  # out of range nodes will not included in the graph
        for _id, content in nodes.items():
            words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
            words = [word for word in words if word < max_len]
            if len(words) != 0:
                other_nodes.add(_id)
            else:
                oor_nodes.append(_id)

        activated_nodes = set()
        for _id, content in edges.items():
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']
                new_words = [word for word in words if word > -1 and word < max_len]
                if len(new_words) > 0:
                    activated_nodes.add(content['content']['arg1'])
                    activated_nodes.add(content['content']['arg2'])
        oor_nodes.extend(list(other_nodes - activated_nodes))

        # process nodes
        sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
        sum_worthy = []
        nodefreq = []
        nodewords = []
        id2node = {}
        ii = 0
        extracted_labels = []
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:

                words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                # sum_worthy.append(content['InSalientSent'])

                if len(words) != 0:
                    nodewords.append(words)
                    nodefreq.append(len(content['content']))
                    try:
                        sum_worthy.append(content[key])
                    except KeyError:
                        sum_worthy.append(0)
                    id2node[_id] = ii
                    ii += 1
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            # print('warning! no nodes in this sample')
            nodewords = [[0], [2]]
            nodefreq.extend([1, 1])
            sum_worthy.extend([0, 0])
        nodelength = [len(words) for words in nodewords]
        nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]

        # process edges
        acticated_nodes = set()

        triples = []
        edge_freq = []
        relations = []
        sum_worthy_edges = []
        id2edge = {}
        sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))

        ii = 0
        for _id, content in sorted_edges:
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']

                new_words = [word for word in words if word > -1 and word < max_len]
                new_words = new_words[:node_max_len]
                if len(new_words) > 0:
                    node1 = id2node[content['content']['arg1']]
                    node2 = id2node[content['content']['arg2']]
                    edge = int(_id.split('_')[1])
                    edge_freq.append(1)
                    try:
                        sum_worthy_edges.append(content[key])
                    except KeyError:
                        sum_worthy_edges.append(0)
                    triples.append([node1, ii, node2])
                    if adj_type == 'edge_up':
                        nodewords[node1].extend(new_words)
                    elif adj_type == 'edge_down':
                        nodewords[node2].extend(new_words)
                    acticated_nodes.add(content['content']['arg1'])
                    acticated_nodes.add(content['content']['arg2'])
                    id2edge[_id] = ii
                    ii += 1
                    relations.append(new_words)
        if len(relations) == 0:
            # print('warning! no edges in this sample')
            relations = [[1]]
            edge_freq = [1]
            triples = [[0, 0, 1]]
            sum_worthy_edges.append(0)
        rlength = [len(words) for words in relations]

        if not docgraph:
            max_sent = count_max_sent(source_sent, max_src_len)
            node_lists, edge_lists, triples = make_node_lists(subgraphs, paras, oor_nodes, id2edge, id2node, max_sent)
            if adj_type == 'edge_as_node':
                node_num = len(nodewords)
                for i in range(len(triples)):
                    node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]


        if adj_type == 'edge_as_node':
            nodewords = nodewords + relations
            nodelength = nodelength + rlength
            nodefreq = nodefreq + edge_freq
            sum_worthy = sum_worthy + sum_worthy_edges
        if docgraph:
            return source, target, (nodewords, nodelength, sum_worthy, word_freq_feat, nodefreq), (relations, rlength, triples)
        else:
            return source, target, (nodewords, node_lists, sum_worthy, word_freq_feat, nodefreq), (relations, rlength, triples)

    batch = list(map(prepro_one, batch))
    return batch

@curry
def prepro_fn_gat_bert(tokenizer, align, max_src_len, max_tgt_len, batch, node_max_len=30,
                       key='summary_worthy', adj_type='edge_as_node', docgraph=True, stride=256):
    # sources, targets, nodes, edges = batch
    # sources = tokenize(max_src_len, sources)
    # targets = tokenize(max_tgt_len, targets)
    # batch = list(zip(sources, targets))
    def prepro_one(sample):
        source, target, nodes, edges, subgraphs, paras = sample
        #source_sent = [sent.strip().split() for sent in source]
        source_sent_tokenized = [tokenizer.tokenize(sent) for sent in source]
        source_sent = source
        source = ' '.join(source).strip()
        target = ' '.join(target).strip()
        source = [tokenizer.bos_token] + tokenizer.tokenize(source)[:max_src_len - 2] + [tokenizer.eos_token]
        target = tokenizer.tokenize(target)[:max_tgt_len] # will add start and end later

        #original_order = ' '.join(source_sent).split(' ')
        # test_order_match = {}
        order_match = {}
        count = 1
        i = 0
        for sents in [' '.join(source_sent)]:
            sent_words = sents.split(' ')
            if len(sent_words) > 0:
                order_match[i] = list(range(count, count + align[sent_words[0]]))
                count += align[sent_words[0]]
                i += 1
                for word in sent_words[1:]:
                    new_word = ' ' + word # if use bpe
                    #new_word = word
                    order_match[i] = list(range(count, count + align[new_word]))
                    # test_order_match[new_word] = [count, count + align[new_word]]
                    count += align[new_word]
                    i += 1


        # source_lists = [source[:BERT_MAX_LEN]]
        # length = len(source) - BERT_MAX_LEN
        # i = 1
        # while length > 0:
        #     source_lists.append(source[i * BERT_MAX_LEN - stride:(i + 1) * BERT_MAX_LEN - stride])
        #     i += 1
        #     length -= (BERT_MAX_LEN - stride)


        #word_freq_feat = create_word_freq_in_para_feat(paras, source_sent, max_src_len)
        #assert len(source) == len(word_freq_feat)

        max_len = len(source) - 1
        #max_len = max_src_len - 1
        # find out of range and useless nodes
        other_nodes = set()
        oor_nodes = []  # out of range nodes will not included in the graph
        for _id, content in nodes.items():
            words = [_ for mention in content['content'] for pos in mention['word_pos'] if
                     pos != -1 and order_match.__contains__(pos) for _ in
                     order_match[pos]]
            words = [word for word in words if word < max_len]
            if len(words) != 0:
                other_nodes.add(_id)
            else:
                oor_nodes.append(_id)

        activated_nodes = set()
        for _id, content in edges.items():
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']
                new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in
                             order_match[word] if _ < max_len]
                if len(new_words) > 0:
                    activated_nodes.add(content['content']['arg1'])
                    activated_nodes.add(content['content']['arg2'])
        oor_nodes.extend(list(other_nodes - activated_nodes))

        # process nodes
        sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
        sum_worthy = []
        nodefreq = []
        nodewords = []
        id2node = {}
        ii = 0
        extracted_labels = []
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:
                words = [_ for mention in content['content'] for pos in mention['word_pos'] if
                         pos != -1 and order_match.__contains__(pos) for _ in order_match[pos]]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                # sum_worthy.append(content['InSalientSent'])

                if len(words) != 0:
                    nodewords.append(words)
                    nodefreq.append(len(content['content']))
                    try:
                        sum_worthy.append(content[key])
                    except KeyError:
                        sum_worthy.append(0)
                    id2node[_id] = ii
                    ii += 1
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            # print('warning! no nodes in this sample')
            nodewords = [[0], [2]]
            nodefreq.extend([1, 1])
            sum_worthy.extend([0, 0])
        nodelength = [len(words) for words in nodewords]
        nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]

        # process edges
        activated_nodes = set()

        triples = []
        edge_freq = []
        relations = []
        sum_worthy_edges = []
        id2edge = {}
        sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))

        ii = 0
        for _id, content in sorted_edges:
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']
                new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in
                             order_match[word] if _ < max_len]
                new_words = new_words[:node_max_len]
                if len(new_words) > 0:
                    node1 = id2node[content['content']['arg1']]
                    node2 = id2node[content['content']['arg2']]
                    edge = int(_id.split('_')[1])
                    edge_freq.append(1)
                    try:
                        sum_worthy_edges.append(content[key])
                    except KeyError:
                        sum_worthy_edges.append(0)
                    triples.append([node1, ii, node2])
                    if adj_type == 'edge_up':
                        nodewords[node1].extend(new_words)
                    elif adj_type == 'edge_down':
                        nodewords[node2].extend(new_words)
                    activated_nodes.add(content['content']['arg1'])
                    activated_nodes.add(content['content']['arg2'])
                    id2edge[_id] = ii
                    ii += 1
                    relations.append(new_words)
        if len(relations) == 0:
            # print('warning! no edges in this sample')
            relations = [[1]]
            edge_freq = [1]
            triples = [[0, 0, 1]]
            sum_worthy_edges.append(0)
        rlength = [len(words) for words in relations]

        if not docgraph:
            max_sent = count_max_sent(source_sent_tokenized, max_src_len-2)
            node_lists, edge_lists, triples = make_node_lists(subgraphs, paras, oor_nodes, id2edge, id2node, max_sent)
            if adj_type == 'edge_as_node':
                node_num = len(nodewords)
                for i in range(len(triples)):
                    node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]


        if adj_type == 'edge_as_node':
            nodewords = nodewords + relations
            nodelength = nodelength + rlength
            nodefreq = nodefreq + edge_freq
            sum_worthy = sum_worthy + sum_worthy_edges
        if docgraph:
            return source, target, (nodewords, nodelength, sum_worthy, nodefreq), (relations, rlength, triples)
        else:
            return source, target, (nodewords, node_lists, sum_worthy, nodefreq), (relations, rlength, triples)

    batch = list(map(prepro_one, batch))
    return batch

@curry
def convert_batch_gat_bert(tokenizer, max_src_len, batch):
    stride = 256
    word2id = tokenizer.encoder
    unk = word2id[tokenizer._unk_token]
    sources, targets, node_infos, edge_infos = list(map(list, unzip(batch)))
    nodewords, nodelengths, sum_worhies, nodefreq = list(unzip(node_infos))
    relations, rlengths, triples = list(unzip(edge_infos))
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
    batch = [sources, list(zip(src_exts, tar_ins, targets, nodewords, nodelengths, sum_worhies, nodefreq, relations, rlengths, triples, src_length))]
    return batch

@curry
def batchify_fn_gat_bert(tokenizer, data, cuda=True,
                     adj_type='concat_triple', mask_type='none', docgraph=True):
    sources, ext_srcs, tar_ins, targets, \
    nodes, nodelengths, sum_worthy, nodefreq, relations, rlengths, triples, src_lens = (data[0], ) + tuple(map(list, unzip(data[1])))
    start = tokenizer.encoder[tokenizer._bos_token]
    end = tokenizer.encoder[tokenizer._eos_token]
    pad = tokenizer.encoder[tokenizer._pad_token]


    if not docgraph:
        node_lists = nodelengths
        if adj_type == 'edge_as_node':
            adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
        else:
            adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))
    else:
        if adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        elif adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        else:
            adjs = [make_adj(triple, len(node), len(node), cuda) for triple, node, relation in zip(triples, nodes, relations)]

    #src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)
    feature_dict = {'node_freq': nodefreq}

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    sum_worthy_label = pad_batch_tensorize(sum_worthy, pad=-1, cuda=cuda)
    sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=cuda).float()

    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()


    ext_vsize = ext_src.max().item() + 1
    if docgraph:
        fw_args = (source, src_lens, tar_in, ext_src, ext_vsize, (_nodes, nmask, node_num, sum_worthy, feature_dict),
                   (_relations, rmask, triples, adjs))
    else:
        fw_args = (source, src_lens, tar_in, ext_src, ext_vsize, (_nodes, nmask, node_num, sum_worthy, feature_dict, node_lists),
                   (_relations, rmask, triples, adjs))

    if 'soft' in mask_type:
        loss_args = (target, sum_worthy_label)
    else:
        loss_args = (target, )
    return fw_args, loss_args

@curry
def convert_batch_gat(unk, word2id, batch):
    sources, targets, node_infos, edge_infos = list(map(list, unzip(batch)))
    nodewords, nodelengths, sum_worhies, word_freq_feat, nodefreq = list(unzip(node_infos))
    relations, rlengths, triples = list(unzip(edge_infos))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets, nodewords, nodelengths, sum_worhies, word_freq_feat, nodefreq, relations, rlengths, triples))
    return batch


@curry
def batchify_fn_gat(pad, start, end, data, cuda=True,
                     adj_type='concat_triple', mask_type='none', decoder_supervision=False, docgraph=True):
    sources, ext_srcs, tar_ins, targets, \
    nodes, nodelengths, sum_worthy, word_freq_feat, nodefreq, relations, rlengths, triples = tuple(map(list, unzip(data)))
    if not docgraph:
        node_lists = nodelengths
        if adj_type == 'edge_as_node':
            adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
        else:
            adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))
    else:
        if adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        elif adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        else:
            adjs = [make_adj(triple, len(node), len(node), cuda) for triple, node, relation in zip(triples, nodes, relations)]

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)
    word_freq = pad_batch_tensorize(word_freq_feat, pad=pad, cuda=cuda)
    feature_dict = {'word_inpara_freq': word_freq,
                    'node_freq': nodefreq}

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    sum_worthy_label = pad_batch_tensorize(sum_worthy, pad=-1, cuda=cuda)
    sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=cuda).float()

    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()



    ext_vsize = ext_src.max().item() + 1
    if docgraph:
        fw_args = (source, src_lens, tar_in, ext_src, ext_vsize, (_nodes, nmask, node_num, sum_worthy, feature_dict),
                   (_relations, rmask, triples, adjs))
    else:
        fw_args = (source, src_lens, tar_in, ext_src, ext_vsize, (_nodes, nmask, node_num, sum_worthy, feature_dict, node_lists),
                   (_relations, rmask, triples, adjs))
    if 'soft' in mask_type and decoder_supervision:
        raise Exception('not implemented yet')
        #loss_args = (target, sum_worthy_label, extracted_labels)
    elif 'soft' in mask_type:
        loss_args = (target, sum_worthy_label)
    elif decoder_supervision:
        raise Exception('not implemented yet')
        #loss_args = (target, extracted_labels)
    else:
        loss_args = (target, )
    return fw_args, loss_args


@curry
def convert_batch_gat_copy_from_graph(unk, word2id, batch):
    sources, targets, node_infos, edge_infos = list(map(list, unzip(batch)))
    nodewords, nodelengths, sum_worhies = list(unzip(node_infos))
    nodewords = [list(nodeword) for nodeword in nodewords]
    all_node_words = [list(concat(nodeword)) for nodeword in nodewords] # position in article
    sum_worhies = [list(sum_worhy) for sum_worhy in sum_worhies]
    gold_copy_masks = []
    ext_node_aligns = []
    for _bid, nodeword in enumerate(list(nodewords)):
        ext_node_align = []
        gold_mask = []
        for _i, words in enumerate(nodeword):
            align = [_i for _ in range(len(words))]
            ext_node_align.extend(align)
            _mask = [1 if sum_worhies[_bid][_i] else 0 for _ in range(len(words))]
            gold_mask.extend(_mask)
        gold_copy_masks.append(gold_mask)

        ext_node_aligns.append(ext_node_align)

    relations, rlengths, triples = list(unzip(edge_infos))
    ext_word2id = dict(word2id)
    for _bid, words in enumerate(all_node_words):
        for word in words:
            original_word = sources[_bid][word]
            if original_word not in ext_word2id:
                ext_word2id[original_word] = len(ext_word2id)
    # for source in sources:
    #     for word in source:
    #         if word not in ext_word2id:
    #             ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    node_exts = []
    for _i, words in enumerate(all_node_words):
        node_ext = []
        for word in words:
            node_ext.append(src_exts[_i][word])
        node_exts.append(node_ext)


    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(
        zip(sources, node_exts, tar_ins, targets, nodewords, nodelengths, sum_worhies, relations, rlengths, triples,
            all_node_words, ext_node_aligns, gold_copy_masks))
    return batch




@curry
def prepro_graph(max_src_len, max_tgt_len, adj_type, batch, docgraph=True, reward_data_dir=None):
    if reward_data_dir is not None:
        sources, targets, nodes, edges, subgraphs, paras, questions = batch
    else:
        sources, targets, nodes, edges, subgraphs, paras = batch
    tokenized_sents = list(map(tokenize(None), sources))
    sources = tokenize(max_src_len, [' '.join(source) for source in sources])
    targets = tokenize(max_tgt_len, targets)

    paras = list(paras)

    word_freq_feats = [create_word_freq_in_para_feat(para, source_sent, max_src_len) for para, source_sent in zip(paras, tokenized_sents)]

    nodewords, nodelength, nodefreq, sum_worthy, triples, relations, _ = \
        list(zip(*[process_nodes(node, edge, len(list(concat(tokenized_sent))[:max_src_len]), len(tokenized_sent), key='summary_worthy', adj_type=adj_type,
                                 source_sent=tokenized_sent, max_src_len=max_src_len, paras=para, subgraphs=subgraph, docgraph=docgraph)
    for node, edge, tokenized_sent, para, subgraph in zip(nodes, edges, tokenized_sents, paras, subgraphs)]))

    if reward_data_dir is not None:
        if docgraph:
            batch = list(zip(sources, targets, nodewords, word_freq_feats, nodefreq, relations, triples, questions))
        else:
            node_lists = nodelength
            batch = list(zip(sources, targets, nodewords, word_freq_feats, nodefreq, relations, triples, node_lists, questions))
    else:
        if docgraph:
            batch = list(zip(sources, targets, nodewords, word_freq_feats, nodefreq, relations, triples))
        else:
            node_lists = nodelength
            batch = list(zip(sources, targets, nodewords, word_freq_feats, nodefreq, relations, triples, node_lists))
    return batch

@curry
def convert_batch_graph_rl(unk, word2id, batch, docgraph=True, reward_data_dir=None):
    if reward_data_dir is not None:
        if docgraph:
            raw_sources, raw_targets, nodewords, word_freq_feats, nodefreq, relations, triples, questions = map(list, unzip(batch))
        else:
            raw_sources, raw_targets, nodewords, word_freq_feats, nodefreq, relations, triples, node_lists, questions = map(list, unzip(batch))
    else:
        if docgraph:
            raw_sources, raw_targets, nodewords, word_freq_feats, nodefreq, relations, triples = map(list, unzip(batch))
        else:
            raw_sources, raw_targets, nodewords, word_freq_feats, nodefreq, relations, triples, node_lists = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in raw_sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, raw_sources)
    sources = conver2id(unk, word2id, raw_sources)
    if docgraph:
        batch = list(zip(sources, src_exts, nodewords, word_freq_feats, nodefreq, relations, triples))
    else:
        batch = list(zip(sources, src_exts, nodewords, word_freq_feats, nodefreq, relations, triples, node_lists))
    if reward_data_dir is not None:
        return (batch, ext_word2id, raw_sources, raw_targets, questions)
    else:
        return (batch, ext_word2id, raw_sources, raw_targets)

@curry
def batchify_fn_graph_rl(pad, start, end, data, cuda=True, adj_type='concat_triple', docgraph=True, reward_data_dir=None):
    if reward_data_dir is not None:
        batch, ext_word2id, raw_articles, raw_targets, questions = data
    else:
        batch, ext_word2id, raw_articles, raw_targets = data
        questions = []
    if docgraph:
        sources, ext_srcs, nodes, word_freq_feat, nodefreq, relations, triples = tuple(map(list, unzip(batch)))
        if adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
        elif adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
        else:
            adjs = [make_adj(triple, len(node), len(node), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
    else:
        sources, ext_srcs, nodes, word_freq_feat, nodefreq, relations, triples, node_lists = tuple(map(list, unzip(batch)))
        if adj_type == 'edge_as_node':
            adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
        else:
            adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))




    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)
    word_freq = pad_batch_tensorize(word_freq_feat, pad=pad, cuda=cuda)
    feature_dict = {'word_inpara_freq': word_freq,
                    'node_freq': nodefreq}
    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()



    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]


    source = pad_batch_tensorize(sources, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    extend_vsize = len(ext_word2id)
    ext_id2word = {_id:_word for _word, _id in ext_word2id.items()}
    #print('ext_size:', ext_vsize, extend_vsize)
    if docgraph:
        fw_args = (source, src_lens, ext_src, extend_vsize, _nodes, nmask, node_num, feature_dict, adjs,
               START, END, UNK, 100)
    else:
        fw_args = (source, src_lens, ext_src, extend_vsize, _nodes, nmask, node_num, feature_dict, node_lists, adjs,
                   START, END, UNK, 100)

    loss_args = (raw_articles, ext_id2word, raw_targets, questions)

    return fw_args, loss_args

@curry
def prepro_graph_bert(tokenizer, align, max_src_len, max_tgt_len, adj_type, batch, docgraph=True, reward_data_dir=None):
    if reward_data_dir is not None:
        sources, targets, nodes, edges, subgraphs, paras, questions = batch
    else:
        sources, targets, nodes, edges, subgraphs, paras = batch

    old_sources = sources
    sources = [' '.join(raw_sents) for raw_sents in sources]
    sources = [[tokenizer.bos_token] + tokenizer.tokenize(source)[:max_src_len - 2] + [
        tokenizer.eos_token] for source in sources]

    targets = [tokenizer.tokenize(target)[:max_tgt_len] for target in targets]

    source_sents_tokenized = [[tokenizer.tokenize(sent) for sent in source] for source in old_sources]
    max_sents = list(map(count_max_sent(max_source_num=max_src_len - 2), source_sents_tokenized))

    #tokenized_sents = list(map(tokenize(None), old_sources))

    paras = list(paras)

    #word_freq_feats = [create_word_freq_in_para_feat(para, source_sent, max_src_len) for para, source_sent in zip(paras, tokenized_sents)]

    nodewords, nodelength, nodefreq, sum_worthy, triples, relations = \
        list(zip(*[process_nodes_bert(align, node, edge, len(source)-1, max_sent, key='summary_worthy', adj_type=adj_type,
                                 source_sent=sent, paras=para, subgraphs=subgraph, docgraph=docgraph, source=source)
    for node, edge, sent, para, subgraph, source, max_sent in zip(nodes, edges, old_sources, paras, subgraphs, sources, max_sents)]))

    if reward_data_dir is not None:
        if docgraph:
            batch = list(zip(sources, targets, nodewords, nodefreq, relations, triples, questions))
        else:
            node_lists = nodelength
            batch = list(zip(sources, targets, nodewords, nodefreq, relations, triples, node_lists, questions))
    else:
        if docgraph:
            batch = list(zip(sources, targets, nodewords, nodefreq, relations, triples))
        else:
            node_lists = nodelength
            batch = list(zip(sources, targets, nodewords, nodefreq, relations, triples, node_lists))
    return batch

@curry
def convert_batch_graph_rl_bert(tokenizer, max_src_len, batch, docgraph=True, reward_data_dir=None):
    stride = 256
    word2id = tokenizer.encoder
    unk = word2id[tokenizer._unk_token]
    ext_word2id = dict(word2id)
    if reward_data_dir is not None:
        if docgraph:
            raw_sources, raw_targets, nodewords, nodefreq, relations, triples, questions = map(list, unzip(batch))
        else:
            raw_sources, raw_targets, nodewords, nodefreq, relations, triples, node_lists, questions = map(list, unzip(batch))
    else:
        if docgraph:
            raw_sources, raw_targets, nodewords, nodefreq, relations, triples = map(list, unzip(batch))
        else:
            raw_sources, raw_targets, nodewords, nodefreq, relations, triples, node_lists = map(list, unzip(batch))
    src_lens = [len(src) for src in raw_sources]
    for source in raw_sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, raw_sources)
    sources = raw_sources
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
    tar_ins = conver2id(unk, word2id, raw_targets)
    targets = conver2id(unk, ext_word2id, raw_targets)
    if docgraph:
        batch = [sources, list(zip(src_exts, nodewords, nodefreq, relations, triples, src_lens, tar_ins, targets))]
    else:
        batch = [sources, list(zip(src_exts, nodewords, nodefreq, relations, triples, node_lists, src_lens, tar_ins, targets))]
    if reward_data_dir is not None:
        return (batch, ext_word2id, raw_sources, raw_targets, questions)
    else:
        return (batch, ext_word2id, raw_sources, raw_targets)

@curry
def batchify_fn_graph_rl_bert(tokenizer, data, cuda=True, adj_type='concat_triple', docgraph=True, reward_data_dir=None):
    start = tokenizer.encoder[tokenizer._bos_token]
    end = tokenizer.encoder[tokenizer._eos_token]
    pad = tokenizer.encoder[tokenizer._pad_token]
    unk = tokenizer.encoder[tokenizer._unk_token]
    if reward_data_dir is not None:
        batch, ext_word2id, raw_articles, raw_targets, questions = data
    else:
        batch, ext_word2id, raw_articles, raw_targets = data

        questions = []
    if docgraph:
        sources, ext_srcs, nodes, nodefreq, relations, triples, src_lens, tar_ins, targets = (batch[0],) + tuple(map(list, unzip(batch[1])))
        if adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
        elif adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
        else:
            adjs = [make_adj(triple, len(node), len(node), cuda) for triple, node, relation in
                    zip(triples, nodes, relations)]
    else:
        sources, ext_srcs, nodes, nodefreq, relations, triples, node_lists, src_lens, tar_ins, targets = (batch[0],) + tuple(map(list, unzip(batch[1])))
        if adj_type == 'edge_as_node':
            adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
        else:
            adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))




    nodefreq = pad_batch_tensorize(nodefreq, pad=0, cuda=cuda)
    feature_dict = {'node_freq': nodefreq}
    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    #src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]


    source = pad_batch_tensorize(sources, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    extend_vsize = len(ext_word2id)
    ext_id2word = {_id:_word for _word, _id in ext_word2id.items()}
    #print('ext_size:', ext_vsize, extend_vsize)
    if docgraph:
        fw_args = (source, src_lens, ext_src, extend_vsize, _nodes, nmask, node_num, feature_dict, adjs,
               start, end, unk, 150, tar_in)
    else:
        fw_args = (source, src_lens, ext_src, extend_vsize, _nodes, nmask, node_num, feature_dict, node_lists, adjs,
                   start, end, unk, 150, tar_in)

    loss_args = (raw_articles, ext_id2word, raw_targets, questions, target)

    return fw_args, loss_args