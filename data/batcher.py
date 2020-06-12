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

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d[0], d[1]
        word_num = len(' '.join(source_sents).split(' '))
        return source_sents and extracts and word_num > 5
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

def coll_fn_gat(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, nodes = d[0], d[1], d[2]
        word_num = len(' '.join(source_sents).split(' '))
        filter_extracts = [extract for extract in extracts if extract < 60]
        return source_sents and extracts and word_num > 5 and len(d[2]) < 200 and filter_extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

def coll_fn_extract_entity(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, clusters = d
        return (source_sents and extracts) and clusters
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch
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

@curry
def prepro_fn_extract(max_src_len, max_src_num, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    return batch

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

@curry
def prepro_fn_extract_gat(tokenizer, start, end, align, batch, max_len=1024, stride=256, node_max_len=30, key='summary_worthy', adj_type='edge_as_node'):
    assert max_len in [512, 1024, 1536, 2048]
    def prepro_one(sample):
        source_sents, extracts, nodes, edges, subgraphs, paras = sample
        #original_order = ' '.join(source_sents).split(' ')
        order_match = {}
        count = 1
        i = 0
        for sents in source_sents:
            sent_words = sents.split(' ')
            if len(sent_words) > 0:
                order_match[i] = list(range(count, count + align[sent_words[0]]))
                count += align[sent_words[0]]
                i += 1
                for word in sent_words[1:]:
                    new_word = ' ' + word
                    order_match[i] = list(range(count, count + align[new_word]))
                    count += align[new_word]
                    i += 1
        max_len = count

        tokenized_sents = [tokenizer.tokenize(source_sent) for source_sent in source_sents]
        tokenized_sents[0] = [start] + tokenized_sents[0]
        tokenized_sents[-1] = tokenized_sents[-1] + [end]
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
        max_sent = len(truncated_word_num)
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
        # if max_node_num > max_len:
        #     print('Warning, wrong align')
        #     max_len = max_node_num

        # find out of range and useless nodes
        other_nodes = set()
        oor_nodes = [] # out of range nodes will not included in the graph
        for _id, content in nodes.items():
            words = [_ for mention in content['content'] for pos in mention['word_pos'] if pos != -1 and order_match.__contains__(pos) for _ in
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
                new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in order_match[word] if _ < max_len]
                if len(new_words) > 0:
                    activated_nodes.add(content['content']['arg1'])
                    activated_nodes.add(content['content']['arg2'])
        oor_nodes.extend(list(other_nodes - activated_nodes))

        # process nodes
        sorted_nodes = sorted(nodes.items(), key=lambda x:int(x[0].split('_')[1]))
        sum_worthy = []
        nodewords = []
        nodefreq = []
        id2node = {}
        ii = 0
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:
                words = [_ for mention in content['content'] for pos in mention['word_pos'] if pos != -1 and order_match.__contains__(pos) for _ in order_match[pos]]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                #sum_worthy.append(content['InSalientSent'])
                sum_worthy.append(content[key])
                if len(words) != 0:
                    nodewords.append(words)
                    nodefreq.append(len(content['content']))
                    id2node[_id] = ii
                    ii += 1
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            #print('warning! no nodes in this sample')
            nodefreq.extend([1, 1])
            nodewords = [[0],[2]]
            sum_worthy.extend([0, 0])
        nodelength = [len(words) for words in nodewords]

        # process edges
        acticated_nodes = set()
        triples = []
        relations = []
        sum_worthy_edges = []
        edge_freq = []
        id2edge = {}
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
                    edge_freq.append(1)
                    sum_worthy_edges.append(content[key])
                    triples.append([node1, ii, node2])
                    acticated_nodes.add(content['content']['arg1'])
                    acticated_nodes.add(content['content']['arg2'])
                    id2edge[_id] = ii
                    ii += 1
                    relations.append(new_words)
        if len(relations) == 0:
            #print('warning! no edges in this sample')
            relations = [[1]]
            edge_freq = [1]
            sum_worthy_edges.extend([0])
            triples = [[0, 0, 1]]
        rlength = [len(words) for words in relations]

        nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]

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

        if adj_type == 'edge_as_node':
            node_num = len(nodewords)
            nodewords = nodewords + relations
            nodelength = nodelength + rlength
            sum_worthy = sum_worthy + sum_worthy_edges
            nodefreq = nodefreq + edge_freq
            for i in range(len(triples)):
                node_lists[i] = node_lists[i] + [edge+node_num for edge in edge_lists[i]]
        # print('length:', len(tokenized_sents))
        # print(tokenized_sents)
        # print('node:', max(nodewords))
        # print('order match:', order_match)
        # print(align['ryan'])
        # for _key, value in test_order_match.items():
        #     print(_key, tokenized_sents[value[0]:value[1]])
        #     print(value[0], value[1])

        return tokenized_sents_lists, (cleaned_extracts, truncated_word_num), (nodewords, nodelength, sum_worthy), (relations, rlength, triples, nodefreq, node_lists)
    batch = list(map(prepro_one, batch))
    return batch

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
def prepro_fn_extract_gat_nobert(batch, max_sent_len=100, max_sent=60, node_max_len=30,
                                 key='summary_worthy', adj_type='concat_triple'):
    assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'concat_triple', 'edge_as_node']
    def prepro_one(sample):
        source_sents, extracts, nodes, edges, paras = sample
        tokenized_sents = tokenize(max_sent_len, source_sents)[:max_sent]
        tokenized_sents_2 = tokenize(None, source_sents)[:max_sent]
        tokenized_article = list(concat(tokenized_sents_2))
        max_len = len(tokenized_article)
        indexes = []
        for e in extracts:
            if e > len(tokenized_sents)-1:
                indexes.append(extracts.index(e))
        indexes = list(sorted(indexes, reverse=True))
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat = create_word_freq_in_para_feat(paras, tokenized_sents, tokenized_article)


        # find out of range and useless nodes
        other_nodes = set()
        oor_nodes = [] # out of range nodes will not included in the graph
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
        sorted_nodes = sorted(nodes.items(), key=lambda x:int(x[0].split('_')[1]))
        sum_worthy = []
        nodewords = []
        nodefreq = []
        nodetype = []
        nodeinsent = []
        id2node = {}
        ii = 0
        extracted_labels = []
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:
                words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                #sum_worthy.append(content['InSalientSent'])
                sum_worthy.append(content[key])
                if len(words) != 0:
                    nodewords.append(words)
                    nodefreq.append(len(content['content']))
                    nodetype.append(1)
                    nodeinsent.append([mention['sent_pos'] for mention in content['content'] if mention['sent_pos'] < len(tokenized_sents)])
                    id2node[_id] = ii
                    ii += 1
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            #print('warning! no nodes in this sample')
            nodewords = [[0],[2]]
            sum_worthy.extend([0, 0])
            nodefreq.extend([1, 1])
            nodetype.extend([1, 1])
            nodeinsent.extend([[0], [0]])
            extracted_label = [0 for _ in range(len(cleaned_extracts))]
            extracted_labels.append(extracted_label)
            extracted_labels.append(extracted_label)
        nodelength = [len(words) for words in nodewords]
        nodefreq = [freq if freq < MAX_FREQ-1 else MAX_FREQ-1 for freq in nodefreq]

        # process edges
        acticated_nodes = set()
        triples = []
        relations = []
        edge_freq = []
        edgeinsent = []
        edgetype = []
        sum_worthy_edges = []
        edge_extracted_labels = []
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
                    try:
                        sent_pos = [content['content']['sent_pos']]
                    except KeyError:
                        sent_pos = [content['content']['arg1_original'][0]['sent_pos']]
                    sum_worthy_edges.append(content[key])
                    triples.append([node1, ii, node2])
                    edge_freq.append(1)
                    edgeinsent.append(sent_pos)
                    edgetype.append(2)
                    if adj_type == 'edge_up':
                        nodewords[node1].extend(new_words)
                    elif adj_type == 'edge_down':
                        nodewords[node2].extend(new_words)
                    acticated_nodes.add(content['content']['arg1'])
                    acticated_nodes.add(content['content']['arg2'])
                    ii += 1
                    relations.append(new_words)
        if len(relations) == 0:
            #print('warning! no edges in this sample')
            relations = [[1]]
            triples = [[0, 0, 1]]
            edge_freq = [1]
            edgeinsent.append([0])
            edgetype.append(2)
            edge_extracted_label = [0 for _ in range(len(cleaned_extracts))]
            edge_extracted_labels.append(edge_extracted_label)
            sum_worthy_edges.extend([0])
        rlength = [len(words) for words in relations]
        if adj_type == 'edge_as_node':
            nodewords = nodewords + relations
            nodelength = nodelength + rlength
            sum_worthy = sum_worthy + sum_worthy_edges
            extracted_labels = extracted_labels + edge_extracted_labels
            nodefreq = nodefreq + edge_freq
            nodetype = nodetype + edgetype
            nodeinsent = nodeinsent + edgeinsent

        sent_node_aligns = create_sent_node_align(nodeinsent, len(tokenized_sents))





        return tokenized_sents, (cleaned_extracts, tokenized_article, extracted_labels), (nodewords, nodelength, sum_worthy, sent_node_aligns), \
               (relations, rlength, triples, nodefreq, word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat)
    batch = list(map(prepro_one, batch))
    return batch



@curry
def prepro_fn_extract_bert(tokenizer, batch, max_len=512):
    def prepro_one(sample):
        source_sents, extracts = sample
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
                truncated_word_num.append(max_len - total_count)
                break
            total_count += num
        tokenized_sents = list(concat(tokenized_sents))[:max_len]
        cleaned_extracts = list(filter(lambda e: e < len(truncated_word_num),
                                       extracts))
        return tokenized_sents, (cleaned_extracts, truncated_word_num)
    batch = list(map(prepro_one, batch))
    return batch

@curry
def prepro_fn_extract_bert_stride(tokenizer, batch, max_len=1024, stride=256):
    assert max_len in [512, 1024, 1536, 2048]
    assert stride in [128, 256]
    def prepro_one(sample):
        source_sents, extracts = sample
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

        return tokenized_sents_lists, (cleaned_extracts, truncated_word_num)
    batch = list(map(prepro_one, batch))
    return batch

@curry
def prepro_fn_extract_bert_sent(tokenizer, batch, max_len=75, max_sent_len=60):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
        tokenized_sents = tokenized_sents[:max_sent_len]
        tokenized_sents = [['[CLS]'] + tokenized_sent[:max_len-1] for tokenized_sent in tokenized_sents]
        word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
        cleaned_extracts = list(filter(lambda e: e < len(word_num),
                                       extracts))
        return tokenized_sents, (cleaned_extracts, word_num)
    batch = list(map(prepro_one, batch))
    return batch

@curry
def prepro_fn_extract_entity(max_src_len, max_src_num, batch, pad=0, split_token='<split>', args = {}):
    # split will be "split token"
    def prepro_one(sample):
        source_sents, extracts, clusters = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        # merge cluster
        cluster_words = []
        cluster_wpos = []
        cluster_spos = []
        for cluster in clusters:
            scluster_word = []
            scluster_wpos = []
            scluster_spos = []
            for mention in cluster:
                if mention['position'][0] > max_src_num-2:
                    continue
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
                    if word_end > 99:
                        word_end = 99
                    scluster_word += tokenized_sents[sent_num][word_start:word_end]
                    scluster_wpos += list(range(word_start, word_end))
                    scluster_spos += [mention['position'][0] + 1 for _ in
                                      range(word_start+1, word_end+1)]
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
                    print(scluster_word)
                    print(scluster_wpos)
                    print('cluster:', cluster)
                assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)

        rel = args.get('enable_rel', True)
        cluster_num = len(cluster_words) + 1
        if rel:
            #adj = torch.zeros(cluster_num+1, cluster_num+1)
            rels = []
            for idx, spos in enumerate(cluster_spos):
                real_spos = set([num for num in spos if num > 0])
                for _idx, _spos in enumerate(cluster_spos):
                    _real_spos = set([num for num in _spos if num > 0])
                    if _idx < idx:
                        continue
                    if _idx != idx and len(real_spos & _real_spos) > 0:
                        rels.extend([0])
            adj = [[0 for _ in range(cluster_num+len(rels))] for __ in range(cluster_num+len(rels))]
            for _i in range(len(rels)):
                adj[cluster_num+_i][cluster_num+_i] = 1.
            adj[0][0] += 1
        else:
            adj = [[0 for _ in range(cluster_num)] for __ in range(cluster_num)]
            rels = []
            adj[0][0] += 1
        rel_num = 0
        for idx, spos in enumerate(cluster_spos):
            real_spos = set([num for num in spos if num > 0])
            for _idx, _spos in enumerate(cluster_spos):
                if _idx < idx:
                    continue
                _real_spos = set([num for num in _spos if num > 0])
                if _idx != idx:
                    if len(real_spos & _real_spos) > 0 and rel:
                        adj[idx+1][cluster_num + rel_num] = 1.
                        adj[cluster_num + rel_num][idx+1] = 1.
                        adj[cluster_num + rel_num][_idx+1] = 1.
                        adj[_idx+1][cluster_num + rel_num] = 1.
                        rel_num += 1
                    elif len(real_spos & _real_spos) > 0 and not rel:
                        adj[idx+1][_idx+1] = 1.
                        adj[_idx+1][idx+1] = 1.
                else:
                    adj[idx+1][idx+1] = 1.


        clusters = (cluster_words, cluster_wpos, cluster_spos, adj, rels)

        return tokenized_sents, cleaned_extracts, clusters
    batch = list(map(prepro_one, batch))
    return batch


@curry
def prepro_fn_extract_hardattn(max_src_len, max_src_num, batch, pad=0, split_token='<split>'):
    # split will be "split token"
    return batch

@curry
def prepro_fn_extract_graph(max_src_len, max_src_num, batch, pad=0, split_token='<split>', args = {}):
    # split will be "split token"
    def prepro_one_entity(sample):
        source_sents, extracts, clusters = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        # merge cluster
        cluster_words = []
        cluster_wpos = []
        cluster_spos = []
        for cluster in clusters:
            scluster_word = []
            scluster_wpos = []
            scluster_spos = []
            for mention in cluster:
                if mention['position'][0] > max_src_num-2:
                    continue
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
                    if word_end > 99:
                        word_end = 99
                    scluster_word += tokenized_sents[sent_num][word_start:word_end]
                    scluster_wpos += list(range(word_start, word_end))
                    scluster_spos += [mention['position'][0] + 1 for _ in
                                      range(word_start+1, word_end+1)]
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
                    print(scluster_word)
                    print(scluster_wpos)
                    print('cluster:', cluster)
                assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)

        rel = args.get('enable_rel', True)
        cluster_num = len(cluster_words) + 1
        if rel:
            #adj = torch.zeros(cluster_num+1, cluster_num+1)
            rels = []
            for idx, spos in enumerate(cluster_spos):
                real_spos = set([num for num in spos if num > 0])
                for _idx, _spos in enumerate(cluster_spos):
                    _real_spos = set([num for num in _spos if num > 0])
                    if _idx < idx:
                        continue
                    if _idx != idx and len(real_spos & _real_spos) > 0:
                        rels.extend([0])
            adj = [[0 for _ in range(cluster_num+len(rels))] for __ in range(cluster_num+len(rels))]
            for _i in range(len(rels)):
                adj[cluster_num+_i][cluster_num+_i] = 1.
            adj[0][0] += 1
        else:
            adj = [[0 for _ in range(cluster_num)] for __ in range(cluster_num)]
            rels = []
            adj[0][0] += 1
        rel_num = 0
        for idx, spos in enumerate(cluster_spos):
            real_spos = set([num for num in spos if num > 0])
            for _idx, _spos in enumerate(cluster_spos):
                if _idx < idx:
                    continue
                _real_spos = set([num for num in _spos if num > 0])
                if _idx != idx:
                    if len(real_spos & _real_spos) > 0 and rel:
                        adj[idx+1][cluster_num + rel_num] = 1.
                        adj[cluster_num + rel_num][idx+1] = 1.
                        adj[cluster_num + rel_num][_idx+1] = 1.
                        adj[_idx+1][cluster_num + rel_num] = 1.
                        rel_num += 1
                    elif len(real_spos & _real_spos) > 0 and not rel:
                        adj[idx+1][_idx+1] = 1.
                        adj[_idx+1][idx+1] = 1.
                else:
                    adj[idx+1][idx+1] = 1.


        clusters = (cluster_words, cluster_wpos, cluster_spos, adj, rels)

        return tokenized_sents, cleaned_extracts, clusters

    def prepro_one(sample):
        source_sents, extracts, clusters = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        # merge cluster
        sent_num = len(tokenized_sents)
        adj_dim = sent_num + 1 + 2 * sent_num - 2
        adj = [[0 if _ != __ else 1. for _ in range(adj_dim)] for __ in range(adj_dim)]
        rels = [0 if _id % 2 == 0 else 1 for _id in range(2 * (sent_num-1))]
        for i in range(sent_num):
            adj[i][sent_num] = 1.
            adj[sent_num][i] = 1.
        for i in range(sent_num-1):
            adj[i][sent_num + 1 + 2 * i] = 1.
            adj[sent_num + 1 + 2 * i][i+1] = 1.
            adj[sent_num + 1 + 1 + 2 * i][i] = 1.
            adj[i + 1][sent_num + 1 + 1 + 2 * i] = 1.



        clusters = ([], [], [], adj, rels)

        return tokenized_sents, cleaned_extracts, clusters

    entity = args.get('entity', False)
    if entity is True:
        batch = list(map(prepro_one_entity, batch))
    else:
        batch = list(map(prepro_one, batch))
    return batch




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


@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_stop(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(source_sents))
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_bert(tokenizer, batch):
    def convert_one(sample):
        tokenized_sents, (extracts, word_num) = sample
        id_sents = tokenizer.convert_tokens_to_ids(tokenized_sents)

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(word_num))
        return id_sents, extracts, word_num
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_bert_stride(tokenizer, batch):
    def convert_one(sample):
        tokenized_sents_lists, (extracts, word_num) = sample
        id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(word_num))
        return id_sents, extracts, word_num
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_bert_sentence(tokenizer, batch):
    def convert_one(sample):
        tokenized_sents, (extracts, word_num) = sample
        id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sent) for tokenized_sent in tokenized_sents]
        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(word_num))
        return id_sents, extracts, word_num
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_gat(tokenizer, batch):
    def convert_one(sample):
        tokenized_sents_lists, (extracts, word_num), (nodes, nlength, sum_worthy), (relations, rlength, triples, nodefreq, node_lists) = sample
        id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(word_num))
        return id_sents, extracts, word_num, nodes, nlength, sum_worthy, relations, rlength, triples, nodefreq, node_lists
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_gat_nobert(unk, word2id, batch):
    @curry
    def convert_one(word2id, sample):
        source_sents, (extracts, tokenized_article, extracted_labels), (nodes, nlength, sum_worthy, sent_node_align), \
        (relations, rlength, triples, nodefreq, word_freq, word_inpara_freq, sent_freq_feat, sent_inpara_freq_feat) = sample
        id_sents = conver2id(unk, word2id, source_sents)
        word2id = defaultdict(lambda: unk, word2id)
        id_article = [word2id[word] for word in tokenized_article]

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(source_sents))
        stop_labels = [0 for _ in range(len(extracts))]
        stop_labels[-1] = 1
        extracted_labels = [extracted_label + [0] for extracted_label in extracted_labels]
        extracted_labels.append(stop_labels)
        return id_sents, extracts, id_article, nodes, nlength, sum_worthy, sent_node_align, relations, rlength, triples, \
               nodefreq, word_freq, word_inpara_freq, sent_freq_feat, sent_inpara_freq_feat, extracted_labels
    batch = list(map(convert_one(word2id), batch))
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

@curry
def batchify_fn_extract_ptr_gat(pad, data, cuda=True, adj_type='edge_as_node', mask_type='soft'):
    source_lists, targets, word_nums, nodes, nlength, sum_worthy, relations, rlength, triples, nodefreq, node_lists = tuple(map(list, unzip(data)))
    # adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
    if adj_type == 'edge_as_node':
        batch_adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
    else:
        batch_adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))

    src_nums = list(map(len, word_nums))
    word_nums = list(word_nums)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    source_lists = [source for source_list in source_lists for source in source_list]
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    sum_worthy_label = pad_batch_tensorize(sum_worthy, pad=-1, cuda=cuda)
    sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=cuda).float()

    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()

    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)


    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    feature_dict = {'node_freq': nodefreq}


    fw_args = (src_nums, tar_in, (sources, word_nums, feature_dict), (_nodes, nmask, node_num, sum_worthy, None), (_relations, rmask, triples, batch_adjs, node_lists, None))
    #loss_args = (target, )
    if 'soft' in mask_type:
        loss_args = (target, sum_worthy_label)
    # elif decoder_supervision:
    #     loss_args = (target, extracted_labels)
    else:
        loss_args = (target, )
    return fw_args, loss_args

def normalize_adjs(adjs):
    d = adjs.sum(1, keepdim=True)
    d[d == 0] = 1e-8
    adjs = adjs / d
    return adjs

@curry
def batchify_fn_extract_ptr_gat_nobert(pad, data, cuda=True, adj_type='concat_triple', mask_type='none', decoder_supervision=False, model_type='gat'):
    assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'concat_triple', 'edge_as_node']
    source_lists, targets, source_articles, nodes, nlength, sum_worthy, sent_node_aligns, relations, rlength, triples, \
    nodefreq, word_freq, word_inpara_freq, sent_word_freq, sent_word_inpara_freq, extracted_labels = tuple(map(list, unzip(data)))
    if model_type == 'ggnn':
        assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'edge_as_node']
        if adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in_bidirectional(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        else:
            adjs = [make_adj_bidirectional(triple, len(node), len(node), cuda) for triple, node, relation in zip(triples, nodes, relations)]

        adjs = pad_batch_tensorize_adjs(adjs)
    elif model_type == 'gcn':
        assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'edge_as_node']
        if adj_type == 'edge_as_node':
            adjs = [[make_adj_edge_in(triple, len(node), len(relation), cuda)] for triple, node, relation in
                    zip(triples, nodes, relations)]
        else:
            adjs = [[make_adj(triple, len(node), len(node), cuda)] for triple, node, relation in
                    zip(triples, nodes, relations)]

        adjs = pad_batch_tensorize_adjs(adjs)
        adjs = normalize_adjs(adjs)
    else:
        if adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        elif adj_type == 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), cuda) for triple, node, relation in zip(triples, nodes, relations)]
        else:
            adjs = [make_adj(triple, len(node), len(node), cuda) for triple, node, relation in zip(triples, nodes, relations)]

    src_nums = [len(source_list) for source_list in source_lists]
    source_articles = pad_batch_tensorize(source_articles, pad=pad, cuda=cuda)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    # source_lists = [source for source_list in source_lists for source in source_list]
    # sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    # extra features
    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)
    word_inpara_freq = pad_batch_tensorize(word_inpara_freq, pad=pad, cuda=cuda)
    word_freq = pad_batch_tensorize(word_freq, pad=pad, cuda=cuda)
    sent_word_freq = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), sent_word_freq))
    sent_word_inpara_freq = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), sent_word_inpara_freq))

    extracted_labels = pad_batch_tensorize_3d(extracted_labels, pad=-1, cuda=cuda)
    extracted_labels = extracted_labels.permute(0, 2, 1).contiguous()
    #print('extracted labels:', extracted_labels)

    sum_worthy_label = pad_batch_tensorize(sum_worthy, pad=-1, cuda=cuda)
    sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=cuda).float()


    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()


    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )
    feature_dict = {'sent_word_freq': sent_word_freq,
                    'word_freq': word_freq,
                    'sent_inpara_freq': sent_word_inpara_freq,
                    'word_inpara_freq': word_inpara_freq,
                    'node_freq': nodefreq}


    fw_args = (src_nums, tar_in, (sources, source_articles, feature_dict), (_nodes, nmask, node_num, sum_worthy, sent_node_aligns), (_relations, rmask, triples, adjs))
    if 'soft' in mask_type and decoder_supervision:
        loss_args = (target, sum_worthy_label, extracted_labels)
    elif 'soft' in mask_type:
        loss_args = (target, sum_worthy_label)
    elif decoder_supervision:
        loss_args = (target, extracted_labels)
    else:
        loss_args = (target, )
    return fw_args, loss_args

@curry
def convert_batch_extract_ff(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
    return batch


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


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_bert_stride(pad, data, cuda=True):
    source_lists, targets, word_nums = tuple(map(list, unzip(data)))

    src_nums = list(map(len, word_nums))
    word_nums = list(word_nums)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    source_lists = [source for source_list in source_lists for source in source_list]
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (sources, word_nums))
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_bert(pad, data, cuda=True):
    source_lists, targets, word_nums = tuple(map(list, unzip(data)))

    src_nums = list(map(len, word_nums))
    word_nums = list(word_nums)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (sources, word_nums))
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_bert_sentence(pad, data, cuda=True):
    source_lists, targets, word_nums = tuple(map(list, unzip(data)))

    src_nums = list(map(len, word_nums))
    word_nums = [len(source_list) for source_list in source_lists]
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    source_lists = [sent for source_list in source_lists for sent in source_list]
    sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (sources, word_nums))
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_graph(pad, data, cuda=True, entity=False):
    source_lists, targets, clusters_infos = tuple(map(list, unzip(data)))
    (cluster_lists, cluster_wpos, cluster_spos, adjs, rels) = list(zip(*clusters_infos))

    src_nums = list(map(len, source_lists))
    cl_nums = list(map(len, cluster_lists))
    if entity:
        cluster_nums = [cl_num for cl_num in cl_nums]
        feature_nums = [cl_num+src_num+len(rel)+2 for src_num, cl_num, rel in zip(src_nums, cl_nums, rels)]
    else:
        cluster_nums = [cl_num for cl_num in cl_nums]
        feature_nums = [src_num + len(rel) + 1 for src_num, rel in
                        zip(src_nums, rels)]

    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    cluster_lists = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_lists)) # list of tensors, each tensor padded
    cluster_wpos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_wpos))
    cluster_spos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_spos))

    # PAD is -1 (dummy extraction index) for using sequence loss
    adjs = [torch.cuda.FloatTensor(adj) if cuda else torch.FloatTensor(adj) for adj in adjs]
    rels = [torch.cuda.LongTensor(rel) if cuda else torch.LongTensor(rel) for rel in rels]
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (cluster_lists, cluster_wpos, cluster_spos, adjs, rels), (cluster_nums, feature_nums))
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_entity(pad, data, cuda=True):
    source_lists, targets, clusters_infos = tuple(map(list, unzip(data)))
    (cluster_lists, cluster_wpos, cluster_spos, adjs, rels) = list(zip(*clusters_infos))

    src_nums = list(map(len, source_lists))
    cl_nums = list(map(len, cluster_lists))
    cluster_nums = [cl_num for cl_num in cl_nums]
    feature_nums = [cl_num+len(rel)+1 if cl_num != 0 else 1 for cl_num, rel in zip(cl_nums, rels)]
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    clusters = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_lists)) # list of tensors, each tensor padded
    cluster_wpos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_wpos))
    cluster_spos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), cluster_spos))

    # PAD is -1 (dummy extraction index) for using sequence loss
    adjs = [torch.cuda.FloatTensor(adj) if cuda else torch.FloatTensor(adj) for adj in adjs]
    rels = [torch.cuda.LongTensor(rel) if cuda else torch.LongTensor(rel) for rel in rels]
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (clusters, cluster_wpos, cluster_spos, adjs, rels), (cluster_nums, feature_nums))
    loss_args = (target, )
    return fw_args, loss_args



@curry
def convert_batch_extract_ptr_entity(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos = sample
        (cluster_lists, cluster_wpos, clust_spos) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos)

        return id_sents, extracts, cluster_infos
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_entity_stop(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos = sample
        (cluster_lists, cluster_wpos, clust_spos, adj, rel) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos, adj, rel)
        extracts.append(len(source_sents))
        return id_sents, extracts, cluster_infos
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_graph(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos = sample
        (cluster_lists, cluster_wpos, clust_spos, adj, rel) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos, adj, rel)
        extracts.append(len(source_sents))
        return id_sents, extracts, cluster_infos
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_entity_hardattn(unk, word2id, batch):

    return batch

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_nnse(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #target = tensor_type(list(concat(targets)))
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    loss_target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums, target)
    loss_args = (loss_target, )
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
