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
from data.batcher import pad_batch_tensorize, pad_batch_tensorize_3d
from data.batcher import make_adj_edge_in, make_adj, create_word_freq_in_para_feat

BERT_MAX_LEN = 512
MAX_FREQ = 100

@curry
def tokenize(max_len, texts):
    return [t.strip().lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

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
def prepro_fn_subgraph_nobert(batch, max_sent_len=100, max_sent=60, node_max_len=30,
                                 key='summary_worthy', adj_type='edge_as_node'):
    assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'concat_triple', 'edge_as_node']
    def prepro_one(sample):
        source_sents, extracts, nodes, edges, subgraphs, paras = sample
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
        sent_align_para = []
        last_idx = 0
        for sent in range(len(tokenized_sents)):
            flag = False
            for _idx, para in enumerate(paras):
                if sent in para:
                    sent_align_para.append([_idx])
                    last_idx = _idx
                    flag = True
                    break
            if not flag:
                sent_align_para.append([last_idx])
        assert len(sent_align_para) == len(tokenized_sents)
        sent_align_para.append([last_idx+1])

        segment_feat_para = [sent_align_para[_sid][0]+1 for _sid, sent in enumerate(tokenized_sents_2) for word in sent]
        segment_feat_sent = [[sent_align_para[_sid][0]+1 for word in sent]for _sid, sent in enumerate(tokenized_sents)]

        word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat = create_word_freq_in_para_feat(paras,
                                                                                                                tokenized_sents,
                                                                                                                tokenized_article)


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
        id2node = {}
        ii = 0
        extracted_labels = []
        for _id, content in sorted_nodes:
            if _id not in oor_nodes:
                #extracted_label = content['extracted_label']
                #for _index in indexes:
                #    del extracted_label[_index]
                #extracted_labels.append(extracted_label)
                words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
                words = [word for word in words if word < max_len]
                words = words[:node_max_len]
                #sum_worthy.append(content['InSalientSent'])
                sum_worthy.append(content[key])
                if len(words) != 0:
                    nodefreq.append(len(content['content']))
                    nodewords.append(words)
                    id2node[_id] = ii
                    ii += 1
                else:
                    oor_nodes.append(_id)
        if len(nodewords) == 0:
            #print('warning! no nodes in this sample')
            nodewords = [[0],[2]]
            sum_worthy.extend([0, 0])
            nodefreq.extend([1, 1])
            extracted_label = [0 for _ in range(len(cleaned_extracts))]
            extracted_labels.append(extracted_label)
            extracted_labels.append(extracted_label)

        # process edges
        activated_nodes = set()
        sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))
        sum_worthy_edges = []
        edge_extracted_labels = []
        relations = []
        edge_freq = []
        id2edge = {}

        ii = 0
        for _id, content in sorted_edges:
            if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
                words = content['content']['word_pos']

                new_words = [word for word in words if word > -1 and word < max_len]
                new_words = new_words[:node_max_len]
                if len(new_words) > 0:
                    #extracted_label = content['extracted_label']
                    #for _index in indexes:
                    #    del extracted_label[_index]
                    #edge_extracted_labels.append(extracted_label)
                    node1 = id2node[content['content']['arg1']]
                    node2 = id2node[content['content']['arg2']]

                    edge_freq.append(1)
                    sum_worthy_edges.append(content[key])
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
            #print('warning! no edges in this sample')
            relations = [[1]]
            edge_freq = [1]
            #edge_extracted_label = [0 for _ in range(len(cleaned_extracts))]
            #edge_extracted_labels.append(edge_extracted_label)
            sum_worthy_edges.extend([0])

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
            if paraid > max_sent-1:
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


        if adj_type == 'edge_as_node':
            node_num = len(nodewords)
            nodewords = nodewords + relations
            nodefreq = nodefreq + edge_freq
            sum_worthy = sum_worthy + sum_worthy_edges
            # extracted_labels = extracted_labels + edge_extracted_labels
            for i in range(len(triples)):
                node_lists[i] = node_lists[i] + [edge+node_num for edge in edge_lists[i]]


        gold_dec_selection_label = [0 for i in range(len(node_lists))]
        for sent in cleaned_extracts:
            for i, para in enumerate(paras):
                if sent in para:
                    gold_dec_selection_label[i] = 1
        # gold_dec_selection_label.append(1)




        return tokenized_sents, (cleaned_extracts, tokenized_article), \
               (nodewords, sum_worthy, gold_dec_selection_label), (relations, triples, node_lists, sent_align_para, segment_feat_sent, segment_feat_para, nodefreq, word_inpara_feat, sent_inpara_freq_feat)
    batch = list(map(prepro_one, batch))
    return batch


@curry
def convert_batch_subgraph_nobert(unk, word2id, batch):
    @curry
    def convert_one(word2id, sample):
        source_sents, (extracts, tokenized_article), (nodes, sum_worthy, dec_selection_mask), \
        (relations, triples, node_lists, sent_align_para, segment_feat_sent, segment_feat_para, nodefreq, word_inpara_feat, sent_inpara_freq_feat) = sample
        id_sents = conver2id(unk, word2id, source_sents)
        word2id = defaultdict(lambda: unk, word2id)
        id_article = [word2id[word] for word in tokenized_article]

        #id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(source_sents))
        return id_sents, extracts, id_article, nodes, sum_worthy, relations, triples, node_lists, dec_selection_mask, sent_align_para, segment_feat_sent, segment_feat_para, nodefreq, word_inpara_feat, sent_inpara_freq_feat
    batch = list(map(convert_one(word2id), batch))
    return batch

@curry
def batchify_fn_subgraph_nobert(pad, data, cuda=True, adj_type='edge_as_node', mask_type='none', model_type='gat'):
    assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'concat_triple', 'edge_as_node']
    source_lists, targets, source_articles, nodes, sum_worthy, relations, triples, node_lists, dec_selection_mask, \
    sent_align_paras, segment_feat_sent, segment_feat_para, nodefreq, word_inpara_freq, sent_word_inpara_freq = tuple(map(list, unzip(data)))
    if adj_type == 'edge_as_node':
        batch_adjs = list(map(subgraph_make_adj_edge_in(cuda=cuda), zip(triples, node_lists)))
    else:
        batch_adjs = list(map(subgraph_make_adj(cuda=cuda), zip(triples, node_lists)))
    # print('adj:', batch_adjs[0][0])
    # print('node list:', node_lists[0][0])
    # print('triple:', triples[0][0])


    src_nums = [len(source_list) for source_list in source_lists]
    source_articles = pad_batch_tensorize(source_articles, pad=pad, cuda=cuda)
    segment_feat_para = pad_batch_tensorize(segment_feat_para, pad=pad, cuda=cuda)
    #sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    segment_feat_sent = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), segment_feat_sent))

    nodefreq = pad_batch_tensorize(nodefreq, pad=pad, cuda=cuda)
    word_inpara_freq = pad_batch_tensorize(word_inpara_freq, pad=pad, cuda=cuda)
    sent_word_inpara_freq = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), sent_word_inpara_freq))
    # source_lists = [source for source_list in source_lists for source in source_list]
    # sources = pad_batch_tensorize(source_lists, pad=pad, cuda=cuda)
    #print('extracted labels:', extracted_labels)

    sum_worthy_label = pad_batch_tensorize(sum_worthy, pad=-1, cuda=cuda)
    sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=cuda).float()

    dec_selection_mask = pad_batch_tensorize(dec_selection_mask, pad=0, cuda=cuda).float()


    node_num = [len(_node) for _node in nodes]
    _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=cuda)
    _relations = pad_batch_tensorize_3d(relations, pad=0, cuda=cuda)
    nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=cuda).ne(-1).float()
    rmask = pad_batch_tensorize_3d(relations, pad=-1, cuda=cuda).ne(-1).float()
    segment_features = pad_batch_tensorize


    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )
    feature_dict = {'seg_para': segment_feat_para,
                    'seg_sent': segment_feat_sent,
                    'sent_inpara_freq': sent_word_inpara_freq,
                    'word_inpara_freq': word_inpara_freq,
                    'node_freq': nodefreq
                    }


    fw_args = (src_nums, tar_in, (sources, source_articles, feature_dict), (_nodes, nmask, node_num, sum_worthy, dec_selection_mask),
               (_relations, rmask, triples, batch_adjs, node_lists, sent_align_paras))
    if 'soft' in mask_type:
        loss_args = (target, sum_worthy_label)
    # elif decoder_supervision:
    #     loss_args = (target, extracted_labels)
    else:
        loss_args = (target, )
    return fw_args, loss_args