from torch.utils.data import DataLoader

from data.data import CnnDmDataset
from data.batcher import tokenize


import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle
from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
#from data.abs_batcher import create_word_freq_in_para_feat
from data.batcher import create_word_freq_in_para_feat, create_sent_node_align
from data.batcher import pad_batch_tensorize, pad_batch_tensorize_3d, conver2id
from utils import PAD, UNK, START, END
from data.batcher import make_adj_triple, make_adj, make_adj_edge_in
from transformers import RobertaTokenizer
import pickle



MAX_FREQ = 100

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class RLDataset_graph(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key='nodes_pruned2'):
        super().__init__(split, DATA_DIR)
        self.node_key = key
        self.edge_key = key.replace('nodes', 'edges')
        print('using key: ', key)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        nodes = js_data[self.node_key]
        edges = js_data[self.edge_key]
        subgraphs = js_data['subgraphs']
        paras = js_data['paragraph_merged']
        return art_sents, abs_sents, nodes, edges, subgraphs, paras

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

def process_nodes(nodes, edges, max_len, max_sent_num, node_max_len=30, key='InSalientSent', adj_type='edge_as_node', docgraph=True,
                  source_sent=None, subgraphs=None, paras=None, max_src_len=800):
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
    nodeinsent = []
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
                nodeinsent.append([mention['sent_pos'] for mention in content['content'] if
                                   mention['sent_pos'] < max_sent_num])
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
        nodeinsent.extend([[0], [0]])
    nodelength = [len(words) for words in nodewords]
    nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]

    # process edges
    acticated_nodes = set()

    triples = []
    edge_freq = []
    relations = []
    edgeinsent = []
    sum_worthy_edges = []
    sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))

    ii = 0
    id2edge = {}
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
                edge_freq.append(1)
                edgeinsent.append(sent_pos)
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
        edgeinsent.append([0])
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
        nodeinsent = nodeinsent + edgeinsent

    sent_node_aligns = create_sent_node_align(nodeinsent, max_sent_num)
    if docgraph:
        return nodewords, nodelength, nodefreq, sum_worthy, triples, relations, sent_node_aligns
    else:
        return nodewords, node_lists, nodefreq, sum_worthy, triples, relations, sent_node_aligns

def process_subgraphs(nodes, edges, subgraphs, paras, max_len, max_sent, node_max_len=30, key='InSalientSent', adj_type='edge_as_node'):
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
    nodewords = []
    nodefreq = []
    id2node = {}
    ii = 0
    extracted_labels = []
    for _id, content in sorted_nodes:
        if _id not in oor_nodes:
            # extracted_label = content['extracted_label']
            # for _index in indexes:
            #    del extracted_label[_index]
            # extracted_labels.append(extracted_label)
            words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
            words = [word for word in words if word < max_len]
            words = words[:node_max_len]
            # sum_worthy.append(content['InSalientSent'])
            sum_worthy.append(content[key])
            if len(words) != 0:
                nodefreq.append(len(content['content']))
                nodewords.append(words)
                id2node[_id] = ii
                ii += 1
            else:
                oor_nodes.append(_id)
    if len(nodewords) == 0:
        # print('warning! no nodes in this sample')
        nodewords = [[0], [2]]
        sum_worthy.extend([0, 0])
        nodefreq.extend([1, 1])


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
                # extracted_label = content['extracted_label']
                # for _index in indexes:
                #    del extracted_label[_index]
                # edge_extracted_labels.append(extracted_label)
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
        # print('warning! no edges in this sample')
        relations = [[1]]
        edge_freq = [1]
        # edge_extracted_label = [0 for _ in range(len(cleaned_extracts))]
        # edge_extracted_labels.append(edge_extracted_label)
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

    if adj_type == 'edge_as_node':
        node_num = len(nodewords)
        nodewords = nodewords + relations
        nodefreq = nodefreq + edge_freq
        sum_worthy = sum_worthy + sum_worthy_edges
        # extracted_labels = extracted_labels + edge_extracted_labels
        for i in range(len(triples)):
            node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]

    return nodewords, node_lists, nodefreq, sum_worthy, triples, relations



def build_batchers_graph(batch_size, node_key, adj_type, sum_worthy_key, docgraph, paragraph):
    assert not all([docgraph, paragraph])
    @curry
    def coll(key, adj_type, batch):
        split_token = '<split>'
        pad = 0
        art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras = unzip(batch)

        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts, nodes, edges, subgraphs, paras = d
            return source_sents and extracts and nodes and subgraphs and paras

        art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras  = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras)))))
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        inputs = []
        # merge cluster
        for art_sent, nodes, edges, subgraphs, paras in zip(art_sents, all_nodes, all_edges, all_subgraphs, all_paras):
            max_len = len(list(concat(art_sent)))
            _, word_inpara_freq_feat, _, sent_inpara_freq_feat = create_word_freq_in_para_feat(paras, art_sent, list(concat(art_sent)))
            nodewords, nodelength, nodefreq, sum_worthy, triples, relations, sent_node_aligns = process_nodes(nodes, edges, max_len, max_sent_num=len(list(art_sent)),
                                                                                                              key=key, adj_type=adj_type)
            if paragraph:
                nodewords, node_lists, nodefreq, sum_worthy, triples, relations = process_subgraphs(
                    nodes, edges, subgraphs, paras, max_len, max_sent=len(list(art_sent)),
                    key=key, adj_type=adj_type
                )
                sent_align_para = []
                last_idx = 0
                for sent in range(len(art_sent)):
                    flag = False
                    for _idx, para in enumerate(paras):
                        if sent in para:
                            sent_align_para.append([_idx])
                            last_idx = _idx
                            flag = True
                            break
                    if not flag:
                        sent_align_para.append([last_idx])
                assert len(sent_align_para) == len(art_sent)
                sent_align_para.append([last_idx + 1])

            if docgraph:
                inputs.append((art_sent, nodewords, nodefreq, word_inpara_freq_feat, sent_inpara_freq_feat, triples, relations, sent_node_aligns))
            elif paragraph:
                inputs.append((art_sent, nodewords, nodefreq, word_inpara_freq_feat, sent_inpara_freq_feat, triples,
                               relations, sent_align_para, node_lists))
            else:
                raise Exception('wrong graph type')
        assert len(inputs) == len(abs_sents)
        return inputs, abs_sents


    loader = DataLoader(
        RLDataset_graph('train', node_key), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll(sum_worthy_key, adj_type)
    )
    val_loader = DataLoader(
        RLDataset_graph('val', node_key), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll(sum_worthy_key, adj_type)
    )
    return cycle(loader), val_loader


def build_batchers_graph_bert(batch_size, node_key, adj_type, max_src_len, docgraph, paragraph):
    assert not all([docgraph, paragraph])
    @curry
    def coll(tokenizer, align, max_src_len, adj_type, batch):
        art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras = unzip(batch)

        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts, nodes, edges, subgraphs, paras = d
            return source_sents and extracts and nodes and subgraphs and paras

        art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras  = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch, all_nodes, all_edges, all_subgraphs, all_paras)))))
        old_sources = art_batch
        art_sents = [[tokenizer.tokenize(source_sent) for source_sent in source_sents] for source_sents in art_batch]
        for _i in range(len(art_sents)):
            art_sents[_i][0] = [tokenizer.bos_token] + art_sents[_i][0]
            art_sents[_i][-1] = art_sents[_i][-1] + [tokenizer.eos_token]
        truncated_word_nums = []
        word_nums = [[len(sent) for sent in art_sent] for art_sent in art_sents]
        for word_num in word_nums:
            truncated_word_num = []
            total_count = 0
            for num in word_num:
                if total_count + num < max_src_len:
                    truncated_word_num.append(num)
                else:
                    truncated_word_num.append(max_src_len - total_count)
                    break
                total_count += num
            truncated_word_nums.append(truncated_word_num)
        sources = [list(concat(art_sent))[:max_src_len] for art_sent in art_sents]
        raw_art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        max_sents = list(map(count_max_sent(max_source_num=max_src_len), art_sents))

        inputs = []
        # merge cluster

        nodewords, nodelength, nodefreq, sum_worthy, triples, relations = \
            list(zip(*[process_nodes_bert(align, node, edge, len(source) - 1, max_sent, key='InSalientSent',
                                          adj_type=adj_type,
                                          source_sent=sent, paras=para, subgraphs=subgraph, docgraph=docgraph,
                                          source=source)
                       for node, edge, sent, para, subgraph, source, max_sent in
                       zip(all_nodes, all_edges, old_sources, all_paras, all_subgraphs, sources, max_sents)]))

        # for art_sent, nodes, edges, subgraphs, paras in zip(art_sents, all_nodes, all_edges, all_subgraphs, all_paras):
        #     max_len = len(list(concat(art_sent)))
        #
        #     nodewords, nodelength, nodefreq, sum_worthy, triples, relations, sent_node_aligns = process_nodes(nodes, edges, max_len, max_sent_num=len(list(art_sent)), key=key, adj_type=adj_type)
        #     if paragraph:
        #         nodewords, node_lists, nodefreq, sum_worthy, triples, relations = process_subgraphs(
        #             nodes, edges, subgraphs, paras, max_len, max_sent=len(list(art_sent)),
        #             key=key, adj_type=adj_type
        #         )
        #         sent_align_para = []
        #         last_idx = 0
        #         for sent in range(len(art_sent)):
        #             flag = False
        #             for _idx, para in enumerate(paras):
        #                 if sent in para:
        #                     sent_align_para.append([_idx])
        #                     last_idx = _idx
        #                     flag = True
        #                     break
        #             if not flag:
        #                 sent_align_para.append([last_idx])
        #         assert len(sent_align_para) == len(art_sent)
        #         sent_align_para.append([last_idx + 1])

            # if docgraph:
            #     inputs.append((art_sent, nodewords, nodefreq, triples, relations, sent_node_aligns))
            # elif paragraph:
            #     inputs.append((art_sent, nodewords, nodefreq, triples,
            #                    relations, sent_align_para, node_lists))
            # else:
            #     raise Exception('wrong graph type')
        if docgraph:
            inputs = list(zip(raw_art_sents, sources, nodewords, nodefreq, triples, relations, truncated_word_nums))
        else:
            node_lists = nodelength
            inputs = list(zip(raw_art_sents, sources, nodewords, nodefreq, triples, relations, node_lists, truncated_word_nums))



        assert len(inputs) == len(abs_sents)
        return inputs, abs_sents

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    try:
        with open('/data/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
            align = pickle.load(f)
    except FileNotFoundError:
        with open('/data2/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
            align = pickle.load(f)

    try:
        with open('/data/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl',
                  'rb') as f:
            align2 = pickle.load(f)
    except FileNotFoundError:
        with open('/data2/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl',
                  'rb') as f:
            align2 = pickle.load(f)
    align.update(align2)
    loader = DataLoader(
        RLDataset_graph('train', node_key), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll(tokenizer, align, max_src_len, adj_type)
    )
    val_loader = DataLoader(
        RLDataset_graph('val', node_key), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll(tokenizer, align, max_src_len, adj_type)
    )
    return cycle(loader), val_loader

@curry
def prepro_rl_graph(tokenized_sents, nodes, edges, paras, subgraphs, adj_type='edge_as_node', docgraph=True):
    max_len = len(list(concat(tokenized_sents)))
    _, word_inpara_freq_feat, _, sent_inpara_freq_feat = create_word_freq_in_para_feat(paras, tokenized_sents,
                                                                                       list(concat(tokenized_sents)))
    if docgraph:
        nodewords, nodelength, nodefreq, sum_worthy, triples, relations, sent_node_aligns = process_nodes(nodes, edges,max_len,
                                                                                                      max_sent_num=len(list(tokenized_sents)),
                                                                                                      key='InSalientSent',
                                                                                                      adj_type=adj_type)
        nodes = (nodewords, nodefreq, word_inpara_freq_feat, sent_inpara_freq_feat, triples, relations, sent_node_aligns)
    else:
        nodewords, node_lists, nodefreq, sum_worthy, triples, relations = process_subgraphs(
            nodes, edges, subgraphs, paras, max_len, max_sent=len(list(tokenized_sents)),
            key='InSalientSent', adj_type=adj_type
        )
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
        sent_align_para.append([last_idx + 1])
        nodes = (nodewords, nodefreq, word_inpara_freq_feat, sent_inpara_freq_feat, triples, relations, sent_align_para, node_lists)


    return nodes

def process_nodes_bert(align, nodes, edges, max_len, max_sent_num, node_max_len=30, key='InSalientSent', adj_type='edge_as_node', docgraph=True,
                  source_sent=None, subgraphs=None, paras=None, source=None):
    assert source_sent is not None
    #original_order = ' '.join(source_sent).split(' ')
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
                new_word = ' ' + word
                order_match[i] = list(range(count, count + align[new_word]))
                # test_order_match[new_word] = [count, count + align[new_word]]
                count += align[new_word]
                i += 1



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
        max_sent = max_sent_num
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
        return nodewords, nodelength, nodefreq, sum_worthy, triples, relations
    else:
        return nodewords, node_lists, nodefreq, sum_worthy, triples, relations

@curry
def prepro_rl_graph_bert(align, old_sources, sources, tokenized_sents, max_src_len, nodes, edges, paras, subgraphs, adj_type='edge_as_node', docgraph=True):
    #max_len = len(list(concat(tokenized_sents)))
    max_len = len(sources)
    max_sent = count_max_sent(tokenized_sents, max_source_num=max_src_len)
    if docgraph:
        nodewords, nodelength, nodefreq, sum_worthy, triples, relations = process_nodes_bert(align, nodes, edges, max_len - 1, max_sent,key='InSalientSent',adj_type=adj_type,source_sent=old_sources,docgraph=False,subgraphs=subgraphs,paras=paras,source=sources)
        nodes = (nodewords, nodefreq, triples, relations)
    else:
        nodewords, nodelength, nodefreq, sum_worthy, triples, relations = process_nodes_bert(align, nodes, edges, max_len - 1, max_sent, key='InSalientSent',adj_type=adj_type, source_sent=old_sources, docgraph=False, subgraphs=subgraphs, paras=paras, source=sources)

        nodes = (nodewords, nodefreq, triples, relations, nodelength)


    return nodes