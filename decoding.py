""" decoding utilities"""
import json
import re
import os
from os.path import join
import pickle as pkl
from itertools import starmap

from cytoolz import curry, concat

import torch

from utils import PAD, UNK, START, END
from model.copy_summ import CopySumm
from model.copy_summ_multiencoder import CopySummGAT, CopySummParagraph
from model.extract import ExtractSumm, PtrExtractSumm, NNSESumm, PtrExtractSummGAT, PtrExtractSummSubgraph
from model.rl import ActorCritic, SelfCritic, SelfCriticEntity
from data.batcher import conver2id, pad_batch_tensorize, pad_batch_tensorize_3d
from data.data import CnnDmDataset
from collections import defaultdict
from data.batcher import make_adj_triple, make_adj, make_adj_edge_in
from model.rl_ext import SelfCriticGraph
from data.abs_batcher import create_word_freq_in_para_feat, make_node_lists, count_max_sent
from data.ExtractBatcher import subgraph_make_adj, subgraph_make_adj_edge_in
from toolz.sandbox import unzip
import pickle

MAX_FREQ = 100
BERT_MAX_LEN = 512

try:
    DATASET_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents

class AbsDecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        art_sents = ' '.join(art_sents)
        return [art_sents]

class AbsDecodeDatasetGAT(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, docgraph):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)
        self._docgraph = docgraph

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        if self._docgraph:
            art_sents, nodes, edges = js_data['article'], js_data['nodes_pruned2'], js_data['edges_pruned2']
        else:
            art_sents, nodes, edges = js_data['article'], js_data['nodes'], js_data['edges']
        subgraphs, paras = js_data['subgraphs'], js_data['paragraph_merged']
        #art_sents = ' '.join(art_sents)
        return art_sents, nodes, edges, subgraphs, paras

class DecodeDatasetEntity(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key='filtered_rule23_6_input_mention_cluster'):
        pass



class DecodeDatasetGAT(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)
        assert key in ['nodes', 'nodes_pruned2', 'nodes_sw']
        self._key = key
        self._edge_key = key.replace('nodes', 'edges')

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, nodes, edges, paras, subgraphs = js_data['article'], js_data[self._key], js_data[self._edge_key], js_data['paragraph_merged'], js_data['subgraphs']
        return art_sents, nodes, edges, paras, subgraphs

class DecodeDatasetGATSubgraph(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)
        assert key in ['nodes', 'nodes_pruned2', 'nodes_sw']
        self._key = key
        self._edge_key = key.replace('nodes', 'edges')

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, nodes, edges, subgraphs, paras  = js_data['article'], js_data[self._key], js_data[self._edge_key], \
                                                     js_data['subgraphs'], js_data['paragraph_merged']
        try:
            extracts = js_data['extracted_combine']
        except KeyError:
            extracts = [0]

        return art_sents, nodes, edges, subgraphs, paras, extracts

def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


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


class Abstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True, min_len=0, reverse=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir, reverse)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = CopySumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self._min_len = min_len
        self._bert = abstractor._bert
        self._bert_length = abstractor._bert_max_length
        if self._bert:
            self._tokenizer = abstractor._bert_model._tokenizer
            self._end = self._tokenizer.encoder[self._tokenizer._eos_token]
            self._unk = self._tokenizer.encoder[self._tokenizer._unk_token]
        else:
            self._end = END
            self._unk = UNK

    def _prepro(self, raw_article_sents):
        if self._bert:
            sources = [' '.join(raw_sents) for raw_sents in raw_article_sents]
            sources = [[self._tokenizer.bos_token] + self._tokenizer.tokenize(source)[:self._bert_length - 2] + [self._tokenizer.eos_token] for
                       source in sources]
            stride = 256
            word2id = self._tokenizer.encoder

            unk = word2id[self._tokenizer._unk_token]
            start = self._tokenizer.encoder[self._tokenizer._bos_token]
            end = self._tokenizer.encoder[self._tokenizer._eos_token]
            pad = self._tokenizer.encoder[self._tokenizer._pad_token]
            art_lens = [len(src) for src in sources]
            ext_word2id = dict(word2id)
            ext_id2word = dict(self._tokenizer.decoder)
            for source in sources:
                for word in source:
                    if word not in ext_word2id:
                        ext_word2id[word] = len(ext_word2id)
                        ext_id2word[len(ext_id2word)] = word
            extend_arts = conver2id(unk, ext_word2id, sources)
            if self._bert_length > BERT_MAX_LEN:
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
            extend_vsize = len(ext_word2id)
            article = pad_batch_tensorize(sources, pad, cuda=False
                                          ).to(self._device)
            extend_art = pad_batch_tensorize(extend_arts, pad, cuda=False
                                            ).to(self._device)


            dec_args = (article, art_lens, extend_art, extend_vsize,
                        start, end, unk, self._max_len)
        else:
            ext_word2id = dict(self._word2id)
            ext_id2word = dict(self._id2word)
            for raw_words in raw_article_sents:
                for w in raw_words:
                    if not w in ext_word2id:
                        ext_word2id[w] = len(ext_word2id)
                        ext_id2word[len(ext_id2word)] = w
            articles = conver2id(UNK, self._word2id, raw_article_sents)
            art_lens = [len(art) for art in articles]
            article = pad_batch_tensorize(articles, PAD, cuda=False
                                         ).to(self._device)
            extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
            extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                            ).to(self._device)
            extend_vsize = len(ext_word2id)
            dec_args = (article, art_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        decs, attns = self._net.batch_decode(*dec_args)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for id_, attn in zip(decs, attns):
                if id_[i] == self._end:
                    break
                elif id_[i] == self._unk:
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents

class BeamAbstractorGAT(object):
    def __init__(self, abs_dir, max_len=100, cuda=True, min_len=0, reverse=True, key='summary_worthy', docgraph=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir, reverse)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        print('abs args:', abs_args)
        if docgraph:
            abstractor = CopySummGAT(**abs_args)
        else:
            abstractor = CopySummParagraph(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._cuda = cuda
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self._min_len = min_len
        print('max len: {}, min len {}'.format(self._max_len, self._min_len))
        self._adj_type = self._net._adj_type
        self._mask_type = self._net._mask_type
        self._key = key
        self._copy_from_node = self._net._copy_from_node
        self._docgraph = docgraph
        self._bert = abstractor._bert

        if self._bert:
            self._bert_length = abstractor._bert_max_length
            self._tokenizer = abstractor._bert_model._tokenizer
            try:
                with open('/data/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                    self._align = pickle.load(f)
            except FileNotFoundError:
                with open('/data2/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                    self._align = pickle.load(f)
            try:
                with open('/data/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                    align2 = pickle.load(f)
            except FileNotFoundError:
                with open('/data2/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                    align2 = pickle.load(f)
            self._align.update(align2)
            self._end = self._tokenizer.encoder[self._tokenizer._eos_token]
            self._unk = self._tokenizer.encoder[self._tokenizer._unk_token]
        else:
            self._end = END
            self._unk = UNK


    def __call__(self, batch, beam_size=5, diverse=1.0):
        self._net.eval()
        raw_article_sents = batch[0]
        if self._copy_from_node:
            dec_args, id2word, raw_node_exts = self._prepro_copy_from_node(batch)
        else:
            dec_args, id2word = self._prepro(batch, self._docgraph)
        dec_args = (*dec_args, beam_size, diverse, self._min_len)
        all_beams = self._net.batched_beamsearch(*dec_args)
        if self._copy_from_node:
            all_beams = list(starmap(_process_beam(id2word, unk=self._unk),
                                     zip(all_beams, raw_node_exts)))
        else:
            all_beams = list(starmap(_process_beam(id2word, unk=self._unk),
                                 zip(all_beams, raw_article_sents)))
        return all_beams

    def _prepro(self, batch, docgraph=True):
        raw_article_sents, all_nodes, all_edges, subgraphs, paras, raw_article_batch, max_src_len = batch
        if self._bert:
            sources = [' '.join(raw_sents) for raw_sents in raw_article_batch]
            sources = [[self._tokenizer.bos_token] + self._tokenizer.tokenize(source)[:self._bert_length - 2] + [
                self._tokenizer.eos_token] for
                       source in sources]

            stride = 256
            word2id = self._tokenizer.encoder

            unk = word2id[self._tokenizer._unk_token]
            start = self._tokenizer.encoder[self._tokenizer._bos_token]
            end = self._tokenizer.encoder[self._tokenizer._eos_token]
            pad = self._tokenizer.encoder[self._tokenizer._pad_token]
            art_lens = [len(src) for src in sources]
            ext_word2id = dict(word2id)
            ext_id2word = dict(self._tokenizer.decoder)
            for source in sources:
                for word in source:
                    if word not in ext_word2id:
                        ext_word2id[word] = len(ext_word2id)
                        ext_id2word[len(ext_id2word)] = word
            extend_arts = conver2id(unk, ext_word2id, sources)
            if self._bert_length > BERT_MAX_LEN:
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
            articles = conver2id(unk, word2id, sources)
            extend_vsize = len(ext_word2id)
            article = pad_batch_tensorize(articles, pad, cuda=False
                                          ).to(self._device)
            extend_art = pad_batch_tensorize(extend_arts, pad, cuda=False
                                             ).to(self._device)


        else:
            source_sents = [[sent.lower().split(' ') for sent in arts] for arts in raw_article_batch]
            ext_word2id = dict(self._word2id)
            ext_id2word = dict(self._id2word)
            for raw_words in raw_article_sents:
                for w in raw_words:
                    if not w in ext_word2id:
                        ext_word2id[w] = len(ext_word2id)
                        ext_id2word[len(ext_id2word)] = w
            articles = conver2id(UNK, self._word2id, raw_article_sents)
            art_lens = [len(art) for art in articles]
            article = pad_batch_tensorize(articles, PAD, cuda=False
                                          ).to(self._device)
            extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
            extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                             ).to(self._device)
            extend_vsize = len(ext_word2id)

        @curry
        def prepro_one(graph, node_max_len=30, key='summary_worthy', adj_type='no_edge', max_src_len=None, docgraph=True):
            nodes, edges, article, subgraphs, paras, source_sent = graph
            word_freq_feat = create_word_freq_in_para_feat(paras, source_sent, max_src_len=max_src_len)

            max_len = len(article)
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
            nodewords = []
            nodefreq = []
            sum_worthy = []
            id2node = {}
            ii = 0
            for _id, content in sorted_nodes:
                if _id not in oor_nodes:
                    words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
                    words = [word for word in words if word < max_len]
                    words = words[:node_max_len]
                    try:
                        sum_worthy.append(content[key])
                    except KeyError:
                        sum_worthy.append(0)
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
                        edge_freq.append(1)
                        if adj_type == 'edge_up':
                            nodewords[node1].extend(new_words)
                        elif adj_type == 'edge_down':
                            nodewords[node2].extend(new_words)
                        edge = int(_id.split('_')[1])
                        triples.append([node1, ii, node2])
                        sum_worthy_edges.append(content[key])
                        acticated_nodes.add(content['content']['arg1'])
                        acticated_nodes.add(content['content']['arg2'])
                        id2edge[_id] = ii
                        ii += 1
                        relations.append(new_words)
            if len(relations) == 0:
                # print('warning! no edges in this sample')
                edge_freq = [1]
                relations = [[1]]
                triples = [[0, 0, 1]]
            rlength = [len(words) for words in relations]
            if not docgraph:
                max_sent = count_max_sent(source_sent, max_src_len)
                node_lists, edge_lists, triples = make_node_lists(subgraphs, paras, oor_nodes, id2edge, id2node, max_sent)
            if adj_type == 'edge_as_node':
                node_num = len(nodewords)
                if not docgraph:
                    for i in range(len(triples)):
                        node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]
                nodewords = nodewords + relations
                nodefreq = nodefreq + edge_freq
                nodelength = nodelength + rlength
                sum_worthy = sum_worthy + sum_worthy_edges
            if not docgraph:
                return nodewords, sum_worthy, relations, triples, word_freq_feat, nodefreq, node_lists
            else:
                return nodewords, sum_worthy, relations, triples, word_freq_feat, nodefreq

        @curry
        def prepro_one_bert(graph, node_max_len=30, key='summary_worthy', adj_type='no_edge', max_src_len=None, docgraph=True,
                            tokenizer=None, align=None):
            nodes, edges, article, subgraphs, paras, source = graph
            source_sent = [sent.strip().split() for sent in source]
            source_sent_tokenized = [tokenizer.tokenize(sent) for sent in source]
            # source_sent = source
            # source = ' '.join(source).strip()
            # source = [self._tokenizer.bos_token] + self._tokenizer.tokenize(source)[:max_src_len - 2] + [self._tokenizer.eos_token]
            # target = ' '.join(target).strip()
            # target = self._tokenizer.tokenize(target)[:max_tgt_len]  # will add start and end later

            #original_order = list(concat(source_sent))
            order_match = {}
            count = 1
            i = 0
            # for word in original_order:
            #     order_match[i] = list(range(count, count + align[word]))
            #     count += align[word]
            #     i += 1
            for sents in [' '.join(source)]:
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

            # source_lists = [source[:BERT_MAX_LEN]]
            # length = len(source) - BERT_MAX_LEN
            # i = 1
            # while length > 0:
            #     source_lists.append(source[i * BERT_MAX_LEN - stride:(i + 1) * BERT_MAX_LEN - stride])
            #     i += 1
            #     length -= (BERT_MAX_LEN - stride)

            # word_freq_feat = create_word_freq_in_para_feat(paras, source_sent, max_src_len)
            # assert len(source) == len(word_freq_feat)

            max_len = len(article) - 1
            # max_len = max_src_len - 1
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
                max_sent = count_max_sent(source_sent_tokenized, max_src_len - 2)
                node_lists, edge_lists, triples = make_node_lists(subgraphs, paras, oor_nodes, id2edge, id2node,
                                                                  max_sent)
                if adj_type == 'edge_as_node':
                    node_num = len(nodewords)
                    for i in range(len(triples)):
                        node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]

            if adj_type == 'edge_as_node':
                nodewords = nodewords + relations
                nodelength = nodelength + rlength
                nodefreq = nodefreq + edge_freq
                sum_worthy = sum_worthy + sum_worthy_edges

            if not docgraph:
                return nodewords, sum_worthy, relations, triples, nodefreq, node_lists
            else:
                return nodewords, sum_worthy, relations, triples, nodefreq


        if self._bert:
            batch_data = list(zip(all_nodes, all_edges, extend_arts, subgraphs, paras, raw_article_batch))
            batch = list(map(prepro_one_bert(key=self._key, adj_type=self._adj_type, max_src_len=max_src_len, docgraph=docgraph,
                                        tokenizer=self._tokenizer, align=self._align), batch_data))
            if docgraph:
                nodes, sum_worthy, edges, triples, nodefreqs = list(zip(*batch))
            else:
                nodes, sum_worthy, edges, triples, nodefreqs, node_lists = list(zip(*batch))
        else:
            batch_data = list(zip(all_nodes, all_edges, articles, subgraphs, paras, source_sents))
            batch = list(
                map(prepro_one(key=self._key, adj_type=self._adj_type, max_src_len=max_src_len, docgraph=docgraph),
                    batch_data))
            if docgraph:
                nodes, sum_worthy, edges, triples, word_freq_feats, nodefreqs = list(zip(*batch))
            else:
                nodes, sum_worthy, edges, triples, word_freq_feats, nodefreqs, node_lists = list(zip(*batch))
        node_num = [len(_node) for _node in nodes]
        _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=False).to(self._device)
        sum_worthy = pad_batch_tensorize(sum_worthy, pad=0, cuda=False).float().to(self._device)
        _relations = pad_batch_tensorize_3d(edges, pad=0, cuda=False).to(self._device)
        nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=False).ne(-1).float().to(self._device)
        rmask = pad_batch_tensorize_3d(edges, pad=-1, cuda=False).ne(-1).float().to(self._device)
        # features
        nodefreq = pad_batch_tensorize(nodefreqs, pad=0, cuda=False).to(self._device)
        if not self._bert:
            word_freq = pad_batch_tensorize(word_freq_feats, pad=0, cuda=False).to(self._device)
            feature_dict = {'word_inpara_freq': word_freq,
                        'node_freq': nodefreq}
        else:
            feature_dict = {'node_freq': nodefreq}


        if docgraph:
            if self._adj_type == 'concat_triple':
                adjs = [make_adj_triple(triple, len(node), len(relation), self._cuda) for triple, node, relation in
                        zip(triples, nodes, edges)]
            elif self._adj_type== 'edge_as_node':
                adjs = [make_adj_edge_in(triple, len(node), len(relation), self._cuda) for triple, node, relation in
                        zip(triples, nodes, edges)]
            else:
                adjs = [make_adj(triple, len(node), len(node), self._cuda) for triple, node, relation in
                        zip(triples, nodes, edges)]
        else:
            if self._adj_type == 'edge_as_node':
                adjs = list(map(subgraph_make_adj_edge_in(cuda=self._cuda), zip(triples, node_lists)))
            else:
                adjs = list(map(subgraph_make_adj(cuda=self._cuda), zip(triples, node_lists)))

        if docgraph:
            node_info = (_nodes, nmask, node_num, sum_worthy, feature_dict)
        else:
            node_info = (_nodes, nmask, node_num, sum_worthy, feature_dict, node_lists)
        edge_info = (_relations, rmask, triples, adjs)

        if self._bert:
            dec_args = (article, art_lens, extend_art, extend_vsize,
                    node_info, edge_info, None,
                    start, end, unk, self._max_len)
        else:
            dec_args = (article, art_lens, extend_art, extend_vsize,
                        node_info, edge_info, None,
                        START, END, UNK, self._max_len)

        return dec_args, ext_id2word


    def _prepro_copy_from_node(self, batch):
        raw_article_sents, all_nodes, all_edges = batch

        @curry
        def prepro_one(graph, node_max_len=30, key='summary_worthy', adj_type='no_edge'):
            nodes, edges, article = graph
            max_len = len(article)
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
            nodewords = []
            sum_worthy = []
            id2node = {}
            ii = 0
            for _id, content in sorted_nodes:
                if _id not in oor_nodes:
                    words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
                    words = [word for word in words if word < max_len]
                    words = words[:node_max_len]
                    if len(words) != 0:
                        sum_worthy.append(content[key])
                        nodewords.append(words)
                        id2node[_id] = ii
                        ii += 1
                    else:
                        oor_nodes.append(_id)
            if len(nodewords) == 0:
                # print('warning! no nodes in this sample')
                nodewords = [[0], [2]]
                sum_worthy.extend([0, 0])
            nodelength = [len(words) for words in nodewords]
            # process edges
            acticated_nodes = set()
            triples = []
            relations = []
            sum_worthy_edges = []
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
                        if adj_type == 'edge_up':
                            nodewords[node1].extend(new_words)
                        elif adj_type == 'edge_down':
                            nodewords[node2].extend(new_words)
                        edge = int(_id.split('_')[1])
                        triples.append([node1, ii, node2])
                        sum_worthy_edges.append(content[key])
                        acticated_nodes.add(content['content']['arg1'])
                        acticated_nodes.add(content['content']['arg2'])
                        ii += 1
                        relations.append(new_words)
            if len(relations) == 0:
                # print('warning! no edges in this sample')
                relations = [[1]]
                triples = [[0, 0, 1]]
                sum_worthy_edges.append(0)
            rlength = [len(words) for words in relations]
            if adj_type == 'edge_as_node':
                nodewords = nodewords + relations
                nodelength = nodelength + rlength
                sum_worthy = sum_worthy + sum_worthy_edges

            return nodewords, sum_worthy, relations, triples





        batch_data = list(zip(all_nodes, all_edges, raw_article_sents))
        batch = list(map(prepro_one(key=self._key, adj_type=self._adj_type), batch_data))
        nodes, sum_worthies, edges, triples = list(zip(*batch))

        all_node_words = [list(concat(nodeword)) for nodeword in nodes]  # position in article
        sum_worhies = [list(sum_worhy) for sum_worhy in sum_worthies]
        gold_copy_masks = []
        ext_node_aligns = []
        for _bid, nodeword in enumerate(list(nodes)):
            ext_node_align = []
            gold_mask = []
            for _i, words in enumerate(nodeword):
                align = [_i for _ in range(len(words))]
                ext_node_align.extend(align)
                _mask = [1 if sum_worhies[_bid][_i] else 0 for _ in range(len(words))]
                gold_mask.extend(_mask)
            gold_copy_masks.append(gold_mask)
            ext_node_aligns.append(ext_node_align)
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for _bid, words in enumerate(all_node_words):
            for word in words:
                original_word = raw_article_sents[_bid][word]
                if original_word not in ext_word2id:
                    ext_word2id[original_word] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = original_word
        src_exts = conver2id(UNK, ext_word2id, raw_article_sents)
        raw_node_exts = []
        for _i, words in enumerate(all_node_words):
            node_ext = []
            for word in words:
                node_ext.append(raw_article_sents[_i][word])
            raw_node_exts.append(node_ext)
        node_exts = conver2id(UNK, ext_word2id, raw_node_exts)
        node_exts = pad_batch_tensorize(node_exts, pad=0, cuda=False).to(self._device)

        extend_vsize = len(ext_word2id)
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                      ).to(self._device)
        # extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        # extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
        #                                  ).to(self._device)


        node_num = [len(_node) for _node in nodes]
        _nodes = pad_batch_tensorize_3d(nodes, pad=0, cuda=False).to(self._device)
        sum_worthy = pad_batch_tensorize(sum_worthies, pad=0, cuda=False).float().to(self._device)
        _relations = pad_batch_tensorize_3d(edges, pad=0, cuda=False).to(self._device)
        nmask = pad_batch_tensorize_3d(nodes, pad=-1, cuda=False).ne(-1).float().to(self._device)
        rmask = pad_batch_tensorize_3d(edges, pad=-1, cuda=False).ne(-1).float().to(self._device)

        all_node_word = pad_batch_tensorize(all_node_words, pad=0, cuda=False).to(self._device)
        all_node_mask = pad_batch_tensorize(all_node_words, pad=-1, cuda=False).ne(-1).float().to(self._device)
        ext_node_aligns = pad_batch_tensorize(ext_node_aligns, pad=0, cuda=False).to(self._device)
        gold_copy_mask = pad_batch_tensorize(gold_copy_masks, pad=0, cuda=False).float().to(self._device)




        if self._adj_type == 'concat_triple':
            adjs = [make_adj_triple(triple, len(node), len(relation), self._cuda) for triple, node, relation in
                    zip(triples, nodes, edges)]
        elif self._adj_type== 'edge_as_node':
            adjs = [make_adj_edge_in(triple, len(node), len(relation), self._cuda) for triple, node, relation in
                    zip(triples, nodes, edges)]
        else:
            adjs = [make_adj(triple, len(node), len(node), self._cuda) for triple, node, relation in
                    zip(triples, nodes, edges)]
        node_info = (_nodes, nmask, node_num, sum_worthy)
        edge_info = (_relations, rmask, triples, adjs)
        ext_info = (all_node_word, all_node_mask, ext_node_aligns, gold_copy_mask)

        dec_args = (article, art_lens, node_exts, extend_vsize,
                    node_info, edge_info, ext_info,
                    START, END, UNK, self._max_len)


        return dec_args, ext_id2word, raw_node_exts









class BeamAbstractor(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse, self._min_len)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word, unk=self._unk),
                                 zip(all_beams, raw_article_sents)))
        return all_beams

class BeamAbstractor_cnn(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse, self._min_len)
        all_beams = self._net.batched_beamsearch_cnn(*dec_args)
        all_beams = list(starmap(_process_beam(id2word, unk=self._unk),
                                 zip(all_beams, raw_article_sents)))
        return all_beams

@curry
def _process_beam(id2word, beam, art_sent, unk=UNK):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == unk:
                copy_word = art_sent[max(range(len(art_sent)),
                                         key=lambda j: attn[j].item())]
                seq.append(copy_word)
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        del hyp.attns
        #del hyp.coverage
        return hyp
    return list(map(process_hyp, beam))


class Extractor(object):
    def __init__(self, ext_dir, max_ext=5, cuda=True, force_ext=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        if ext_meta['net'] == 'ml_ff_extractor':
            ext_cls = ExtractSumm
        elif ext_meta['net'] == 'ml_rnn_extractor':
            ext_cls = PtrExtractSumm
        elif ext_meta['net'] == 'ml_nnse_extractor':
            ext_cls = NNSESumm
        elif ext_meta["net_args"]['extractor']['net'] == "ml_rnn_extractor":
            ext_cls = PtrExtractSumm
        else:
            raise ValueError()
        ext_ckpt = load_best_ckpt(ext_dir)
        ext_args = ext_meta['net_args']
        if ext_args.__contains__('extractor'):
            ext_args = ext_meta['net_args']['extractor']['net_args']
        extractor = ext_cls(**ext_args)
        extractor.load_state_dict(ext_ckpt)
        word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext
        self.force_ext = force_ext
        try:
            self._bert = self._net._bert
        except:
            self._bert = False

    def __call__(self, raw_article_sents):
        if not self._bert:
            self._net.eval()
            n_art = len(raw_article_sents)
            articles = conver2id(UNK, self._word2id, raw_article_sents)
            article = pad_batch_tensorize(articles, PAD, cuda=False, max_num=5
                                         ).to(self._device)
        else:
            self._net.eval()
            article = raw_article_sents
            n_art = 1
        if not self.force_ext:
            indices = self._net.extract([article], k=min(n_art, self._max_ext), force_ext=self.force_ext)
        else:
            indices = self._net.extract([article], k=min(n_art, self._max_ext))
        return indices

class ExtractorGAT(object):
    def __init__(self, ext_dir, max_ext=5, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        self._subgraph = False
        if ext_meta['net'] == 'ml_gat_extractor':
            ext_cls = PtrExtractSummGAT
        elif ext_meta['net'] == 'ml_subgraph_gat_extractor':
            ext_cls = PtrExtractSummSubgraph
            self._subgraph = True
            print(ext_meta['net'])
        elif ext_meta['net_args']['extractor']['net'] == 'ml_gat_extractor':
            ext_cls = PtrExtractSummGAT
        else:
            raise ValueError()
        ext_ckpt = load_best_ckpt(ext_dir)
        if ext_meta['net'] in ['ml_gat_extractor', 'ml_subgraph_gat_extractor']:
            ext_args = ext_meta['net_args']
            word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        elif ext_meta['net_args']['extractor']['net'] == 'ml_gat_extractor':
            ext_args = ext_meta['net_args']['extractor']['net_args']
            word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        else:
            raise ValueError()
        print(ext_args)
        extractor = ext_cls(**ext_args)
        extractor.load_state_dict(ext_ckpt)

        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        #print(self._net)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext
        try:
            self._bert = self._net._bert
        except:
            self._bert = False

    def __call__(self, article, article_sents, nodes, edges, output_attn=False):
        self._net.eval()
        n_art = 1
        if self._bert:
            if output_attn:
                indices, attns = self._net.extract(article, [article_sents], nodes, edges, k=min(n_art, self._max_ext),
                                            output_attn=output_attn)
                return indices, attns
            else:
                indices = self._net.extract(article, [article_sents], nodes, edges, k=min(n_art, self._max_ext), output_attn=output_attn)
                return indices

        else:
            article_sents = conver2id(UNK, self._word2id, article_sents)
            word2id = defaultdict(lambda: UNK, self._word2id)
            article = [word2id[word] for word in article]
            article = pad_batch_tensorize([article], PAD, cuda=False, max_num=5
                                                ).to(self._device)
            article_sents = pad_batch_tensorize(article_sents, PAD, cuda=False, max_num=5
                                          ).to(self._device)
            if output_attn:
                indices, attns = self._net.extract(article, [article_sents], nodes, edges, k=min(n_art, self._max_ext),
                                            output_attn=output_attn)
                return indices, attns
            else:
                indices = self._net.extract(article, [article_sents], nodes, edges, k=min(n_art, self._max_ext), output_attn=output_attn)
                return indices



class ExtractorEntity(object):
    def __init__(self):
        pass

class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False, max_num=5
                                     ).to(self._device)
        return article

class ArticleBatcherGraph(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        full_article = list(concat(articles))
        full_article = pad_batch_tensorize([full_article], PAD, cuda=False).to(self._device)
        article = pad_batch_tensorize(articles, PAD, cuda=False, max_num=5
                                     ).to(self._device)
        return article, full_article


class RLExtractor(object):
    def __init__(self, ext_dir, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        assert ext_meta['net'] == 'rnn-ext_abs_rl'
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda))
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}

    def __call__(self, raw_article_sents):
        self._net.eval()
        indices = self._net(raw_article_sents)
        return indices

class SCExtractor(object):
    def __init__(self, ext_dir, cuda=True, docgraph=False, paragraph=False):
        assert not all([docgraph, paragraph])
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        assert ext_meta['net'] == 'rnn-ext_abs_rl'
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        if docgraph:
            extractor = PtrExtractSummGAT(**ext_args)
            agent = SelfCriticGraph(extractor,
                                    ArticleBatcherGraph(word2id, cuda),
                                    cuda,
                                    docgraph,
                                    paragraph
                               )
        else:
            extractor = PtrExtractSummSubgraph(**ext_args)
            agent = SelfCriticGraph(extractor,
                                    ArticleBatcherGraph(word2id, cuda),
                                    cuda,
                                    docgraph,
                                    paragraph
            )
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}

        #self.entity = entity

    def __call__(self, raw_article_sents):
        self._net.eval()
        with torch.no_grad():
            indices, _, _ = self._net(raw_article_sents, validate=True)
        return indices
