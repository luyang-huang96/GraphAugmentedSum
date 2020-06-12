from model.extract import Bert_model, MeanSentEncoder
from model.graph_enc import gat_encode
from torch import nn
import torch.nn.functional as F
import torch

BERT_MAX_LEN = 512

class graph_classification(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=256):
        super().__init__()
        self._input_dim = input_dim
        self._out_dim = out_dim

        self._linear1 = nn.Linear(input_dim, hidden_dim)
        self._linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, nodes):
        out = F.relu(self._linear1(nodes))
        out = self._linear2(out)
        if self._out_dim == 1:
            out = out.squeeze(2)

        return out


class GraphClassification(nn.Module):
    def __init__(self, out_dim, emb_dim=300, bertmodel='bert-base-uncased', bert_stride=256, baseline=False, gat_args={}):
        super().__init__()
        self._bertmodel = bertmodel
        self._bert_stride = bert_stride
        self._emb_dim = emb_dim
        if 'bert-large' in bertmodel:
            self._bert_linear = nn.Linear(4096, emb_dim)
        else:
            self._bert_linear = nn.Linear(3072, emb_dim)
        self._bert_relu = nn.PReLU()
        self._bert_model = Bert_model(bertmodel)
        self._class_layer = graph_classification(emb_dim, out_dim)

        self._node_enc = MeanSentEncoder()
        self._baseline = baseline
        if not baseline:
            self._graph_layer_num = 2
            self._graph_enc = nn.ModuleList([gat_encode(gat_args) for _ in range(self._graph_layer_num)])


    def forward(self, sent_nums, target, articles, ninfo, rinfo):
        articles, word_nums = articles
        (nodes, nmask, node_num) = ninfo
        (relations, rmask, triples, adjs) = rinfo

        articles, _ = self._encode_bert(articles, word_nums)

        nodes = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num)
        out = self._class_layer(nodes)

        return out





    def _encode_bert(self, articles, word_nums):
        source_nums = [sum(word_num) for word_num in word_nums]
        with torch.no_grad():
            bert_out = self._bert_model(articles)
        bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
        bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
        del bert_out
        hsz = bert_hidden.size(2)
        batch_id = 0
        max_source = max(source_nums)
        bert_hiddens = []
        max_len = bert_hidden.size(1)
        for source_num in source_nums:
            source = torch.zeros(max_source, hsz).to(bert_hidden.device)
            if source_num < BERT_MAX_LEN:
                source[:source_num, :] += bert_hidden[batch_id, :source_num, :]
                batch_id += 1
            else:
                source[:BERT_MAX_LEN, :] += bert_hidden[batch_id, :BERT_MAX_LEN, :]
                batch_id += 1
                start = BERT_MAX_LEN
                while start < source_num:
                    # print(start, source_num, max_source)
                    if start - self._bert_stride + BERT_MAX_LEN < source_num:
                        end = start - self._bert_stride + BERT_MAX_LEN
                        batch_end = BERT_MAX_LEN
                    else:
                        end = source_num
                        batch_end = source_num - start + self._bert_stride
                    source[start:end, :] += bert_hidden[batch_id, self._bert_stride:batch_end, :]
                    batch_id += 1
                    start += (BERT_MAX_LEN - self._bert_stride)
            bert_hiddens.append(source)
        bert_hidden = torch.stack(bert_hiddens)
        articles = bert_hidden
        article_sents = []
        # for i, word_num in enumerate(word_nums):
        #     max_word_num = max(word_num)
        #     if max_word_num < 5:  # in case smaller than CNN stride
        #         max_word_num = 5
        #     new_word_num = []
        #     start_num = 0
        #     for num in word_num:
        #         new_word_num.append((start_num, start_num + num))
        #         start_num += num
        #     article_sents.append(
        #         torch.stack(
        #             [torch.cat([bert_hidden[i, num[0]:num[1], :],
        #                         torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
        #              if (num[1] - num[0]) != max_word_num
        #              else bert_hidden[i, num[0]:num[1], :]
        #              for num in new_word_num
        #              ]
        #         )
        #     )
        return articles, article_sents

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, triples, adjs, node_num):
        d_word = articles.size(-1)

        bs, n_node, n_word = nodes.size()
        nodes = nodes.view(bs, -1).unsqueeze(2).expand(bs, n_node * n_word, d_word)
        nodes = articles.gather(1, nodes).view(bs, n_node, n_word, d_word).contiguous()
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)
        nodes = self._node_enc(nodes, mask=nmask)

        bs, nr, nw = relations.size()
        edges = relations.view(bs, -1).unsqueeze(2).expand(bs, nr * nw, d_word)
        edges = articles.gather(1, edges).view(bs, nr, nw, d_word)
        rmask = rmask.unsqueeze(3).expand(bs, nr, nw, d_word)
        edges = self._node_enc(edges, mask=rmask)
        # print('layer {} nodes: {}'.format(-1, nodes))
        # print('layer {} edges: {}'.format(-1, edges))

        if not self._baseline:
            for i_layer in range(self._graph_layer_num):
                triple_reps = []
                for batch_id, ts in enumerate(triples):
                    triple_reps.append(
                        torch.stack(
                        [
                            torch.cat([nodes[batch_id, i, :],
                                       edges[batch_id, j, :],
                                       nodes[batch_id, k, :]], dim=-1)
                            for i,j,k in ts
                        ])
                    )

                nodes, edges = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
                # print('layer {} nodes: {}'.format(i_layer, nodes))
                # print('layer {} edges: {}'.format(i_layer, edges))

        return nodes

class GraphClassification(nn.Module):
    def __init__(self, out_dim, emb_dim=300, bertmodel='bert-base-uncased', bert_stride=256, baseline=False, gat_args={}):
        super().__init__()
        self._bertmodel = bertmodel
        self._bert_stride = bert_stride
        self._emb_dim = emb_dim
        if 'bert-large' in bertmodel:
            self._bert_linear = nn.Linear(4096, emb_dim)
        else:
            self._bert_linear = nn.Linear(3072, emb_dim)
        self._bert_relu = nn.PReLU()
        self._bert_model = Bert_model(bertmodel)
        self._class_layer = graph_classification(emb_dim, out_dim)

        self._node_enc = MeanSentEncoder()
        self._baseline = baseline
        if not baseline:
            self._graph_layer_num = 2
            self._graph_enc = nn.ModuleList([gat_encode(gat_args) for _ in range(self._graph_layer_num)])


    def forward(self, sent_nums, target, articles, ninfo, rinfo):
        articles, word_nums = articles
        (nodes, nmask, node_num) = ninfo
        (relations, rmask, triples, adjs) = rinfo

        articles, _ = self._encode_bert(articles, word_nums)

        nodes = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num)
        out = self._class_layer(nodes)

        return out





    def _encode_bert(self, articles, word_nums):
        source_nums = [sum(word_num) for word_num in word_nums]
        with torch.no_grad():
            bert_out = self._bert_model(articles)
        bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
        bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
        del bert_out
        hsz = bert_hidden.size(2)
        batch_id = 0
        max_source = max(source_nums)
        bert_hiddens = []
        max_len = bert_hidden.size(1)
        for source_num in source_nums:
            source = torch.zeros(max_source, hsz).to(bert_hidden.device)
            if source_num < BERT_MAX_LEN:
                source[:source_num, :] += bert_hidden[batch_id, :source_num, :]
                batch_id += 1
            else:
                source[:BERT_MAX_LEN, :] += bert_hidden[batch_id, :BERT_MAX_LEN, :]
                batch_id += 1
                start = BERT_MAX_LEN
                while start < source_num:
                    # print(start, source_num, max_source)
                    if start - self._bert_stride + BERT_MAX_LEN < source_num:
                        end = start - self._bert_stride + BERT_MAX_LEN
                        batch_end = BERT_MAX_LEN
                    else:
                        end = source_num
                        batch_end = source_num - start + self._bert_stride
                    source[start:end, :] += bert_hidden[batch_id, self._bert_stride:batch_end, :]
                    batch_id += 1
                    start += (BERT_MAX_LEN - self._bert_stride)
            bert_hiddens.append(source)
        bert_hidden = torch.stack(bert_hiddens)
        articles = bert_hidden
        article_sents = []
        # for i, word_num in enumerate(word_nums):
        #     max_word_num = max(word_num)
        #     if max_word_num < 5:  # in case smaller than CNN stride
        #         max_word_num = 5
        #     new_word_num = []
        #     start_num = 0
        #     for num in word_num:
        #         new_word_num.append((start_num, start_num + num))
        #         start_num += num
        #     article_sents.append(
        #         torch.stack(
        #             [torch.cat([bert_hidden[i, num[0]:num[1], :],
        #                         torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
        #              if (num[1] - num[0]) != max_word_num
        #              else bert_hidden[i, num[0]:num[1], :]
        #              for num in new_word_num
        #              ]
        #         )
        #     )
        return articles, article_sents

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, triples, adjs, node_num):
        d_word = articles.size(-1)

        bs, n_node, n_word = nodes.size()
        nodes = nodes.view(bs, -1).unsqueeze(2).expand(bs, n_node * n_word, d_word)
        nodes = articles.gather(1, nodes).view(bs, n_node, n_word, d_word).contiguous()
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)
        nodes = self._node_enc(nodes, mask=nmask)

        bs, nr, nw = relations.size()
        edges = relations.view(bs, -1).unsqueeze(2).expand(bs, nr * nw, d_word)
        edges = articles.gather(1, edges).view(bs, nr, nw, d_word)
        rmask = rmask.unsqueeze(3).expand(bs, nr, nw, d_word)
        edges = self._node_enc(edges, mask=rmask)
        # print('layer {} nodes: {}'.format(-1, nodes))
        # print('layer {} edges: {}'.format(-1, edges))

        if not self._baseline:
            for i_layer in range(self._graph_layer_num):
                triple_reps = []
                for batch_id, ts in enumerate(triples):
                    triple_reps.append(
                        torch.stack(
                        [
                            torch.cat([nodes[batch_id, i, :],
                                       edges[batch_id, j, :],
                                       nodes[batch_id, k, :]], dim=-1)
                            for i,j,k in ts
                        ])
                    )

                nodes, edges = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
                # print('layer {} nodes: {}'.format(i_layer, nodes))
                # print('layer {} edges: {}'.format(i_layer, edges))

        return nodes