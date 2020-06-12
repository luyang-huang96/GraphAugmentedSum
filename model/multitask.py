from model.extract import Bert_model, MeanSentEncoder
from model.graph_enc import gat_encode
from torch import nn
import torch.nn.functional as F
import torch
from model.auxilary import graph_classification
from model.extract import LSTMEncoder, MeanSentEncoder
from torch.nn import init
from .rnn import MultiLayerLSTMCells

BERT_MAX_LEN = 512
INI = 1e-2

class LSTMPointerNet_multitask(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden,
                 dropout, side_dim, attention_type):
        # attention type: seneca, bidaf, mask
        assert attention_type in ['seneca', 'bidaf', 'mask']
        n_layer = 1
        n_hop = 1
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

        # side info attention
        if attention_type == 'seneca':
            self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
            self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self.side_v = nn.Parameter(torch.Tensor(n_hidden))
            init.xavier_normal_(self.side_wm)
            init.xavier_normal_(self.side_wq)
            init.uniform_(self.side_v, -INI, INI)


        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_ws)


        # pad entity put in graph enc now
        # self._pad_entity = nn.Parameter(torch.Tensor(side_dim))
        # init.uniform_(self._pad_entity)

        # stop token
        self._stop = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._stop, -INI, INI)


    def forward(self, attn_mem, mem_sizes, lstm_in, side_mem, side_sizes):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""

        batch_size, max_sent_num, input_dim = attn_mem.size()
        side_dim = side_mem.size(2)

        attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
        for i, sent_num in enumerate(mem_sizes):
            attn_mem[i, sent_num, :] += self._stop
        mem_sizes = [mem_size+1 for mem_size in mem_sizes]

        # side_mem = torch.cat([side_mem, torch.zeros(batch_size, 1, side_dim).to(side_mem.device)], dim=1) #b * ns * s
        # for i, side_size in enumerate(side_sizes):
        #     side_mem[i, side_size, :] += self._pad_entity
        # side_sizes = [side_size+1 for side_size in side_sizes]


        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)

        side_feat = self._prepare_side(side_mem) #b * ns * side_h


        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)


        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet_multitask.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

        side_e = LSTMPointerNet_multitask.attention(side_feat, query, self.side_v, self.side_wq, side_sizes)

        output = LSTMPointerNet_multitask.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq, self._attn_ws)

        return output  # unormalized extraction logit


    def extract(self, attn_mem, mem_sizes, k, side_mem, side_sizes):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        # side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)

        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        if not self._hard_attention:
            side_feat = self._prepare_side(side_mem)
        else:
            side_feat = side_mem
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        if self._hard_attention:
            max_side = side_mem.size(1)
            side_dim = side_mem.size(2)
            context = self._start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)

        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet_multitask.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            side_e = LSTMPointerNet_multitask.attention(side_feat, query, self.side_v, self.side_wq, side_sizes)
                #print('context:', context)
            score = LSTMPointerNet_multitask.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                              self._attn_ws)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            if ext == max_sent and sent_num != 0:
                break
            elif sent_num == 0 and ext == max_sent:
                ext = score.topk(2, dim=0)[1][1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    def _prepare_side(self, side_mem):
        side_feat = torch.matmul(side_mem, self.side_wm.unsqueeze(0))
        return side_feat

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention_wiz_side(attention, query, side, v, w, s):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2) + torch.matmul(side, s.unsqueeze(0)).unsqueeze(2)
        # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet_entity.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


    @staticmethod
    def hard_attention(attention, query, w_bi, wq, _start, ground_truth=None):
        """ attention context vector"""
        # ground truth B * Nsent * Nside
        # attention B * Nside * Side
        # output = ground_truth.unsqueeze(3) * attention.unsqueeze(1) # B*Nsent*Nside*Side teacher forcing
        # output = output.sum(dim=2) # B*Nsent*Side
        side_dim = attention.size(2)
        n_side = attention.size(1)
        batch_size = attention.size(0)
        n_sent = query.size(1)
        all_output = torch.zeros(batch_size, n_sent, side_dim).to(attention.device)
        context = _start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
        all_selection = torch.zeros(batch_size, n_sent, n_side).to(attention.device)
        for sent_id in range(n_sent):
            _query = query[:, sent_id, :].unsqueeze(1)
            bilinear = w_bi(context.unsqueeze(2).repeat(1,1,n_side,1), attention.unsqueeze(1).repeat(1, 1, 1, 1)).squeeze(3) # B*1*Nside
            selection = bilinear + torch.matmul(_query, wq.unsqueeze(0))
            all_selection[:, sent_id, :] = selection.squeeze(1)
            selected = F.sigmoid(selection) # B*1*Nside
            selected = selected.gt(0.5).float()
            output = selected.unsqueeze(3) * attention.unsqueeze(1)
            output = output.sum(dim=2)  # B*Nsent*Side
            all_output[:, sent_id, :] = output.squeeze(1)
            context = context + output

        return all_output, all_selection




class PtrExtractSummGAT_multitask(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size,
                 lstm_hidden, dropout=0.0, bert=False, bert_stride=0, bertmodel='bert-large-uncased-whole-word-masking', gat_args={}):
        super().__init__()
        enc_out_dim = emb_dim

        self._sent_enc = MeanSentEncoder()
        self._node_enc = MeanSentEncoder()

        self._graph_layer_num = 2
        self._graph_enc = nn.ModuleList([gat_encode(gat_args) for _ in range(self._graph_layer_num)])


        self._art_enc = LSTMEncoder(
            emb_dim, emb_dim / 2, 1,
            dropout=dropout, bidirectional=True
        )
        self._extractor = LSTMPointerNet_entity(
            emb_dim, lstm_hidden, 1,
            dropout, 1, emb_dim
        )


        self._bert = bert
        self._bertmodel = bertmodel
        self._bert_stride = bert_stride
        if bert:
            self._emb_dim = emb_dim
            if 'bert-large' in bertmodel:
                self._bert_linear = nn.Linear(4096, emb_dim)
            else:
                self._bert_linear = nn.Linear(3072, emb_dim)
            self._bert_relu = nn.PReLU()
            self._bert_model = Bert_model(bertmodel)

    def forward(self, sent_nums, target, articles, ninfo, rinfo):
        articles, word_nums = articles
        (nodes, nmask, node_num) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        #get bert embedding
        if self._bert:
            articles, article_sents = self._encode_bert(articles, word_nums)
        else:
            raise ('not implemented yet')


        enc_out = self._encode(article_sents, sent_nums)
        nodes = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num)


        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )

        output = self._extractor(enc_out, sent_nums, ptr_in, nodes, node_num)


        return output


    def extract(self, article_sents, clusters, sent_nums=None, cluster_nums=None, k=4, force_ext=True):
        # node implemented yet
        enc_out = self._encode(article_sents, sent_nums)
        if not self._context:
            entity_out = self._encode_entity(clusters, cluster_nums)
        else:
            entity_out = self._encode_entity(clusters, cluster_nums, enc_out)
        _, _, (entity_out, entity_mask) = self._graph_enc(clusters[3], clusters[4], (
            entity_out, torch.tensor(cluster_nums, device=entity_out.device)))

        #cluster_nums = [cluster_num + 1 for cluster_num in cluster_nums]

        output = self._extractor.extract(enc_out, sent_nums, k, entity_out, cluster_nums)
        return output

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

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
        for i, word_num in enumerate(word_nums):
            max_word_num = max(word_num)
            if max_word_num < 5:  # in case smaller than CNN stride
                max_word_num = 5
            new_word_num = []
            start_num = 0
            for num in word_num:
                new_word_num.append((start_num, start_num + num))
                start_num += num
            article_sents.append(
                torch.stack(
                    [torch.cat([bert_hidden[i, num[0]:num[1], :],
                                torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
                     if (num[1] - num[0]) != max_word_num
                     else bert_hidden[i, num[0]:num[1], :]
                     for num in new_word_num
                     ]
                )
            )
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

    def set_embedding(self, embedding):
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)