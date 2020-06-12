import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from data.batcher import pad_batch_tensorize
from .attention import step_attention, badanau_attention, copy_from_node_attention, hierarchical_attention
from .util import len_mask, sequence_mean
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs
from utils import PAD, UNK, START, END
from model.graph_enc import gat_encode, node_mask
from model.extract import MeanSentEncoder
from model.rnn import lstm_multiembedding_encoder
from model.graph_enc import subgraph_encode
from model.roberta import RobertaEmbedding
from .rnn import MultiLayerLSTMCells

MAX_FREQ = 100
INIT = 1e-2
BERT_MAX_LEN = 512

class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, side_dim1, side_dim2=None, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        self._v_s1 = nn.Parameter(torch.Tensor(side_dim1))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        init.uniform_(self._v_s1, -INIT, INIT)
        if side_dim2 is not None:
            self._v_s2 = nn.Parameter(torch.Tensor(side_dim2))
            init.uniform_(self._v_s2, -INIT, INIT)
        else:
            self._v_s2 = None

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self._b = None

    def forward(self, context, state, input_, side1, side2=None):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1))
                  + torch.matmul(side1, self._v_s1.unsqueeze(1)))
        if side2 is not None and self._v_s2 is not None:
            output += torch.matmul(side2, self._v_s2.unsqueeze(1))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output



class CopySummGAT(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, side_dim, dropout=0.0, gat_args={},
                 adj_type='no_edge', mask_type='none', feed_gold=False,
                 graph_layer_num=1, copy_from_node=False, copy_bank='node',
                 feature_banks=[], bert=False, bert_length=512, decoder_supervision=False):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)
        self._bert = bert
        feat_emb_dim = emb_dim // 4
        if self._bert:
            self._bert_model = RobertaEmbedding()
            self._embedding = self._bert_model._embedding
            self._embedding.weight.requires_grad = False
            emb_dim = self._embedding.weight.size(1)
            self._bert_max_length = bert_length
            self._enc_lstm = nn.LSTM(
                emb_dim, n_hidden, n_layer,
                bidirectional=bidirectional, dropout=dropout
            )
            self._projection = nn.Sequential(
                nn.Linear(2 * n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, emb_dim, bias=False)
            )
            self._dec_lstm = MultiLayerLSTMCells(
                emb_dim * 2, n_hidden, n_layer, dropout=dropout
            )
            self._bert_stride = 256

        if copy_from_node:
            self._copy = _CopyLinear(n_hidden, n_hidden, 2*emb_dim, side_dim)
        else:
            self._copy = _CopyLinear(n_hidden, n_hidden, 2*emb_dim, side_dim, side_dim)
        self._feature_banks = feature_banks

        graph_hsz = n_hidden
        if 'nodefreq' in self._feature_banks:
            graph_hsz += feat_emb_dim
            self._node_freq_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        gat_args['graph_hsz'] = graph_hsz
        self._graph_layer_num = graph_layer_num
        self._graph_enc = nn.ModuleList([gat_encode(gat_args) for _ in range(self._graph_layer_num)])
        self._node_enc = MeanSentEncoder()

        if feed_gold or mask_type == 'encoder':
            self._graph_mask = node_mask(mask_type='gold')
        elif mask_type == 'soft':
            self._graph_mask = nn.ModuleList([node_mask(mask_type=mask_type, emb_dim=graph_hsz*(i+1)) for i in range(self._graph_layer_num+1)])
        else:
            self._graph_mask = node_mask(mask_type='none')

        self._gold = feed_gold
        self._adj_type = adj_type
        self._mask_type = mask_type
        self._copy_bank = copy_bank

        enc_lstm_in_dim = emb_dim
        if 'freq' in self._feature_banks:
            enc_lstm_in_dim += feat_emb_dim
            self._freq_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        if 'inpara_freq' in self._feature_banks:
            enc_lstm_in_dim += feat_emb_dim
            self._inpara_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(
            enc_lstm_in_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )

        # node attention
        self._attn_s1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_wm = nn.Parameter(torch.Tensor(graph_hsz, n_hidden))
        self._attns_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attns_wm)
        init.xavier_normal_(self._attns_wq)
        init.xavier_normal_(self._attn_s1)
        init.uniform_(self._attns_v, -INIT, INIT)
        self._graph_proj = nn.Linear(graph_hsz, graph_hsz)

        if copy_from_node:
            self._projection_decoder = nn.Sequential(
                nn.Linear(4 * n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, emb_dim, bias=False)
            )
        else:
            self._projection_decoder = nn.Sequential(
                nn.Linear(3 * n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, emb_dim, bias=False)
            )
        self._decoder_supervision = False

        self._copy_from_node = copy_from_node
        if self._copy_from_node:
            self._attn_copyx = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self._attn_copyn = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self._attn_copyh = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self._attn_copyv = nn.Parameter(torch.Tensor(n_hidden))
            init.xavier_normal_(self._attn_copyx)
            init.xavier_normal_(self._attn_copyh)
            init.xavier_normal_(self._attn_copyn)
            init.uniform_(self._attn_copyv, -INIT, INIT)
        else:
            self._attn_copyx = None
            self._attn_copyn = None
            self._attn_copyh = None
            self._attn_copyv = None

        self._decoder = CopyLSTMDecoderGAT(
            self._copy, self._attn_s1, self._attns_wm, self._attns_wq, self._attn_v, copy_from_node,
            self._attn_copyh, self._attn_copyv, None, None, False,
            self._embedding, self._dec_lstm, self._attn_wq, self._projection_decoder, self._attn_wb, self._attn_v,
        )



    def forward(self, article, art_lens, abstract, extend_art, extend_vsize, ninfo, rinfo, ext_ninfo=None):
        (nodes, nmask, node_num, sw_mask, feature_dict) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        if self._copy_from_node:
            (all_node_words, all_node_mask, ext_node_aligns, gold_copy_mask) = ext_ninfo
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None



        if self._mask_type == 'soft' or self._mask_type == 'none':
            nodes, masks = self._encode_graph(attention, nodes, nmask, relations,
                                              rmask, triples, adjs, node_num, node_mask=None, nodefreq=feature_dict['node_freq'])
        else:
            nodes = self._encode_graph(attention, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])

        if self._copy_from_node:
            bs = attention.size(0)
            nnum = all_node_words.size(1)
            d_word = attention.size(-1)
            all_node_words = all_node_words.unsqueeze(2).expand(bs, nnum, d_word)
            ext_node_words = attention.gather(1, all_node_words)
            ext_node_words = ext_node_words * all_node_mask.unsqueeze(2)
            ext_node_aligns = ext_node_aligns.unsqueeze(2).expand(bs, nnum, d_word)
            ext_node_reps = nodes.gather(1, ext_node_aligns)
            ext_node_words = torch.matmul(ext_node_words, self._attn_copyx)
            ext_node_reps = torch.matmul(ext_node_reps, self._attn_copyn)
            if self._copy_bank == 'gold':
                ext_info = (ext_node_words, ext_node_reps, gold_copy_mask)
            elif self._copy_bank == 'node':
                ext_info = (ext_node_words, ext_node_reps, all_node_mask)
        else:
            ext_info = None


        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        logit, selections = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract,
            nodes,
            node_num,
            sw_mask,
            False,
            ext_info
        )
        logit = (logit, )
        if 'soft' in self._mask_type:
            logit += (masks, )


        return logit

    def encode(self, article, feature_dict, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        feature_embeddings = {}
        if 'freq' in self._feature_banks:
            feature_embeddings['freq'] = self._freq_embedding
        if 'inpara_freq' in self._feature_banks:
            feature_embeddings['inpara_freq'] = self._inpara_embedding
        if self._bert:
            if self._bert_max_length > 512:
                source_nums = art_lens
            with torch.no_grad():
                bert_out = self._bert_model(article)
            bert_hidden = bert_out[0]
            if self._bert_max_length > 512:
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
            #article = self._bert_relu(self._bert_linear(bert_hidden))
            article = bert_hidden
            enc_art, final_states = lstm_multiembedding_encoder(
                article, self._enc_lstm, art_lens,
                init_enc_states, None, {}, {}
            )
        else:
            enc_art, final_states = lstm_multiembedding_encoder(
                article, self._enc_lstm, art_lens,
                init_enc_states, self._embedding,
                feature_embeddings, feature_dict
            )
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out)

    def encode_bert(self, article, feature_dict, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        feature_embeddings = {}
        if 'freq' in self._feature_banks:
            feature_embeddings['freq'] = self._freq_embedding
        if 'inpara_freq' in self._feature_banks:
            feature_embeddings['inpara_freq'] = self._inpara_embedding

        enc_art, final_states = lstm_multiembedding_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding,
            feature_embeddings, feature_dict
        )
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out)

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, triples, adjs, node_num, node_mask=None, nodefreq=None):
        d_word = articles.size(-1)

        masks = []
        bs, n_node, n_word = nodes.size()
        nodes = nodes.view(bs, -1).unsqueeze(2).expand(bs, n_node * n_word, d_word)
        nodes = articles.gather(1, nodes).view(bs, n_node, n_word, d_word).contiguous()
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)
        nodes = self._node_enc(nodes, mask=nmask)
        if 'nodefreq' in self._feature_banks:
            assert nodefreq is not None
            nodefreq = self._node_freq_embedding(nodefreq)
            nodes = torch.cat([nodes, nodefreq], dim=-1)


        nodes_no_mask = nodes
        if self._mask_type == 'encoder':
            nodes, node_mask = self._graph_mask(nodes, node_mask)
        elif self._mask_type == 'soft':
            nodes, node_mask = self._graph_mask[0](nodes, _input=nodes)
            masks.append(node_mask.squeeze(2))
        # print('node_mask:', node_mask[0:2, :])
        # print('n_mask:', nmask[0:2, :])
        # print('triples:', triples[0:2])
        init_nodes = nodes

        if self._adj_type == 'triple':
            bs, nr, nw = relations.size()
            edges = relations.view(bs, -1).unsqueeze(2).expand(bs, nr * nw, d_word)
            edges = articles.gather(1, edges).view(bs, nr, nw, d_word)
            rmask = rmask.unsqueeze(3).expand(bs, nr, nw, d_word)
            edges = self._node_enc(edges, mask=rmask)
        else:
            edges = nodes

        for i_layer in range(self._graph_layer_num):
            if self._adj_type == 'concat_triple':

                triple_reps = []
                for batch_id, ts in enumerate(triples):
                    if self._mask_type == 'encoder':
                        triple_reps.append(
                            torch.stack(
                                [
                                    torch.cat([nodes[batch_id, i, :],
                                               edges[batch_id, j, :] * node_mask[batch_id, i] * node_mask[batch_id, k],
                                               nodes[batch_id, k, :]], dim=-1)
                                    for i, j, k in ts
                                ])
                        )
                    else:
                        triple_reps.append(
                            torch.stack(
                            [
                                torch.cat([nodes[batch_id, i, :],
                                           # edges[batch_id, j, :] * node_mask[batch_id, i] * node_mask[batch_id, k],
                                           edges[batch_id, j, :],
                                           nodes[batch_id, k, :]], dim=-1)
                                for i,j,k in ts
                            ])
                        )
            else:
                triple_reps = nodes

            #print('before layer {}, nodes: {}'.format(i_layer, nodes[0:2,:,:10]))

            nodes, edges = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
            if self._mask_type == 'encoder':
                nodes, node_mask = self._graph_mask(nodes, node_mask)
            elif self._mask_type == 'soft':
                if i_layer == 0:
                    _input = nodes_no_mask
                _input = torch.cat([nodes, nodes_no_mask], dim=-1)
                original_nodes = nodes
                nodes, node_mask = self._graph_mask[i_layer+1](nodes, _input=_input)
                masks.append(node_mask.squeeze(2))
                nodes_no_mask = torch.cat([nodes_no_mask, original_nodes], dim=-1)

        # add initial reps
        nodes = self._graph_proj(init_nodes) + nodes
        if 'soft' in self._mask_type:
            return nodes, masks
        else:
            return nodes

    def encode_general(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len):
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        _nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                          None, None, adjs, node_num, node_mask=None,
                                          nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)

        return attention, init_dec_states, _nodes

    def rl_step(self, article, art_lens, extend_art, extend_vsize,
                     _ns, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len, attention, init_dec_states, nodes, sample=False):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        tok = torch.LongTensor([go] * batch_size).to(article.device)
        outputs = []
        attns = []
        seqLogProbs = []
        states = init_dec_states
        for i in range(max_len):
            if sample:
                tok, states, attn_score, sampleProb = self._decoder.sample_step(
                    tok, states, attention, nodes, node_num)
                seqLogProbs.append(sampleProb)
            else:
                tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                tok, states, attention, nodes, node_num)
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        if sample:
            return outputs, attns, seqLogProbs
        else:
            return outputs, attns

    def ml_step(self, abstract, node_num, attention, init_dec_states, nodes):
        logit, selections = self._decoder(
            attention,
            init_dec_states, abstract,
            nodes,
            node_num,
            None,
            False,
            None
        )
        return logit

    def greedy(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len, tar_in): #deprecated
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                              None, None, adjs, node_num, node_mask=None, nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                tok, states, attention, nodes, node_num)
            #print('greedy tok:', tok.size())
            if i == 0:
                unfinished = (tok != eos)
                #print('greedy tok:', tok)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns

    def sample(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len, abstract, ml): #deprecated
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                          None, None, adjs, node_num, node_mask=None,
                                          nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = ((init_dec_states[0][0].clone(), init_dec_states[0][1].clone()), init_dec_states[1].clone())
        seqLogProbs = []
        for i in range(max_len):
            tok, states, attn_score, sampleProb = self._decoder.sample_step(
                tok, states, attention, nodes, node_num)
            #print('sample tok:', tok)
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok = tok.masked_fill(tok >= vsize, unk)
            seqLogProbs.append(sampleProb)
            if unfinished.data.sum() == 0:
                break

        if ml:
            assert abstract is not None
            logit, _ = self._decoder(
                attention,
                init_dec_states, abstract,
                nodes,
                node_num,
                None,
                False,
                None
            )
            return outputs, attns, seqLogProbs, logit

        return outputs, attns, seqLogProbs

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           ninfo, rinfo, ext_ninfo,
                           go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
        (nodes, nmask, node_num, sw_mask, feature_dict) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        if self._copy_from_node:
            (all_node_words, all_node_mask, ext_node_aligns, gold_copy_mask) = ext_ninfo
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        if self._mask_type == 'soft':
            nodes, masks = self._encode_graph(attention, nodes, nmask, relations,
                                              rmask, triples, adjs, node_num, node_mask=None, nodefreq=feature_dict['node_freq'])
        else:
            nodes = self._encode_graph(attention, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])

        if self._copy_from_node:
            batch_size = attention.size(0)
            nnum = all_node_words.size(1)
            d_word = attention.size(-1)
            all_node_words = all_node_words.unsqueeze(2).expand(batch_size, nnum, d_word)
            ext_node_words = attention.gather(1, all_node_words)
            ext_node_words = ext_node_words * all_node_mask.unsqueeze(2)
            ext_node_aligns = ext_node_aligns.unsqueeze(2).expand(batch_size, nnum, d_word)
            ext_node_reps = nodes.gather(1, ext_node_aligns)
            ext_node_words = torch.matmul(ext_node_words, self._attn_copyx)
            ext_node_reps = torch.matmul(ext_node_reps, self._attn_copyn)
            if self._copy_bank == 'gold':
                ext_info = (ext_node_words, ext_node_reps, gold_copy_mask)
            elif self._copy_bank == 'node':
                ext_info = (ext_node_words, ext_node_reps, all_node_mask)
            ext_info = ([ext_info[0][i, :, :] for i in range(batch_size)],
                        [ext_info[1][i, :, :] for i in range(batch_size)],
                        [ext_info[2][i, :] for i in range(batch_size)])
        else:
            ext_info = None



        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        max_node_num = max(node_num)
        if sw_mask is not None:
            all_nodes = [(nodes[i, :, :], node_num[i], sw_mask[i, :]) for i in range(len(node_num))]
        else:
            all_nodes = [(nodes[i, :, :], node_num[i]) for i in range(len(node_num))]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            filtered_nodes = torch.stack([all_nodes[i][0] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            filtered_node_num = [all_nodes[i][1] for i, _beam in enumerate(all_beams) if _beam != []]
            if sw_mask is not None:
                filtered_sw_mask = torch.stack([all_nodes[i][2] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            else:
                filtered_sw_mask = None

            if self._copy_from_node:
                filtered_ext_info = (torch.stack([ext_info[0][i] for i, _beam in enumerate(all_beams) if _beam != []], dim=0),
                                     torch.stack([ext_info[1][i] for i, _beam in enumerate(all_beams) if _beam != []], dim=0),
                                     torch.stack([ext_info[2][i] for i, _beam in enumerate(all_beams) if _beam != []], dim=0))
            else:
                filtered_ext_info = None

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, filtered_nodes, filtered_node_num,
                max_node_num=max_node_num, side_mask=filtered_sw_mask,  force_not_stop=force_not_stop, filtered_ext_info=filtered_ext_info, eos=eos)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                    ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs





class CopyLSTMDecoderGAT(AttentionalLSTMDecoder):
    def __init__(self, copy, attn_s1, attns_wm, attns_wq, attns_v,
                 copy_from_node, attn_copyh=None, attn_copyv=None,
                 para_wm=None, para_v=None, hierarchical_attention=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy
        self._attn_s1 = attn_s1
        self._attns_wm = attns_wm
        self._attns_wq = attns_wq
        self._attns_v = attns_v
        self._attn_copyh = attn_copyh
        self._attn_copyv = attn_copyv
        self._copy_from_node = copy_from_node

        self._hierarchical_attn = hierarchical_attention
        if self._hierarchical_attn:
            assert para_wm is not None and para_v is not None
            self._para_wm = para_wm
            self._para_v = para_v


    def __call__(self, attention, init_states, target, nodes, node_num, side_mask=None, output_attn=None, ext_info=None, paras=None):
        max_len = target.size(1)
        states = init_states
        logits = []
        score_ns = []
        for i in range(max_len):
            tok = target[:, i:i+1]
            logit, states, _, score_n, out_attns = self._step(tok, states, attention, nodes, node_num,
                                                              side_mask=side_mask, output_attn=True, ext_info=ext_info, paras=paras)
            logits.append(logit)
            score_ns.append(score_n)
        logit = torch.stack(logits, dim=1)
        return logit, score_ns

    def _step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None, paras=None):
        out_attentions = {}
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention
        if self._copy_from_node:
            ext_node_words, ext_node_reps, all_node_mask = ext_info
            query_copy = torch.matmul(lstm_out, self._attn_copyh)
            context_node, score_copy = copy_from_node_attention(query_copy, ext_node_words, ext_node_reps, ext_node_words,
                                                                mem_mask=all_node_mask.unsqueeze(-2), v=self._attn_copyv)


        # nodes attention
        if self._hierarchical_attn:
            node_reps, node_length, para_node_aligns = paras
            para_reps = nodes
            node_reps = torch.matmul(node_reps, self._attns_wm)
            para_reps = torch.matmul(para_reps, self._para_wm)
            query_s = torch.mm(lstm_out, self._attns_wq)
            nmask = len_mask(node_length, attention.device).unsqueeze(-2)
            para_length = node_num
        else:
            nodes = torch.matmul(nodes, self._attns_wm)
            query_s = torch.mm(lstm_out, self._attns_wq)
            nmask = len_mask(node_num, attention.device).unsqueeze(-2)
        if self._hierarchical_attn:
            side_n, score_n = hierarchical_attention(query_s, node_reps, node_reps, self._attn_v, para_reps, self._para_v, para_node_aligns, mem_mask=nmask, hierarchical_length=para_length)
        else:
            if side_mask is not None:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=side_mask.unsqueeze(-2), v=self._attns_v, sigmoid=False)
            else:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v, sigmoid=False)
        out_attentions['node_attention'] = score_n.detach()
        side_n = torch.mm(side_n, self._attn_s1)

        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)
        out_attentions['word_attention'] = score.detach()

        if self._copy_from_node:
            dec_out = self._projection(torch.cat([lstm_out, context, side_n, context_node], dim=1))
        else:
            dec_out = self._projection(torch.cat([lstm_out, context, side_n], dim=1))
            score_copy = score

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        if self._copy_from_node:
            copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in, side_n, context_node))
        else:
            copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in, side_n))
        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy),
                src=score_copy * copy_prob
        ) + 1e-10)  # numerical stability for log
        if output_attn:
            return lp, (states, dec_out), score, score_n, out_attentions
        else:
            return lp, (states, dec_out), score, score_n

    def decode_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None, paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn, ext_info, paras)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score, score_n

    def sample_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None, paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn, ext_info, paras)
        #logprob = F.log_softmax(logit, dim=1)
        logprob = logit
        score = torch.exp(logprob)
        #out = torch.multinomial(score, 1).detach()
        out = torch.multinomial(score, 1)
        sampleProb = logprob.gather(1, out)
        return out, states, score, sampleProb

    def topk_step(self, tok, states, attention, k, nodes, node_num, max_node_num, side_mask=None, force_not_stop=False, filtered_ext_info=None, filtered_para_info=None, eos=END):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        out_attentions = {}
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]
        query = torch.matmul(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # nodes attention
        if self._hierarchical_attn:
            node_reps, node_length, para_node_aligns, max_subgraph_node_num = filtered_para_info
            para_reps = nodes
            node_reps = torch.matmul(node_reps, self._attns_wm)
            para_reps = torch.matmul(para_reps, self._para_wm)
            query_s = torch.matmul(lstm_out, self._attns_wq)
            nmask = len_mask(node_length, attention.device, max_num=max_subgraph_node_num).unsqueeze(-2)
            para_length = node_num
        else:
            nodes = torch.matmul(nodes, self._attns_wm)
            query_s = torch.matmul(lstm_out, self._attns_wq)
            nmask = len_mask(node_num, attention.device, max_num=max_node_num).unsqueeze(-2)
        if self._hierarchical_attn:
            side_n, score_n = hierarchical_attention(query_s, node_reps, node_reps, self._attn_v, para_reps, self._para_v, para_node_aligns, mem_mask=nmask,
                                                     hierarchical_length=para_length, max_para_num=max_node_num)
        else:
            if side_mask is not None:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=side_mask.unsqueeze(-2), v=self._attns_v)
            else:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v)

        if self._copy_from_node:
            ext_node_words, ext_node_reps, all_node_mask = filtered_ext_info
            query_copy = torch.matmul(lstm_out, self._attn_copyh)
            context_node, score_copy = copy_from_node_attention(query_copy, ext_node_words, ext_node_reps, ext_node_words,
                                                                mem_mask=all_node_mask.unsqueeze(-2), v=self._attn_copyv)

        out_attentions['node_attention'] = score_n.detach()
        side_n = torch.matmul(side_n, self._attn_s1)


        # attention is beamable

        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)
        if self._copy_from_node:
            dec_out = self._projection(torch.cat([lstm_out, context, side_n, context_node], dim=-1))
        else:
            dec_out = self._projection(torch.cat([lstm_out, context, side_n], dim=-1))
            score_copy = score
        #dec_out = self._projection(torch.cat([lstm_out, context, side_n], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)

        if self._copy_from_node:
            copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in_beamable, side_n, context_node)).contiguous().view(-1, 1)
        else:
            copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in_beamable, side_n)).contiguous().view(-1, 1)
        # copy_prob = torch.sigmoid(
        #     self._copy(context, lstm_out, lstm_in_beamable, side_n)
        # ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy).contiguous().view(
                    beam*batch, -1),
                src=score_copy.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)
        if force_not_stop:
            lp[:, :, eos] = -1e8

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score_copy

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        logit = torch.mm(dec_out, self._embedding.weight.t())
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize-vsize
                                    ).to(logit.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy


class CopySummParagraph(CopySummGAT):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, side_dim, dropout=0.0, gat_args={},
                 adj_type='no_edge', mask_type='none', pe=False, feed_gold=False,
                 graph_layer_num=1, copy_from_node=False, copy_bank='node',
                 feature_banks=[], hierarchical_attn=False, bert=False, bert_length=512, decoder_supervision=False):
        super().__init__(vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, side_dim, dropout, gat_args,
                 adj_type, mask_type, pe, feed_gold,
                 graph_layer_num, copy_from_node, copy_bank, feature_banks, bert, bert_length)

        feat_emb_dim = emb_dim // 4
        graph_hsz = n_hidden
        if 'nodefreq' in self._feature_banks:
            graph_hsz += feat_emb_dim
        gat_args['graph_hsz'] = graph_hsz

        gat_args['graph_hsz'] = graph_hsz

        self._graph_enc = subgraph_encode(gat_args)

        self._bert = bert
        if self._bert:
            self._bert_model = RobertaEmbedding()
            self._embedding = self._bert_model._embedding
            self._embedding.weight.requires_grad = False
            emb_dim = self._embedding.weight.size(1)
            self._bert_max_length = bert_length
            self._enc_lstm = nn.LSTM(
                emb_dim, n_hidden, n_layer,
                bidirectional=bidirectional, dropout=dropout
            )
            self._projection = nn.Sequential(
                nn.Linear(2 * n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, emb_dim, bias=False)
            )
            self._dec_lstm = MultiLayerLSTMCells(
                emb_dim * 2, n_hidden, n_layer, dropout=dropout
            )
            self._bert_stride = 256



        if mask_type == 'encoder':
            self._graph_mask = node_mask(mask_type='gold')
        elif mask_type == 'soft':
            self._graph_mask = node_mask(mask_type=mask_type, emb_dim=graph_hsz)
        else:
            self._graph_mask = node_mask(mask_type='none')

        self._hierarchical_attn = hierarchical_attn
        if self._hierarchical_attn:
            self._para_wm = nn.Parameter(torch.Tensor(graph_hsz, n_hidden))
            self._para_v = nn.Parameter(torch.Tensor(n_hidden))
            init.xavier_normal_(self._para_wm)
            init.uniform_(self._para_v, -INIT, INIT)
        else:
            self._para_wm = None
            self._para_v = None

        self._decoder = CopyLSTMDecoderGAT(
            self._copy, self._attn_s1, self._attns_wm, self._attns_wq, self._attn_v, copy_from_node,
            self._attn_copyh, self._attn_copyv, self._para_wm, self._para_v, self._hierarchical_attn,
            self._embedding, self._dec_lstm, self._attn_wq, self._projection_decoder, self._attn_wb, self._attn_v
        )



    def forward(self, article, art_lens, abstract, extend_art, extend_vsize, ninfo, rinfo, ext_ninfo=None):
        (nodes, nmask, node_num, sw_mask, feature_dict, node_lists) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None

        if self._mask_type == 'soft' or self._mask_type == 'none':
            outputs = self._encode_graph(attention, nodes, nmask, relations,
                                              rmask, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        else:
            outputs = self._encode_graph(attention, nodes, nmask, relations, rmask, adjs, node_lists, sw_mask,
                                       nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None
        ext_info = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        logit, selections = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract,
            topics[0],
            topics[1],
            sw_mask,
            False,
            ext_info,
            paras
        )
        logit = (logit,)
        if 'soft' in self._mask_type:
            logit += (masks,)

        return logit

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, batch_adjs, node_lists, node_mask=None, nodefreq=None):
        d_word = articles.size(-1)

        masks = []
        bs, n_node, n_word = nodes.size()
        nodes = nodes.view(bs, -1).unsqueeze(2).expand(bs, n_node * n_word, d_word)
        nodes = articles.gather(1, nodes).view(bs, n_node, n_word, d_word).contiguous()
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)
        nodes = self._node_enc(nodes, mask=nmask)
        if 'nodefreq' in self._feature_banks:
            assert nodefreq is not None
            nodefreq = self._node_freq_embedding(nodefreq)
            nodes = torch.cat([nodes, nodefreq], dim=-1)
        if self._mask_type == 'encoder':
            nodes, node_mask = self._graph_mask(nodes, node_mask)
        elif self._mask_type == 'soft':
            nodes, node_mask = self._graph_mask(nodes, _input=nodes)
            masks.append(node_mask.squeeze(2))

        # topics, topic_length = self._graph_enc(batch_adjs, nodes, node_lists)
        if self._hierarchical_attn:
            (topics, topic_length), (node_reps, node_length, node_align_paras) = self._graph_enc(batch_adjs, nodes, node_lists, output_node_rep=True)
        else:
            topics, topic_length = self._graph_enc(batch_adjs, nodes, node_lists)

        results = ((topics, topic_length),)

        if 'soft' in self._mask_type:
            results += (masks,)

        if self._hierarchical_attn:
            node_para_aligns = pad_batch_tensorize(node_align_paras, pad=0, cuda=False).to(node_reps.device)
            results += ((node_reps, node_length, node_para_aligns),)

        return results
        # if 'soft' in self._mask_type:
        #     return (topics, topic_length), masks
        # else:
        #     return (topics, topic_length)


    def greedy(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, node_lists, adjs,
                     go, eos, unk, max_len, tar_in):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)
        #
        # (topics, topiclength), masks = self._encode_graph(attention, nodes, nmask,
        #                                       None, None, adjs, node_lists, None, nodefreq=feature_dict['node_freq'])

        outputs = self._encode_graph(attention, nodes, nmask, None,
                                              None, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None



        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                tok, states, attention, topics[0], topics[1], paras=paras)
            #print('greedy tok:', tok.size())
            if i == 0:
                unfinished = (tok != eos)
                #print('greedy tok:', tok)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score)
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns

    def sample(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, node_lists, adjs,
                     go, eos, unk, max_len, abstract, ml):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        # (topics, topiclength), masks = self._encode_graph(attention, nodes, nmask,
        #                                                   None, None, adjs, node_lists, None,
        #                                                   nodefreq=feature_dict['node_freq'])
        outputs = self._encode_graph(attention, nodes, nmask, None,
                                     None, adjs, node_lists, node_mask=None,
                                     nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            paras = None
        else:
            topics = outputs
            paras = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = init_dec_states
        seqLogProbs = []
        for i in range(max_len):
            tok, states, attn_score, sampleProb = self._decoder.sample_step(
                tok, states, attention, topics[0], topics[1], paras=paras)
            #print('sample tok:', tok)
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok = tok.masked_fill(tok >= vsize, unk)
            seqLogProbs.append(sampleProb)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns, seqLogProbs

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     ninfo, rinfo, ext_ninfo,
                     go, eos, unk, max_len, beam_size, diverse=1.0, min_len=0):
        """ greedy decode support batching"""

        return 0

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           ninfo, rinfo, ext_ninfo,
                           go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
        (nodes, nmask, node_num, sw_mask, feature_dict, node_lists) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        if self._copy_from_node:
            (all_node_words, all_node_mask, ext_node_aligns, gold_copy_mask) = ext_ninfo
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        if self._mask_type == 'soft' or self._mask_type == 'none':
            outputs = self._encode_graph(attention, nodes, nmask, relations,
                                              rmask, adjs, node_lists, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        else:
            outputs = self._encode_graph(attention, nodes, nmask, relations, rmask, adjs, node_lists, sw_mask,
                                       nodefreq=feature_dict['node_freq'])
        if self._hierarchical_attn:
            topics, masks, paras = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
            nodes = topics[0]
            node_num = topics[1]
            paras = None
        else:
            topics = outputs
            nodes = topics[0]
            node_num = topics[1]
            paras = None
        ext_info = None

        # if self._mask_type == 'soft' or self._mask_type == 'none':
        #     (topics, topiclength), masks = self._encode_graph(attention, nodes, nmask, relations,
        #                                       rmask, adjs, node_lists, node_mask=None,
        #                                       nodefreq=feature_dict['node_freq'])
        # else:
        #     (topics, topiclength) = self._encode_graph(attention, nodes, nmask, relations, rmask, adjs, node_lists, sw_mask,
        #                                nodefreq=feature_dict['node_freq'])
        # nodes = topics
        # node_num = topiclength


        ext_info = None



        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        if self._hierarchical_attn:
            max_node_num = max(topics[1])
        else:
            max_node_num = max(node_num)
        if self._hierarchical_attn:
            all_nodes = [(topics[0][i, :, :], topics[1][i]) for i in range(len(topics[1]))]
            all_paras = [(paras[0][i, :, :], paras[1][i], paras[2][i, :]) for i in range(len(paras[1]))]
            max_subgraph_node_num = max(paras[1])
        else:
            if sw_mask is not None:
                all_nodes = [(nodes[i, :, :], node_num[i], sw_mask[i, :]) for i in range(len(node_num))]
            else:
                all_nodes = [(nodes[i, :, :], node_num[i]) for i in range(len(node_num))]

        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            filtered_nodes = torch.stack([all_nodes[i][0] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            filtered_node_num = [all_nodes[i][1] for i, _beam in enumerate(all_beams) if _beam != []]
            if sw_mask is not None:
                filtered_sw_mask = torch.stack([all_nodes[i][2] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            else:
                filtered_sw_mask = None
            if self._hierarchical_attn:
                filtered_paras = [torch.stack([all_paras[i][0] for i, _beam in enumerate(all_beams) if _beam != []], dim=0),
                                  [all_paras[i][1] for i, _beam in enumerate(all_beams) if _beam != []],
                                  torch.stack([all_paras[i][2] for i, _beam in enumerate(all_beams) if _beam != []],dim=0),
                                  max_subgraph_node_num]
            else:
                filtered_paras = None


            filtered_ext_info = None

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, filtered_nodes, filtered_node_num,
                max_node_num=max_node_num, side_mask=filtered_sw_mask,  force_not_stop=force_not_stop, filtered_ext_info=filtered_ext_info, filtered_para_info=filtered_paras, eos=eos)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                    ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs