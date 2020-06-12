import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet, LSTMPointerNet_entity
from data.batcher import pad_batch_tensorize
from data.ExtractBatcher import subgraph_make_adj_edge_in, subgraph_make_adj
from pytorch_transformers import BertTokenizer, BertModel, BertConfig
from model.extract import PtrExtractSummGAT, PtrExtractSummSubgraph
from data.batcher import make_adj_edge_in, make_adj_triple, make_adj
from cytoolz import identity, curry, concat


INI = 1e-2
BERT_MAX_LEN = 512

class ArticleBatcherGraph_bert(object):
    def __init__(self, tokenizer, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._tokenizer = tokenizer
        self._device = torch.device('cuda' if cuda else 'cpu')


    def __call__(self, raw_article_sents):
        tokenized_sents = raw_article_sents
        stride = 256
        tokenized_sents_lists = [tokenized_sents[:BERT_MAX_LEN]]
        length = len(tokenized_sents) - BERT_MAX_LEN
        i = 1
        while length > 0:
            tokenized_sents_lists.append(tokenized_sents[i * BERT_MAX_LEN - stride:(i + 1) * BERT_MAX_LEN - stride])
            i += 1
            length -= (BERT_MAX_LEN - stride)
        id_sents = [self._tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]

        pad = self._tokenizer.encoder[self._tokenizer._pad_token]

        sources = pad_batch_tensorize(id_sents, pad=pad, cuda=False).to(self._device)

        return sources

class SCExtractorRLGraph(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet_entity)

        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # side info attention
        self.side_wm = nn.Parameter(ptr_net.side_wm.clone())
        self.side_wq = nn.Parameter(ptr_net.side_wq.clone())
        self.side_v = nn.Parameter(ptr_net.side_v.clone())

        self._n_hop = ptr_net._n_hop
        self._side_attn_type = ptr_net._side_attn_type

        self._attn_ws = nn.Parameter(ptr_net._attn_ws.clone())

        self._pad_entity = nn.Parameter(ptr_net._pad_entity.clone())

        try:
            self._stop = nn.Parameter(ptr_net._stop.clone())
        except KeyError:
            print('Initialize stop tensor')
            self._stop = nn.Parameter(torch.Tensor(ptr_net._hop_wm.size(0)))
            init.uniform_(self._stop, -INI, INI)

        self._hierarchical_attn = ptr_net._hierarchical_attn
        if self._hierarchical_attn:
            self.para_wm = nn.Parameter(ptr_net.para_wm)
            self.para_wq = nn.Parameter(ptr_net.para_wq)
            self.para_v = nn.Parameter(ptr_net.para_v)
            self._pad_para = nn.Parameter(ptr_net._pad_para)



    def forward(self, attn_mem, side_mem, aligns=None, paras=None):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        side_dim = side_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        if self._hierarchical_attn:
            assert paras is not None
            para_reps, para_node_aligns, para_length = paras
            para_node_aligns = torch.cat([para_node_aligns, torch.zeros(batch_size, 1).long().to(para_node_aligns.device)], dim=1)
            para_reps = torch.cat([para_reps, torch.zeros(batch_size, 1, side_dim).to(para_reps.device)], dim=1)
            for i, para_size in enumerate(para_length):
                para_reps[i, para_size, :] += self._pad_para
                para_node_aligns[i, para_size] = para_size
            para_sizes = [para_size+1 for para_size in para_length]


        side_feat = self._prepare_side(side_mem)
        if self._hierarchical_attn:
            para_feat = self._prepare_side(para_reps)
        # print('side feat:', side_feat)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        if self._side_attn_type == 'one-hop':
            assert aligns is not None
            side_sent_reps = []
            for sent2node in aligns[0]:
                side_sent_reps.append(side_feat[0, sent2node, :].mean(dim=-2))
            side_sent_reps = torch.stack(side_sent_reps, dim=0).unsqueeze(0)
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)

            if self._side_attn_type == 'two-hop':
                if self._hierarchical_attn:
                    side_e = LSTMPointerNet_entity.hierarchical_attention(query, para_feat, side_feat, self.para_v,
                                                                          self.para_wq,
                                                                          self.side_v, self.side_wq,
                                                                          para_node_aligns, None, para_sizes)
                else:
                    side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
                score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                             self._attn_ws)
            elif self._side_attn_type == 'one-hop':
                score = LSTMPointerNet_entity.attention_onehop(attn_feat, query, side_sent_reps, self._attn_v,
                                                               self._attn_wq)

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

    def sample(self, attn_mem, side_mem, aligns=None, paras=None):
        """sample k sentences, decode only, batch_size==1"""
        eps = 1e-8
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        side_dim = side_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)

        side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        if self._hierarchical_attn:
            assert paras is not None
            para_reps, para_node_aligns, para_length = paras
            para_node_aligns = torch.cat([para_node_aligns, torch.zeros(batch_size, 1).long().to(para_node_aligns.device)], dim=1)
            para_reps = torch.cat([para_reps, torch.zeros(batch_size, 1, side_dim).to(para_reps.device)], dim=1)
            for i, para_size in enumerate(para_length):
                para_reps[i, para_size, :] += self._pad_para
                para_node_aligns[i, para_size] = para_size
            para_sizes = [para_size+1 for para_size in para_length]

        side_feat = self._prepare_side(side_mem)
        if self._hierarchical_attn:
            para_feat = self._prepare_side(para_reps)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        log_scores = []
        all_lstm_states = []
        if self._side_attn_type == 'one-hop':
            assert aligns is not None
            side_sent_reps = []
            for sent2node in aligns[0]:
                side_sent_reps.append(side_feat[0, sent2node, :].mean(dim=-2))
            side_sent_reps = torch.stack(side_sent_reps, dim=0).unsqueeze(0)

        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)

            if self._side_attn_type == 'two-hop':
                if self._hierarchical_attn:
                    side_e = LSTMPointerNet_entity.hierarchical_attention(query, para_feat, side_feat, self.para_v,
                                                                          self.para_wq,
                                                                          self.side_v, self.side_wq,
                                                                          para_node_aligns, None, para_sizes)
                else:
                    side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
                score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                             self._attn_ws)
                # side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
                # score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                #                                              self._attn_ws)
            elif self._side_attn_type == 'one-hop':
                score = LSTMPointerNet_entity.attention_onehop(attn_feat, query, side_sent_reps, self._attn_v,
                                                               self._attn_wq)

            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            softmax_score = F.softmax(score)
            ext = softmax_score.multinomial(num_samples=1)
            _score = softmax_score.gather(0, ext)
            if ext.item() == max_sent and sent_num != 0:
                log_scores.append(torch.log(_score))
                all_lstm_states.append(lstm_states)
                break
            elif sent_num == 0 and ext.item() == max_sent:
                # force model to sample a largest one
                while(ext.item() == max_sent):
                    ext = softmax_score[:-1].multinomial(num_samples=1)
                    _score = softmax_score.gather(0, ext)
            extracts.append(ext.item())
            log_scores.append(torch.log(_score + eps))
            lstm_states = (h, c)
            all_lstm_states.append(lstm_states)
            lstm_in = attn_mem[:, ext.item(), :].squeeze(1)

        return extracts, log_scores

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
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            SCExtractorRLGraph.attention_score(attention, query, v, w), dim=-1)
        output =  torch.matmul(score, attention)
        return output

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

class SelfCriticGraph(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, net, art_batcher, cuda=True, docgraph=True, paragraph=False):
        assert not all([docgraph, paragraph])
        assert any([docgraph, paragraph])
        super().__init__()
        self._sent_enc = net._sent_enc
        self._art_enc = net._art_enc
        self._node_enc = net._node_enc
        self._graph_enc = net._graph_enc
        self._graph_mask = net._graph_mask
        # some arguments
        self._adj_type = net._adj_type
        self._mask_type = net._mask_type
        self._bert = net._bert
        self._bertmodel = net._bertmodel
        self._bert_stride = net._bert_stride
        self._feature_banks = net._feature_banks
        self._hierarchical_attn = net._hierarchical_attn
        if 'nodefreq' in self._feature_banks:
            self._node_freq_embedding = net._node_freq_embedding

        if not self._bert:
            self._pe = net._pe
        else:
            self._bert_model = net._bert_model
            self._bert_relu = net._bert_relu
            self._bert_linear = net._bert_linear
        self._enable_docgraph = docgraph

        if isinstance(net, PtrExtractSummGAT):
            self._graph_proj = net._graph_proj
            self._graph_layer_num = net._graph_layer_num
            self._graph_model = net._graph_model

        if isinstance(net, PtrExtractSummGAT or isinstance(net, PtrExtractSummSubgraph)):
            self._net = SCExtractorRLGraph(net._extractor)
            self._net_type = 'doc-graph'
        if self._bert:
            self._tokenizer = net._bert_model._tokenizer
            self._batcher = ArticleBatcherGraph_bert(tokenizer=self._tokenizer)
        else:
            self._batcher = art_batcher
        self._cuda = cuda
        self.time_variant = False

    def forward(self, raw_input, n_abs=None, sample_time=1, validate=False):

        if self._enable_docgraph:
            if self._bert:
                raise NotImplementedError
            else:
                raw_article_sents, nodes, nodefreq, word_freq_feat, sent_word_freq, triples, relations, sent_aligns = raw_input
            if self._adj_type == 'concat_triple':
                adjs = [make_adj_triple(triples, len(nodes), len(relations), self._cuda)]
            elif self._adj_type == 'edge_as_node':
                adjs = [make_adj_edge_in(triples, len(nodes), len(relations), self._cuda)]
            else:
                adjs = [make_adj(triples, len(nodes), len(nodes), self._cuda)]
        else:
            if self._bert:
                _, raw_article_sents, nodes, nodefreq, triples, relations, node_lists, word_nums = raw_input
            else:
                raw_article_sents, nodes, nodefreq, word_freq_feat, sent_word_freq, triples, relations, sent_aligns, node_lists = raw_input
            if self._adj_type == 'edge_as_node':
                adjs = [subgraph_make_adj_edge_in((triples, node_lists), cuda=self._cuda)]
            else:
                adjs = [subgraph_make_adj((triples, node_lists), cuda=self._cuda)]


        if not self._bert:
            sent_word_freq = pad_batch_tensorize(sent_word_freq, pad=0, max_num=5, cuda=self._cuda)
            word_freq_feat = pad_batch_tensorize([word_freq_feat], pad=0, cuda=self._cuda)

        nodenum = [len(nodes)]
        sentnum = [len(raw_article_sents)]
        nmask = pad_batch_tensorize(nodes, pad=-1, cuda=self._cuda).ne(-1).float().unsqueeze(0)
        nodes = pad_batch_tensorize(nodes, pad=0, cuda=self._cuda).unsqueeze(0)
        nodefreq = pad_batch_tensorize([nodefreq], pad=0, cuda=self._cuda)



        if self._bert:
            articles = self._batcher(raw_article_sents)
            articles, article_sent = self._encode_bert(articles, [word_nums])
            enc_sent = self._sent_enc(article_sent[0], None, None, None).unsqueeze(0)
            enc_art = self._art_enc(enc_sent)
            sent_aligns = None

        else:
            article_sent, articles = self._batcher(raw_article_sents)
            articles = self._sent_enc._embedding(articles)
            sent_aligns = [sent_aligns]
            if self._pe:
                bs, max_art_len, _ = articles.size()
                src_pos = torch.tensor([[i for i in range(max_art_len)] for _ in range(bs)]).to(articles.device)
                src_pos = self._sent_enc.poisition_enc(src_pos)
                articles = torch.cat([articles, src_pos], dim=-1)
            if 'inpara_freq' in self._feature_banks:
                word_inpara_freq = self._sent_enc._inpara_embedding(word_freq_feat)
                articles = torch.cat([articles, word_inpara_freq], dim=-1)
            enc_sent = self._sent_enc(article_sent, None, sent_word_freq, None).unsqueeze(0)
            enc_art = self._art_enc(enc_sent)


        # print('enc_Art:', enc_art)
        if self._enable_docgraph:
            nodes = self._encode_docgraph(articles, nodes, nmask, adjs, nodenum, enc_out=enc_art, sent_nums=sentnum, nodefreq=nodefreq)
        else:
            outputs = self._encode_paragraph(articles, nodes, nmask, adjs, [node_lists], enc_out=enc_art, sent_nums=sentnum,nodefreq=nodefreq)
            if self._hierarchical_attn:
                (topics, topic_length), masks, (nodes, node_length, node_align_paras) = outputs
                node_align_paras = pad_batch_tensorize(node_align_paras, pad=0, cuda=False).to(nodes.device)
            elif 'soft' in self._mask_type:
                (nodes, topic_length), masks = outputs
                topics = None
                node_align_paras = None
            else:
                nodes, topic_length = outputs
                topics = None
                node_align_paras = None


        nodes = nodes.squeeze(0)




        # print('entity out:', entity_out)

        if not validate:
            greedy = self._net(enc_art, nodes, aligns=sent_aligns, paras=(topics, node_align_paras, topic_length))
            samples = []
            probs = []
            sample, prob= self._net.sample(enc_art, nodes, aligns=sent_aligns, paras=(topics, node_align_paras, topic_length))
            samples.append(sample)
            probs.append(prob)
        else:
            greedy = self._net(enc_art, nodes, aligns=sent_aligns, paras=(topics, node_align_paras, topic_length))

            samples = []
            probs = []
            # for i in range(sample_time):
            #     sample, prob = self._net.sample(enc_art, nodes, aligns=sent_aligns, paras=(topics, node_align_paras, topic_length))
            #     samples.append(sample)
            #     probs.append(prob)
        return greedy, samples, probs

    def _encode_docgraph(self, articles, nodes, nmask, adjs, node_num, node_mask=None, enc_out=None, sent_nums=None, nodefreq=None):
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
        elif self._mask_type == 'soft+sent':
            nodes, node_mask = self._graph_mask[0](nodes, _input=nodes, _sents=enc_out, sent_nums=sent_nums)
            masks.append(node_mask.squeeze(2))
        # print('node_mask:', node_mask[0:2, :])
        # print('n_mask:', nmask[0:2, :])
        # print('triples:', triples[0:2])
        init_nodes = nodes


        for i_layer in range(self._graph_layer_num):
            triple_reps = nodes

            #print('before layer {}, nodes: {}'.format(i_layer, nodes[0:2,:,:10]))
            if self._graph_model in ['ggnn', 'gcn']:
                nodes = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
            else:
                edges = nodes
                nodes, edges = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)

            if self._mask_type == 'encoder':
                nodes, node_mask = self._graph_mask(nodes, node_mask)
            elif self._mask_type == 'soft':
                if i_layer == 0:
                    _input = nodes_no_mask
                _input = torch.cat([nodes, nodes_no_mask], dim=-1)
                nodes, node_mask = self._graph_mask[i_layer+1](nodes, _input=_input)
                masks.append(node_mask.squeeze(2))
            elif self._mask_type == 'soft+sent':
                if i_layer == 0:
                    _input = nodes_no_mask
                _input = torch.cat([nodes, nodes_no_mask], dim=-1)
                nodes, node_mask = self._graph_mask[i_layer+1](nodes, _input=_input, _sents=enc_out, sent_nums=sent_nums)
                masks.append(node_mask.squeeze(2))


        # add initial reps

        nodes = self._graph_proj(init_nodes) + nodes

        return nodes

    def _encode_paragraph(self, articles, nodes, nmask, batch_adjs, node_lists, node_mask=None, enc_out=None, sent_nums=None, nodefreq=None):
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
            nodes, node_mask = self._graph_mask(nodes, _input=nodes)
            masks.append(node_mask.squeeze(2))
        elif self._mask_type == 'soft+sent':
            nodes, node_mask = self._graph_mask(nodes, _input=nodes, _sents=enc_out, sent_nums=sent_nums)
            masks.append(node_mask.squeeze(2))
        # print('node_mask:', node_mask[0:2, :])
        # print('n_mask:', nmask[0:2, :])
        # print('triples:', triples[0:2])
        init_nodes = nodes
        if self._hierarchical_attn:
            (topics, topic_length), (node_reps, node_length, node_align_paras) = self._graph_enc(batch_adjs, nodes, node_lists, output_node_rep=True)
        else:
            topics, topic_length = self._graph_enc(batch_adjs, nodes, node_lists)

        # if 'soft' in self._mask_type:
        #     nodes = nodes * node_mask

        # add initial reps
        # nodes = self._graph_proj(init_nodes) + nodes

        results = ((topics, topic_length),)

        if 'soft' in self._mask_type:
            results += (masks,)

        if self._hierarchical_attn:
            results += ((node_reps, node_length, node_align_paras),)

        return results

    def _encode_bert(self, articles, word_nums):
        source_nums = [sum(word_num) for word_num in word_nums]
        with torch.no_grad():
            bert_out = self._bert_model(articles)
        # bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
        bert_hidden = bert_out[0]
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