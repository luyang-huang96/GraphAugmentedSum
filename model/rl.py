import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet, LSTMPointerNet_entity
from data.batcher import pad_batch_tensorize
from pytorch_transformers import BertTokenizer, BertModel, BertConfig


INI = 1e-2
BERT_MAX_LEN = 512

# FIXME eccessing 'private members' of pointer net module is bad

class Bert_model(object):
    def __init__(self):
        # self._bert_config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking',
        #                                     output_hidden_states=True,
        #                                     output_attentions=False)
        self._bert_model = BertModel.from_pretrained('/data2/luyang/bert_model/bert-large-uncased-wwm/', output_hidden_states=True)
        self._tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        for param in self._bert_model.parameters():
            param.requires_grad = False
        self._bert_model.cuda()
        self._bert_model.eval()
        print('BERT initialized')
        self._pad = self._tokenizer.vocab['[PAD]']
        print('PAD:', self._pad)

    def __call__(self, input_ids):
        attention_mask = (input_ids != self._pad).float()
        return self._bert_model(input_ids, attention_mask=attention_mask)

class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
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

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = (h, c)
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output = torch.mm(score, attention)
        return output


class PtrExtractorRLStop(PtrExtractorRL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
        self._stop = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, n_ext=None):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            for o in outputs:
                score[0, o.item()] = -1e18
            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == max_step:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (trining)
            return outputs, dists
        else:
            return outputs


class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
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

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(F.tanh(sum_), v.unsqueeze(0)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher

    def forward(self, raw_article_sents, n_abs=None):
        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._art_enc(enc_sent).squeeze(0)
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_art)
        else:
            outputs = self._ext(enc_art, n_abs)
        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_art, n_abs)
            return outputs, scores
        else:
            return outputs

class SCExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
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

        self._stop = nn.Parameter(ptr_net._stop.clone())


    def forward(self, attn_mem):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)
            score = self.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
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

    def sample(self, attn_mem, time_varient=False):
        """sample k sentences, decode only, batch_size==1"""
        eps = 1e-8
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        log_scores = []
        all_lstm_states = []
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)
            score = self.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
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

        if time_varient:
            all_extracts = []
            for ind, lstm_states in enumerate(all_lstm_states[:-1]):
                _extracts = extracts[:ind+1]
                lstm_in = attn_mem[:, extracts[ind], :].squeeze(1)
                flag = 0
                for sent_num in range(max_sent-ind-1):
                    h, c = self._lstm_cell(lstm_in, lstm_states)
                    query = h[-1]
                    for _ in range(self._n_hop):
                        query = self.attention(
                            hop_feat, query, self._hop_v, self._hop_wq)
                    score = self.attention_score(
                        attn_feat, query, self._attn_v, self._attn_wq)
                    score = score.squeeze()
                    for e in _extracts:
                        score[e] = -1e6
                    softmax_score = F.softmax(score)
                    ext = softmax_score.max(dim=0)[1]
                    if ext.item() == max_sent:
                        flag = 1
                        #log_scores.append(torch.log(_score))
                        all_extracts.append(_extracts)
                        break
                    _extracts.append(ext.item())
                    #log_scores.append(torch.log(_score))
                    lstm_states = (h, c)
                    lstm_in = attn_mem[:, ext.item(), :].squeeze(1)
                if not flag:
                    all_extracts.append(_extracts)
            return extracts, log_scores, all_extracts
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
            SCExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output =  torch.matmul(score, attention)
        return output

class SCExtractorRLEntity(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net, rnn_entity=False):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet_entity)
        try:
            self._hard_attention = ptr_net._hard_attention
        except:
            self._hard_attention = False
        if self._hard_attention:
            try:
                self.side_dim = ptr_net.side_dim
            except:
                if not rnn_entity:
                    self.side_dim = 300
                else:
                    self.side_dim = 512


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
        if not self._hard_attention:
            self.side_wm = nn.Parameter(ptr_net.side_wm.clone())
            self.side_wq = nn.Parameter(ptr_net.side_wq.clone())
            self.side_v = nn.Parameter(ptr_net.side_v.clone())
        else:
            self.side_wq = nn.Parameter(ptr_net.side_wq.clone())
            self.side_wbi = nn.Bilinear(self.side_dim, self.side_dim, 1)
            self.side_wbi.weight.data.copy_(ptr_net.side_wbi.weight)
            self._start = nn.Parameter(ptr_net._start.clone())
            self._eos_entity = nn.Parameter(ptr_net._eos_entity.clone())

        self._attn_ws = nn.Parameter(ptr_net._attn_ws.clone())

        # self._pad_entity = nn.Parameter(ptr_net._pad_entity.clone())

        try:
            self._stop = nn.Parameter(ptr_net._stop.clone())
        except KeyError:
            print('Initialize stop tensor')
            self._stop = nn.Parameter(torch.Tensor(ptr_net._hop_wm.size(0)))
            init.uniform_(self._stop, -INI, INI)


    def forward(self, attn_mem, side_mem):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        # side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        # if self._hard_attention:
        #     side_mem = torch.cat([side_mem, self._eos_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        # print('side:', side_mem.size())
        # print('side:', side_mem)

        side_feat = self._prepare_side(side_mem)
        # print('side feat:', side_feat)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        # if self._hard_attention:
        #     max_side = side_mem.size(1)
        #     context = self._start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)
            if not self._hard_attention:
                side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
            else:
                side_e, selected = self.hard_attention_decoding(side_feat, query, self.side_wbi, self.side_wq, context)
                context = context + side_e
            score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
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

    def sample(self, attn_mem, side_mem, time_varient=False):
        """sample k sentences, decode only, batch_size==1"""
        eps = 1e-8
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)


        side_feat = self._prepare_side(side_mem)
        lstm_in = lstm_in.squeeze(1)
        extracts = []
        log_scores = []
        all_lstm_states = []
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = self.attention(
                    hop_feat, query, self._hop_v, self._hop_wq)
            if not self._hard_attention:
                side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
            else:
                side_e, selected = self.hard_attention_decoding(side_feat, query, self.side_wbi, self.side_wq, context)
                context = context + side_e
            score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                             self._attn_ws)
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

        if time_varient:
            all_extracts = []
            for ind, lstm_states in enumerate(all_lstm_states[:-1]):
                _extracts = extracts[:ind+1]
                lstm_in = attn_mem[:, extracts[ind], :].squeeze(1)
                flag = 0
                if self._hard_attention:
                    max_side = side_mem.size(1)
                    context = self._start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
                for sent_num in range(max_sent-ind-1):
                    h, c = self._lstm_cell(lstm_in, lstm_states)
                    query = h[-1]
                    for _ in range(self._n_hop):
                        query = self.attention(
                            hop_feat, query, self._hop_v, self._hop_wq)
                    if not self._hard_attention:
                        side_e = self.attention(side_feat, query, self.side_v, self.side_wq)
                    else:
                        side_e, selected = self.hard_attention_decoding(side_feat, query, self.side_wbi, self.side_wq,
                                                                        context)
                        context = context + side_e
                    score = self.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                    self._attn_ws)
                    score = score.squeeze()
                    for e in _extracts:
                        score[e] = -1e6
                    softmax_score = F.softmax(score)
                    ext = softmax_score.max(dim=0)[1]
                    if ext.item() == max_sent:
                        flag = 1
                        #log_scores.append(torch.log(_score))
                        all_extracts.append(_extracts)
                        break
                    _extracts.append(ext.item())
                    #log_scores.append(torch.log(_score))
                    lstm_states = (h, c)
                    lstm_in = attn_mem[:, ext.item(), :].squeeze(1)
                if not flag:
                    all_extracts.append(_extracts)
            return extracts, log_scores, all_extracts
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
            SCExtractorRL.attention_score(attention, query, v, w), dim=-1)
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

    @staticmethod
    def hard_attention_decoding(attention, query, w_bi, wq, context):
        n_side = attention.size(1)
        batch_size = attention.size(0)
        n_sent = query.size(1) #should equal 1
        bilinear = w_bi(context.unsqueeze(2).repeat(1, 1, n_side, 1),
                        attention.unsqueeze(1).repeat(1, n_sent, 1, 1)).squeeze(3)  # B*Nsent*Nside
        selection = bilinear + torch.matmul(query, wq.unsqueeze(0))
        selected = F.sigmoid(selection)
        selected = selected.gt(0.5).float()
        output = selected.unsqueeze(3) * attention.unsqueeze(1)  # B*Nsent*Nside*Side
        output = output.sum(dim=2)  # B*Nsent*Side
        return output, selected



class SelfCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, net, art_batcher, time_variant=False):
        super().__init__()
        self._sent_enc = net._sent_enc
        self._art_enc = net._art_enc
        self._net = SCExtractorRL(net._extractor)
        self._batcher = art_batcher
        self.time_variant = time_variant

        self._bert = net._bert
        self._bert_sent = net._bert_sent
        self._bert_stride = net._bert_stride
        if self._bert:
            self._bert_linear = nn.Linear(4096, net._emb_dim)
            self._bert_linear.weight.data.copy_(net._bert_linear.weight)
            self._bert_relu = nn.PReLU()
            self._bert_relu.weight.data.copy_(net._bert_relu.weight)
            self._bert_model = Bert_model()


    def forward(self, raw_article_sents, n_abs=None, sample_time=1, validate=False):
        if self._bert:
            if self._bert_sent:
                _, article, word_num = raw_article_sents
                article = pad_batch_tensorize(article, pad=0, cuda=True)
                mask = (article != 0).detach().float()
                with torch.no_grad():
                    bert_out = self._bert_model(article)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                bert_hidden = bert_hidden * mask.unsqueeze(2)
                article_sent = bert_hidden
                enc_sent = self._sent_enc(article_sent).unsqueeze(0)
                # print('enc_sent:', enc_sent)
                enc_art = self._art_enc(enc_sent)
            else:
                _, articles, word_num = raw_article_sents
                if self._bert_stride != 0:
                    source_num = sum(word_num)
                    articles = pad_batch_tensorize(articles, pad=0, cuda=True)
                else:
                    articles = torch.tensor(articles, device='cuda')
                with torch.no_grad():
                    bert_out = self._bert_model(articles)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                hsz = bert_hidden.size(2)
                if self._bert_stride != 0:
                    batch_id = 0
                    source = torch.zeros(source_num, hsz).to(bert_hidden.device)
                    if source_num < BERT_MAX_LEN:
                        source[:source_num, :] += bert_hidden[batch_id, :source_num, :]
                        batch_id += 1
                    else:
                        source[:BERT_MAX_LEN, :] += bert_hidden[batch_id, :BERT_MAX_LEN, :]
                        batch_id += 1
                        start = BERT_MAX_LEN
                        while start < source_num:
                            #print(start, source_num, max_source)
                            if start - self._bert_stride + BERT_MAX_LEN < source_num:
                                end = start - self._bert_stride + BERT_MAX_LEN
                                batch_end = BERT_MAX_LEN
                            else:
                                end = source_num
                                batch_end = source_num - start + self._bert_stride
                            source[start:end, :] += bert_hidden[batch_id, self._bert_stride:batch_end, :]
                            batch_id += 1
                            start += (BERT_MAX_LEN - self._bert_stride)
                    bert_hidden = source.unsqueeze(0)
                    del source
                max_word_num = max(word_num)
                # if max_word_num < 5:
                #     max_word_num = 5
                new_word_num = []
                start_num = 0
                for num in word_num:
                    new_word_num.append((start_num, start_num + num))
                    start_num += num
                article_sent = torch.stack(
                        [torch.cat([bert_hidden[0, num[0]:num[1], :],
                                    torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
                         if (num[1] - num[0]) != max_word_num
                         else bert_hidden[0, num[0]:num[1], :]
                         for num in new_word_num
                         ]
                    )
                # print('article_sent:', article_sent)
                enc_sent = self._sent_enc(article_sent).unsqueeze(0)
                # print('enc_sent:', enc_sent)
                enc_art = self._art_enc(enc_sent)
                # print('enc_art:', enc_art)

        else:
            article_sent = self._batcher(raw_article_sents)
            enc_sent = self._sent_enc(article_sent).unsqueeze(0)
            enc_art = self._art_enc(enc_sent)

        if self.time_variant and not validate:
            greedy = self._net(enc_art)
            samples = []
            probs = []
            sample, prob, new_greedy = self._net.sample(enc_art, time_varient=self.time_variant)
            samples.append(sample)
            probs.append(prob)
            greedy = [greedy] + new_greedy
            if len(greedy) != len(prob):
                print(len(enc_art[0]))
                print(greedy)
                print(new_greedy)
                print(sample)
                print(prob)
            assert len(greedy) == len(prob)
        else:
            greedy = self._net(enc_art)
            samples = []
            probs = []
            for i in range(sample_time):
                sample, prob = self._net.sample(enc_art, time_varient=False)
                samples.append(sample)
                probs.append(prob)
            # print('samle:', samples)
            # print('greedy:', greedy)

        return greedy, samples, probs

class SelfCriticEntity(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, net, art_batcher, time_variant=False):
        super().__init__()
        self._sent_enc = net._sent_enc
        self._art_enc = net._art_enc
        self._entity_enc = net._entity_enc
        self._graph_enc = net._graph_enc
        if net._rnn_entity:
            self._mention_cluster_enc = net._mention_cluster_enc

        self._rnn_entity = net._rnn_entity
        self._net = SCExtractorRLEntity(net._extractor, rnn_entity=self._rnn_entity)
        self._batcher = art_batcher
        self.time_variant = time_variant
        self._context = net._context

    def forward(self, raw_input, n_abs=None, sample_time=1, validate=False):
        raw_article_sents, raw_clusters = raw_input
        clusters = (self._batcher(raw_clusters[0]),
                    pad_batch_tensorize(raw_clusters[1], pad=0, max_num=5),
                    pad_batch_tensorize(raw_clusters[2], pad=0, max_num=5),
                    torch.cuda.FloatTensor(raw_clusters[3]),
                    torch.cuda.LongTensor(raw_clusters[4]))

        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._art_enc(enc_sent)
        # print('enc_Art:', enc_art)

        entity_out = self._encode_entity(clusters, cluster_nums=None)
        _, _, (entity_out, entity_mask) = self._graph_enc([clusters[3]], [clusters[4]], (
        entity_out.unsqueeze(0), torch.tensor([len(raw_clusters[0])], device=entity_out.device)))
        entity_out = entity_out.squeeze(0)

        # print('entity out:', entity_out)

        if self.time_variant and not validate:
            greedy = self._net(enc_art, entity_out)
            samples = []
            probs = []
            sample, prob, new_greedy = self._net.sample(enc_art, entity_out, time_varient=self.time_variant)
            samples.append(sample)
            probs.append(prob)
            greedy = [greedy] + new_greedy
            if len(greedy) != len(prob):
                print(len(enc_art[0]))
                print(greedy)
                print(new_greedy)
                print(sample)
                print(prob)
            assert len(greedy) == len(prob)
        else:
            greedy = self._net(enc_art, entity_out)

            samples = []
            probs = []
            for i in range(sample_time):
                sample, prob = self._net.sample(enc_art, entity_out, time_varient=False)
                samples.append(sample)
                probs.append(prob)
        return greedy, samples, probs

    def _encode_entity(self, clusters, cluster_nums, context=None):
        if cluster_nums is None: # test-time excode only
            if context is None:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context)
            else:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context[0, :, :])
        else:
            #enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos) for cluster_words, cluster_wpos, cluster_spos in list(zip(*clusters))]
            if context is None:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos) for cluster_words, cluster_wpos, cluster_spos in list(zip(*clusters))]
            else:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos, context[id, :, :])
                                for id, (cluster_words, cluster_wpos, cluster_spos) in enumerate(list(zip(*clusters)))]
            max_n = max(cluster_nums)
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            enc_entity = torch.stack(
                [torch.cat([s, zero(max_n - n, s.device)], dim=0)
                 if n != max_n
                 else s
                 for s, n in zip(enc_entities, cluster_nums)],
                dim=0
            )
        if self._rnn_entity:
            enc_entity = self._mention_cluster_enc(enc_entity.unsqueeze(0), cluster_nums).squeeze(0)

        return enc_entity
