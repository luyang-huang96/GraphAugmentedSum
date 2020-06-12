import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .attention import step_attention, badanau_attention
from .util import len_mask
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs
from utils import PAD, UNK, START, END
from model.roberta import RobertaEmbedding
from model.rnn import lstm_encoder
from model.util import sequence_mean, len_mask
from .rnn import MultiLayerLSTMCells

INIT = 1e-2
BERT_MAX_LEN = 512


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.regiser_module(None, '_b')

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySumm(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0, bert=False, bert_length=512):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)
        self._bert = bert
        if self._bert:
            self._bert_model = RobertaEmbedding()
            self._eos = self._bert_model._eos
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
                emb_dim*2, n_hidden, n_layer, dropout=dropout
            )
            self._bert_stride = 256

        self._copy = _CopyLinear(n_hidden, n_hidden, 2*emb_dim)


        self._decoder = CopyLSTMDecoder(
            self._copy, self._embedding, self._dec_lstm,
            self._attn_wq, self._projection, self._attn_wb, self._attn_v
        )

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract
        )
        return logit

    def encode(self, article, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
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
            enc_art, final_states = lstm_encoder(
                article, self._enc_lstm, art_lens,
                init_enc_states, None
            )
        else:
            enc_art, final_states = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding
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

    def greedy(self, article, art_lens, extend_art, extend_vsize,
                     go, eos, unk, max_len, min_len=0, tar_in=None):
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
            if i > min_len-1:
                force_not_stop = False
            else:
                force_not_stop = True
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention, force_not_stop=force_not_stop, eos=self._eos)
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
                     go, eos, unk, max_len, min_len=0, abstract=None, ml=False):
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
        seqLogProbs = []
        for i in range(max_len):
            if i > min_len-1:
                force_not_stop = False
            else:
                force_not_stop = True
            tok, states, attn_score, sampleProb = self._decoder.sample_step(
                tok, states, attention, force_not_stop=force_not_stop, eos=self._eos)
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
                           go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
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

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, force_not_stop=force_not_stop, eos=eos)

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

    def batched_beamsearch_cnn(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
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

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, force_not_stop=force_not_stop, eos=eos)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam_cnn(
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


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention
        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))
        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_prob
        ) + 1e-8)  # numerical stability for log
        return lp, (states, dec_out), score


    def topk_step(self, tok, states, attention, k, force_not_stop=False, eos=END):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
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

        # attention is beamable
        query = torch.matmul(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention


        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(
                    beam*batch, -1),
                src=score.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)
        if force_not_stop:
            lp[:, :, eos] = -1e8

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score

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
