import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize, prob_normalize_sigmoid
from .myownutils import get_sinusoid_encoding_table
import math
from model.graph_enc import graph_encode, gat_encode, node_mask, subgraph_encode
from pytorch_transformers import BertTokenizer, BertModel, BertConfig
from transformers import RobertaTokenizer
from data.batcher import pad_batch_tensorize
from model.roberta import RobertaEmbedding

INI = 1e-2
BERT_MAX_LEN = 512
MAX_SENT_LEN = 100
MAX_FREQ = 100

class Bert_model(object):
    def __init__(self, bertmodel='bert-large-uncased-whole-word-masking'):
        # self._bert_config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking',
        #                                     output_hidden_states=True,
        #                                     output_attentions=False)
        if bertmodel == 'bert-large-uncased-whole-word-masking':
            print('bertmodel:', bertmodel)
            self._bert_model = BertModel.from_pretrained('/data2/luyang/bert_model/bert-large-uncased-wwm/', output_hidden_states=True)
        elif bertmodel == 'bert-base-uncased':
            print('bertmodel:', bertmodel)
            self._bert_model = BertModel.from_pretrained(bertmodel,
                                                         output_hidden_states=True)
        self._tokenizer = BertTokenizer.from_pretrained(bertmodel)
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



class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout=0.1, embedding=None, pe=False, feature_banks=[], **kwargs):
        super().__init__()

        if embedding is None:
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:
            self._embedding = embedding

        self._feature_banks = feature_banks
        feat_emb_dim = emb_dim // 4
        if 'freq' in self._feature_banks:
            self._freq_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        if 'inpara_freq' in self._feature_banks:
            self._inpara_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)
        if 'segmentation' in self._feature_banks:
            self._seg_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)


        self._pe = pe
        if pe:
            # sentence level pos enc
            self.poisition_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(6000, emb_dim, padding_idx=0),
                freeze=True)
            emb_dim = emb_dim * 2
        if 'freq' in self._feature_banks:
            emb_dim += feat_emb_dim
        if 'inpara_freq' in self._feature_banks:
            emb_dim += feat_emb_dim
        if 'segmentation' in self._feature_banks:
            emb_dim += feat_emb_dim


        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                         for i in range(3, 6)])

        self._dropout = dropout
        self._grad_handle = None


    def forward(self, input_, sent_word_freq=None, sent_inpara_freq=None, seg=None):
        emb_input = self._embedding(input_)
        if self._pe:
            mask_ = input_.gt(0)
            sent_len = mask_.sum(1)
            sent_num, max_sent_length = input_.size()
            src_pos = torch.zeros(sent_num, max_sent_length).long().to(input_.device)
            total_word_num = 0
            for i in range(sent_num):
                src_pos[i, :] = torch.arange(1, max_sent_length + 1) + total_word_num
                total_word_num += sent_len[i]
                # src_pos[i, :] = torch.arange(1, max_sent_length + 1)
            src_pos = src_pos * mask_.long()
            src_pos[src_pos > 5999] = 5999  # in order for out ouf bound
            src_pos = self.poisition_enc(src_pos)
            emb_input = torch.cat([emb_input, src_pos], dim=-1)
        if 'freq' in self._feature_banks:
            assert type(sent_word_freq) == type(emb_input)
            word_freq = self._freq_embedding(sent_word_freq)
            emb_input = torch.cat([emb_input, word_freq], dim=-1)
        if 'inpara_freq' in self._feature_banks:
            assert type(sent_inpara_freq) == type(emb_input)
            word_inpara_freq = self._inpara_embedding(sent_inpara_freq)
            emb_input = torch.cat([emb_input, word_inpara_freq], dim=-1)
        if 'segmentation' in self._feature_banks:
            assert type(seg) == type(emb_input)
            seg = self._seg_embedding(seg)
            emb_input = torch.cat([emb_input, seg], dim=-1)


        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)

        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

class MeanSentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_, sent_word_freq=None, sent_inpara_freq=None, seg=None, mask=None):
        if mask is None:
            mask = input_.sum(dim=-1).ne(0).sum(dim=-1).detach()
            mask = mask.float()
            mask = mask.unsqueeze(1)
        else:
            mask = mask.sum(dim=-2)
            mask[mask == 0] = 1.
            mask = mask.detach()
        output = input_.sum(dim=-2) / mask
        # output[output == -float('Inf')] = 0.
        # output[output == float('Inf')] = 0.
        return output





class ConvEntityEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout, embedding=None, pe=False, **kwargs):
        super().__init__()
        if embedding is None:
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:
            self._embedding = embedding

        if pe:
            # sentence level pos enc
            self.poisition_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(10000, emb_dim, padding_idx=0),
                freeze=True)

        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])

        self._dropout = dropout
        self._grad_handle = None
        self._hsz = n_hidden * 3


    def forward(self, input_, input_wpos, input_spos, context=None):
        if self.position_encoder:
            input_wpos[input_wpos > 512] = 512
            emb_input = self._embedding(input_) + self.poisition_enc(input_wpos)
        else:
            emb_input = self._embedding(input_)
        if self._context:
            context_dim = context.size(1)
            context = torch.cat(
                [
                    torch.zeros(1, context_dim).to(context.device),
                    context
                ],
                dim = 0
            )
            # print('embed_input:', emb_input.size())
            # print('input_spos:', input_spos.size())
            # print('context:', context.size())
            n_side, n_word = input_spos.size()
            spos = input_spos.view(-1)
            context_embed = context.index_select(0, spos).view(n_side, n_word, -1)
            context_embed = self._linear(context_embed)
            emb_input = torch.cat([emb_input, context_embed], dim=2)
            # print('check:', context)
            # print('check:', input_spos)
            # print('check:', context_embed)
            # print(emb_input.size())

        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        # conv_in = emb_input.transpose(1, 2)
        # output = torch.cat([F.tanh(conv(conv_in)).max(dim=2)[0]
        #                     for conv in self._convs], dim=1)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        # print('output:', output)
        # print(output.size())
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None, return_states=False):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, final_states = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        if return_states:
            return lstm_out.transpose(0, 1), final_states
        else:
            return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class ExtractSumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0, pe=False, petrainable=False, stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, pe, petrainable)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat(
            [s[:n] for s, n in zip(saliency, sent_nums)], dim=0)
        content = self._sent_linear(
            torch.cat([s[:n] for s, n in zip(enc_sent, sent_nums)], dim=0)
        )
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4, force_ext=True):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if force_ext:
            if sent_nums is None:  # test-time extract only
                assert len(article_sents) == 1
                n_sent = logit.size(1)
                extracted = logit[0].topk(
                    k if k < n_sent else n_sent, sorted=False  # original order
                )[1].tolist()
            else:
                extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                             for n, l in zip(sent_nums, logit)]
        else:
            logit = F.sigmoid(logit)
            extracted = logit.gt(0.2)
            if extracted.sum() < 1:
                extracted = logit[0].topk(1)[1].tolist()
            else:
                extracted = [i for i, x in enumerate(extracted[0].tolist()) if x == 1]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = F.tanh(
            self._art_linear(sequence_mean(lstm_out, sent_nums, dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)



class NNSESumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0, pe=False, petrainable=False, stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, pe=pe, petrainable=petrainable)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.LSTM = nn.LSTM(3*conv_hidden, lstm_out_dim, batch_first=True)
        self._sent_linear = nn.Linear(lstm_out_dim + lstm_out_dim, 1)
        self.step = 0

    @staticmethod
    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
        return hidden

    def forward(self, article_sents, sent_nums, target):
        enc_sent, enc_art, hidden = self._encode(article_sents, sent_nums)
        batch_size, max_num_sent, input_dim = enc_sent.size()
        hidden = (self._fix_enc_hidden(hidden[0]), self._fix_enc_hidden(hidden[1]))
        all_decoder_output = torch.zeros(batch_size, max_num_sent, 1).to(enc_sent.device)
        j = 0
        teacher_forcing = torch.rand(1) > math.exp(- self.step / 10000)
        for i in range(max_num_sent):
            if i == 0:
                lstm_in = enc_art[:, i, :].unsqueeze(1)
            else:
                if teacher_forcing:
                    lstm_in = enc_art[:, i, :].unsqueeze(1) * F.sigmoid(output)
                else:
                    lstm_in = enc_art[:, i, :].unsqueeze(1) * target[:, i-1].unsqueeze(1).unsqueeze(1).float()
            output, hidden = self.LSTM(lstm_in, hidden)
            output = self._sent_linear(torch.cat([output, enc_sent[:, i, :].unsqueeze(1)], dim=2))
            all_decoder_output[:, i, :] = output.squeeze(1)
        #mask
        mask = target.ne(-1)
        final_output = torch.masked_select(all_decoder_output.squeeze(2), mask)
        self.step += 1

        return final_output

    def extract(self, article_sents, sent_nums=None, k=4, force_ext=True):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art, hidden = self._encode(article_sents, sent_nums)
        batch_size, max_num_sent, input_dim = enc_sent.size()
        hidden = (self._fix_enc_hidden(hidden[0]), self._fix_enc_hidden(hidden[1]))
        all_decoder_output = torch.zeros(batch_size, max_num_sent, 1).to(enc_sent.device)
        for i in range(max_num_sent):
            if i == 0:
                lstm_in = enc_art[:, i, :].unsqueeze(1)
            else:
                lstm_in = enc_art[:, i, :].unsqueeze(1) * F.sigmoid(output)
            output, hidden = self.LSTM(lstm_in, hidden)
            output = self._sent_linear(torch.cat([output, enc_sent[:, i, :].unsqueeze(1)], dim=2))
            all_decoder_output[:, i, :] = output.squeeze(1)
        logit = all_decoder_output.squeeze(2)
        if force_ext:
            if sent_nums is None:
                assert len(article_sents) == 1
                n_sent = logit.size(1)
                extracted = logit[0].topk(
                    k if k < n_sent else n_sent, sorted=False  # original order
                )[1].tolist()
            else:
                extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                             for n, l in zip(sent_nums, logit)]
        else:
            logit = F.sigmoid(logit)
            extracted = logit.gt(0.15)
            if extracted.sum() < 1:
                extracted = logit[0].topk(1)[1].tolist()
            else:
                extracted = [i for i, x in enumerate(extracted[0].tolist()) if x == 1]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            ) # [batch_size, max_num_sent, input_dim]
        lstm_out, final_hidden = self._art_enc(enc_sent, sent_nums, return_states=True)

        return lstm_out, enc_sent, final_hidden

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, **kwargs):
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

        # stop token
        self._stop = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._stop, -INI, INI)
        self.stop = True

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        batch_size, max_sent_num, input_dim = attn_mem.size()
        if self.stop:
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
            for i, sent_num in enumerate(mem_sizes):
                attn_mem[i, sent_num, :] += self._stop
            mem_sizes = [mem_size+1 for mem_size in mem_sizes]
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)


        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        if self.stop:
            for sent_num in range(max_sent):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                score = LSTMPointerNet.attention_score(
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
        else:
            for _ in range(k):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                score = LSTMPointerNet.attention_score(
                    attn_feat, query, self._attn_v, self._attn_wq)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        return extracts

    def sample(self, attn_mem, mem_sizes, k=4):
        assert self.stop == True
        """sample k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        log_scores = []
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            softmax_score = F.softmax(score)
            ext = softmax_score.multinomial(num_samples=1)
            _score = softmax_score.gather(0, ext)
            if ext.item() == max_sent and sent_num != 0:
                break
            elif sent_num == 0 and ext.item() == max_sent:
                # force model to sample a largest one
                while(ext.item() == max_sent):
                    ext = softmax_score.multinomial(num_samples=1)
                    _score = softmax_score.gather(0, ext)
            extracts.append(ext.item())
            log_scores.append(torch.log(_score))
            lstm_states = (h, c)
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
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output



class LSTMPointerNet_entity(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, side_dim, attention_type, decoder_supervision, side_attn_type='two-hop',
                 hierarchical_attn=False, **kwargs):
        super().__init__()
        assert attention_type in ['glimpse', 'dual']
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

        self.attn_type = attention_type
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
        self._side_attn_type = side_attn_type
        assert side_attn_type in ['two-hop', 'one-hop']

        # side info attention
        if attention_type == 'dual':
            self._biattn = nn.Parameter(torch.Tensor(input_dim, side_dim))
            init.xavier_normal_(self._biattn)
            self._attn_wt = nn.Parameter(torch.Tensor(side_dim, n_hidden))
            init.xavier_normal_(self._attn_wt)


        self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self.side_wm)
        init.xavier_normal_(self.side_wq)
        init.uniform_(self.side_v, -INI, INI)


        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_ws)


        # pad entity put in graph enc now
        self._pad_entity = nn.Parameter(torch.Tensor(side_dim))
        init.uniform_(self._pad_entity)

        # stop token
        self._stop = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._stop, -INI, INI)
        self.stop = True

        self._supervision = decoder_supervision
        self._hierarchical_attn = hierarchical_attn
        if hierarchical_attn:
            self.para_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
            self.para_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self.para_v = nn.Parameter(torch.Tensor(n_hidden))
            init.xavier_normal_(self.para_wm)
            init.xavier_normal_(self.para_wq)
            init.uniform_(self.para_v, -INI, INI)
            self._pad_para = nn.Parameter(torch.Tensor(side_dim))
            init.uniform_(self._pad_para)



    def forward(self, attn_mem, mem_sizes, lstm_in, side_mem, side_sizes, ground_entity=None, side_mask=None, aligns=None, paras=None):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""

        batch_size, max_sent_num, input_dim = attn_mem.size()
        side_dim = side_mem.size(2)
        if self.stop:
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
            for i, sent_num in enumerate(mem_sizes):
                attn_mem[i, sent_num, :] += self._stop
            mem_sizes = [mem_size+1 for mem_size in mem_sizes]

        side_mem = torch.cat([side_mem, torch.zeros(batch_size, 1, side_dim).to(side_mem.device)], dim=1) #b * ns * s
        for i, side_size in enumerate(side_sizes):
            side_mem[i, side_size, :] += self._pad_entity
        side_sizes = [side_size+1 for side_size in side_sizes]
        if self._hierarchical_attn:
            assert paras is not None
            para_reps, para_length, para_node_aligns = paras
            para_node_aligns = torch.cat([para_node_aligns, torch.zeros(batch_size, 1).long().to(para_node_aligns.device)], dim=1)
            para_reps = torch.cat([para_reps, torch.zeros(batch_size, 1, side_dim).to(para_reps.device)], dim=1)
            for i, para_size in enumerate(para_length):
                para_reps[i, para_size, :] += self._pad_para
                para_node_aligns[i, para_size] = para_size
            para_sizes = [para_size+1 for para_size in para_length]


        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)

        side_feat = self._prepare_side(side_mem) #b * ns * side_h
        if self._hierarchical_attn:
            para_feat = self._prepare_side(para_reps)

        if self.attn_type == 'dual':
            tilde_s, _ = self.biattn(attn_mem, side_mem, mem_sizes, side_sizes)


        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)


        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet_entity.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        if side_mask is not None:
            side_mask = torch.cat([side_mask, torch.zeros(batch_size, 1).to(side_mask.device)], dim=1)
            for i, side_size in enumerate(side_sizes):
                side_mask[i, side_size-1] += 1.

        if self._side_attn_type == 'two-hop':
            if self._hierarchical_attn:
                side_e = LSTMPointerNet_entity.hierarchical_attention(query, para_feat, side_feat, self.para_v, self.para_wq,
                                                                      self.side_v, self.side_wq,
                                                                      para_node_aligns, side_sizes, para_sizes)
            else:
                side_e, node_selection = LSTMPointerNet_entity.attention(side_feat, query, self.side_v, self.side_wq, side_sizes,
                                                         gold_mask=side_mask, sigmoid=self._supervision, output_attn=True)




            if self.attn_type == 'dual':
                output = LSTMPointerNet_entity.attention_biattn(attn_feat, tilde_s, query, side_e,
                                                                self._attn_v, self._attn_wq, self._attn_ws, self._attn_wt)
            else:
                output = LSTMPointerNet_entity.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq, self._attn_ws)
        elif self._side_attn_type == 'one-hop':
            assert aligns is not None
            side_sent_reps = []
            for _bid, align in enumerate(aligns):
                side_sent_rep = []
                for sent2node in align:
                    side_sent_rep.append(side_feat[_bid, sent2node, :].mean(dim=-2))
                side_sent_rep = torch.stack(side_sent_rep, dim=0)
                side_sent_reps.append(side_sent_rep)
            max_s = max(mem_sizes)
            feat_dim = side_sent_reps[0].size(1)
            side_sent_reps = torch.stack([torch.cat([side_sent_rep, torch.zeros(max_s-side_sent_rep.size(0), feat_dim).to(side_sent_rep.device)], dim=0)
                                          if side_sent_rep.size(1) != max_s else side_sent_rep
                for side_sent_rep, mem_size in zip(side_sent_reps, mem_sizes)
            ], dim=0)

            output = LSTMPointerNet_entity.attention_onehop(attn_feat, query, side_sent_reps, self._attn_v, self._attn_wq)
        else:
            raise Exception('Wrong side attention type')


        if self._supervision:
            return output, node_selection
        else:
            return output  # unormalized extraction logit


    def extract(self, attn_mem, mem_sizes, k, side_mem, side_sizes, output_attn=False, side_mask=None, aligns=None):
        """extract k sentences, decode only, batch_size==1"""
        attn_dict = {}
        attn_dict['node-rep'] = side_mem.detach()
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)

        attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        side_mem = torch.cat([side_mem, self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        side_sizes = [side_size+1 for side_size in side_sizes]

        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)

        side_feat = self._prepare_side(side_mem)
        if side_mask is not None:
            side_mask = torch.cat([side_mask, torch.ones(batch_size, 1).to(side_mask.device)], dim=1)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        attentions = []
        if self.attn_type == 'dual':
            tilde_s, weight = self.biattn(attn_mem, side_mem, mem_sizes, side_sizes)

        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

            if self._side_attn_type == 'two-hop':
                side_e, attention = LSTMPointerNet_entity.attention(side_feat, query,
                                                                        self.side_v,
                                                                    self.side_wq, side_sizes,
                                                                    output_attn=True, gold_mask=side_mask, sigmoid=self._supervision)
                if self.attn_type == 'dual':
                    score = LSTMPointerNet_entity.attention_biattn(attn_feat, tilde_s, query, side_e,
                                                                    self._attn_v, self._attn_wq, self._attn_ws,
                                                                    self._attn_wt)
                else:
                    score = LSTMPointerNet_entity.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                              self._attn_ws)
            elif self._side_attn_type == 'one-hop':
                side_sent_reps = []
                for sent2node in aligns[0]:
                    side_sent_reps.append(side_feat[0, sent2node, :].mean(dim=-2))
                side_sent_reps = torch.stack(side_sent_reps, dim=0).unsqueeze(0)
                score = LSTMPointerNet_entity.attention_onehop(attn_feat, query, side_sent_reps, self._attn_v, self._attn_wq)

                # print('ws:', self._attn_ws.pow(2).sum())
                # print('wq:', self._attn_wq.pow(2).sum())
                # print('wm:', self._attn_wm.pow(2).sum())
            if self._side_attn_type == 'two-hop':
                attentions.append(attention.detach())
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
        if output_attn:
            if self.attn_type == 'dual':
                attn_dict['bi-attn'] = weight.detach()
            attn_dict['side-attn'] = attentions
            return extracts, attn_dict
        else:
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

    def _prepare_para(self, para_mem):
        side_feat = torch.matmul(para_mem, self.para_wm.unsqueeze(0))
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
    def attention_onehop(attention, query, side, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2) + side.unsqueeze(1)
        # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention_biattn(attention, tilde_s, query, side, v, w, s, t):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + \
               torch.matmul(query, w.unsqueeze(0)).unsqueeze(2) + \
               torch.matmul(side, s.unsqueeze(0)).unsqueeze(2) + \
               torch.matmul(tilde_s, t.unsqueeze(0)).unsqueeze(1)
        # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score


    @staticmethod
    def attention(attention, query, v, w, mem_sizes, output_attn=False, gold_mask=None, sigmoid=False):
        """ attention context vector"""
        score = LSTMPointerNet_entity.attention_score(attention, query, v, w)
        if gold_mask is not None:
            gold_mask = gold_mask.unsqueeze(1).repeat(1, score.size(1), 1)
            if sigmoid:
                norm_score = prob_normalize_sigmoid(score, gold_mask)
            else:
                norm_score = prob_normalize(score, gold_mask)
        else:
            if mem_sizes is None:
                if sigmoid:
                    norm_score = F.sigmoid(score, dim=-1)
                else:
                    norm_score = F.softmax(score, dim=-1)
            else:
                mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
                if sigmoid:
                    norm_score = prob_normalize_sigmoid(score, mask)
                else:
                    norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        if output_attn:
            return output, norm_score
        else:
            return output

    @staticmethod
    def hierarchical_attention(query, attention_para, attention_node, vp, wp, vn, wn, node_para_align, node_length=None, para_length=None):
        score_node = LSTMPointerNet_entity.attention_score(attention_node, query, vn, wn)
        score_para = LSTMPointerNet_entity.attention_score(attention_para, query, vp, wp)
        para_mask = len_mask(para_length, score_para.device).unsqueeze(-2)
        norm_score_para = prob_normalize(score_para, para_mask)
        if node_length is not None:
            node_mask = len_mask(node_length, score_node.device).unsqueeze(-2)
            norm_score_node = prob_normalize(score_node, node_mask)
        else:
            norm_score_node = F.softmax(score_node, dim=-1)

        nq = score_para.size(1)
        node_para_align = node_para_align.unsqueeze(1).repeat(1, nq, 1)
        score_para_node = norm_score_para.gather(2, node_para_align)

        score_node = torch.mul(norm_score_node, score_para_node)
        score_node = score_node / score_node.sum(dim=-1).unsqueeze(-1)

        output = torch.matmul(score_node, attention_node)
        return output




    def biattn(self, _in1, _in2, in1_sizes, in2_sizes):
        sim_matrix = torch.bmm(
            torch.matmul(_in1, self._biattn.unsqueeze(0)), _in2.transpose(1, 2)
        )

        def attn_on_dim(sim_matrix, _in, _in_sizes, dim):
            assert dim in [1, 2]
            mask = torch.zeros(sim_matrix.size(), dtype=torch.int8).to(sim_matrix.device)
            for _i, in_size in enumerate(_in_sizes):
                if dim == 2:
                    mask[_i, :, in_size:] = 1
                else:
                    mask[_i, in_size:, :] = 1
            mask = mask > 0
            sim_matrix.masked_fill_(mask, -float('Inf'))
            weight = F.softmax(sim_matrix, dim=dim)
            if dim == 2:
                in_rep = torch.matmul(weight, _in)
            else:
                in_rep = torch.matmul(weight.permute(0, 2, 1), _in)
            return in_rep, weight

        in1_rep, weight = attn_on_dim(sim_matrix, _in2, in2_sizes, dim=2)
        #in2_rep = attn_on_dim(sim_matrix, _in1, in1_sizes, dim=1)

        # mask_1 = torch.zeros(sim_matrix.size(), dtype=torch.int8).to(sim_matrix.device)
        # for _i, in2_size in enumerate(in2_sizes):
        #     mask_1[_i, :, in2_size:] = 1
        # mask_1 = mask_1 > 0
        # sim_matrix.masked_fill_(mask_1, -float('Inf'))
        # print(sim_matrix)
        #
        # #weight1 = F.softmax(sim_matrix, dim=1)
        # weight2 = F.softmax(sim_matrix, dim=2)
        # print(weight2)
        #
        # #in2_rep = torch.matmul(weight1.permute(0, 2, 1), _in1)
        # in1_rep = torch.matmul(weight2, _in2)
        return in1_rep, weight





class LSTMPointerNet_graph(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, side_dim, stop, hard_attention=False):
        super().__init__()
        # self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        # self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_h = nn.Linear(input_dim, n_hidden)
        self._init_c = nn.Linear(input_dim, n_hidden)
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        # init.uniform_(self._init_h, -INI, INI)
        # init.uniform_(self._init_c, -INI, INI)
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

        # # side info attention
        # if not hard_attention:
        #     self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        #     self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        #     self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        #     init.xavier_normal_(self.side_wm)
        #     init.xavier_normal_(self.side_wq)
        #     init.uniform_(self.side_v, -INI, INI)
        # else:
        #     self.side_wq = nn.Parameter(torch.Tensor(n_hidden, 1))
        #     self.side_wbi = nn.Bilinear(side_dim, side_dim, 1)
        #     init.xavier_normal_(self.side_wq)
        #     self._start = nn.Parameter(torch.Tensor(side_dim))
        #     init.uniform_(self._start)

        # if not hard_attention:
        #     self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        #     init.xavier_normal_(self._attn_ws)
        # else:
        #     self._attn_ws = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        #     init.xavier_normal_(self._attn_ws)

        # pad entity put in graph enc now
        # self._pad_entity = nn.Parameter(torch.Tensor(side_dim))
        # init.uniform_(self._pad_entity)

        # stop token
        if stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform_(self._stop, -INI, INI)
        self.stop = stop

        self._hard_attention = hard_attention
        if self._hard_attention:
            self.side_dim = side_dim


    def forward(self, attn_mem, mem_sizes, lstm_in, feature_sizes, entity=False):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""

        batch_size, max_sent_num, input_dim = attn_mem.size()

        if self.stop:
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
            for i, sent_num in enumerate(feature_sizes):
                attn_mem[i, sent_num, :] += self._stop
            feature_sizes = [feature_size+1 for feature_size in feature_sizes]

        # side_mem = torch.cat([side_mem, torch.zeros(batch_size, 1, side_dim).to(side_mem.device)], dim=1) #b * ns * s
        # for i, side_size in enumerate(side_sizes):
        #     side_mem[i, side_size, :] += self._pad_entity
        # side_sizes = [side_size+1 for side_size in side_sizes]


        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem, torch.tensor(mem_sizes, device=attn_mem.device))

        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)


        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet_graph.attention(
                hop_feat, query, self._hop_v, self._hop_wq, feature_sizes)

        output = LSTMPointerNet_graph.attention_score(attn_feat, query, self._attn_v, self._attn_wq)

        return output  # unormalized extraction logit


    def extract(self, attn_mem, mem_sizes, k, side_mem, side_sizes):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        # side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)

        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem, torch.tensor([max_sent], device=attn_mem.device))

        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []

        if self.stop:
            for sent_num in range(max_sent):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

                score = LSTMPointerNet_graph.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
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
        else:
            for _ in range(k):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

                score = LSTMPointerNet_graph.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem, sents_num):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))

        bs = attn_mem.size(0)
        input_dim = attn_mem.size(2)

        context = torch.gather(attn_mem, dim=1, index=sents_num.unsqueeze(1).unsqueeze(2).expand(bs, 1, input_dim))
        _init_h = self._init_h(context).permute(1, 0, 2)
        _init_c = self._init_c(context).permute(1, 0, 2)


        lstm_states = (_init_h.contiguous(),
                       _init_c.contiguous())
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
        score = LSTMPointerNet_graph.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output





class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, bert=False, sent_enc_type='cnn', bert_sent=False, bert_stride=0
                 , bertmodel='bert-large-uncased-whole-word-masking', **kwargs):
        super().__init__()
        assert sent_enc_type in ['cnn', 'mean']
        if sent_enc_type == 'cnn':
            self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, bert=bert)
        else:
            self._sent_enc = MeanSentEncoder()
            emb_dim = 3 * conv_hidden

        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )
        self._bert = bert
        self._bert_stride = bert_stride
        self._bertmodel = bertmodel
        self._bert_sent = bert_sent
        if bert:
            self._emb_dim = emb_dim
            if 'bert-large' in bertmodel:
                self._bert_linear = nn.Linear(4096, emb_dim)
            else:
                self._bert_linear = nn.Linear(3072, emb_dim)
            self._bert_relu = nn.PReLU()
            self._bert_model = Bert_model(bertmodel)

    def forward(self, article_sents, sent_nums, target, bertargs=None):
        if self._bert:
            word_nums = bertargs[1]
            articles = bertargs[0]
            if self._bert_sent:
                bs = articles.size(0)
                sent_nums = word_nums
                mask = (articles != 0).detach().float()
                with torch.no_grad():
                    bert_out = self._bert_model(articles)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                del bert_out
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                bert_hidden = bert_hidden * mask.unsqueeze(2)
                article_sents = []
                start = 0
                for sent_num in sent_nums:
                    article_sents.append(bert_hidden[start:start+sent_num])
                    start += sent_num
                del bert_hidden
            else:
                if self._bert_stride != 0:
                    source_nums = [sum(word_num) for word_num in word_nums]
                with torch.no_grad():
                    bert_out = self._bert_model(articles)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                del bert_out
                hsz = bert_hidden.size(2)
                if self._bert_stride != 0:
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
                        bert_hiddens.append(source)
                    bert_hidden = torch.stack(bert_hiddens)

                #max_word_num = max(max(word_nums))
                article_sents = []
                for i, word_num in enumerate(word_nums):
                    max_word_num = max(word_num)
                    if max_word_num < 5: # in case smaller than CNN stride
                        max_word_num = 5
                    new_word_num = []
                    start_num = 0
                    for num in word_num:
                        new_word_num.append((start_num, start_num+num))
                        start_num += num
                    article_sents.append(
                        torch.stack(
                            [torch.cat([bert_hidden[i, num[0]:num[1], :], torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
                             if (num[1] - num[0]) != max_word_num
                            else bert_hidden[i, num[0]:num[1], :]
                             for num in new_word_num
                            ]
                        )
                    )

            #article_sents = [torch.cat(article_sents, dim=0)]

        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def _bert_encode4decoding(self, bertargs):
        if self._bert:
            word_nums = bertargs[1]
            articles = bertargs[0]
            if self._bert_sent:
                bs = articles.size(0)
                sent_nums = word_nums
                mask = (articles != 0).detach().float()
                with torch.no_grad():
                    bert_out = self._bert_model(articles)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                del bert_out
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                bert_hidden = bert_hidden * mask.unsqueeze(2)
                article_sents = []
                start = 0
                for sent_num in sent_nums:
                    article_sents.append(bert_hidden[start:start+sent_num])
                    start += sent_num
                del bert_hidden
            else:
                if self._bert_stride != 0:
                    source_nums = [sum(word_num) for word_num in word_nums]
                with torch.no_grad():
                    bert_out = self._bert_model(articles)
                bert_hidden = torch.cat([bert_out[-1][_] for _ in [-4, -3, -2, -1]], dim=-1)
                bert_hidden = self._bert_relu(self._bert_linear(bert_hidden))
                del bert_out
                hsz = bert_hidden.size(2)
                if self._bert_stride != 0:
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
                        bert_hiddens.append(source)
                    bert_hidden = torch.stack(bert_hiddens)

                #max_word_num = max(max(word_nums))
                article_sents = []
                for i, word_num in enumerate(word_nums):
                    max_word_num = max(word_num)
                    if max_word_num < 5: # in case smaller than CNN stride
                        max_word_num = 5
                    new_word_num = []
                    start_num = 0
                    for num in word_num:
                        new_word_num.append((start_num, start_num+num))
                        start_num += num
                    article_sents.append(
                        torch.stack(
                            [torch.cat([bert_hidden[i, num[0]:num[1], :], torch.zeros(max_word_num - num[1] + num[0], hsz).to(bert_hidden.device)], dim=0)
                             if (num[1] - num[0]) != max_word_num
                            else bert_hidden[i, num[0]:num[1], :]
                             for num in new_word_num
                            ]
                        )
                    )
        return article_sents


    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def sample(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output, log_scores = self._extractor.sample(enc_out, sent_nums)
        return output, log_scores

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

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)



class PtrExtractSummGAT(nn.Module):
    """ gat model"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, bert=False, bert_stride=0, bertmodel='bert-large-uncased-whole-word-masking', gat_args={},
                 attn_type='glimpse', feed_gold=False, adj_type='concat_triple', mask_type='none', pe=False, decoder_supervision=False,
                 side_attn_type='two-hop', feature_banks=[]):
        super().__init__()
        assert adj_type in ['no_edge', 'edge_up', 'edge_down', 'concat_triple', 'edge_as_node']
        assert mask_type in ['none', 'encoder', 'soft', 'soft+sent']
        for feature_type in feature_banks:
            assert feature_type in ['freq', 'inpara_freq', 'nodefreq', 'segmentation']
        self._feature_banks = feature_banks
        feat_emb_dim = emb_dim // 4
        if 'nodefreq' in self._feature_banks:
            self._node_freq_embedding = nn.Embedding(MAX_FREQ, feat_emb_dim, padding_idx=0)

        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)

        if bert:
            emb_dim = 3 * conv_hidden
            self._sent_enc = MeanSentEncoder()
            self._node_enc = MeanSentEncoder()
        else:
            self._pe = pe
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self._sent_enc = ConvSentEncoder(vocab_size, emb_dim, conv_hidden, dropout, embedding=self._embedding, pe=pe, feature_banks=feature_banks)
            self._node_enc = MeanSentEncoder()
            if pe:
                emb_dim = emb_dim * 2
            if 'freq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'inpara_freq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'nodefreq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'segmentation' in self._feature_banks:
                emb_dim += feat_emb_dim
            gat_args['graph_hsz'] = emb_dim

        self._graph_layer_num = 1
        self._graph_model = gat_args['graph_model']
        if gat_args['graph_model'] in ['ggnn', 'gcn']:
            assert self._graph_layer_num == 1
        self._graph_enc = nn.ModuleList([gat_encode(gat_args) for _ in range(self._graph_layer_num)])
        if feed_gold or mask_type == 'encoder':
            self._graph_mask = node_mask(mask_type='gold')
        elif mask_type == 'soft':
            self._graph_mask = nn.ModuleList([node_mask(mask_type=mask_type, emb_dim=emb_dim*(i+1)) for i in range(self._graph_layer_num+1)])
        elif mask_type == 'soft+sent':
            self._graph_mask = nn.ModuleList(
                [node_mask(mask_type=mask_type, emb_dim=emb_dim * (i + 1), feat_dim=enc_out_dim) for i in range(self._graph_layer_num + 1)])
        else:
            self._graph_mask = node_mask(mask_type='none')
        self._graph_proj = nn.Linear(emb_dim, emb_dim)


        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )


        self._extractor = LSTMPointerNet_entity(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, emb_dim, attn_type, decoder_supervision, side_attn_type=side_attn_type
        )

        self._decoder_supervision = decoder_supervision
        self._adj_type = adj_type
        self._mask_type = mask_type
        self._bert = bert
        self._bertmodel = bertmodel
        self._bert_stride = bert_stride
        # if bert:
        #     self._emb_dim = emb_dim
        #     if 'bert-large' in bertmodel:
        #         self._bert_linear = nn.Linear(4096, emb_dim)
        #     else:
        #         self._bert_linear = nn.Linear(3072, emb_dim)
        #     self._bert_relu = nn.PReLU()
        #     self._bert_model = Bert_model(bertmodel)

        self._gold = feed_gold

    def forward(self, sent_nums, target, articles, ninfo, rinfo):
        (nodes, nmask, node_num, sw_mask, sent_node_aligns) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        #get bert embedding
        if self._bert:
            articles, word_nums = articles
            articles, article_sents = self._encode_bert(articles, word_nums)
        else:
            article_sents, articles, feature_dict = articles
            article_mask = articles.gt(0)
            articles = self._sent_enc._embedding(articles)
            if self._pe:
                bs, max_art_len = article_mask.size()
                src_pos = torch.tensor([[i for i in range(max_art_len)] for _ in range(bs)]).to(articles.device)
                src_pos = src_pos * article_mask.long()
                src_pos = self._sent_enc.poisition_enc(src_pos)
                articles = torch.cat([articles, src_pos], dim=-1)
            if 'freq' in self._feature_banks:
                word_freq = self._sent_enc._freq_embedding(feature_dict['word_freq'])
                articles = torch.cat([articles, word_freq], dim=-1)
            if 'inpara_freq' in self._feature_banks:
                word_inpara_freq = self._sent_enc._inpara_embedding(feature_dict['word_inpara_freq'])
                articles = torch.cat([articles, word_inpara_freq], dim=-1)

        if self._adj_type == 'concat_triple':
            node_num = [len(ts) for ts in triples]
        bs = len(article_sents)
        sent_word_freq = feature_dict.get('sent_word_freq', [[] for _ in range(bs)])
        sent_inpara_freq = feature_dict.get('sent_inpara_freq', [[] for _ in range(bs)])
        enc_out = self._encode(article_sents, sent_nums, sent_word_freq, sent_inpara_freq)
        if self._mask_type == 'soft':
            nodes, masks = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])
        elif self._mask_type == 'soft+sent':
            nodes, masks = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num,
                                              sw_mask, enc_out, sent_nums, nodefreq=feature_dict['node_freq'])
        else:
            nodes = self._encode_graph(articles, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])


        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )
        if self._decoder_supervision:
            output, selection = self._extractor(enc_out, sent_nums, ptr_in, nodes, node_num)
        elif self._gold:
            output = self._extractor(enc_out, sent_nums, ptr_in, nodes, node_num, side_mask=sw_mask)
        else:
            output = self._extractor(enc_out, sent_nums, ptr_in, nodes, node_num, aligns=sent_node_aligns)


        output = (output, )

        if 'soft' in self._mask_type:
            output += (masks, )
        if self._decoder_supervision:
            output += (selection, )

        return output


    def extract(self, article, article_sents, ninfo, rinfo, sent_nums=None, k=4, output_attn=False):
        # node implemented yet
        (nodes, nmask, node_num, sw_mask, feature_dict, sent_node_aligns) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        if not self._bert:
            article_mask = article.gt(0)
            article = self._sent_enc._embedding(article)
            if self._pe:
                bs, max_art_len = article_mask.size()
                src_pos = torch.tensor([[i for i in range(max_art_len)] for _ in range(bs)]).to(article.device)
                src_pos = src_pos * article_mask.long()
                src_pos[src_pos > 5999] = 5999
                src_pos = self._sent_enc.poisition_enc(src_pos)
                article = torch.cat([article, src_pos], dim=-1)
                if 'freq' in self._feature_banks:
                    word_freq = self._sent_enc._freq_embedding(feature_dict['word_freq'])
                    article = torch.cat([article, word_freq], dim=-1)
                if 'inpara_freq' in self._feature_banks:
                    word_inpara_freq = self._sent_enc._inpara_embedding(feature_dict['word_inpara_freq'])
                    article = torch.cat([article, word_inpara_freq], dim=-1)

        bs = len(article_sents)
        sent_word_freq = feature_dict.get('sent_word_freq', [[] for _ in range(bs)])
        sent_inpara_freq = feature_dict.get('sent_inpara_freq', [[] for _ in range(bs)])
        segment = feature_dict.get('segment', [[] for _ in range(bs)])
        enc_out = self._encode(article_sents, sent_nums, sent_word_freq, sent_inpara_freq, segment)
        #nodes = self._encode_graph(article, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask)
        if self._adj_type == 'concat_triple':
            node_num = [len(ts) for ts in triples]
        if self._mask_type == 'soft':
            nodes, masks = self._encode_graph(article, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])
        elif self._mask_type == 'soft+sent':
            nodes, masks = self._encode_graph(article, nodes, nmask, relations, rmask, triples, adjs, node_num,
                                              sw_mask, enc_out, sent_nums, nodefreq=feature_dict['node_freq'])
        else:
            nodes = self._encode_graph(article, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask, nodefreq=feature_dict['node_freq'])
            masks = None

        if self._gold:
            output, attns = self._extractor.extract(enc_out, sent_nums, k, nodes, node_num, output_attn=True,
                                                    side_mask=sw_mask)
        else:
            output, attns = self._extractor.extract(enc_out, sent_nums, k, nodes, node_num, output_attn=True, aligns=sent_node_aligns)

        attns['mask'] = masks

        #cluster_nums = [cluster_num + 1 for cluster_num in cluster_nums]
        if output_attn:
            return output, attns
        else:
            return output

    def _encode(self, article_sents, sent_nums, sent_word_freq=[], sent_inpara_freq=[], segment=[]):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0], sent_word_freq[0], sent_inpara_freq[0], segment[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            if len(sent_word_freq) != len(article_sents):
                sent_word_freq = [[] for _ in range(len(article_sents))]
            if len(sent_inpara_freq) != len(article_sents):
                sent_inpara_freq = [[] for _ in range(len(article_sents))]
            if len(segment) != len(article_sents):
                segment = [[] for _ in range(len(article_sents))]
            enc_sents = [self._sent_enc(art_sent, sent_word_freq[i], sent_inpara_freq[i], segment[i])
                         for i, art_sent in enumerate(article_sents)]
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

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, triples, adjs, node_num, node_mask=None, enc_out=None, sent_nums=None, nodefreq=None):
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

        if self._adj_type == 'concat_triple':
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
                    if self._mask_type == 'encoder' or 'soft' in self._mask_type:
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
            if self._graph_model in ['ggnn', 'gcn']:
                nodes = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
            else:
                nodes, edges = self._graph_enc[i_layer](adjs, triple_reps, nodes, node_num, edges)
            if self._adj_type != 'concat_triple':
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
        if self._adj_type != 'concat_triple':
            nodes = self._graph_proj(init_nodes) + nodes
        if 'soft' in self._mask_type:
            return nodes, masks
        else:
            return nodes

    def _encode_entity(self, clusters, cluster_nums, context=None):
        if cluster_nums is None: # test-time encode only
            if context is None:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context)
            else:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context[0, :, :])
        else:
            if context is None:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos) for cluster_words, cluster_wpos, cluster_spos in list(zip(*clusters))]
            else:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos, context[id, :, :])
                                for id, (cluster_words, cluster_wpos, cluster_spos) in enumerate(list(zip(*clusters)))]
            max_n = max(cluster_nums)

            def zero(n, device):
                z = torch.zeros(n, self._entity_enc._hsz).to(device)
                return z

            # if 0 in cluster_nums:
            #     print('0 cluster exists')
            #     print([enc_entity.size() for enc_entity in enc_entities])
            #     print(cluster_nums)
            #     cluster_nums = [cluster_num if cluster_num != 0 else 1 for cluster_num in cluster_nums]
            enc_entity = torch.stack(
                [torch.cat([s, zero(max_n - n, s.device)], dim=0)
                 if n != max_n
                 else s
                 for s, n in zip(enc_entities, cluster_nums)],
                dim=0
            )

        return enc_entity


    def set_embedding(self, embedding):
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)



class PtrExtractSummSubgraph(PtrExtractSummGAT):
    "subgraph"
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, bert=False, bert_stride=0, bertmodel='roberta-base', gat_args={},
                 attn_type='glimpse', feed_gold=False, adj_type='concat_triple', mask_type='none', pe=False, decoder_supervision=False,
                 side_attn_type='two-hop', feature_banks=[], hierarchical_attn=False):
        super().__init__(emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop, dropout, bert, bert_stride, bertmodel, gat_args,
                 attn_type, feed_gold, adj_type, mask_type, pe, decoder_supervision, side_attn_type, feature_banks)

        feat_emb_dim = emb_dim // 4
        if bert:
            emb_dim = 3 * conv_hidden
            self._emb_dim = emb_dim
            if 'large' in bertmodel:
                self._bert_linear = nn.Linear(1024, emb_dim)
            else:
                self._bert_linear = nn.Linear(768, emb_dim)
            self._bert_relu = nn.PReLU()
            self._bert_model = RobertaEmbedding(bertmodel)
            if 'nodefreq' in self._feature_banks:
                emb_dim += feat_emb_dim
            gat_args['graph_hsz'] = emb_dim


        else:
            if pe:
                emb_dim = emb_dim * 2
            if 'freq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'inpara_freq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'nodefreq' in self._feature_banks:
                emb_dim += feat_emb_dim
            if 'segmentation' in self._feature_banks:
                emb_dim += feat_emb_dim
            gat_args['graph_hsz'] = emb_dim
        # if bert:
        #     self._emb_dim = emb_dim
        #     if 'large' in bertmodel:
        #         self._bert_linear = nn.Linear(1024, emb_dim)
        #     else:
        #         self._bert_linear = nn.Linear(768, emb_dim)
        #     self._bert_relu = nn.PReLU()
        #     self._bert_model = RobertaEmbedding(bertmodel)

        self._graph_enc = subgraph_encode(gat_args)

        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        if mask_type == 'encoder':
            self._graph_mask = node_mask(mask_type='gold')
        elif mask_type == 'soft':
            self._graph_mask = node_mask(mask_type=mask_type, emb_dim=emb_dim)
        elif mask_type == 'soft+sent':
            self._graph_mask = node_mask(mask_type=mask_type, emb_dim=emb_dim, feat_dim=enc_out_dim)
        else:
            self._graph_mask = node_mask(mask_type='none')

        self._hierarchical_attn = hierarchical_attn
        if hierarchical_attn:
            self._extractor = LSTMPointerNet_entity(
                enc_out_dim, lstm_hidden, lstm_layer,
                dropout, n_hop, emb_dim, attn_type, decoder_supervision,
                side_attn_type=side_attn_type, hierarchical_attn=hierarchical_attn
            )

    def forward(self, sent_nums, target, articles, ninfo, rinfo):
        (nodes, nmask, node_num, sw_mask, gold_dec_mask) = ninfo
        (relations, rmask, triples, batch_adjs, batch_node_lists, sent_para_aligns) = rinfo
        if self._bert:
            articles, word_nums, feature_dict = articles
            articles, article_sents = self._encode_bert(articles, word_nums)
        else:
            article_sents, articles, feature_dict = articles
            article_mask = articles.gt(0)
            articles = self._sent_enc._embedding(articles)
            if self._pe:
                bs, max_art_len = article_mask.size()
                src_pos = torch.tensor([[i for i in range(max_art_len)] for _ in range(bs)]).to(articles.device)
                src_pos = src_pos * article_mask.long()
                src_pos = self._sent_enc.poisition_enc(src_pos)
                articles = torch.cat([articles, src_pos], dim=-1)
            if 'freq' in self._feature_banks:
                word_freq = self._sent_enc._freq_embedding(feature_dict['word_freq'])
                articles = torch.cat([articles, word_freq], dim=-1)
            if 'inpara_freq' in self._feature_banks:
                word_inpara_freq = self._sent_enc._inpara_embedding(feature_dict['word_inpara_freq'])
                articles = torch.cat([articles, word_inpara_freq], dim=-1)
            if 'segmentation' in self._feature_banks:
                seg_para = self._sent_enc._seg_embedding(feature_dict['seg_para'])
                articles = torch.cat([articles, seg_para], dim=-1)

        bs = len(article_sents)
        sent_word_freq = feature_dict.get('sent_word_freq', [[] for _ in range(bs)])
        sent_inpara_freq = feature_dict.get('sent_inpara_freq', [[] for _ in range(bs)])
        segment = feature_dict.get('seg_sent', [[] for _ in range(bs)])
        enc_out = self._encode(article_sents, sent_nums, sent_word_freq, sent_inpara_freq, segment)
        if self._mask_type == 'soft':
            outputs = self._encode_graph(articles, nodes, nmask, relations, rmask, batch_adjs, batch_node_lists, sw_mask, nodefreq=feature_dict['node_freq'])
        elif self._mask_type == 'soft+sent':
            outputs = self._encode_graph(articles, nodes, nmask, relations, rmask, batch_adjs, batch_node_lists,
                                              sw_mask, enc_out, sent_nums, nodefreq=feature_dict['node_freq'])
        else:
            outputs = self._encode_graph(articles, nodes, nmask, relations, rmask, batch_adjs, batch_node_lists, sw_mask, nodefreq=feature_dict['node_freq'])

        if self._hierarchical_attn:
            topics, masks, nodes = outputs
        elif 'soft' in self._mask_type:
            topics, masks = outputs
        else:
            topics = outputs


        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )

        if self._gold:
            output = self._extractor(enc_out, sent_nums, ptr_in, topics[0], topics[1], side_mask=gold_dec_mask)
        elif self._hierarchical_attn:
            node_para_aligns = pad_batch_tensorize(nodes[2], pad=0, cuda=False).to(enc_out.device)
            output = self._extractor(enc_out, sent_nums, ptr_in, nodes[0], nodes[1], paras=(topics[0], topics[1], node_para_aligns))
        else:
            output = self._extractor(enc_out, sent_nums, ptr_in, topics[0], topics[1], aligns=sent_para_aligns)

        output = (output,)

        if 'soft' in self._mask_type:
            output += (masks,)

        return output

    def _encode_graph(self, articles, nodes, nmask, relations, rmask, batch_adjs, node_lists,
                      node_mask=None, enc_out=None, sent_nums=None, nodefreq=None):
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
        #nodes = self._graph_proj(init_nodes) + nodes
        results = ((topics, topic_length), )

        if 'soft' in self._mask_type:
            results += (masks, )

        if self._hierarchical_attn:
            results += ((node_reps, node_length, node_align_paras), )

        return results

    def extract(self, article, article_sents, ninfo, rinfo, sent_nums=None, k=4, output_attn=False):
        # node implemented yet
        (nodes, nmask, node_num, sw_mask, gold_dec_mask, feature_dict) = ninfo
        (relations, rmask, triples, batch_adjs, batch_node_lists, sent_para_aligns) = rinfo
        if not self._bert:
            article_mask = article.gt(0)
            article = self._sent_enc._embedding(article)
            if self._pe:
                bs, max_art_len = article_mask.size()
                src_pos = torch.tensor([[i for i in range(max_art_len)] for _ in range(bs)]).to(article.device)
                src_pos = src_pos * article_mask.long()
                src_pos[src_pos > 5999] = 5999
                src_pos = self._sent_enc.poisition_enc(src_pos)
                article = torch.cat([article, src_pos], dim=-1)
            if 'freq' in self._feature_banks:
                word_freq = self._sent_enc._freq_embedding(feature_dict['word_freq'])
                article = torch.cat([article, word_freq], dim=-1)
            if 'inpara_freq' in self._feature_banks:
                word_inpara_freq = self._sent_enc._inpara_embedding(feature_dict['word_inpara_freq'])
                article = torch.cat([article, word_inpara_freq], dim=-1)
            if 'segmentation' in self._feature_banks:
                seg_para = self._sent_enc._seg_embedding(feature_dict['seg_para'])
                article = torch.cat([article, seg_para], dim=-1)

        bs = len(article_sents)
        sent_word_freq = feature_dict.get('sent_word_freq', [[] for _ in range(bs)])
        sent_inpara_freq = feature_dict.get('sent_inpara_freq', [[] for _ in range(bs)])
        segment = feature_dict.get('seg_sent', [[] for _ in range(bs)])

        enc_out = self._encode(article_sents, sent_nums, sent_word_freq, sent_inpara_freq, segment)
        #nodes = self._encode_graph(article, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask)
        if self._mask_type == 'soft':
            topics, masks = self._encode_graph(article, nodes, nmask, relations, rmask, batch_adjs, batch_node_lists, sw_mask, nodefreq=feature_dict['node_freq'])
        elif self._mask_type == 'soft+sent':
            topics, masks = self._encode_graph(article, nodes, nmask, relations, rmask, batch_adjs, batch_node_lists,
                                              sw_mask, enc_out, sent_nums, nodefreq=feature_dict['node_freq'])
        else:
            topics = self._encode_graph(article, nodes, nmask, relations, rmask, triples, batch_adjs, batch_node_lists, sw_mask, nodefreq=feature_dict['node_freq'])

        if self._gold:
            output, attns = self._extractor.extract(enc_out, sent_nums, k, topics[0], topics[1], output_attn=True,
                                                    side_mask=gold_dec_mask)
        else:
            output, attns = self._extractor.extract(enc_out, sent_nums, k, topics[0], topics[1], output_attn=True, aligns=sent_para_aligns)

        if 'soft' in self._mask_type:
            attns['mask'] = masks

        #cluster_nums = [cluster_num + 1 for cluster_num in cluster_nums]
        if output_attn:
            return output, attns
        else:
            return output


