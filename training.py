""" module providing basic training utilities"""
#from coherence_interface.coherence_inference import coherence_infer, batch_global_infer
import os
from os.path import join
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce, concat

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX
from utils import PAD, UNK, START, END
from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ
from nltk import sent_tokenize
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').disabled = True
logging.basicConfig(level=logging.ERROR)
from copy import deepcopy
from model.util import sequence_loss
from torch.nn import functional as F


def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        try:
            grad_norm = grad_norm.item()
        except AttributeError:
            grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

def get_loss_args(net_out, bw_args):
    if isinstance(net_out, tuple):
        loss_args = net_out + bw_args
    else:
        loss_args = (net_out, ) + bw_args
    return loss_args

@curry
def compute_loss(net, criterion, fw_args, loss_args):
    net_out = net(*fw_args)
    all_loss_args = get_loss_args(net_out, loss_args)
    loss = criterion(*all_loss_args)
    return loss

@curry
def val_step(loss_step, fw_args, loss_args):
    loss = loss_step(fw_args, loss_args)
    try:
        n_data = loss.size(0)
        return n_data, loss.sum().item()
    except:
        n_data = 1
        return n_data, loss.item()

@curry
def multitask_val_step(loss_step, fw_args, loss_args):
    losses = loss_step(fw_args, loss_args)
    try:
        n_data = losses[0].size(0)
        return n_data, losses[0].sum().item(), losses[1].sum().item()
    except IndexError:
        n_data = 1
        return n_data, losses[0].item(), losses[1].item()

@curry
def basic_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net, criterion))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}

@curry
def multitask_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = multitask_val_step(compute_loss(net, criterion))
        n_data, tot_loss, tot_aux_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]),
            starmap(validate_fn, val_batches),
            (0, 0, 0)
        )
    val_loss = tot_loss / n_data
    val_aux_loss = tot_aux_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    print('validation auxilary loss: {:.4f} ... '.format(val_aux_loss))
    return {'loss': val_loss, 'aux_loss': val_aux_loss}


@curry
def rl_validate(net, val_batches, reward_func=None, reward_coef = 0.01, local_coh_func=None, local_coh_coef=0.005, bert=False):
    print('running validation ... ', end='')
    if bert:
        tokenizer = net._bert_model._tokenizer
        end = tokenizer.encoder[tokenizer._eos_token]
        unk = tokenizer.encoder[tokenizer._unk_token]
    def argmax(arr, keys):
        return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
    def sum_id2word(raw_article_sents, decs, attns):
        if bert:
            dec_sents = []
            for i, raw_words in enumerate(raw_article_sents):
                dec = []
                for id_, attn in zip(decs, attns):
                    if id_[i] == end:
                        break
                    elif id_[i] == unk:
                        dec.append(argmax(raw_words, attn[i]))
                    else:
                        dec.append(id2word[id_[i].item()])
                dec_sents.append(dec)
        else:
            dec_sents = []
            for i, raw_words in enumerate(raw_article_sents):
                dec = []
                for id_, attn in zip(decs, attns):
                    if id_[i] == END:
                        break
                    elif id_[i] == UNK:
                        dec.append(argmax(raw_words, attn[i]))
                    else:
                        dec.append(id2word[id_[i].item()])
                dec_sents.append(dec)
        return dec_sents
    net.eval()
    start = time()
    i = 0
    score = 0
    score_reward = 0
    score_local_coh = 0
    score_r2 = 0
    score_r1 = 0
    bl_r2 = []
    bl_r1 = []
    with torch.no_grad():
        for fw_args, bw_args in val_batches:
            raw_articles = bw_args[0]
            id2word = bw_args[1]
            raw_targets = bw_args[2]
            if reward_func is not None:
                questions = bw_args[3]
            greedies, greedy_attns = net.greedy(*fw_args)
            greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns)
            bl_scores = []
            if reward_func is not None:
                bl_coh_inputs = []
            if local_coh_func is not None:
                bl_local_coh_scores = []
            for baseline, target in zip(greedy_sents, raw_targets):
                if bert:
                    text = ''.join(baseline)
                    baseline = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                                 errors=tokenizer.errors)
                    baseline = baseline.split(' ')
                    text = ''.join(target)
                    target = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                               errors=tokenizer.errors)
                    target = target.split(' ')


                bss = sent_tokenize(' '.join(baseline))
                if reward_func is not None:
                    bl_coh_inputs.append(bss)

                bss = [bs.split(' ') for bs in bss]
                tgs = sent_tokenize(' '.join(target))
                tgs = [tg.split(' ') for tg in tgs]
                bl_score = compute_rouge_l_summ(bss, tgs)
                bl_r1.append(compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1))
                bl_r2.append(compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
                bl_scores.append(bl_score)
            bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)

            if reward_func is not None:
                bl_reward_scores = reward_func.score(questions, bl_coh_inputs)
                bl_reward_scores = torch.tensor(bl_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)
                score_reward += bl_reward_scores.mean().item() * 100

            reward = bl_scores.mean().item()
            i += 1
            score += reward * 100
            score_r2 += torch.tensor(bl_r2, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100
            score_r1 += torch.tensor(bl_r1, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100


    val_score = score / i
    score_r2 = score_r2 / i
    score_r1 = score_r1 / i
    if reward_func is not None:
        val_reward_score = score_reward / i
    else:
        val_reward_score = 0
    val_local_coh_score = 0
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation reward: {:.4f} ... '.format(val_score))
    print('validation r2: {:.4f} ... '.format(score_r2))
    print('validation r1: {:.4f} ... '.format(score_r1))
    if reward_func is not None:
        print('validation reward: {:.4f} ... '.format(val_reward_score))
    if local_coh_func is not None:
        val_local_coh_score = score_local_coh / i
        print('validation {} reward: {:.4f} ... '.format(local_coh_func.__name__, val_local_coh_score))
    print('n_data:', i)
    return {'score': val_score,
            'score_reward:': val_reward_score}

class BasicPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()


    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        #self._net.zero_grad()
        fw_args, bw_args = next(self._batches)
        net_out = self._net(*fw_args)

        # get logs and output for logging, backward
        log_dict = {}
        loss_args = self.get_loss_args(net_out, bw_args)

        # backward and update ( and optional gradient monitoring )
        loss = self._criterion(*loss_args).mean()

        loss.backward()
        log_dict['loss'] = loss.item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        #torch.cuda.empty_cache()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()

class AbsSelfCriticalPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, optim, grad_fn=None, reward_fn=None, reward_coef=0.01, local_coh_func=None, local_coh_coef=0.005,
                 accumulate_g_step=1, weights=[0., 0.5, 0.5], bert=False, multigpu=False, ml_loss=False):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn
        self._reward_fn = reward_fn
        self._reward_w8 = reward_coef
        self._w8 = weights

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()
        self._local_coh_fun = local_coh_func
        self._local_co_coef = local_coh_coef
        self._accumulate_g_step = accumulate_g_step
        self._bert = bert
        if bert:
            print('use bert in model')
            self._tokenizer = self._net._bert_model._tokenizer
            self._end = self._tokenizer.encoder[self._tokenizer._eos_token]
            self._unk = self._tokenizer.encoder[self._tokenizer._unk_token]
            self._pad = self._tokenizer.encoder[self._tokenizer._pad_token]
        else:
            self._end = END
            self._unk = UNK
            self._pad = PAD

        self._multigpu = multigpu
        self._ml_loss = ml_loss
        self._ml_loss_w8 = 0.01
        if self._ml_loss:
            nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)

            def criterion(logits, targets):
                return sequence_loss(logits, targets, nll, pad_idx=self._pad)
            print('Add ML loss')
            print('PAD id:', self._pad)
            self._ml_criterion = criterion

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args


    def train_step(self, sample_time=1):
        torch.autograd.set_detect_anomaly(True)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        def sum_id2word(raw_article_sents, decs, attns, id2word):
            if self._bert:
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
            else:
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
        def pack_seq(seq_list):
            return torch.cat([_.unsqueeze(1) for _ in seq_list], 1)
        # forward pass of model
        self._net.train()
        #self._net.zero_grad()
        total_loss = None
        for i in range(self._accumulate_g_step):
            fw_args, bw_args = next(self._batches)
            raw_articles = bw_args[0]
            id2word = bw_args[1]
            raw_targets = bw_args[2]
            if self._reward_fn is not None:
                questions = bw_args[3]
            targets = bw_args[4]

            # encode
            # attention, init_dec_states, nodes = self._net.encode_general(*fw_args)
            #fw_args += (attention, init_dec_states, nodes)
            # _init_dec_states = ((init_dec_states[0][0].clone(), init_dec_states[0][1].clone()), init_dec_states[1].clone())
            with torch.no_grad():
                # g_fw_args = fw_args + (attention, _init_dec_states, nodes, False)
                # greedies, greedy_attns = self._net.rl_step(*g_fw_args)
                greedies, greedy_attns = self._net.greedy(*fw_args)
            greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns, id2word)
            bl_scores = []
            if self._reward_fn is not None:
                bl_reward_scores = []
                bl_reward_inputs = []
            if self._local_coh_fun is not None:
                bl_local_coh_scores = []
            for baseline, target in zip(greedy_sents, raw_targets):
                if self._bert:
                    text = ''.join(baseline)
                    baseline = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                          errors=self._tokenizer.errors)
                    baseline = baseline.strip().lower().split(' ')
                    text = ''.join(target)
                    target = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                                 errors=self._tokenizer.errors)
                    target = target.strip().lower().split(' ')


                bss = sent_tokenize(' '.join(baseline))
                tgs = sent_tokenize(' '.join(target))
                if self._reward_fn is not None:
                    bl_reward_inputs.append(bss)
                if self._local_coh_fun is not None:
                    local_coh_score = self._local_coh_fun(bss)
                    bl_local_coh_scores.append(local_coh_score)
                bss = [bs.split(' ') for bs in bss]
                tgs = [tg.split(' ') for tg in tgs]

                #bl_score = compute_rouge_l_summ(bss, tgs)
                bl_score = (self._w8[2] * compute_rouge_l_summ(bss, tgs) + \
                           self._w8[0] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1) + \
                            self._w8[1] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
                bl_scores.append(bl_score)
            bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)


            # sample
            # s_fw_args = fw_args + (attention, init_dec_states, nodes, True)
            # samples, sample_attns, seqLogProbs = self._net.rl_step(*s_fw_args)
            fw_args += (self._ml_loss, )
            if self._ml_loss:
                samples, sample_attns, seqLogProbs, ml_logit = self._net.sample(*fw_args)
            else:
                samples, sample_attns, seqLogProbs = self._net.sample(*fw_args)
            sample_sents = sum_id2word(raw_articles, samples, sample_attns, id2word)
            sp_seqs = pack_seq(samples)
            _masks = (sp_seqs > PAD).float()
            sp_seqLogProb = pack_seq(seqLogProbs)
            #loss_nll = - sp_seqLogProb.squeeze(2)
            loss_nll = - sp_seqLogProb.squeeze(2) * _masks.detach().type_as(sp_seqLogProb)
            sp_scores = []
            if self._reward_fn is not None:
                sp_reward_inputs = []
            for sample, target in zip(sample_sents, raw_targets):
                if self._bert:
                    text = ''.join(sample)
                    sample = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                          errors=self._tokenizer.errors)
                    sample = sample.strip().lower().split(' ')
                    text = ''.join(target)
                    target = bytearray([self._tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                                 errors=self._tokenizer.errors)
                    target = target.strip().lower().split(' ')

                sps = sent_tokenize(' '.join(sample))
                tgs = sent_tokenize(' '.join(target))
                if self._reward_fn is not None:
                    sp_reward_inputs.append(sps)

                sps = [sp.split(' ') for sp in sps]
                tgs = [tg.split(' ') for tg in tgs]
                #sp_score = compute_rouge_l_summ(sps, tgs)
                sp_score = (self._w8[2] * compute_rouge_l_summ(sps, tgs) + \
                            self._w8[0] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=1) + \
                            self._w8[1]* compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=2))
                sp_scores.append(sp_score)
            sp_scores = torch.tensor(sp_scores, dtype=torch.float32, device=greedy_attns[0].device)
            if self._reward_fn is not None:
                sp_reward_scores, bl_reward_scores = self._reward_fn.score_two_seqs(questions, sp_reward_inputs, bl_reward_inputs)
                sp_reward_scores = torch.tensor(sp_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)
                bl_reward_scores = torch.tensor(bl_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)

            reward = sp_scores.view(-1, 1) - bl_scores.view(-1, 1)
            if self._reward_fn is not None:
                reward += self._reward_w8 * (sp_reward_scores.view(-1, 1) - bl_reward_scores.view(-1, 1))
            reward.requires_grad_(False)

            loss = reward.contiguous().detach() * loss_nll
            loss = loss.sum()
            full_length = _masks.data.float().sum()
            loss = loss / full_length
            if self._ml_loss:
                ml_loss = self._ml_criterion(ml_logit, targets)
                loss += self._ml_loss_w8 * ml_loss.mean()
            # if total_loss is None:
            #     total_loss = loss
            # else:
            #     total_loss += loss

            loss = loss / self._accumulate_g_step
            loss.backward()

        log_dict = {}
        if self._reward_fn is not None:
            log_dict['reward'] = bl_scores.mean().item()
            log_dict['question_reward'] = bl_reward_scores.mean().item()
            log_dict['sample_question_reward'] = sp_reward_scores.mean().item()
            log_dict['sample_reward'] = sp_scores.mean().item()
        else:
            log_dict['reward'] = bl_scores.mean().item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())

        self._opt.step()
        self._net.zero_grad()
        #torch.cuda.empty_cache()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, BasicPipeline) or isinstance(pipeline, AbsSelfCriticalPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None


    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        if 'question_reward' in log_dict:
            print('train step: {}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}\r'.format(
                self._step,
                'reward',
                log_dict['reward'],
                'question_reward',
                log_dict['question_reward'],
                'sample_question_reward',
                log_dict['sample_question_reward'],
                'sample_reward',
                log_dict['sample_reward']
                ), end='')
        else:
            print('train step: {}, {}: {:.4f}\r'.format(
                self._step,
                'loss' if 'loss' in log_dict else 'reward',
                self._running_loss), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()




class MultiTaskTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, BasicPipeline) or isinstance(pipeline, AbsSelfCriticalPipeline) or isinstance(pipeline, MultiTaskPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        self._running_aux_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        aux_loss = log_dict['aux_loss'] if 'aux_loss' in log_dict else 0
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss

        if self._running_aux_loss is not None and aux_loss != 0:
            self._running_aux_loss = 0.99*self._running_aux_loss + 0.01*aux_loss
        elif aux_loss != 0:
            self._running_aux_loss = aux_loss


        if self._running_aux_loss is not None:
            print('train step: {}, {}: {:.4f}, {}:{:.4f}\r'.format(
                self._step,
                'auxilary loss',
                self._running_aux_loss,
                'loss', self._running_loss), end='')
        else:
            print('train step: {}, {}: {:.4f}\r'.format(
                self._step,
                'loss' if 'loss' in log_dict else 'reward',
                self._running_loss), end='')

        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        elif 'aux_loss' in val_log:
            val_metric = (val_log['loss'], val_log['aux_loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric[0])
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        val_metric = val_metric[0]
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()


class MultiTaskPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn
        self._aux_w = 0.1


        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()


    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        #self._net.zero_grad()
        fw_args, bw_args = next(self._batches)
        net_out = self._net(*fw_args)

        # get logs and output for logging, backward
        log_dict = {}
        loss_args = self.get_loss_args(net_out, bw_args)

        # backward and update ( and optional gradient monitoring )
        losses = self._criterion(*loss_args)
        loss = losses[0]
        for _loss in losses[1:]:
            loss += self._aux_w * _loss

        loss.backward()
        log_dict['loss'] = losses[0].item()
        log_dict['aux_loss'] = losses[1].item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        #torch.cuda.empty_cache()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None and len(val_metric) == 2:
            name = 'ckpt-{:6f}-{:3f}-{}'.format(val_metric[0], val_metric[1], step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()