from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os.path import join
import re, os, json
import torch
import argparse
from cytoolz import curry, concat
from tqdm import trange, tqdm
from train_roberta_multiple_choice import Bert_choice, _count_data, pad_batch_tensorize_3d
from metric import compute_rouge_l_summ, compute_rouge_n
from collections import OrderedDict
import multiprocessing as mp
from pytorch_transformers import BertTokenizer
import random, math

MAX_LEN = 250

def load_best_ckpt(model_dir):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[2]), reverse=True)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=lambda storage, loc: storage
    )['state_dict']
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
            new_ckpt[name] = v
        else:
            name = k  # remove `module.`
            new_ckpt[name] = v
    return new_ckpt

@curry
def process_one(blank, tokenizer, _q):
    questions, context = _q
    idxs = [i for i in range(len(questions))]

    if len(idxs) > 1:
        random.shuffle(idxs)
        new_questions = [questions[idx] for idx in idxs[:1]]
        questions = new_questions


    context = ' '.join(context)
    choicess = [[question['answer'], question['choice1'], question['choice2'], question['choice3']] for question
                in
                questions]
    questions = [question['question'].replace('<\\blank>', blank) for question in questions]
    questions = [[tokenizer.tokenize(qp.lower()) for qp in question.split(blank)] for question in
                 questions]
    new_questions = []
    for question in questions:
        new_q = ['[CLS]']
        for q in question:
            new_q += q + [blank]
        new_q.pop()
        new_questions.append(new_q)
    questions = new_questions
    contexts = [['[SEP]'] + tokenizer.tokenize(context.lower()) for _ in range(len(questions))]
    choicess = [[[tokenizer.tokenize(c.lower()) for c in choice] for choice in choices] for choices in
                choicess]
    choicess = [
        [['[SEP]'] + choice[0] + ['[SEP]'] + choice[1] if len(choice) == 2 else ['[SEP]'] + choice[0] for choice in choices]
        for
        choices in choicess]
    _inputs = [
        [tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
        question, context, choices in zip(questions, contexts, choicess)]
    return _inputs

@curry
def process_one_two_deqs(blank, tokenizer, _q):
    questions, context_sample, context_greedy = _q
    idxs = [i for i in range(len(questions))]

    if len(idxs) > 1:
        random.shuffle(idxs)
        new_questions = [questions[idx] for idx in idxs[:1]]
        questions = new_questions


    context1 = ' '.join(context_sample)
    context2 = ' '.join(context_greedy)
    choicess = [[question['answer'], question['choice1'], question['choice2'], question['choice3']] for question
                in
                questions]
    questions = [question['question'].replace('<\\blank>', blank) for question in questions]
    questions = [[tokenizer.tokenize(qp.lower()) for qp in question.split(blank)] for question in
                 questions]
    new_questions = []
    for question in questions:
        new_q = ['[CLS]']
        for q in question:
            new_q += q + [blank]
        new_q.pop()
        new_questions.append(new_q)
    questions = new_questions
    context_sample = [['[SEP]'] + tokenizer.tokenize(context1.lower()) for _ in range(len(questions))]
    context_greedy = [['[SEP]'] + tokenizer.tokenize(context2.lower()) for _ in range(len(questions))]
    choicess = [[[tokenizer.tokenize(c.lower()) for c in choice] for choice in choices] for choices in
                choicess]
    choicess = [
        [['[SEP]'] + choice[0] + ['[SEP]'] + choice[1] if len(choice) == 2 else ['[SEP]'] + choice[0] for choice in choices]
        for
        choices in choicess]
    _inputs_greedy = [
        [tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
        question, context, choices in zip(questions, context_sample, choicess)]
    _inputs_sample = [
        [tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
        question, context, choices in zip(questions, context_greedy, choicess)]
    return _inputs_greedy, _inputs_sample


class cloze_reward():
    def __init__(self, model_dir, device='cuda', bs=32):
        model = Bert_choice()
        tokenizer = model._tokenizer
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        ckpt = load_best_ckpt(model_dir)
        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._blank = '[unused0]'
        self._device = device
        self._mini_bs = bs

    def score_two_seqs(self, questions, abstracts_sample, abstracts_baseline):
        pscores = []
        #_inputs, batch_index = self.process_data(questions, abstracts)
        pss = []
        assert len(abstracts_sample) == len(abstracts_baseline)
        data_num = len(abstracts_sample)
        _inputs, batch_index = self.process_data_mp_two_seqs(questions, abstracts_sample, abstracts_baseline)
        bs, cn, length = _inputs.size()
        for _i in range(math.ceil(bs / self._mini_bs)):
            _bs = min(self._mini_bs, bs - _i*self._mini_bs)
            labels = torch.tensor([0 for _ in range(_bs)]).to(self._device)
            with torch.no_grad():
                scores, batch_ids = self._model.evaluation(_inputs[_i*self._mini_bs:(_i+1)*self._mini_bs, :], labels)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            pss.extend(scores[:, 0].tolist())
        total_count = 0
        for _index in batch_index:
            ps = pss[total_count:total_count+_index]
            total_count += _index
            if len(ps) == 0:
                prob = 0
            else:
                prob = sum(ps) / len(ps)
            pscores.append(prob)
        sample_scores = pscores[:data_num]
        greedy_scores = pscores[data_num:]

        return sample_scores, greedy_scores

    def score(self, questions, abstracts):
        pscores = []
        #_inputs, batch_index = self.process_data(questions, abstracts)
        pss = []
        _inputs, batch_index = self.process_data_mp(questions, abstracts)
        bs, cn, length = _inputs.size()
        for _i in range(math.ceil(bs / self._mini_bs)):
            _bs = min(self._mini_bs, bs - _i*self._mini_bs)
            labels = torch.tensor([0 for _ in range(_bs)]).to(self._device)
            with torch.no_grad():
                scores, batch_ids = self._model.evaluation(_inputs[_i*self._mini_bs:(_i+1)*self._mini_bs, :], labels)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            pss.extend(scores[:, 0].tolist())
        total_count = 0
        for _index in batch_index:
            ps = pss[total_count:total_count+_index]
            if len(ps) == 0:
                prob = 0
            else:
                prob = sum(ps) / len(ps)
            pscores.append(prob)

        return pscores

    def process_data(self, batch_questions, contexts):
        batch_inputs = []
        batch_indexs = []
        for (questions, context) in zip(batch_questions, contexts):
            context = ' '.join(context)
            choicess = [[question['answer'], question['choice1'], question['choice2'], question['choice3']] for question
                        in
                        questions]
            questions = [question['question'].replace('<\\blank>', self._blank) for question in questions]
            questions = [[self._tokenizer.tokenize(qp.lower()) for qp in question.split(self._blank)] for question in
                         questions]
            new_questions = []
            for question in questions:
                new_q = ['[CLS]']
                for q in question:
                    new_q += q + [self._blank]
                new_q.pop()
                new_questions.append(new_q)
            questions = new_questions
            contexts = [['[SEP]'] + self._tokenizer.tokenize(context.lower()) for _ in range(len(questions))]
            choicess = [[[self._tokenizer.tokenize(c.lower()) for c in choice] for choice in choices] for choices in
                        choicess]
            choicess = [
                [['[SEP]'] + choice[0] + ['[SEP]'] + choice[1] if len(choice) == 2 else ['[SEP]'] + choice[0] for choice in choices]
                for
                choices in choicess]
            _inputs = [
                [self._tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
                question, context, choices in zip(questions, contexts, choicess)]
            batch_inputs.extend(_inputs)
            batch_indexs.append(len(_inputs))

        batch_inputs = pad_batch_tensorize_3d(batch_inputs, pad=0, cuda=False)
        return batch_inputs.to(self._device), batch_indexs


    def process_data_mp(self, batch_questions, contexts):
        with mp.Pool(processes=4) as pool:
            batch_inputs = list(pool.map(process_one(self._blank, self._tokenizer),
                                 zip(batch_questions, contexts), chunksize=1))
        batch_indexs = [len(_inputs) for _inputs in batch_inputs]
        batch_inputs = list(concat(batch_inputs))
        batch_inputs = pad_batch_tensorize_3d(batch_inputs, pad=0, cuda=False)
        return batch_inputs.to(self._device), batch_indexs

    def process_data_mp_two_seqs(self, batch_questions, sample_contexts, greedy_contexts):
        with mp.Pool(processes=4) as pool:
            batch_inputs = list(pool.map(process_one_two_deqs(self._blank, self._tokenizer),
                                 zip(batch_questions, sample_contexts, greedy_contexts), chunksize=1))
            batch_inputs = [batch_input[0] for batch_input in batch_inputs] + [batch_input[1] for batch_input in batch_inputs]
        batch_indexs = [len(_inputs) for _inputs in batch_inputs]
        batch_inputs = list(concat(batch_inputs))
        batch_inputs = pad_batch_tensorize_3d(batch_inputs, pad=0, cuda=False)
        return batch_inputs.to(self._device), batch_indexs
