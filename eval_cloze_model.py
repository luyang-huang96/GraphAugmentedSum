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
from scipy import stats
import numpy as np

MAX_LEN = 512


class ClozeDataset(Dataset):
    def __init__(self, split: str, path: str, system_path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)
        self._system_path = system_path

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js_data = json.loads(f.read())
        context = []
        try:
            with open(join(self._system_path, 'output', '{}.dec'.format(i))) as g:
                for line in g:
                    context.append(line.strip())
        except FileNotFoundError:
            with open(join(self._system_path, 'test', '{}.ref'.format(i))) as g:
                for line in g:
                    context.append(line.strip())
        #context = ' '.join(context)
        try:
            abs = js_data['abstract']
            abs = [_abstract.lower().split(' ') for _abstract in abs]
        except KeyError:
            abs = []

        try:
            _id = js_data['id']
            questions = js_data['questions']
        except KeyError:
            questions = []
            _id = '0'

        return questions, context, _id, abs

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

def data_loader(split, tokenizer, args):
    dataset = ClozeDataset(split, args.data_path, args.system_path)
    @curry
    def coll(tokenizer, batch):
        blank = '[unused0]'
        questions, context, _ids, abstract = list(filter(bool, list(zip(*batch))))
        system = context
        # print('q:', questions)
        # print('c:', context)
        if len(abstract) > 0:
            rouges = {
                'RLr': compute_rouge_l_summ([_c.lower().split(' ') for _c in context[0]], abstract[0], mode='r'),
                'RLf': compute_rouge_l_summ([_c.lower().split(' ') for _c in context[0]], abstract[0], mode='f'),
                'R1': compute_rouge_n(' '.join(context[0]).split(' '), list(concat(abstract[0])), mode='f', n=1),
                'R2': compute_rouge_n(' '.join(context[0]).split(' '), list(concat(abstract[0])), mode='f', n=2)
            }
        if len(questions[0]) != 0:
            questions = questions[0]
            context = context[0]
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
                [['[SEP]'] + choice[0] + ['[SEP]'] + choice[1] if len(choice) == 2 else ['[SEP]'] + choice[0] for choice in choices] for
                choices in choicess]
            _inputs = [
                [tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
                question, context, choices in zip(questions, contexts, choicess)]
            _inputs = pad_batch_tensorize_3d(_inputs, pad=0, cuda=False)
        else:
            _inputs = []


        return (_inputs, rouges, system, abstract)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=coll(tokenizer)
    )
    return loader



def eval(args):
    split = 'test'
    model = Bert_choice()
    tokenizer = model._tokenizer
    ckpt = load_best_ckpt(args.model_dir)
    model.load_state_dict(ckpt)

    model = model.to(args.device)
    model.eval()

    dataloader = data_loader(split, tokenizer, args)
    epoch_iterator = tqdm(dataloader, desc="Iteration")

    score_dict = {
        'accuracy': [],
        'mrr': [],
        'prob': [],
        'rlr': [],
        'rlf': [],
        'r1': [],
        'r2': []
    }
    total_scores = {}
    for step, batch in enumerate(epoch_iterator):
        _inputs, rouges, system, human = batch
        if len(_inputs) == 0:
            continue
        _inputs = _inputs.to(args.device)
        bs, cn, length = _inputs.size()
        labels = torch.tensor([0 for _ in range(bs)]).to(args.device)
        with torch.no_grad():
            scores, _ids = model.evaluation(_inputs, labels)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            ps = scores[:, 0].tolist()
            trues = [1 if _id == 0 else 0 for _id in _ids]
            ranks = scores.argsort(dim=-1, descending=True).tolist()
            mrr_scores = []
            for _i in range(bs):
                mrr_scores.append(1 / (ranks[_i].index(0)+1))
        total_scores[step] = {
            'mrr': mrr_scores,
            'prob': ps,
            'accuracy': trues,
            'system': system,
            'human': human
        }
        accuracy = sum(trues) / len(trues)
        mrr = sum(mrr_scores) / len(mrr_scores)
        prob = sum(ps) / len(ps)
        rlr = rouges['RLr']
        rlf = rouges['RLf']
        r1 = rouges['R1']
        r2 = rouges['R2']
        score_dict['r1'].append(r1)
        score_dict['r2'].append(r2)
        score_dict['rlf'].append(rlf)
        score_dict['rlr'].append(rlr)
        score_dict['prob'].append(prob)
        score_dict['mrr'].append(mrr)
        score_dict['accuracy'].append(accuracy)
    with open(join(args.system_path, 'entity-cloze-score.json'), 'w') as f:
        json.dump(total_scores, f)
    with open(join(args.system_path, 'entity-scores.json'), 'w') as f:
        json.dump(score_dict, f)
    matrix = [[] for _ in range(len(score_dict.items()))]
    i = 0
    for _name, score1 in score_dict.items():
        for _name2, score2 in score_dict.items():
            pearsonr = stats.pearsonr(np.array(score1), np.array(score2))[0]
            matrix[i].append(pearsonr)
            print("pearson correlation {} {}: {}".format(_name, _name2, pearsonr))
        i += 1
        print('{}: {}'.format(_name, sum(score1) / len(score1)))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset'))
    parser.add_argument('--system_path', required=True, help='path of system output')
    parser.add_argument('--data_path', required=True, help='path to data')
    parser.add_argument('--model_dir', required=True, help='path to model')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.device = 'cuda'
    torch.cuda.set_device(args.gpu_id)
    eval(args)
