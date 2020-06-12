from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os.path import join
import re, os, json
import torch.nn as nn
import torch
from pytorch_transformers import BertForMultipleChoice
from pytorch_transformers import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch import optim
import argparse
from cytoolz import curry, concat
from torch.nn.utils import clip_grad_norm_
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import time
from datetime import timedelta

MAX_LEN = 300

class MultipleChoiceDataset(Dataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js_data = json.loads(f.read())
        question, context = js_data['question'], js_data['context']
        answer, choice1, choice2, choice3 = js_data['answer'], js_data['choice1'], js_data['choice2'], js_data['choice3']
        return question, context, [answer, choice1, choice2, choice3]

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def batcher(path, bs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    blank = '[unused0]'

    @curry
    def coll(tokenizer, batch):
        questions, contexts, choicess = list(filter(bool, list(zip(*batch))))
        questions = [question.replace('<\\blank>', blank) for question in questions]
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
        contexts = [['[SEP]'] + tokenizer.tokenize(' '.join(context).lower()) for context in contexts]

        choicess = [[[tokenizer.tokenize(c.lower()) for c in choice] for choice in choices] for choices in
                    choicess]
        choicess = [[['[SEP]'] + choice[0] + ['[SEP]'] + choice[1] if len(choice) == 2 else ['[SEP]'] + choice[0] for choice in choices] for choices in choicess]

        _inputs = [
            [tokenizer.convert_tokens_to_ids((question + context + choice)[:MAX_LEN]) for choice in choices] for
            question, context, choices in zip(questions, contexts, choicess)]
        _inputs = pad_batch_tensorize_3d(_inputs, pad=0, cuda=False)

        return (_inputs)

    train_loader = DataLoader(
        MultipleChoiceDataset('train', path), batch_size=bs, shuffle=True, num_workers=4,
        collate_fn=coll(tokenizer)
    )
    test_loader = DataLoader(MultipleChoiceDataset('val', path), batch_size=bs, shuffle=False, num_workers=4,
                             collate_fn=coll(tokenizer))

    return train_loader, test_loader


def pad_batch_tensorize_3d(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max([len(x) for _input in inputs for x in _input])
    if len(inputs) > 1:
        assert len(inputs[0]) == len(inputs[1])
    tensor_shape = (batch_size, len(inputs[0]), max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        for j, _input in enumerate(ids):
            tensor[i, j, :len(_input)] = tensor_type(_input)
    return tensor

class Bert_choice(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        self._blank = '[unused0]'
        self._question = '[SEP]'
        self._context = '[SEP]'
        self._choice = '[SEP]'
        self._choice_split = '[SEP]'
        self._device = device

    def forward(self, _inputs, labels):
        #questions, contexts, choicess = self.prepare(questions, contexts, choicess)
        outputs = self._model(_inputs, labels=labels)
        loss, classification_scores = outputs[:2]
        _, _ids = torch.max(classification_scores, 1)
        #ccr = sum([1 if _id == 0 else 0 for _id in _ids]) / len(_ids)


        return loss.mean(), _ids

    def evaluation(self, _inputs, labels):
        outputs = self._model(_inputs, labels=labels)
        loss, classification_scores = outputs[:2]
        _, _ids = torch.max(classification_scores, 1)

        return classification_scores, _ids


    def prepare(self, questions, contexts, choicess):
        questions = [question.replace('<\\blank>', self._blank) for question in questions]
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
        contexts = [[self._tokenizer.tokenize(sent.lower()) for sent in context] for context in contexts]
        new_contexts = []
        for context in contexts:
            new_c = [self._context]
            for c in context:
                new_c += c + ['[SEP]']
            new_c.pop()
            new_contexts.append(new_c)
        contexts = new_contexts

        choicess = [[[self._tokenizer.tokenize(c.lower()) for c in choice] for choice in choices] for choices in
                    choicess]
        new_choicess = []
        for choices in choicess:
            new_choices = []
            for choice in choices:
                if len(choice) == 1:
                    new_choice = [self._choice] + choice[0]
                else:
                    new_choice = [self._choice] + choice[0] + [self._choice_split] + choice[1]
                new_choices.append(new_choice)
            new_choicess.append(new_choices)
        choicess = new_choicess
        return questions, contexts, choicess

def evaluate(model, loader, args):
    print('start validation: ')
    model.eval()
    total_ccr = 0
    total_loss = 0
    step = 0
    start = time.time()
    for _i, batch in enumerate(loader):
        with torch.no_grad():
            #questions, contexts, choicess = batch
            _inputs = batch.to(args.device)
            bs, cn, length = _inputs.size()
            labels = torch.tensor([0 for _ in range(bs)]).to(args.device)
            loss, _ids = model(_inputs, labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            ccr = sum([1 if _id == 0 else 0 for _id in _ids]) / len(_ids)

            total_ccr += ccr
            total_loss += loss
            step += 1
    print('validation ccr: {:.4f} loss {:.4f}'.format(total_ccr / step, total_loss / step))
    print('validation finished in {} '.format(timedelta(seconds=int(time.time() - start))))
    model.train()
    return total_ccr / step, total_loss / step


def train(args):
    save_path = join(args.save_path, 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.train_batch_size = args.bs * max(1, args.n_gpu)
    train_loader, val_loader = batcher(args.path, args.train_batch_size)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    print(t_total / args.num_train_epochs)
    tb_writer = SummaryWriter(log_dir=join(args.save_path, 'tensorboard'))
    model = Bert_choice()
    if args.cuda:
        model = model.cuda()
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)



    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    global_ccr = 0
    global_step = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        tr_loss, logging_loss = 0, 0
        tr_ccr = 0
        for step, batch in enumerate(epoch_iterator):
            #questions, contexts, choicess = batch
            _inputs = batch.to(args.device)
            bs, cn, length = _inputs.size()
            labels = torch.tensor([0 for _ in range(bs)]).to(args.device)

            loss, _ids = model(_inputs, labels)
            ccr = sum([1 if _id == 0 else 0 for _id in _ids]) / len(_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            tr_loss += loss.item()
            tr_ccr += ccr / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 2)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                global_ccr = global_ccr * 0.01 + tr_ccr
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
                tb_writer.add_scalar('ccr', global_ccr, global_step)
                global_step += 1
                #print('loss: {:.4f} ccr {:.4f}\r'.format(tr_loss, ccr), end='')
                logging_loss = tr_loss
                tr_ccr = 0
                if global_step % args.ckpt == 0:
                    total_ccr, total_loss = evaluate(model, val_loader, args)
                    name = 'ckpt-{:4f}-{:4f}-{}'.format(total_loss, total_ccr, global_step)
                    save_dict = {}
                    save_dict['state_dict'] = model.state_dict()
                    save_dict['optimizer'] = optimizer.state_dict()
                    torch.save(save_dict, join(save_path, name))






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset'))
    parser.add_argument('--path', required=True, help='path of data')
    parser.add_argument('--save_path', required=True, help='path to store/eval')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--bs', type=int, action='store', default=5,
                        help='the training batch size')
    parser.add_argument('--num_train_epochs', type=int, action='store', default=10,
                        help='the training batch size')
    parser.add_argument('--ckpt', type=int, action='store', default=10000,
                        help='ckpt per global step')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    #torch.cuda.set_device(args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    print('use {} gpus'.format(args.n_gpu))
    args.device = 'cuda'
    train(args)

