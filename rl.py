""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ
from training import BasicPipeline
from nltk import sent_tokenize


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]
            all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch = next(loader)
    for raw_arts in art_batch:
        (inds, ms), bs = agent(raw_arts)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts)]
    with torch.no_grad():
        summaries = abstractor(ext_sents)
    i = 0
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch):
        rs = ([reward_fn(summaries[i+j], abss[j])
              for j in range(min(len(inds)-1, len(abss)))]

              + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
              + [stop_coeff*stop_reward_fn(
                  list(concat(summaries[i:i+len(inds)-1])),
                  list(concat(abss)))])
        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices))) # divide by T*B
    critic_loss = F.mse_loss(baseline, reward)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            try:
                grad_log['grad_norm'+n] = tot_grad.item()
            except AttributeError:
                grad_log['grad_norm' + n] = tot_grad
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        try:
            grad_norm = grad_norm.item()
        except AttributeError:
            grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

def compute_rouge_with_marginal_increase(reward_fn, targets, abss, step, gamma=0.95):
    reward = 0
    if reward_fn.__name__ != 'compute_rouge_l_summ':
        if step == len(targets): # this time step greedy generates "stop"
            reward += reward_fn(list(concat(targets[:])), list(concat(abss)))
        else:
            sent_reward = [reward_fn(list(concat(targets[:i+1+step])), list(concat(abss))) for i in range(len(targets) - step)]
            for ind in range(len(sent_reward)):
                if ind != 0:
                    reward += math.pow(gamma, ind) * (sent_reward[ind] - sent_reward[ind-1])
                else:
                    reward += sent_reward[ind]
    else:
        if step == len(targets): # this time step greedy generates "stop"
            reward += reward_fn(list(concat(targets[:])), abss)
        else:
            sent_reward = [reward_fn(targets[:i+1+step], abss) for i in range(len(targets) - step)]
            for ind in range(len(sent_reward)):
                if ind != 0:
                    reward += math.pow(gamma, ind) * (sent_reward[ind] - sent_reward[ind-1])
                else:
                    reward += sent_reward[ind]

    return reward




# def sc_train_step(agent, abstractor, loader, opt, grad_fn,
#                    reward_fn=compute_rouge_l, sample_time=1, entity=False):
#     gamma = 0.95
#     opt.zero_grad()
#     art_batch, abs_batch = next(loader)
#     all_loss = []
#     reward = 0
#     advantage = 0
#     i = 0
#     for idx, raw_arts in enumerate(art_batch):
#         greedy, samples, all_log_probs = agent(raw_arts, sample_time=sample_time)
#         if agent.time_variant:
#             bs = []
#             abss = abs_batch[idx]
#             for _ind, gd in enumerate(greedy):
#                 greedy_sents = [raw_arts[ind] for ind in gd]
#                 with torch.no_grad():
#                     greedy_sents = [[word for sent in greedy_sents for word in sent]]
#                     greedy_sents = abstractor(greedy_sents)
#                     greedy_sents = sent_tokenize(' '.join(greedy_sents[0]))
#                     greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
#                 if reward_fn.__name__ != 'compute_rouge_l_summ':
#                     if _ind != len(greedy)-1:
#                         baseline = compute_rouge_with_marginal_increase(reward_fn, greedy_sents, abss, _ind, gamma=gamma)
#                     else:
#                         baseline = reward_fn(list(concat(greedy_sents)), list(concat(abss)))
#                 else:
#                     if _ind != len(greedy)-1:
#                         baseline = compute_rouge_with_marginal_increase(reward_fn, greedy_sents, abss, _ind, gamma=gamma)
#                     else:
#                         baseline = reward_fn(greedy_sents, abss)
#                 bs.append(baseline)
#             sample_sents = [raw_arts[ind] for ind in samples[0]]
#
#             with torch.no_grad():
#                 sample_sents = [[word for sent in sample_sents for word in sent]]
#                 sample_sents = abstractor(sample_sents)
#                 sample_sents = sent_tokenize(' '.join(sample_sents[0]))
#                 sample_sents = [sent.strip().split(' ') for sent in sample_sents]
#
#             if reward_fn.__name__ != 'compute_rouge_l_summ':
#                 rewards = [reward_fn(list(concat(sample_sents[:i+1])), list(concat(abss))) for i in range(len(sample_sents))]
#                 all_rewards = []
#                 for index in range(len(rewards)):
#                     rwd = 0
#                     for _index in range(len(rewards)-index):
#                         if _index != 0:
#                             rwd += (rewards[_index+index] - rewards[_index+index-1]) * math.pow(gamma, _index)
#                         else:
#                             rwd += rewards[_index+index]
#                     all_rewards.append(rwd)
#                 all_rewards.append(
#                     compute_rouge_n(list(concat(sample_sents)), list(concat(abss)))
#                 )
#             else:
#                 rewards = [reward_fn(sample_sents[:i + 1], abss) for i in
#                            range(len(sample_sents))]
#                 all_rewards = []
#                 for index in range(len(rewards)):
#                     rwd = 0
#                     for _index in range(len(rewards) - index):
#                         if _index != 0:
#                             rwd += (rewards[_index + index] - rewards[_index + index - 1]) * math.pow(gamma, _index)
#                         else:
#                             rwd += rewards[_index + index]
#                     all_rewards.append(rwd)
#                 all_rewards.append(
#                     compute_rouge_n(list(concat(sample_sents)), list(concat(abss)))
#                 )
#             # print('greedy:', greedy)
#             # print('sample:', samples[0])
#             # print('baseline:', bs)
#             # print('rewars:', all_rewards)
#             reward += bs[-1]
#             advantage += (all_rewards[-1] - bs[-1])
#             i += 1
#             advs = [torch.tensor([_bs - rwd], dtype=torch.float).to(all_log_probs[0][0].device) for _bs, rwd in zip(bs, all_rewards)]
#             for log_prob, adv in zip(all_log_probs[0], advs):
#                 all_loss.append(log_prob * adv)
#         else:
#             if entity:
#                 raw_arts = raw_arts[0]
#             greedy_sents = [raw_arts[ind] for ind in greedy]
#             with torch.no_grad():
#                 greedy_sents = [[word for sent in greedy_sents for word in sent]]
#                 greedy_sents = abstractor(greedy_sents)
#                 greedy_sents = sent_tokenize(' '.join(greedy_sents[0]))
#                 greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
#             abss = abs_batch[idx]
#             if reward_fn.__name__ != 'compute_rouge_l_summ':
#                 bs = reward_fn(list(concat(greedy_sents)), list(concat(abss)))
#             else:
#                 bs = reward_fn(greedy_sents, abss)
#             for sample, log_probs in zip(samples, all_log_probs):
#                 sample_sents = [raw_arts[ind] for ind in sample]
#                 with torch.no_grad():
#                     sample_sents = [[word for sent in sample_sents for word in sent]]
#                     sample_sents = abstractor(sample_sents)
#                     sample_sents = sent_tokenize(' '.join(sample_sents[0]))
#                     sample_sents = [sent.strip().split(' ') for sent in sample_sents]
#                 if reward_fn.__name__ != 'compute_rouge_l_summ':
#                     rwd = reward_fn(list(concat(sample_sents)), list(concat(abss)))
#                 else:
#                     rwd = reward_fn(sample_sents, abss)
#                 reward += bs
#                 advantage += (rwd - bs)
#                 i += 1
#                 adv = torch.tensor([bs - rwd], dtype=torch.float).to(log_probs[0].device)
#                 for log_prob in log_probs:
#                     all_loss.append(log_prob * adv)
#     reward = reward / i
#     advantage = advantage / i
#
#     # backprop and update
#     loss = torch.cat(all_loss, dim=0).mean()
#     loss.backward()
#     grad_log = grad_fn()
#     opt.step()
#     log_dict = {}
#     log_dict.update(grad_log)
#     log_dict['reward'] = reward
#     log_dict['advantage'] = advantage
#     log_dict['mse'] = 0
#     assert not math.isnan(log_dict['grad_norm'])
#     return log_dict

def sc_train_step(agent, abstractor, loader, opt, grad_fn,
                   reward_fn=compute_rouge_l, sample_time=1, graph=False, bert=False):
    gamma = 0.95
    opt.zero_grad()
    art_batch, abs_batch = next(loader)
    all_loss = []
    reward = 0
    advantage = 0
    i = 0
    greedy_inputs = []
    sample_inputs = []
    sample_log_probs = []
    for idx, raw_arts in enumerate(art_batch):
        greedy, samples, all_log_probs = agent(raw_arts, sample_time=sample_time)

        if graph or bert:
            raw_arts = raw_arts[0]
        greedy_sents = [raw_arts[ind] for ind in greedy]
        greedy_sents = [word for sent in greedy_sents for word in sent]
        greedy_inputs.append(greedy_sents)
        sample_sents = [raw_arts[ind] for ind in samples[0]]
        sample_sents = [word for sent in sample_sents for word in sent]
        sample_inputs.append(sample_sents)
        sample_log_probs.append(all_log_probs[0])
    if not agent.time_variant:
        with torch.no_grad():
            greedy_outs = abstractor(greedy_inputs)
            sample_outs = abstractor(sample_inputs)
        for greedy_sents, sample_sents, log_probs, abss in zip(greedy_outs, sample_outs, sample_log_probs, abs_batch):
            greedy_sents = sent_tokenize(' '.join(greedy_sents))
            greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
            if reward_fn.__name__ != 'compute_rouge_l_summ':
                bs = reward_fn(list(concat(greedy_sents)), list(concat(abss)))
            else:
                bs = reward_fn(greedy_sents, abss)
            sample_sents = sent_tokenize(' '.join(sample_sents))
            sample_sents = [sent.strip().split(' ') for sent in sample_sents]
            if reward_fn.__name__ != 'compute_rouge_l_summ':
                rwd = reward_fn(list(concat(sample_sents)), list(concat(abss)))
            else:
                rwd = reward_fn(sample_sents, abss)
            reward += bs
            advantage += (rwd - bs)
            i += 1
            adv = torch.tensor([bs - rwd], dtype=torch.float).to(log_probs[0].device)
            for log_prob in log_probs:
                all_loss.append(log_prob * adv)
    reward = reward / i
    advantage = advantage / i

    # backprop and update
    loss = torch.cat(all_loss, dim=0).mean()
    loss.backward()
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = reward
    log_dict['advantage'] = advantage
    log_dict['mse'] = 0
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


# def sc_validate(agent, abstractor, loader, entity=False):
#     agent.eval()
#     start = time()
#     print('start running validation...', end='')
#     avg_reward = 0
#     i = 0
#     with torch.no_grad():
#         for art_batch, abs_batch in loader:
#             for idx, raw_arts in enumerate(art_batch):
#                 greedy, sample, log_probs = agent(raw_arts, sample_time=1, validate=True)
#                 if entity:
#                     raw_arts = raw_arts[0]
#                 sample = sample[0]
#                 log_probs = log_probs[0]
#                 greedy_sents = [raw_arts[ind] for ind in greedy]
#                 #greedy_sents = list(concat(greedy_sents))
#                 with torch.no_grad():
#                     greedy_sents = [[word for sent in greedy_sents for word in sent]]
#                     greedy_sents = abstractor(greedy_sents)
#                     greedy_sents = sent_tokenize(' '.join(greedy_sents[0]))
#                     greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
#                 sample_sents = [raw_arts[ind] for ind in sample]
#                 sample_sents = list(concat(sample_sents))
#                 abss = abs_batch[idx]
#
#                 bs = compute_rouge_n(list(concat(greedy_sents)), list(concat(abss)))
#                 avg_reward += bs
#                 i += 1
#     avg_reward /= (i/100)
#     print('finished in {}! avg reward: {:.2f}'.format(
#         timedelta(seconds=int(time()-start)), avg_reward))
#     return {'reward': avg_reward}

def sc_validate(agent, abstractor, loader, entity=False, bert=False):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            greedy_inputs = []
            for idx, raw_arts in enumerate(art_batch):
                greedy, sample, log_probs = agent(raw_arts, sample_time=1, validate=True)
                if entity or bert:
                    raw_arts = raw_arts[0]
                # sample = sample[0]
                # log_probs = log_probs[0]
                greedy_sents = [raw_arts[ind] for ind in greedy]
                #greedy_sents = list(concat(greedy_sents))
                greedy_sents = [word for sent in greedy_sents for word in sent]
                greedy_inputs.append(greedy_sents)
            with torch.no_grad():
                greedy_outputs = abstractor(greedy_inputs)
            greedy_abstracts = []
            for greedy_sents in greedy_outputs:
                greedy_sents = sent_tokenize(' '.join(greedy_sents))
                greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
                greedy_abstracts.append(greedy_sents)
            for idx, greedy_sents in enumerate(greedy_abstracts):
                abss = abs_batch[idx]
                bs = compute_rouge_n(list(concat(greedy_sents)), list(concat(abss)))
                avg_reward += bs
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


class SCPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, entity, bert):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._reward_fn = reward_fn

        self._n_epoch = 0  # epoch not very useful?
        self._entity = entity
        self._bert = bert


    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = sc_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._reward_fn,
            graph=self._entity,
            bert=self._bert
        )
        return log_dict

    def validate(self):
        return sc_validate(self._net, self._abstractor, self._val_batcher, self._entity, self._bert)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing