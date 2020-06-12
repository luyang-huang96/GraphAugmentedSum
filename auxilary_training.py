from coherence_interface.coherence_inference import coherence_infer, batch_global_infer
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
def multi_val_step(loss_step, fw_args, loss_args):
    losses = loss_step(fw_args, loss_args)
    n_data = 1
    losses = [loss.item() for loss in losses]
    return n_data, losses[0],losses[1],losses[2], losses[3]


@curry
def multi_basic_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = multi_val_step(compute_loss(net, criterion))
        n, tot_loss, tot_p, tot_r, tot_f = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4]),
            starmap(validate_fn, val_batches),
            (0, 0, 0, 0, 0)
        )
    val_loss = tot_loss / n
    val_p = tot_p / n
    val_r = tot_r / n
    val_f = tot_f / n
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation precision: {:.4f} ... '.format(val_loss))
    print('validation p: {:.4f} ... '.format(val_p))
    print('validation r: {:.4f} ... '.format(val_r))
    print('validation f: {:.4f} ... '.format(val_f))
    return {'loss': val_loss}