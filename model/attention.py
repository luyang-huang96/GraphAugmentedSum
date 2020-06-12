""" attention functions """
from torch.nn import functional as F
import torch
from model.util import len_mask

def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))



def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    if score.dtype == torch.float16:
        mask = (1 - mask.float()) * (-50000)
        score = score + mask
    else:
        score = score.masked_fill(mask == 0, -1e15)
    norm_score = F.softmax(score, dim=-1)
    return norm_score


def prob_normalize_sigmoid(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e15)
    norm_score = F.sigmoid(score)
    return norm_score

def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def step_attention(query, key, value, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)

def badanau_attention(query, key, value, v, bias=None, mem_mask=None, side=None, sigmoid=False):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    if len(query.size()) == 2:
        score = query.unsqueeze(-2) + key
        if bias is not None:
            score += bias
        if side is not None:
            score += side.unsqueeze(1)
        score = torch.matmul(F.tanh(score), v.unsqueeze(0).unsqueeze(2)).permute(0, 2, 1).contiguous()
        if sigmoid:
            if mem_mask is None:
                norm_score = F.sigmoid(score, dim=-1)
            else:
                norm_score = prob_normalize_sigmoid(score, mem_mask)
        else:
            if mem_mask is None:
                norm_score = F.softmax(score, dim=-1)
            else:
                norm_score = prob_normalize(score, mem_mask)
        output = attention_aggregate(value, norm_score)
    elif len(query.size()) == 3:
        # for batch decoding
        score = query.unsqueeze(-2) + key.unsqueeze(0)
        if bias is not None:
            score += bias
        if side is not None:
            if len(side.size()) == len(query.size()):
                score += side.unsqueeze(-2)
            else:
                score += side.unsqueeze(0).unsqueeze(-2)
        score = torch.matmul(F.tanh(score), v.unsqueeze(0).unsqueeze(2)).permute(0, 1, 3, 2).contiguous()
        if sigmoid:
            if mem_mask is None:
                norm_score = F.sigmoid(score, dim=-1)
            else:
                norm_score = prob_normalize_sigmoid(score, mem_mask.unsqueeze(0).expand_as(score))
        else:
            if mem_mask is None:
                norm_score = F.softmax(score, dim=-1)
            else:
                norm_score = prob_normalize(score, mem_mask.unsqueeze(0).expand_as(score))
        output = attention_aggregate(value, norm_score)


    return output.squeeze(-2), norm_score.squeeze(-2)

def badanau_attention_score(query, key, v, bias=None):
    if len(query.size()) == 3:
        key = key.unsqueeze(0)
    sum_ = query.unsqueeze(-2) + key
    if bias is not None:
        sum_ += bias
    score = torch.matmul(F.tanh(sum_), v.unsqueeze(0).unsqueeze(2))
    if len(query.size()) == 2:
        score = score.permute(0, 2, 1).contiguous()
    else:
        score = score.permute(0, 1, 3, 2).contiguous()
    return score

def hierarchical_attention(query, key, value, v, hierarchical_key, hierarchical_v, hierarchical_align, hierarchical_bias=None, bias=None, mem_mask=None, hierarchical_length=None,
                           max_para_num=0):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    if len(query.size()) == 2:
        score_node = badanau_attention_score(query, key, v, bias)
        score_para = badanau_attention_score(query, hierarchical_key, hierarchical_v, hierarchical_bias)
        hierarchical_mask = len_mask(hierarchical_length, score_para.device).unsqueeze(-2)
        norm_score_para = prob_normalize(score_para, hierarchical_mask)
        norm_score_node = prob_normalize(score_node, mem_mask)
        nq = score_para.size(1)
        hierarchical_align = hierarchical_align.unsqueeze(1).repeat(1, nq, 1)
        score_para_node = norm_score_para.gather(2, hierarchical_align)
        norm_score_node = torch.mul(norm_score_node, score_para_node)
        norm_score = norm_score_node / norm_score_node.sum(dim=-1).unsqueeze(-1)
        output = attention_aggregate(value, norm_score)
    elif len(query.size()) == 3:
        # for batch decoding
        score_node = badanau_attention_score(query, key, v, bias)
        score_para = badanau_attention_score(query, hierarchical_key, hierarchical_v, hierarchical_bias)
        hierarchical_mask = len_mask(hierarchical_length, score_para.device, max_num=max_para_num).unsqueeze(-2).unsqueeze(0).expand_as(score_para)
        norm_score_para = prob_normalize(score_para, hierarchical_mask)
        norm_score_node = prob_normalize(score_node, mem_mask.unsqueeze(0).expand_as(score_node))
        nq = score_para.size(2)
        beam = score_para.size(0)
        hierarchical_align = hierarchical_align.unsqueeze(1).unsqueeze(0).repeat(beam, 1, nq, 1)
        score_para_node = norm_score_para.gather(-1, hierarchical_align)

        norm_score_node = torch.mul(norm_score_node, score_para_node)
        norm_score = norm_score_node / norm_score_node.sum(dim=-1).unsqueeze(-1)
        output = attention_aggregate(value, norm_score)


    return output.squeeze(-2), norm_score.squeeze(-2)

def copy_from_node_attention(query, key1, key2, value, v, bias=None, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    if len(query.size()) == 2:
        score = query.unsqueeze(-2) + key1 + key2
        if bias is not None:
            score += bias
        score = torch.matmul(F.tanh(score), v.unsqueeze(0).unsqueeze(2)).permute(0, 2, 1).contiguous()
        if mem_mask is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            norm_score = prob_normalize(score, mem_mask)
        output = attention_aggregate(value, norm_score)
    elif len(query.size()) == 3:
        # for batch decoding
        score = query.unsqueeze(-2) + key1.unsqueeze(0) + key2.unsqueeze(0)
        if bias is not None:
            score += bias
        score = torch.matmul(F.tanh(score), v.unsqueeze(0).unsqueeze(2)).permute(0, 1, 3, 2).contiguous()
        if mem_mask is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            norm_score = prob_normalize(score, mem_mask.unsqueeze(0).expand_as(score))
        output = attention_aggregate(value, norm_score)
    else:
        raise Exception

        # score = query.unsqueeze(-2) + key.unsqueeze(0)
        # if bias is not None:
        #     score += bias
        # if side is not None:
        #     if len(side.size()) == len(query.size()):
        #         score += side.unsqueeze(-2)
        #     else:
        #         score += side.unsqueeze(0).unsqueeze(-2)
        # score = torch.matmul(F.tanh(score), v.unsqueeze(0).unsqueeze(2)).permute(0, 1, 3, 2).contiguous()
        # if sigmoid:
        #     if mem_mask is None:
        #         norm_score = F.sigmoid(score, dim=-1)
        #     else:
        #         norm_score = prob_normalize_sigmoid(score, mem_mask.unsqueeze(0).expand_as(score))
        # else:
        #     if mem_mask is None:
        #         norm_score = F.softmax(score, dim=-1)
        #     else:
        #         norm_score = prob_normalize(score, mem_mask.unsqueeze(0).expand_as(score))
        # output = attention_aggregate(value, norm_score)


    return output.squeeze(-2), norm_score.squeeze(-2)
