""" ROUGE utils"""
import os
import threading
import subprocess as sp
from collections import Counter, deque

from cytoolz import concat, curry
from nltk.stem import porter
import re


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count

@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    reference = tokenize(' '.join(reference))
    output = tokenize(' '.join(output))
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score

@curry
def compute_rouge_n_tuple(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference, comnination p(n n)"""
    assert mode in list('fpr')  # F-1, precision, recall

    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score

def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]

@curry
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


@curry
def compute_rouge_l_sent_level(outputs, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    # reference list of sents sents are list of words
    # output list of sents sents are list of words
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = 0
    word_count = 0
    lcs_r = 0
    sum_count = 0
    for output in outputs:
        lcs += _lcs_len(output, list(concat(reference)))
        word_count += len(output)
    for ref in reference:
        lcs_r += _lcs_len(ref, list(concat(output)))
        sum_count += len(ref)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / word_count
        recall = lcs_r / sum_count
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score



def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""

    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    summs = [tokenize(' '.join(summ)) for summ in summs]
    refs = [tokenize(' '.join(ref)) for ref in refs]
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None
class Meteor(object):
    def __init__(self):
        assert _METEOR_PATH is not None
        cmd = 'java -Xmx2G -jar {} - - -l en -norm -stdio'.format(_METEOR_PATH)
        self._meteor_proc = sp.Popen(
            cmd.split(),
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
            universal_newlines=True, encoding='utf-8', bufsize=1
        )
        self._lock = threading.Lock()

    def __call__(self, summ, ref):
        self._lock.acquire()
        score_line = 'SCORE ||| {} ||| {}\n'.format(
            ' '.join(ref), ' '.join(summ))
        self._meteor_proc.stdin.write(score_line)
        stats = self._meteor_proc.stdout.readline().strip()
        eval_line = 'EVAL ||| {}\n'.format(stats)
        self._meteor_proc.stdin.write(eval_line)
        score = float(self._meteor_proc.stdout.readline().strip())
        self._lock.release()
        return score

    def __del__(self):
        self._lock.acquire()
        self._meteor_proc.stdin.close()
        self._meteor_proc.kill()
        self._meteor_proc.wait()
        self._lock.release()

def tokenize(text, stemmer=porter.PorterStemmer()):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
      text: A text blob to tokenize.
      stemmer: An optional stemmer.

    Returns:
      A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", text)

    tokens = re.split(r"\s+", text)
    if stemmer:
      # Only stem words more than 3 characters long.
      tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]

    return tokens
