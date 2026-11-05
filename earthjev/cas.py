"""Select between memory and canvas answers using calibrated confidence."""
import math
import re
import statistics

import numpy as np

from earthjev.calibration import logit, sigmoid

PRESENTATIONS = ('canvas', 'memory')  # iteration order of every per-presentation computation
LETTERS = 'ABCDE'
NOT_FEATURED = re.compile(r"\bdo(?:es)?(?:n't| not) feature\b", re.I)  # the absence option (MME-RS option E)


def platt(ab, x):
    """P = sigmoid(a x + b) of a Platt map ab = (a, b)."""
    return float(sigmoid(ab[0] * np.asarray(x, float) + ab[1]))


def confidence_feature(row, letter, letter_lp, tok_lp):
    """(probability, feature z_r) of one answer. Multiple choice: the first-position probability of the extracted
    option letter, z_r = logit(p); multi-answer rows and answers without a single letter use the probability of the
    whole output (with <|im_end|>). Open-ended: z_r = the mean token log-prob of the output with <|im_end|>,
    p = exp(z_r). letter_lp: log-probs of the letters A-E at the first answer position; tok_lp: log-probs of the
    answer tokens."""
    if row['is_mcq']:
        single = bool(letter) and len(letter) == 1 and letter in LETTERS and not row.get('multi_choice')
        p = math.exp(letter_lp[LETTERS.index(letter)] if single else sum(tok_lp))
        return p, float(logit(p))
    m = statistics.fmean(tok_lp)
    return math.exp(m), m


def select_presentation(P, is_mcq):
    """The presentation with the higher P; exact ties go to memory for multiple choice, canvas otherwise."""
    if P['canvas'] == P['memory']:
        return 'memory' if is_mcq else 'canvas'
    return max(PRESENTATIONS, key=lambda r: P[r])


def presence_probability(view_scores, presence_platt):
    """Max calibrated presence over the evidence views (Eq. pres): P_pres = sigmoid(a logit(p) + b) of each view's
    region probability p = (score + 1) / 2 (view scores are confidences in [-1, 1])."""
    return max(platt(presence_platt, logit((s + 1) / 2)) for s in view_scores)


def veto(row, letter, letter_lp, presence, threshold=0.5):
    """Evidence presence veto: the replacement letter (the other option with the highest first-position log-prob)
    when the answer is an absence option and presence >= threshold; else None (keep the answer)."""
    texts = {o['id']: o['text'] for o in row['options']}
    if presence < threshold or not letter or not NOT_FEATURED.search(texts.get(letter, '')):
        return None
    other = [o['id'] for o in row['options'] if not NOT_FEATURED.search(o['text'])]
    return max(other, key=lambda x: letter_lp[LETTERS.index(x)])


def decide(row, cand, cfg, presence=None):
    """The final answer of one question from its two candidate answers (Eq. 3 and the presence veto).

    cand: {presentation: dict(output, letter, letter_lp, tok_lp)}; cfg: the method configuration; presence: the max
    calibrated presence over the views (presence_probability) for multiple choice, None to skip the veto. Returns
    dict(render, P, raw, out, letter, veto): the chosen presentation, the calibrated P of the final answer, its raw
    probability, the final answer text, its letter and the veto's replacement letter (None if not vetoed).
    """
    maps = cfg['cas']['answer_platt'][row['bench']]
    conf = {r: confidence_feature(row, cand[r]['letter'], cand[r]['letter_lp'], cand[r]['tok_lp'])
            for r in PRESENTATIONS}
    P = {r: platt(maps[r], conf[r][1]) for r in PRESENTATIONS}
    render = select_presentation(P, row['is_mcq'])
    c = cand[render]
    out = dict(render=render, P=P[render], raw=conf[render][0], out=c['output'], letter=c['letter'], veto=None)
    if presence is not None and row['is_mcq']:
        v = veto(row, c['letter'], c['letter_lp'], presence, cfg['cas']['veto_threshold'])
        if v:  # the final answer's confidence: the replacement letter's first-position probability, same map
            prob = float(np.exp(c['letter_lp'][LETTERS.index(v)]))
            out.update(out=v, letter=v, veto=v, raw=prob, P=platt(maps[render], logit(prob)))
    return out
