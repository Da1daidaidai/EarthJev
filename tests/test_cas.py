"""CAS and its calibration primitives: confidence features and their fallbacks, the presentation tie rule, the presence
probability, the presence veto, decide() on hand-made inputs and on logged questions of the method (the decisions of the
original implementation), Platt fitting and AUROC."""
import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from earthjev import cas, data, scoring, spv
from earthjev.calibration import auroc, logit, platt_fit, sigmoid
from earthjev.paths import load_config

CFG = load_config()
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
LP = [math.log(p) for p in (0.1, 0.3, 0.05, 0.05, 0.5)]  # first-position letter probabilities A-E
OPTIONS = [dict(id=x, text=t) for x, t in zip('ABCDE', ('Red', 'Blue', 'Green', 'White',
                                                       "This image doesn't feature the color."))]
MCQ = dict(qid='q', bench='mme_rs', is_mcq=True, multi_choice=False, options=OPTIONS)
OPEN = dict(qid='q', bench='lrs_vqa', is_mcq=False)


def test_confidence_feature():
    tok = [math.log(0.5), math.log(0.9)]
    p, z = cas.confidence_feature(MCQ, 'B', LP, tok)
    assert abs(p - 0.3) < 1e-12 and abs(z - math.log(0.3 / 0.7)) < 1e-9
    for row, letter in ((dict(MCQ, multi_choice=True), 'B'), (MCQ, ''), (MCQ, 'AB'), (MCQ, None)):
        assert abs(cas.confidence_feature(row, letter, LP, tok)[0] - 0.45) < 1e-12  # the whole output's probability
    p, z = cas.confidence_feature(OPEN, None, None, tok)
    assert abs(z - (math.log(0.5) + math.log(0.9)) / 2) < 1e-12 and abs(p - math.exp(z)) < 1e-12


def test_select_presentation_and_presence():
    assert cas.select_presentation({'canvas': 0.4, 'memory': 0.4}, True) == 'memory'
    assert cas.select_presentation({'canvas': 0.4, 'memory': 0.4}, False) == 'canvas'
    assert cas.select_presentation({'canvas': 0.5, 'memory': 0.4}, True) == 'canvas'
    a, b = CFG['cas']['presence_platt']
    assert cas.presence_probability([-1.0, 0.2], (a, b)) == pytest.approx(1 / (1 + math.exp(-(a * logit(0.6) + b))))
    with pytest.raises(ValueError):
        cas.presence_probability([], (a, b))  # multiple choice needs at least one view


def test_veto():
    assert cas.veto(MCQ, 'E', LP, 0.6) == 'B'  # evidence in view: the best other option
    assert cas.veto(MCQ, 'E', LP, 0.5) == 'B' and cas.veto(MCQ, 'E', LP, 0.4) is None
    assert cas.veto(MCQ, 'B', LP, 0.9) is None and cas.veto(MCQ, '', LP, 0.9) is None
    row = dict(MCQ, options=OPTIONS[:4] + [dict(id='E', text='The image does not feature the number.')])
    assert cas.veto(row, 'E', LP, 0.9) == 'B'


def _cand(canvas, memory):
    tok = [math.log(0.9), math.log(0.99)]
    return {r: dict(output=x, letter=x, letter_lp=LP if r == 'memory' else LP[::-1], tok_lp=tok)
            for r, x in (('canvas', canvas), ('memory', memory))}


def test_decide():
    maps = CFG['cas']['answer_platt']['mme_rs']
    cand = _cand('A', 'B')  # canvas: p(A) = 0.5, memory: p(B) = 0.3
    P = {r: cas.platt(maps[r], logit(p)) for r, p in (('canvas', 0.5), ('memory', 0.3))}
    d = cas.decide(MCQ, cand, CFG, presence=0.9)
    assert d['render'] == max(P, key=P.get) and d['P'] == pytest.approx(P[d['render']]) and d['veto'] is None
    assert d['out'] == d['letter'] == cand[d['render']]['output']
    assert d['raw'] == pytest.approx(0.5 if d['render'] == 'canvas' else 0.3)


def test_decide_veto():
    cand = _cand('A', 'E')  # P(memory: E) > P(canvas: A); memory answers the absence option E
    d = cas.decide(MCQ, cand, CFG, presence=0.7)
    assert d['render'] == 'memory' and d['veto'] == d['out'] == d['letter'] == 'B'  # the best other option
    assert d['P'] == pytest.approx(cas.platt(CFG['cas']['answer_platt']['mme_rs']['memory'], logit(0.3)))
    assert d['raw'] == pytest.approx(0.3)
    for presence in (0.3, None):  # no evidence in a view, or no presence given: the answer stays
        d = cas.decide(MCQ, cand, CFG, presence=presence)
        assert d['render'] == 'memory' and d['out'] == d['letter'] == 'E' and d['veto'] is None


def test_decisions_golden():
    """Logged questions of the method on the five benchmarks (tests/fixtures/cas_decisions.json.gz, recorded with the
    original decision code): vetoes and absence answers kept, the nearest presentation ties and synthetic exact ties,
    fast paths, pointer trusts near tau_fast, two-image rows and random questions. From the logged routes, answers,
    views and stated pointers, the pointer trust picks the route and decide() the presentation and final answer: all
    as the original (P_ptr, presence and P within 1e-12)."""
    cases = json.loads(gzip.decompress((FIXTURES / 'cas_decisions.json.gz').read_bytes()))['cases']
    rows, sp = {}, CFG['spv']
    assert len(cases) > 250 and sum(c['golden']['veto'] is not None for c in cases) > 30
    for c in cases:
        row = rows.setdefault(c['bench'], {r['qid']: r for r in data.load_rows(c['bench'])})[c['qid']]
        g, route, trust = c['golden'], c['own'], None
        if c['pointers']:
            trust, _ = spv.pointer_trust(*zip(*c['pointers']), sp['trust_platt'])
            route = spv.route(trust, sp['tau_fast'], sp['tau_keep'])
        assert route == g['route'] and (trust is None) == (g['trust'] is None), c['qid']
        assert trust is None or abs(trust - g['trust']) < 1e-12, c['qid']
        r = c['routes'][route]
        presence = cas.presence_probability(r['views'], CFG['cas']['presence_platt']) if row['is_mcq'] else None
        assert presence is None or abs(presence - g['presence']) < 1e-12, c['qid']
        cfg = CFG
        if c['answer_platt']:  # an exact tie: both presentations give the same answer under the same map
            cfg = copy.deepcopy(CFG)
            cfg['cas']['answer_platt'][c['bench']] = c['answer_platt']
        cand = {k: dict(x, letter=scoring.answer_letter(row, x['output'])) for k, x in r['cand'].items()}
        d = cas.decide(row, cand, cfg, presence)
        assert (d['render'], d['out'], d['veto']) == (g['render'], g['out'], g['veto']), c['qid']
        assert abs(d['P'] - g['P']) < 1e-12 and scoring.score(row, d['out']) == g['ok'], c['qid']


def test_platt_fit():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 2, 20000)
    y = (rng.random(len(s)) < sigmoid(0.7 * s - 0.4)).astype(float)
    a, b = platt_fit(s, y)
    assert abs(a - 0.7) < 0.05 and abs(b + 0.4) < 0.05
    a, b = platt_fit([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])  # separable: the ridge keeps it finite and ordered
    assert math.isfinite(a) and a > 0 and cas.platt((a, b), 0.0) < 0.5 < cas.platt((a, b), 3.0)


def test_auroc():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 300).astype(float)  # many ties
    lab = rng.random(300) < 0.4
    pairs = [(a > b) + 0.5 * (a == b) for a in s[lab] for b in s[~lab]]
    assert auroc(lab, s) == pytest.approx(np.mean(pairs), abs=1e-12) and auroc([1, 1], [0.2, 0.3]) is None
