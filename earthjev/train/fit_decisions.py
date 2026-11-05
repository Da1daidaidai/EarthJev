"""Fit calibration maps and routing thresholds from saved predictions."""
import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from earthjev import cas, data, pointers, scoring, spv
from earthjev.calibration import logit, platt_fit
from earthjev.paths import load_config, weights_dir

ROUTES = {'s1': 'fast', 's2': 'search', 's2x': 'search_without_pointers'}
_RUN = 'spv'  # key of the pointer record in a logged search (last_search.spv)
GRID = [(fast, keep) for fast in (0.5, 0.6, 0.7, 0.8, 0.9) for keep in (0.0, 0.25, 0.5)]  # (tau_fast, tau_keep)
MAP_SETS = {'mme_rs': ('mme_rs',), 'xlrs_lite': ('xlrs_lite',), 'rshr': ('mme_rs', 'xlrs_lite'),
            'lrs_vqa': ('lrs_vqa',), 'lrs_gro': ('gro_train',)}


# ------------------------------------------------------------------ the calibration runs

def _rows(paths):
    out = []
    for path in sorted(paths):
        with gzip.open(path, 'rt') as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


class Unit:
    """One logged calibration question: its question row, the logged route, the stated pointers (verifier p, Yes/No
    p_yn) and per route the two candidate answers {presentation: dict(output, letter, letter_lp, tok_lp, ok)}."""

    def __init__(self, search, answers, row):
        self.row = row
        ls = search['last_search']
        info = ls.get(_RUN) if isinstance(ls, dict) else None  # two-image rows log one search per image
        self.own = 'fast' if info and info['path'] == 'S1' else 'search'
        self.pointers = answers['stem_props']
        self.cand = {}
        for key, c in answers['conds'].items():
            if c['source'].startswith('same:'):  # the answers of another route
                c = answers['conds'][c['source'][5:]]
            self.cand[ROUTES[key]] = {r: self._candidate(c, r) for r in cas.PRESENTATIONS}

    def _candidate(self, c, r):
        text, sc = c['outputs'][r], (c.get('scores') or {}).get(r) or {}
        return dict(output=text, letter=scoring.answer_letter(self.row, text), letter_lp=sc.get('letter_lp'),
                    tok_lp=sc.get('tok_lp'), ok=bool(scoring.score(self.row, text)))


def load_units(bench, runs):
    """{qid: Unit} of one calibration set (runs/<bench>/search_*.jsonl.gz and answers_*.jsonl.gz), in the order of its
    search rows. Multiple-choice questions take their options from the question rows (earthjev.data)."""
    d = Path(runs) / bench
    answers = {r['qid']: r for r in _rows(d.glob('answers_*.jsonl.gz'))}
    options, units = None, {}
    for s in _rows(d.glob('search_*.jsonl.gz')):
        a = answers.get(s['qid'])
        if s.get('error') or a is None or a.get('error'):
            raise ValueError(f"{d}: question {s['qid']} has no error-free search and answers rows")
        row = {k: s.get(k) for k in ('qid', 'bench', 'category', 'source', 'ground_truth', 'is_mcq')}
        if row['is_mcq']:
            options = options or {r['qid']: r for r in data.load_rows(bench)}
            row = dict(options[row['qid']], **row)
        units[s['qid']] = Unit(s, a, row)
    if not units:
        raise FileNotFoundError(f'no logged questions in {d}')
    return units


def presence_probe(cfg, runs):
    """(labels, p) of the LRS-VQA dev probe crops (runs/presence_probe_dev.jsonl.gz holds the cached region-head logits
    z_v): p = the verifier's region probability with the question's directional phrase, label 1 for a crop on the
    question's evidence box."""
    rows = _rows([Path(runs) / 'presence_probe_dev.jsonl.gz'])
    questions = {r['qid']: r for r in data.load_rows('lrs_vqa')}
    verifier = spv.PointerVerifier.load(weights_dir(cfg, 'verifier'), cfg['spv']['temperature'])
    z = np.array([r['z'] for r in rows], np.float32)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r['qid']].append(i)
    p = np.zeros(len(rows))
    for qid, ii in by.items():
        q = questions[qid]
        phrases = [r for r in pointers.parse(q['stem']) if r.kind == 'phrase']
        p[ii] = verifier.score(z[ii], [rows[i]['bbox'] for i in ii], q['image_size'], phrases)[0]
    return np.array([r['label'] for r in rows], float), p


# ------------------------------------------------------------------ the fits

def trust_signal(u):
    """max_j s_j of a question's stated pointers (the argument of the pointer-trust map)."""
    return max(spv.pointer_trust([x['p'] for x in u.pointers], [x['p_yn'] for x in u.pointers], (1.0, 0.0))[1])


def fit_trust(correct, wrong):
    """Platt map of logit(max_j s_j) for 'the stated pointer is correct' over the questions with stated pointers."""
    refs = [(u, 1) for u in correct if u.pointers] + [(u, 0) for u in wrong if u.pointers]
    return platt_fit(logit([trust_signal(u) for u, _ in refs]), [y for _, y in refs])


def route_accuracy(units, sp):
    """Accuracy (%) of the fixed presentation (memory for multiple choice, canvas otherwise; no veto) on the route that
    the pointer trust selects under the parameters sp (the logged route for questions without stated pointers)."""
    ok = []
    for u in units:
        route = u.own
        if u.pointers:
            trust, _ = spv.pointer_trust([x['p'] for x in u.pointers], [x['p_yn'] for x in u.pointers],
                                         sp['trust_platt'])
            route = spv.route(trust, sp['tau_fast'], sp['tau_keep'])
        ok.append(u.cand[route]['memory' if u.row['is_mcq'] else 'canvas']['ok'])
    return 100 * statistics.fmean(ok)


def fit_thresholds(cfg, trust_platt, xlrs, correct, wrong):
    """(tau_fast, tau_keep) maximizing J = mean accuracy over the three {qid: Unit} sets, and {grid point: J}."""
    J = {}
    for fast, keep in GRID:
        sp = dict(cfg['spv'], trust_platt=list(trust_platt), tau_fast=fast, tau_keep=keep)
        x, o, w = (route_accuracy(us.values(), sp) for us in (xlrs, correct, wrong))
        J[fast, keep] = (x + o + w) / 3
    return max(GRID, key=lambda g: (round(J[g], 9), -g[1], g[0])), J


def fit_answer_maps(units):
    """{presentation: (a, b)}: Platt maps of the confidence feature for the correctness of the answers of the logged
    route."""
    feats = {r: [] for r in cas.PRESENTATIONS}
    for u in units:
        for r, c in u.cand[u.own].items():
            feats[r].append((cas.confidence_feature(u.row, c['letter'], c['letter_lp'], c['tok_lp'])[1], c['ok']))
    return {r: platt_fit(*zip(*feats[r])) for r in cas.PRESENTATIONS}


def refit(cfg, runs):
    """The refitted parameters {'spv': {...}, 'cas': {...}} and the J grid, from the calibration runs in the
    directory runs (data/calibration/<config>)."""
    u = {b: load_units(b, runs) for b in ('mme_rs', 'lrs_vqa', 'xlrs_lite', 'gro_train', 'lrsrefdev_ok',
                                          'lrsrefdev_bad', 'grorefdev_ok', 'grorefdev_bad')}
    correct = {**u['lrsrefdev_ok'], **u['grorefdev_ok']}  # {qid: Unit}; the two sets share no question
    wrong = {**u['lrsrefdev_bad'], **u['grorefdev_bad']}
    trust = fit_trust(correct.values(), wrong.values())
    (fast, keep), J = fit_thresholds(cfg, trust, u['xlrs_lite'], correct, wrong)
    maps = {b: fit_answer_maps([x for s in sets for x in u[s].values()]) for b, sets in MAP_SETS.items()}
    y, p = presence_probe(cfg, runs)
    presence = platt_fit(logit(p), y)
    fit = dict(spv=dict(trust_platt=list(trust), tau_fast=fast, tau_keep=keep),
               cas=dict(answer_platt={b: {r: list(m[r]) for r in cas.PRESENTATIONS} for b, m in maps.items()},
                        presence_platt=list(presence)))
    return fit, J


REL_TOL = 1e-12  # NumPy's exp / log kernels differ in the last bits between CPUs (e.g. with and without AVX-512)


def _close(v, c, rel):
    if isinstance(v, (list, tuple)) and isinstance(c, (list, tuple)):
        return len(v) == len(c) and all(_close(a, b, rel) for a, b in zip(v, c))
    if isinstance(v, float) and isinstance(c, float):
        return abs(v - c) <= rel * max(1.0, abs(c))
    return v == c


def compare(fit, cfg, prefix='', rel=REL_TOL):
    """[(key, refitted, configured)] of every refitted leaf that differs from the configuration by more than a relative
    rel (rel=0: bitwise)."""
    out = []
    for k, v in fit.items():
        if isinstance(v, dict):
            out += compare(v, cfg.get(k, {}), f'{prefix}{k}.', rel)
        elif not _close(v, cfg.get(k), rel):
            out.append((prefix + k, v, cfg.get(k)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='qwen35-9b', help='configuration name under configs/ or a JSON path')
    ap.add_argument('--runs', help='directory of the logged calibration runs (default data/calibration/<config>)')
    ap.add_argument('--out', help='write the refitted parameters to this JSON file')
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    fit, J = refit(cfg, a.runs or data.DATA / 'calibration' / Path(a.config).stem)
    print('J (mean accuracy, %) per (tau_fast, tau_keep):', ', '.join(f'{g}: {j:.4f}' for g, j in J.items()))
    print(json.dumps(fit, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(fit, indent=1) + '\n')
    diff = compare(fit, cfg)
    for k, v, c in diff:
        print(f'DIFFERS {k}: refit {v} vs configured {c}')
    if diff:
        print(f'{len(diff)} values differ from the configuration')
    elif compare(fit, cfg, rel=0):
        print(f'refit equals the configuration up to floating-point rounding (relative {REL_TOL:g}; '
              f'{len(compare(fit, cfg, rel=0))} values differ in the last bits on this CPU)')
    else:
        print('refit equals the configuration bitwise')
    return 1 if diff else 0


if __name__ == '__main__':
    sys.exit(main())
