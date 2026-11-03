"""Compare live runs of EarthJev with the reference live records (reference/live/<config>/<bench>.jsonl.gz).

    python scripts/check_live.py RUN.jsonl [RUN.jsonl ...] [--config qwen35-9b]

The reference records (reference/live/<config>/) are, per configuration, 1,612 questions answered by the original
live pipeline on one NVIDIA H100 (bf16, flash-attention 2, the pinned GPU stack): shard 1/12 of MME-RS, 1/24 of
LRS-VQA, 1/10 of XLRS-lite, 1/32 of LRS-GRO and 1/3 of RSHR. Reproduce them with scripts/run_earthjev.py --bench <b>
--shard 1/<n> [--config <config>] (reference/README.md).

--config is the name of a shipped configuration (configs/<name>.json); a run whose records name another configuration
is refused. Several files are merged as in scripts/evaluate.py (earthjev.data.read_runs).

Printed per benchmark: the share of the run's questions whose route (every image), pointer trust P_ptr, views (boxes
and scores), answers of the two presentations, chosen presentation, final answer and correctness equal the reference,
and the largest |difference| of P_ptr and of the final answer's calibrated P. P is compared on the questions without
a veto only: after a veto the Qwen3.5-9B reference holds the vetoed answer's P, the pipeline the replacement letter's.
Expected: on the same GPU model and stack every field reproduces (greedy decoding of the same inputs). On another GPU
configuration the bf16 numerics differ, so scores and log-probs move and some views and answers change
(reference/README.md lists the agreement measured on a MIG partition of an H100).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earthjev import data, paths, scoring  # noqa: E402

FIELDS = ('route', 'trust', 'views', 'canvas', 'memory', 'render', 'final', 'correct')


def trust(rec):
    return [s['trust'] and s['trust']['P'] for s in rec['spv']]


def compare(rec, ref, row):
    """{field: equal} and the |differences| of P_ptr and P of one question."""
    if rec.get('error') is not None:
        return {f: False for f in FIELDS}, None, None
    eq = dict(route=[s['route'] for s in rec['spv']] == [s['route'] for s in ref['spv']],
              trust=trust(rec) == trust(ref), views=rec['views'] == ref['views'],
              canvas=rec['outputs']['canvas'] == ref['outputs']['canvas'],
              memory=rec['outputs']['memory'] == ref['outputs']['memory'], render=rec['render'] == ref['render'],
              final=rec['final'] == ref['final'],
              correct=scoring.score(row, rec['final']) == scoring.score(row, ref['final']))
    pairs = [(a, b) for a, b in zip(trust(rec), trust(ref)) if a is not None and b is not None]
    d_trust = max((abs(a - b) for a, b in pairs), default=None)
    vetoed = rec['decision']['veto'] is not None or ref['decision']['veto'] is not None
    return eq, d_trust, None if vetoed else abs(rec['P'] - ref['P'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--config', default='qwen35-9b', help='the configuration the runs used (its reference records)',
                    choices=sorted(p.stem for p in (paths.REPO / 'configs').glob('*.json')))
    a = ap.parse_args()
    live = paths.REPO / 'reference' / 'live' / a.config
    try:
        runs = data.read_runs(a.runs, paths.load_config(a.config)['name'])
    except data.ConfigMismatch as e:
        sys.exit(f'{e}: compare it with that --config')
    except ValueError as e:
        sys.exit(str(e))
    if not runs:
        sys.exit('no records')
    for bench, recs in sorted(runs.items()):
        if not (live / f'{bench}.jsonl.gz').exists():
            print(f'{bench}: no reference live records')
            continue
        refs = data.read_runs([live / f'{bench}.jsonl.gz'])[bench]
        rows = {r['qid']: r for r in data.load_rows(bench)}
        qids = [q for q in recs if q in refs]
        print(f'\n{bench}: {len(qids)} of {len(recs)} questions have a reference record ({len(refs)} references)')
        if not qids:
            continue
        eq, d_trust, d_p = zip(*(compare(recs[q], refs[q], rows[q]) for q in qids))
        print('  ' + '  '.join(f'{f} {100 * sum(e[f] for e in eq) / len(eq):.1f}%' for f in FIELDS))
        dt = [x for x in d_trust if x is not None]
        dp = [x for x in d_p if x is not None]
        print(f'  max |dP_ptr| {max(dt):.3g} over {len(dt)} questions' if dt else '  no pointer trust computed',
              f'| max |dP| {max(dp):.3g} over {len(dp)} questions without a veto ({sum(x == 0 for x in dp)} equal)'
              if dp else '')


if __name__ == '__main__':
    main()
