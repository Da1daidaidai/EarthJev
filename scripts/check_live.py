"""Compare predictions against the released reference runs."""
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
