"""Score one or more prediction files by benchmark and question group."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earthjev import data, paths, scoring  # noqa: E402

PAPER = json.loads((paths.REPO / 'reference' / 'paper_scores.json').read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--config', default='qwen35-9b', help='the configuration the runs used (its paper numbers)',
                    choices=sorted(p.stem for p in (paths.REPO / 'configs').glob('*.json')))
    a = ap.parse_args()
    paper_scores = PAPER[a.config]
    try:
        runs = data.read_runs(a.runs, paths.load_config(a.config)['name'])
    except data.ConfigMismatch as e:
        sys.exit(f'{e}: evaluate it with that --config')
    except ValueError as e:
        sys.exit(str(e))
    if not runs:
        sys.exit('no records')
    avgs = {}
    for bench, recs in sorted(runs.items()):
        full = set(data.load_split(f'{bench}_full'))
        rows = [r for r in data.load_rows(bench) if r['qid'] in recs]
        answers = {q: None if r.get('error') else r['final'] for q, r in recs.items()}
        ours = scoring.benchmark_scores(bench, rows, answers)
        paper = paper_scores['benchmarks'].get(bench)
        is_full = set(recs) == full
        errors = sum(r.get('error') is not None for r in recs.values())
        print(f"\n{bench}: {len(recs)} of {len(full)} questions"
              f"{'' if is_full else ' (a subset: the paper numbers below are full-set numbers)'}, {errors} errors")
        width = {c: max(8, len(c)) + 1 for c in ours}
        print(f"  {'':10s}" + ''.join(f'{c:>{w}s}' for c, w in width.items()))
        for name, s in (('this run', ours), ('paper', paper)):
            cells = ''.join(f'{"n/a" if not s or s.get(c) is None else f"{s[c]:.2f}":>{w}s}' for c, w in width.items())
            print(f'  {name:10s}{cells}')
        if is_full:
            avgs[bench] = ours['avg']
    if set(avgs) >= set(data.BENCHMARKS):
        print(f"\noverall (mean of the five benchmarks): {scoring.overall(avgs):.2f}, "
              f"paper {paper_scores['overall']:.2f}")


if __name__ == '__main__':
    main()
