"""Run EarthJev live on a benchmark: one JSON line per question (the record of earthjev/pipeline.py).

    python scripts/run_earthjev.py --bench mme_rs [--split full] [--shard 1/12] [--resume | --overwrite] [--limit N]
                                   [--out PATH] [--config qwen35-9b]

Questions: the split's questions sorted by (image, qid); --shard i/n (1 <= i <= n) runs the i-th of n contiguous
chunks whose sizes differ by at most one. The questions on one image are consecutive and share its decoding, but a
chunk boundary can split them. --resume keeps the error-free records already in --out and runs the rest;
--overwrite replaces --out; without either, a non-empty --out is refused. --config: a name under configs/ or a JSON
file (e.g. a modified copy). Default output: $EARTHJEV_OUT/earthjev_<config>/<bench>_<split>_<i>of<n>.jsonl. Needs a
CUDA GPU (the configuration's answering model, bf16 with flash-attention 2), $EARTHJEV_DATA and $EARTHJEV_MODELS.
Score with scripts/evaluate.py; compare with the reference live records with scripts/check_live.py (reference/).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earthjev import data, paths, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', required=True, choices=data.BENCHMARKS)
    ap.add_argument('--split', default='full')
    ap.add_argument('--shard', default='1/1', help='i/n')
    once = ap.add_mutually_exclusive_group()
    once.add_argument('--resume', action='store_true', help='keep the finished questions of --out')
    once.add_argument('--overwrite', action='store_true', help='replace a non-empty --out')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out')
    ap.add_argument('--config', default='qwen35-9b', help='a name under configs/ or a JSON file')
    a = ap.parse_args()
    cfg = paths.load_config(a.config)
    i, n = map(int, a.shard.split('/'))
    qids = set(data.load_split(f'{a.bench}_{a.split}'))
    try:
        rows = data.shard([r for r in data.load_rows(a.bench) if r['qid'] in qids], i, n)[:a.limit]
    except ValueError as e:
        ap.error(str(e))
    out = a.out or paths.out_root() / f'earthjev_{Path(a.config).stem}' / f'{a.bench}_{a.split}_{i}of{n}.jsonl'
    method = None

    def run(row):  # the models load on the first question that runs
        nonlocal method
        method = method or pipeline.EarthJev.load(cfg)
        return method.run_row(row)

    try:
        pipeline.run_rows(rows, run, out, a.resume, a.overwrite)
    except FileExistsError as e:
        ap.error(str(e))


if __name__ == '__main__':
    main()
