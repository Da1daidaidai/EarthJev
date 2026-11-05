"""Load benchmark questions, evaluation splits and source images."""
import gzip
import json
import sys
from collections import defaultdict

from earthjev import paths

BENCHMARKS = ('mme_rs', 'lrs_vqa', 'xlrs_lite', 'lrs_gro', 'rshr')
DATA = paths.REPO / 'data'


def load_rows(bench):
    """The questions of one set (a benchmark or a calibration set 'gro_train', 'lrsrefdev_ok' / 'lrsrefdev_bad',
    'grorefdev_ok' / 'grorefdev_bad') in their fixed order."""
    with open(DATA / 'benchmarks' / f'{bench}.jsonl', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_split(name):
    """A frozen question list (qids), e.g. 'mme_rs_full' or 'lrs_vqa_dev'."""
    with open(DATA / 'splits' / f'{name}.json', encoding='utf-8') as f:
        return json.load(f)


def image_paths(row):
    """Absolute paths of the question's image(s) under $EARTHJEV_DATA (two for the change-detection questions)."""
    root = paths.data_root().absolute()
    return [root / p for p in row['images']] if 'images' in row else [root / row['image']]


def shard(rows, i, n):
    """Shard i of n (1 <= i <= n, else ValueError): the i-th of n contiguous chunks (sizes differ by at most one) of the
    rows sorted by (first image path, qid). The questions on one image are consecutive, but a chunk boundary can split
    them."""
    if not 1 <= i <= n:
        raise ValueError(f'shard {i}/{n}: needs 1 <= i <= n')
    rows = sorted(rows, key=lambda r: (r['images'][0] if 'images' in r else r['image'], r['qid']))
    return rows[(i - 1) * len(rows) // n:i * len(rows) // n]


class ConfigMismatch(ValueError):
    """A run file holds records of another configuration than the one given."""


def config_option(name):
    """The --config value (file name under configs/) of a configuration's name, or the name itself."""
    for p in sorted((paths.REPO / 'configs').glob('*.json')):
        if json.loads(p.read_text()).get('name') == name:
            return p.stem
    return name


def read_runs(files, config=None):
    """{bench: {qid: record}} of run files (scripts/run_earthjev.py; .jsonl or .jsonl.gz). The files are read in sorted
    order, so the result does not depend on the order they are given in. A question in several files keeps an
    error-free record over one with an error, otherwise the record read last; such questions are reported on stderr
    with their files. config: the configuration's name (its 'name'); a record of another configuration is a
    ValueError (records without the field, e.g. the reference records, pass)."""
    runs, seen = defaultdict(dict), defaultdict(list)
    for path in sorted(map(str, files)):
        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
            for k, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{k}: not a complete JSON record ({e.msg}); a run that was cut short is '
                                     'repaired by running it again with --resume') from None
                if config is not None and r.get('config', config) != config:
                    raise ConfigMismatch(f"{path}: records of the configuration --config {config_option(r['config'])}, "
                                         f'not --config {config_option(config)}')
                old = runs[r['bench']].get(r['qid'])
                if old is None or r.get('error') is None or old.get('error') is not None:
                    runs[r['bench']][r['qid']] = r
                seen[r['bench'], r['qid']].append(path)
    shared = defaultdict(int)
    for files_of_q in seen.values():
        if len(files_of_q) > 1:
            shared[tuple(sorted(set(files_of_q)))] += 1
    for names, n in shared.items():
        print(f"warning: {n} questions have several records, in {', '.join(names)}", file=sys.stderr)
    return runs
