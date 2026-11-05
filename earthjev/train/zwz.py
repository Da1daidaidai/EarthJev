"""Build region-head training crops from ZwZ-RL-VQA annotations."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter

from earthjev import paths
from earthjev.search import grid_cells

SEED = 20261009
SPLIT_FRAC = (('train', 0.90), ('val', 0.95), ('calib', 1.0))  # by image: u = sha256(seed|basename) in [0, 1)
CAPS = {'train': 12000, 'val': 1500, 'calib': 1500}  # questions kept per split (val and calib have fewer)
MAX_BOX_FRAC, MIN_BOX_SIDE, MIN_IMAGE_SIDE = 0.25, 8, 672
MIN_CELL = 112  # training grid cells (the search scores cells >= 224 px; smaller ones stand for deeper levels)
KINDS = ('cell_d1', 'cell_d2', 'cell_d2neg', 'centred', 'context')
MANIFEST_SHA256 = '17427785a4b5809d59d0bb8c50d88d4f56911e11e0c531c4865e33abf0866916'


def image_root():
    """$EARTHJEV_DATA/ZwZ-RL-VQA: holds train.parquet and original_images/ (the manifest's image paths)."""
    return paths.data_root() / 'ZwZ-RL-VQA'


def rows_path():
    return paths.out_root() / 'zwz' / 'train_rows.jsonl'


def manifest_path():
    return paths.out_root() / 'zwz' / 'manifest.jsonl'


def export():
    import pyarrow.parquet as pq
    rows = pq.read_table(image_root() / 'train.parquet',
                         columns=['images', 'original_images', 'bbox', 'problem', 'answer']).to_pylist()
    rows_path().parent.mkdir(parents=True, exist_ok=True)
    with open(rows_path(), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{len(rows)} rows -> {rows_path()}')


def stem(problem):
    """First line of the question without the <image> tag (red-box line, options and format line dropped)."""
    return problem.replace('<image>', '').strip().split('\n')[0].strip()


def _unit(text):
    return int(hashlib.sha256(f'{SEED}|{text}'.encode()).hexdigest()[:12], 16) / 16 ** 12


def split_of(image_name):
    u = _unit(image_name.rsplit('/', 1)[-1])
    return next(name for name, hi in SPLIT_FRAC if u < hi)


def build_manifest(rows, image_size):
    """(manifest rows, drop counts) from the exported rows; image_size(name) -> (W, H) of an original image."""
    seen, drop, out = set(), Counter(), []
    for r in rows:
        orig, q = r['original_images'][0], stem(r['problem'])
        if (orig, q) in seen:
            drop['duplicate'] += 1
            continue
        seen.add((orig, q))
        try:
            W, H = image_size(orig)
        except OSError:
            drop['image_missing'] += 1
            continue
        x1, y1, x2, y2 = r['bbox']
        box = [max(0, round(x1)), max(0, round(y1)), min(W, round(x2)), min(H, round(y2))]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) <= 0 or max(bw, bh) < MIN_BOX_SIDE:
            drop['box_degenerate'] += 1
        elif bw * bh > MAX_BOX_FRAC * W * H:
            drop['box_too_large'] += 1
        elif min(W, H) < MIN_IMAGE_SIDE:
            drop['image_too_small'] += 1
        else:
            qid = 'zwz_' + hashlib.sha1(f'{orig}|{q}'.encode()).hexdigest()[:12]
            out.append(dict(qid=qid, image=orig, W=W, H=H, bbox=box, stem=q, answer=r['answer'],
                            overlay='red bounding box' in r['problem'], split=split_of(orig)))
    kept = []
    for name, cap in CAPS.items():
        kept += sorted((r for r in out if r['split'] == name), key=lambda r: _unit(r['qid']))[:cap]
    return kept, drop


def build():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def image_size(name):
        with Image.open(image_root() / name) as im:
            return im.size

    with open(rows_path()) as f:
        kept, drop = build_manifest(map(json.loads, f), image_size)
    with open(manifest_path(), 'w') as f:
        for r in kept:
            f.write(json.dumps(r) + '\n')
    counts = Counter(r['split'] for r in kept)
    print(f'dropped {dict(drop)}; kept {dict(counts)} -> {manifest_path()}')


def load_manifest(splits=('train', 'val', 'calib')):
    with open(manifest_path()) as f:
        return [r for r in map(json.loads, f) if r['split'] in splits]


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _grid(box):
    return [c for c, _ in grid_cells(box, MIN_CELL)]


def _free_square(rng, side, W, H, box, tries=50):
    """A random side x side crop (clipped to the image) that does not intersect box, or None."""
    w, h = min(side, W), min(side, H)
    for _ in range(tries):
        x, y = rng.randint(0, W - w), rng.randint(0, H - h)
        if not _intersects((x, y, x + w, y + h), box):
            return x, y, x + w, y + h
    return None


def crops(q):
    """[{kind, label, bbox, group}] of one manifest row, deterministic per qid (random.Random(seed|qid)):

    cell_d1     the 3x3 grid of the image (cells >= 112 px);
    cell_d2     the 3x3 grid of the d1 cell holding the box centre;
    cell_d2neg  3 random children of one random d1 cell that misses the box;
    centred     a square centred on the box, side max(2 x long side, 448 px), and 2 same-size squares missing it;
    context     2 squares of log-uniform side in [max(1.5 x long side, 224), min(W, H)] holding the whole box, each
                with a same-size square missing it.
    label 1 = holds the box centre (context: the whole box), 0 = does not intersect the box; other crops are not
    emitted. group ('d1' / 'd2') marks sibling cells (kept for the index; unused by the final training loss)."""
    rng = random.Random(f'{SEED}|{q["qid"]}')
    W, H, b = q['W'], q['H'], q['bbox']
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    centre_in = lambda c: c[0] <= cx < c[2] and c[1] <= cy < c[3]
    box_in = lambda c: c[0] <= b[0] and c[1] <= b[1] and b[2] <= c[2] and b[3] <= c[3]
    out = []

    def add(kind, cell, rule, group=None):
        cell = [int(v) for v in cell]
        if not _intersects(cell, b):
            out.append(dict(kind=kind, label=0, bbox=cell, group=group))
        elif rule(cell):
            out.append(dict(kind=kind, label=1, bbox=cell, group=group))

    d1 = _grid((0, 0, W, H))
    for c in d1:
        add('cell_d1', c, centre_in, 'd1')
    hit = [c for c in d1 if centre_in(c)]
    if hit:
        for c in _grid(hit[0]):
            add('cell_d2', c, centre_in, 'd2')
    negs = [c for c in d1 if not _intersects(c, b)]
    if negs:
        kids = _grid(rng.choice(negs))
        for c in rng.sample(kids, min(3, len(kids))):
            add('cell_d2neg', c, centre_in)
    long_side = max(b[2] - b[0], b[3] - b[1])
    half = max(2 * long_side, 448) / 2
    pos = (max(0, round(cx - half)), max(0, round(cy - half)), min(W, round(cx + half)), min(H, round(cy + half)))
    add('centred', pos, centre_in)
    for _ in range(2):
        neg = _free_square(rng, max(pos[2] - pos[0], pos[3] - pos[1]), W, H, b)
        if neg:
            add('centred', neg, centre_in)
    lo, hi = max(1.5 * long_side, 224), min(W, H)
    for _ in range(2 if lo <= hi else 0):
        s = min(hi, round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
        xs, ys = (max(0, b[2] - s), min(b[0], W - s)), (max(0, b[3] - s), min(b[1], H - s))
        if xs[0] <= xs[1] and ys[0] <= ys[1]:
            x, y = rng.randint(*xs), rng.randint(*ys)
            add('context', (x, y, x + s, y + s), box_in)
        neg = _free_square(rng, s, W, H, b)
        if neg:
            add('context', neg, box_in)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['export', 'build'])
    {'export': export, 'build': build}[ap.parse_args(argv).cmd]()


if __name__ == '__main__':
    main()
