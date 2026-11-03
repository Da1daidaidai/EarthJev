"""Questions with inserted spatial pointers: the pointer-trust calibration sets.

A pointer sentence (earthjev.pointers.render, in a format seeded by the question id) is appended to the question stem.
It states either the annotated evidence box ('ok') or a box of the same size at a seeded position whose
neighbourhood misses the evidence ('bad'); the same questions are used for both conditions.

lrsrefdev_<ok|bad>   the LRS-VQA dev questions whose evidence box leaves room for a wrong box: 200 of the 210 dev
                     questions with a box (of 240).
grorefdev_<ok|bad>   LRS-GRO training questions with a box (canonical 0-1024 frame), except those on the images of 2,000
                     held-out LRS-VQA evidence questions (the paper's pointer test, built the same way outside the dev
                     split; 2,049 questions).
They are the fitting data of the pointer-trust map (a_ptr, b_ptr) and, with the XLRS-lite dev questions, of the
thresholds tau_fast / tau_keep (Sec. III-B).
"""
import hashlib
import os
import random

from earthjev.pointers import render

MAX_QUESTIONS = 2000  # LRS-VQA questions per selection
SEED = 20261013
FORMATS = ('px_res', 'px', 'n1000', 'n1', 'point_px')  # the formats of inserted pointer sentences
GRO_FORMAT = '\n Answer the question using a single word or phrase.'  # the LRS-GRO answer format line


def pointer_box(hbox, size, qid, condition):
    """Pointer box (pixels): the clipped evidence box ('ok'), or the same size at a seeded uniform position whose
    neighbourhood (a square of side max(512, 2 x long side) centred on it) misses the evidence box ('bad'; None when
    the image leaves no room for one)."""
    W, H = size
    x1, y1, x2, y2 = max(0, hbox[0]), max(0, hbox[1]), min(W, hbox[2]), min(H, hbox[3])
    if condition == 'ok':
        return [x1, y1, x2, y2]
    w, h = x2 - x1, y2 - y1
    half = max(512, 2 * max(w, h)) / 2
    rng = random.Random(f'{SEED}|{qid}')
    for _ in range(10000):
        bx, by = rng.uniform(0, W - w), rng.uniform(0, H - h)
        cx, cy = bx + w / 2, by + h / 2
        if cx + half <= x1 or cx - half >= x2 or cy + half <= y1 or cy - half >= y2:
            return [round(bx), round(by), round(bx + w), round(by + h)]
    return None


def pointer_format(qid):
    """The seeded sentence format of a question (the same for both conditions)."""
    return FORMATS[int(hashlib.sha256(f'{SEED}|fmt|{qid}'.encode()).hexdigest()[:8], 16) % len(FORMATS)]


def evidence_questions(lrs_rows, dev_ids, dev):
    """LRS-VQA rows (with hbox and image_size) with an evidence box, inside (dev True) or outside the dev split, in
    seeded sha256 order: the first 2,000 that admit a 'bad' box."""
    dev_ids = set(dev_ids)
    rows = sorted((r for r in lrs_rows if r['hbox'] and (r['qid'] in dev_ids) == dev),
                  key=lambda r: hashlib.sha256(f'{SEED}|{r["qid"]}'.encode()).hexdigest())
    return [r for r in rows if pointer_box(r['hbox'], r['image_size'], r['qid'], 'bad')][:MAX_QUESTIONS]


def lrsrefdev(lrs_rows, dev_ids, condition):
    """lrsrefdev_<condition> from the LRS-VQA rows and the LRS-VQA dev qids."""
    out = []
    for r in evidence_questions(lrs_rows, dev_ids, dev=True):
        fmt = pointer_format(r['qid'])
        box = pointer_box(r['hbox'], r['image_size'], r['qid'], condition)
        stem = r['stem'] + ' ' + render(fmt, box, tuple(r['image_size']))
        out.append(dict(r, bench=f'lrsrefdev_{condition}', stem=stem, question=stem + r['question'][len(r['stem']):],
                        ref_format=fmt, ref_box=box))
    return out


def groref(gro_rows, lrs_rows, dev_ids, condition):
    """grorefdev_<condition> from the LRS-GRO training rows (gro_box in the 0-1024 frame: pixels = value x max(W, H) /
    1024; their images are LRS-VQA images, whose sizes the LRS-VQA rows give), skipping the images of the held-out
    evidence questions (evidence_questions outside the dev split) and the boxes that admit no 'bad' position."""
    size = {os.path.basename(r['image']): tuple(r['image_size']) for r in lrs_rows}
    test_images = {os.path.basename(r['image']) for r in evidence_questions(lrs_rows, dev_ids, dev=False)}
    out = []
    for r in gro_rows:
        name = os.path.basename(r['image'])
        if not r['gro_box'] or name in test_images:
            continue
        W, H = size[name]
        hbox = [round(v * max(W, H) / 1024) for v in r['gro_box']]
        if pointer_box(hbox, (W, H), r['qid'], 'bad') is None:
            continue
        box = pointer_box(hbox, (W, H), r['qid'], condition)
        fmt = pointer_format(r['qid'])
        stem = r['stem'] + ' ' + render(fmt, box, (W, H))
        out.append(dict(r, bench=f'grorefdev_{condition}', stem=stem, question=stem + GRO_FORMAT, hbox=hbox,
                        image_size=[W, H], ref_format=fmt, ref_box=box))
    return out


NAMES = tuple(f'{s}_{c}' for s in ('lrsrefdev', 'grorefdev') for c in ('ok', 'bad'))


def load(name):
    """Rows of one of NAMES, built from the base rows of earthjev.data."""
    if name not in NAMES:
        raise ValueError(f'unknown pointer set {name!r}')
    from earthjev import data
    family, condition = name.split('_')
    lrs, dev_ids = data.load_rows('lrs_vqa'), data.load_split('lrs_vqa_dev')
    if family == 'grorefdev':
        return groref(data.load_rows('gro_train'), lrs, dev_ids, condition)
    return lrsrefdev(lrs, dev_ids, condition)
