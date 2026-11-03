"""ZwZ-RL-VQA training data: crop geometry, manifest rules and feature batching; with the data, the released manifest
and crop index hashes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from earthjev.train import zwz
from earthjev.train.extract_features import chunks

GOLDEN = json.loads((Path(__file__).parent / 'fixtures' / 'zwz_golden.json').read_text())
CROP_INDEX_SHA256 = '917b0a6932df8ed192bf07b1d6afb5268095cd261a9b885c5f52d301ce3cb27f'  # 250,351 rows


def test_crop_geometry():
    q = dict(qid='t', W=2250, H=1500, bbox=[1996, 816, 2047, 902])
    cs = zwz.crops(q)
    assert cs == zwz.crops(q) and cs != zwz.crops(dict(q, qid='u'))
    for c in cs:
        x1, y1, x2, y2 = c['bbox']
        assert 0 <= x1 < x2 <= q['W'] and 0 <= y1 < y2 <= q['H']
        assert c['label'] == zwz._intersects(c['bbox'], q['bbox'])  # clean positives or clean negatives
    d1 = [c for c in cs if c['kind'] == 'cell_d1']
    assert len(d1) == 9 and sum(c['label'] for c in d1) == 1 and d1[5]['label'] == 1
    assert [c['bbox'] for c in d1][:2] == [[0, 0, 750, 500], [750, 0, 1500, 500]]
    d2 = [c for c in cs if c['kind'] == 'cell_d2']  # 250 x 166 cells; 3 touch the box without its centre
    assert len(d2) == 6 and sum(c['label'] for c in d2) == 1 and all(c['group'] == 'd2' for c in d2)
    assert {c['kind'] for c in cs} == set(zwz.KINDS)
    cs = zwz.crops(dict(qid='x', W=900, H=900, bbox=[250, 250, 350, 350]))  # box across the d1 cell lines
    d1 = [c['label'] for c in cs if c['kind'] == 'cell_d1']  # touching cells without the centre are dropped
    assert len(d1) == 6 and sum(d1) == 1
    for g in GOLDEN['crops']:  # outputs of the original code
        assert hashlib.sha256(json.dumps(zwz.crops(g['q'])).encode()).hexdigest() == g['sha256']


def test_manifest_rules():
    assert zwz.stem('<image>What color is the hull?\n\nOnly focus on ...\n\nA. black') == 'What color is the hull?'
    assert zwz.split_of('original_images/sa_1.jpg') == zwz.split_of('sa_1.jpg')
    row = lambda name, problem, bbox: dict(original_images=[f'original_images/{name}'], problem=problem, bbox=bbox,
                                           answer='A', images=['x'])
    rows = [row('a.jpg', 'Q1?\nx', [10, 10, 60, 60]), row('a.jpg', 'Q1?\ny', [10, 10, 60, 60]),  # duplicate stem
            row('a.jpg', 'Q2? red bounding box', [-5, 0, 900.4, 40]), row('b.jpg', 'Q3?', [0, 0, 600, 600]),
            row('c.jpg', 'Q4?', [5, 5, 9, 12]), row('d.jpg', 'Q5?', [10, 10, 50, 50]),
            row('e.jpg', 'Q6?', [1, 1, 30, 30])]
    sizes = {'original_images/a.jpg': (1000, 800), 'original_images/b.jpg': (1000, 1000),
             'original_images/c.jpg': (1000, 1000), 'original_images/d.jpg': (1000, 600)}

    def image_size(name):
        if name not in sizes:
            raise FileNotFoundError(name)
        return sizes[name]

    kept, drop = zwz.build_manifest(rows, image_size)
    assert drop == Counter(duplicate=1, box_too_large=1, box_degenerate=1, image_too_small=1, image_missing=1)
    assert sorted(r['stem'] for r in kept) == ['Q1?', 'Q2? red bounding box'] and len({r['split'] for r in kept}) == 1
    assert [r['qid'] for r in kept] == sorted((r['qid'] for r in kept), key=zwz._unit)
    r = next(r for r in kept if r['stem'].startswith('Q2'))
    assert r['bbox'] == [0, 0, 900, 40] and r['overlay'] and (r['W'], r['H']) == (1000, 800)
    assert r['qid'] == 'zwz_' + hashlib.sha1(b'original_images/a.jpg|Q2? red bounding box').hexdigest()[:12]


def test_chunks():
    grid = torch.tensor([[1, 2, 2]] * 20 + [[1, 4, 2]] * 20)  # 4 or 8 pixel rows per image
    lengths = torch.tensor([12] + [5] * 31 + [4] * 8)
    mask = (torch.arange(12)[None] < lengths[:, None]).long()
    batch = dict(input_ids=mask * 5, attention_mask=mask, mm_token_type_ids=mask * 0, image_grid_thw=grid,
                 pixel_values=torch.arange(int(grid.prod(1).sum()))[:, None].float())
    parts = list(chunks(batch, 32))
    assert [len(p['input_ids']) for p in parts] == [32, 8] and [p['input_ids'].shape[1] for p in parts] == [12, 4]
    assert torch.equal(torch.cat([p['pixel_values'] for p in parts]), batch['pixel_values'])
    assert torch.equal(parts[1]['image_grid_thw'], grid[32:]) and list(chunks(batch, 40))[0] is batch


@pytest.mark.data
def test_released_manifest_and_crop_index():
    path = zwz.manifest_path()
    if not path.exists():
        pytest.skip('run `python -m earthjev.train.zwz export` and `build` first')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == zwz.MANIFEST_SHA256
    index = ''.join(json.dumps(dict(id=q['qid'], split=q['split'], j=j, **c)) + '\n'
                    for q in zwz.load_manifest() for j, c in enumerate(zwz.crops(q)))
    assert hashlib.sha256(index.encode()).hexdigest() == CROP_INDEX_SHA256
