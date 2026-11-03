"""Evidence views, the canvas and memory presentations (earthjev.evidence) and image preparation
(earthjev.answerer.prepare_image). The golden digests in fixtures/evidence_golden.json were produced by the code behind
the paper's results on the same synthetic images."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from earthjev.answerer import FACTOR, prepare_image
from earthjev.evidence import (MIN_VIEW_SIDE, compose_canvas, grid_name, grow, iou, proposal_crop, render_memory_views,
                               select_views)

GOLDEN = json.loads((Path(__file__).parent / 'fixtures' / 'evidence_golden.json').read_text())


def pattern(w, h, mode='RGB'):
    """Deterministic high-frequency test image."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    arr = np.stack([(x * 7 + y * 13 + (x * y) % 251) % 256, (x * 3 + y * 5) % 256, (x ^ y) % 256], -1)
    im = Image.fromarray(arr.astype(np.uint8))
    return im if mode == 'RGB' else im.convert(mode)


def digest(im):
    return f'{im.mode}|{im.width}x{im.height}|' + hashlib.sha256(im.tobytes()).hexdigest()


def test_view_geometry():
    W, H = 6000, 4000
    assert grow([0, 0, 200, 100], W, H) == [0, 0, 512, 512]  # small corner box: min side, shifted in
    assert grow([1000, 1000, 2000, 1400], W, H) == [750, 900, 2250, 1500]  # 1.5x context around the centre
    assert grow([0, 0, W, H], W, H) == [0, 0, W, H]  # root node: clipped to the image
    sel = [dict(bbox=[1000, 1000, 1400, 1400], confidence=0.5), dict(bbox=[1040, 1000, 1440, 1400], confidence=0.9),
           dict(bbox=[4000, 3000, 4300, 3300], confidence=0.7), dict(bbox=[100, 100, 300, 300], confidence=0.1),
           dict(bbox=[5000, 100, 5200, 300], confidence=0.2), dict(bbox=[3000, 100, 3200, 300], confidence=0.3)]
    views = select_views(sel, (W, H))
    assert [v['score'] for v in views] == [0.9, 0.7, 0.3, 0.2]  # the merged pair keeps 0.9; top 4 by score
    assert views[0]['bbox'] == [900, 900, 1540, 1500] and views[1]['bbox'] == [3894, 2894, 4406, 3406]
    assert all(r - l >= MIN_VIEW_SIDE and b - t >= MIN_VIEW_SIDE for l, t, r, b in (v['bbox'] for v in views))
    assert all('note' not in v for v in views)
    note = 'the region referenced in the question'
    v2 = select_views(sel + [dict(bbox=[1100, 1050, 1300, 1250], confidence=0.6, note=note)], (W, H))
    assert v2[0]['bbox'] == [900, 894, 1540, 1500] and v2[0]['note'] == note  # a note survives merging
    assert proposal_crop([1000, 1000, 1040, 1020], (W, H)) == [796, 786, 1244, 1234]  # 448 px floor
    assert proposal_crop([0, 0, 400, 300], (W, H)) == [0, 0, 600, 550]  # 2 x long side, clipped
    assert grid_name([0, 0, 10, 10], (W, H)) == 'top-left'
    assert grid_name([5990, 3990, 6000, 4000], (W, H)) == 'bottom-right'
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == 50 / 150


def test_select_views_golden():
    for case in GOLDEN['select_views']:
        assert select_views(case['pool'], tuple(case['size'])) == case['views']


def test_canvas_golden():
    im = pattern(*GOLDEN['canvas']['size'])
    assert compose_canvas(im, []) is im
    for case in GOLDEN['canvas']['cases']:
        assert digest(compose_canvas(im, [dict(bbox=b) for b in case['boxes']])) == case['digest'], case['boxes']


def test_memory_golden():
    for case in GOLDEN['memory']:
        im = pattern(*case['size'], case['mode'])
        renders = render_memory_views(im, [dict(bbox=b) for b in case['boxes']], 'Is there a ship?')
        assert [digest(r) for r in renders] == case['digests'], case['boxes']
        assert all(max(r.size) == 1024 for r in renders)


def test_memory_input_checks():
    im = pattern(800, 600)
    for case in GOLDEN['memory_errors']:
        with pytest.raises(ValueError) as e:
            render_memory_views(im, [dict(bbox=b) for b in case['boxes']], case['stem'])
        assert str(e.value) == case['error']
    assert len(render_memory_views(im, [dict(bbox=[np.int64(1), 2, 30, 40])] * 16, 'q')) == 16


def test_prepare_image_golden():
    for case in GOLDEN['prepare']:
        w, h, px, mn = case['args']
        out = prepare_image(pattern(w, h), px, mn)
        assert digest(out) == case['digest'], case['args']
        assert out.width % FACTOR == out.height % FACTOR == 0 and out.width * out.height <= px
        assert smart_resize(out.height, out.width, FACTOR, out.width * out.height, out.width * out.height) == (
            out.height, out.width)  # a fixed point: the processor keeps it
    strip = prepare_image(Image.new('RGB', (3000, 4), (255, 0, 0)), 1_048_576)
    assert max(strip.size) / min(strip.size) <= 100
    assert prepare_image(Image.new('L', (100, 80)), 1_048_576).mode == 'RGB'
    assert prepare_image(Image.new('RGB', (100, 80)), 1_048_576).size == prepare_image(
        Image.new('RGB', (100, 80)), 1_048_576, 65_536).size  # default floor: the processor's 65,536 px
