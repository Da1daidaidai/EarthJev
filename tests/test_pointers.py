"""earthjev.pointers: pointer formats, stripping, directional phrases, geometry and the directional prior."""
import json
from pathlib import Path

import numpy as np

from earthjev.pointers import ADJ, FORMATS, N_GEO, PREP, REGION, SUPER, geometry, parse, phrase_axes, render, \
    rule_prior, strip

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
S = (10000, 10000)
XL = ('Identify the object category within the given bounding box in the satellite or aerial image. '
      'Image resolution: 10000 x 10000. Bounding box: [5322, 6073, 5573, 6356].')
RSHR = 'Which bounding box best localizes an orange house，resolution: 10000 x 10000'


def test_boxes_points_and_strip():
    (r,) = parse(XL, size=S)
    assert r.kind == 'box' and r.coord == 'px_res' and r.box == (0.5322, 0.6073, 0.5573, 0.6356)
    assert strip(XL) == 'Identify the object category within the given bounding box in the satellite or aerial image.'
    (r,) = parse('Image resolution: 7360 x 4912. Bounding box: [3680, 2456, 3700, 2470].', size=(6460, 4315))
    assert abs(r.box[0] - 0.5) < 1e-9 and abs(r.box[1] - 0.5) < 1e-9  # stated resolution != file: rescaled
    refs = parse(RSHR, [('A', '[3200, 6343, 3327, 6470]'), ('B', '[4288, 4408, 4415, 4560]')], size=S)
    assert [(r.kind, r.option, r.coord) for r in refs] == [('box', 'A', 'px_res'), ('box', 'B', 'px_res')]
    assert strip(RSHR) == 'Which bounding box best localizes an orange house'
    assert parse('Look at [0.1, 0.2, 0.3, 0.4] please.')[0].coord == 'n1'
    r = parse('Look at the box [100, 200, 300, 400] (normalized 0-1000 coordinates).', size=S)[0]
    assert r.coord == 'n1000' and r.box == (0.1, 0.2, 0.3, 0.4)
    r = parse('The target is <|box_start|>(100,200),(300,400)<|box_end|>.', size=S)[0]
    assert r.coord == 'n1000' and r.box == (0.1, 0.2, 0.3, 0.4)
    r = parse('What is the object at pixel (5000, 2500)?', size=S)[0]
    assert r.kind == 'point' and abs((r.box[0] + r.box[2]) / 2 - 0.5) < 1e-9
    assert parse('How many cars (1, 2) are there?', size=S) == []  # no point cue
    r = parse('The target is in the region [100, 200, 300, 400] (pixel coordinates).', size=(2000, 1000))[0]
    assert r.coord == 'px' and r.conf == 1.0 and r.box == (0.05, 0.2, 0.15, 0.4)
    r = parse('The target is in the region [100, 200, 300, 400].', size=S)[0]
    assert r.coord == 'px' and r.conf == 0.5  # cue-less integers <= 1000 in a big image: ambiguous
    r = parse('Region [30, 20, 10, 15] in xywh.', size=(100, 100))[0]  # x2 <= x1: read as [x, y, w, h]
    assert r.coord == 'xywh_px' and np.allclose(r.box, (0.3, 0.2, 0.4, 0.35))


def test_render_round_trip():
    stem = 'What color is the car?'
    for fmt in FORMATS:
        t = stem + ' ' + render(fmt, [1200, 3400, 1300, 3500], (6000, 4000))
        refs = parse(t, size=(6000, 4000))
        assert len(refs) == 1 and refs[0].precise, (fmt, t, refs)
        b = refs[0].box
        assert abs((b[0] + b[2]) / 2 - 1250 / 6000) < 2e-3 and abs((b[1] + b[3]) / 2 - 3450 / 4000) < 2e-3, (fmt, b)
        assert strip(t) == stem, (fmt, strip(t))
    t = stem + ' ' + render('px_res', [1200, 3400, 1300, 3500], (6000, 4000), res=(3000, 2000))
    assert abs(parse(t, size=(6000, 4000))[0].box[0] - 0.2) < 1e-3


def test_phrase_axes():
    cases = {
        'What does the top-most bridge connect?': {'v': (0.0, SUPER)},
        'How many ships are there on the left side of this picture?': {'u': (0.0, REGION)},
        'What is the number of ship in the bottom-left part of the image?': {'v': (1.0, REGION), 'u': (0.0, REGION)},
        'What color is the vehicle to the left of the bottom-most tower crane?': {'v': (1.0, SUPER)},
        'How many people are playing on the football field at the top of the left side of this picture?':
            {'v': (0.0, REGION), 'u': (0.0, REGION)},
        'How many ponds are there to the right of the cluster of houses in the bottom right corner?':
            {'v': (1.0, PREP), 'u': (1.0, PREP)},
        'What is the color of the vest worn by the person on the far left?': {'u': (0.0, PREP)},
        'Is the top-right open field area mostly flat or hilly?': {'v': (0.0, ADJ), 'u': (1.0, ADJ)},
        'How many cargo ships are visible in the southern river channel area?': {'v': (1.0, ADJ)},
        'Is the right-side semicircular landscaped area mainly covered with grass?': {'u': (1.0, ADJ)},
        'What is the shortest route to drive from the large playground at the upper center to the gate?':
            {'v': (0.0, PREP), 'u': (0.5, PREP)},
        'Is the central area suitable for constructing a new residential complex?': {'v': (0.5, ADJ), 'u': (0.5, ADJ)},
        'What activity is in progress at the southern end of the terminal?': {},
        'What type of structure is located directly south of the baseball field?': {},
        'What is the person holding in their right hand?': {},
        'What is the shape of the top rail of the gate?': {},
        'What type of plant is growing in the middle of the road?': {},
        'What color is the puddle on the right side of the road?': {},
        'What is the color of the necklace worn by the person in the center?': {'v': (0.5, PREP), 'u': (0.5, PREP)},
        'Is the north-south road paved?': {},
        'What is the northeast corner building used for?': {'v': (0.0, ADJ), 'u': (1.0, ADJ)},
        'What lies in the northeast corner of the scene?': {'v': (0.0, REGION), 'u': (1.0, REGION)},
        'Where is the playground at the top of the image relative to the coastline?': {'v': (0.0, REGION)},
    }
    for q, want in cases.items():
        assert phrase_axes(q) == want, (q, phrase_axes(q), want)
    assert parse('What color is the boat?') == []
    (r,) = parse('How many ships are in the upper left corner of the picture?')
    assert r.kind == 'phrase' and r.box == (0.0, 0.0, 1 / 3, 1 / 3)
    assert parse('Where is the red car?', [('A', 'top left'), ('B', 'in the bottom right corner of the image')]) == []


def test_geometry_and_prior():
    refs = parse(XL, size=S)
    f, valid = geometry([[0.5, 0.6, 0.6, 0.7], [0.0, 0.0, 0.3, 0.3]], refs)
    assert f.shape == (2, N_GEO) and valid.all() and f[0, 7] == 1 and f[1, 7] == 0 and f[0, 5] == 1 and f[1, 5] == 0
    f, valid = geometry([[0, 0, 1, 1]], [])
    assert not valid.any() and not f.any()
    ph = parse('How many ships are in the upper left corner of the picture?')
    f, _ = geometry([[0, 0, 0.3, 0.3], [0.7, 0.7, 1, 1]], ph)
    assert f[0, 13] == 1 and f[0, 15] < f[1, 15] and f[0, 16] == 1 and f[1, 16] == 0
    opts = parse(RSHR, [('A', '[0, 0, 1000, 1000]'), ('B', '[5000, 5000, 6000, 6000]')], size=S)
    f, _ = geometry([[0.5, 0.5, 0.6, 0.6]], opts)
    assert f[0, 3] == 1 and f[0, 12] == 0.5 and f[0, 5] == 1  # the crop covers option B
    p = rule_prior([[0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1]], ph)  # upper left, strength 1.5 per axis
    assert np.allclose(p, [-1.5 * 0.1 * 2, -1.5 * 0.9 * 2]) and not rule_prior([[0, 0, 1, 1]], refs).any()


def test_golden_parse_and_strip():
    """Real stems (benchmarks and pointer sets) give the references and stripped stems of the original code."""
    cases = json.loads((FIXTURES / 'pointers_golden.json').read_text())
    assert len(cases) > 400
    for c in cases:
        refs = parse(c['stem'], [tuple(o) for o in c['options']], tuple(c['size']))
        got = [dict(kind=r.kind, box=list(r.box), coord=r.coord, option=r.option, conf=r.conf,
                    span=list(r.span) if r.span else None, axes={k: list(v) for k, v in r.axes.items()})
               for r in refs]
        assert got == c['refs'], c['stem']
        assert strip(c['stem']) == c['strip'], c['stem']
