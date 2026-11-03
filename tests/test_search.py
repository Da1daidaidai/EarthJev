"""The hierarchical search against traces recorded from the original ZoomSearch search with stub scorers."""
import hashlib
import json
from pathlib import Path

import pytest

from earthjev.search import grid_cells, hierarchical_search, select_by_sigmoid

FIXTURE = Path(__file__).parent / 'fixtures' / 'search_stub_traces.json'


def stub_hash(box):
    h = hashlib.sha256(repr(tuple(int(v) for v in box)).encode()).digest()
    return int.from_bytes(h[:4], 'big') / 2 ** 31 - 1


def stub_corner(box):
    x1, y1, x2, y2 = box
    return max(-1.0, 1.0 - 2.0 * (x1 + y1) / 30000) - 1e-9 * (x2 - x1)


def stub_leafy(box):
    x1, y1, x2, y2 = box
    if max(x2 - x1, y2 - y1) <= 224:
        return 0.99
    return 0.97 if stub_hash(box) > 0 else 0.5


STUBS = dict(hash=stub_hash, corner=stub_corner, leafy=stub_leafy)


def _run(size, stub):
    s = hierarchical_search(tuple(size), lambda boxes: [stub(b) for b in boxes])
    return dict(final=[[n['bbox'], n['depth'], n['confidence'], '/'.join(n['full_path'])] for n in s['final_selected']],
                iterations=s['iterations'], n_expansions=s['n_expansions'], n_children_scored=s['n_children_scored'])


def test_matches_original_traces():
    cases = json.loads(FIXTURE.read_text())['cases']
    assert len(cases) > 600
    for case in cases:
        expected = {k: case[k] for k in ('final', 'iterations', 'n_expansions', 'n_children_scored')}
        assert _run(case['size'], STUBS[case['stub']]) == expected, (case['stub'], case['size'])


def test_grid_cells():
    cells = grid_cells((0, 0, 1000, 700))
    assert [c for c, _ in cells] == [(0, 0, 333, 233), (333, 0, 666, 233), (666, 0, 1000, 233),
                                     (0, 233, 333, 466), (333, 233, 666, 466), (666, 233, 1000, 466),
                                     (0, 466, 333, 700), (333, 466, 666, 700), (666, 466, 1000, 700)]
    assert cells[4][1] == 'middle-center'
    assert grid_cells((0, 0, 8143, 665)) == []  # rows of 221/221/223 px
    assert [n for _, n in grid_cells((0, 0, 901, 901), min_side=301)] == ['bottom-right']  # 300, 300, 301 px


def test_select_by_sigmoid():
    node = lambda c: dict(confidence=c)
    assert select_by_sigmoid([]) == [] and select_by_sigmoid([node(0.3)]) == [node(0.3)]
    flat = [node(0.1), node(0.2), node(0.15)]  # sigmoid(5 x 0.05) < 0.6: only the most confident one
    assert select_by_sigmoid(flat) == [node(0.2)]
    nodes = [node(c) for c in (0.9, -0.9, 0.8, -0.9, 0.95)]  # kept in their order
    assert select_by_sigmoid(nodes) == [node(0.9), node(0.8), node(0.95)]
    many = [node(c) for c in (0.65, 0.9, 0.7, 0.8, 0.6, 0.95, 0.85, -1.0, -1.0)]  # 7 pass: top 6 by sigmoid
    assert [n['confidence'] for n in select_by_sigmoid(many)] == [0.95, 0.9, 0.85, 0.8, 0.7, 0.65]


def test_edge_cases():
    strip = hierarchical_search((8143, 665), lambda boxes: [0.0] * len(boxes))
    assert strip == dict(final_selected=[dict(bbox=[0, 0, 8143, 665], depth=1, confidence=0.0, full_path=[])],
                         n_expansions=0, n_children_scored=0, iterations=1)
    leaf = hierarchical_search((224, 300), lambda boxes: [])  # the root is split into nothing: root fallback
    assert leaf['final_selected'][0]['bbox'] == [0, 0, 224, 300] and leaf['iterations'] == 1
    with pytest.raises(ValueError):  # longer side below 224 px: math domain error, as the original
        hierarchical_search((223, 100), lambda boxes: [0.0] * len(boxes))
    with pytest.raises(ValueError):
        hierarchical_search((2000, 2000), lambda boxes: [0.0])
