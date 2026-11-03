"""earthjev.spv (pointer views, verifier, trust, routes) and its training code (earthjev.train.pointer_sets,
earthjev.train.train_verifier), against logged searches of the method, the shipped weights and golden values of the
original implementation."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from earthjev import data, pointers, spv
from earthjev.calibration import auroc
from earthjev.paths import load_config, weights_dir
from earthjev.train import pointer_sets
from earthjev.train import train_verifier as tv

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
CFG = load_config()
KEYS = ('bbox', 'crop', 'option', 'coord', 'conf', 'note')


def _verifier(cfg=CFG):
    return spv.PointerVerifier.load(weights_dir(cfg, 'verifier'), cfg['spv']['temperature'])


def test_pointer_proposals():
    stem = ('Is the car at pixel (3000, 2000) the same as the one in [100, 100, 140, 120]? Also [110, 100, 150, 120] '
            'and the top-left corner of the image.')
    opts = [(c, f'[{2000 * i}, 3000, {2000 * i + 300}, 3400]') for i, c in enumerate('ABCDEFGH')]
    props = spv.pointer_proposals(pointers.parse(stem, opts, (20000, 4000)), (20000, 4000))
    q = 'the region referenced in the question'
    assert props[:3] == [
        dict(bbox=[100, 100, 140, 120], crop=[0, 0, 344, 334], option=None, coord='px', conf=1.0, note=q),
        dict(bbox=[2960, 1992, 3040, 2008], crop=[2776, 1776, 3224, 2224], option=None, coord='px', conf=1.0, note=q),
        dict(bbox=[0, 3000, 300, 3400], crop=[0, 2800, 550, 3600], option='A', coord='px', conf=1.0,
             note='the region of option A')]
    # the second stem box overlaps the first view (dropped); at most 8 views (options G and H dropped)
    assert [x['option'] for x in props] == [None, None, 'A', 'B', 'C', 'D', 'E', 'F']
    tiny = spv.pointer_proposals([pointers.Ref('box', (0.5, 0.5, 0.50001, 0.50001))], (1000, 1000))
    assert tiny[0]['bbox'] == [500, 500, 501, 501] and tiny[0]['crop'] == [276, 276, 724, 724]


def test_pointer_replay():
    """Logged searches of the method on XLRS-lite, RSHR-Bench and LRS-VQA questions with a correct or a wrong stated box
    (tests/fixtures/spv_pointer_replay.json.gz, recorded from the original runs: fast paths, no pointer view, phrases,
    option views, several views, two-image rows): the same pointer views, reference count and phrase axes; the logged
    fast path (taken iff max p >= tau_opt in these runs) and pi follow from the logged verifier probabilities."""
    cases = json.loads(gzip.decompress((FIXTURES / 'spv_pointer_replay.json.gz').read_bytes()))['cases']
    n_fast = n_none = 0
    for c in cases:
        for s in c['searches']:  # two-image rows log one search per image
            size, rec = tuple(s['image_size']), s['record']
            refs = pointers.parse(c['stem'], [tuple(o) for o in c['options']], size)
            props = spv.pointer_proposals(refs, size)
            assert props == [{k: x[k] for k in KEYS} for x in rec['proposals']], c['qid']
            phrase = next((r.axes for r in refs if r.kind == 'phrase'), None)
            assert len(refs) == rec['n_refs'] and rec['phrase'] == (
                None if phrase is None else {k: list(v) for k, v in phrase.items()}), c['qid']
            if not props:
                assert rec['path'] == 'none'
                n_none += 1
                continue
            for x, logged in zip(props, rec['proposals']):
                x['p'] = logged['p']
            d = spv.select_path(props, CFG['spv'])
            assert d['pi'] == rec['pi'] and d['admit'] and d['trust'] is None
            assert (d['fast'] is not None) == (rec['path'] == 'S1'), c['qid']
            if d['fast'] is not None:
                n_fast += 1
                assert s['fast_view'] == d['fast']['bbox']
    assert len(cases) > 300 and n_fast > 50 and n_none > 50


def _close(got, want):
    np.testing.assert_allclose(got, np.array(want, float), rtol=0, atol=1e-6)  # None in want: NaN at the same place


def test_verifier_golden():
    """The original verifier's p, gate and mixture on cached region-head logits of LRS-VQA probe crops. The values are
    bitwise equal on the platform the fixture was made on (x86-64 with AVX-512 NumPy); NumPy's exp and the float32
    kernels round differently elsewhere (p moves by up to 2 ulp without AVX-512, and float32 rounding of the network
    alone moves p and the gate by up to 2.1e-7), hence the 1e-6 tolerance."""
    gold = json.loads((FIXTURES / 'spv_golden.json').read_text())
    ver = _verifier()
    assert (ver.T, CFG['spv']['tau_opt']) == (gold['temperature'], gold['tau'])
    for c in gold['cases']:
        refs = [pointers.Ref(r['kind'], tuple(r['box']), r['coord'], r['option'], r['conf'],
                             tuple(r['span']) if r['span'] else None, {k: tuple(v) for k, v in r['axes'].items()})
                for r in c['refs']]
        z = np.asarray(c['z'], np.float32)
        p, g = ver.score(z, c['crops'], tuple(c['size']), refs)
        assert p.dtype == np.float64  # the calibrated sigmoid runs in float64, as in the original
        _close(p, c['p'])
        _close(g, c['gate'])
        for pi, want in c['mixture'].items():
            _close(ver.mixture(z, c['crops'], tuple(c['size']), refs, float(pi)), want)


@pytest.mark.parametrize('config', ['qwen35-9b', 'qwen3-vl-8b'])
def test_weights_match_config(config):
    cfg = load_config(config)
    meta = json.loads((weights_dir(cfg, 'verifier') / 'verifier.json').read_text())
    assert (meta['temperature'], meta['tau']) == (cfg['spv']['temperature'], cfg['spv']['tau_opt'])
    assert meta['width'] == _verifier(cfg).net.zg.in_features and meta['n_geo'] == pointers.N_GEO


def test_trust_route_and_path():
    ab = CFG['spv']['trust_platt']
    approx = lambda x: pytest.approx(x, rel=1e-12, abs=0)  # np.log / np.exp may differ in the last bit by platform
    assert spv.pointer_trust([0.7, 0.2], [0.9, 0.95], ab) == (approx(0.9896954454452027),
                                                              approx([0.820871215252208, 0.6854801408612434]))
    assert spv.pointer_trust([0.05], [0.3], ab) == (approx(0.06839391474329416), approx([0.13057680273413985]))
    assert spv.pointer_trust([1.0], [0.0], ab) == (approx(0.7971847110806014),  # clipped logits
                                                   approx([0.49999999999640554]))
    assert [spv.route(P, 0.6, 0.25) for P in (0.6, 0.59, 0.25, 0.2)] == ['fast', 'search', 'search',
                                                                          'search_without_pointers']
    view = lambda p, option=None: dict(bbox=[0, 0, 1, 1], option=option, p=p)
    sp = CFG['spv']
    with pytest.raises(ValueError):  # one y_j per stem pointer view
        spv.select_path([view(0.05), view(0.9)], sp, y=[0.05])
    opt_only = spv.select_path([view(0.7, 'A'), view(0.5, 'B')], sp, y=[])
    assert opt_only['fast']['option'] == 'A' and opt_only['pi'] == 0.7 and opt_only['trust'] is None
    assert spv.select_path([view(0.62, 'A')], sp)['fast'] is None  # below tau_opt
    stem, opt = view(0.7), view(0.9, 'A')
    trusted = spv.select_path([stem, opt], sp, y=[0.9])  # the fast path uses the best stem pointer
    assert trusted['fast'] is stem and trusted['pi'] == 0.9 and trusted['trust']['s1'] and trusted['admit']
    low = spv.select_path([view(0.05)], sp, y=[0.3])
    assert low['fast'] is None and low['admit'] and low['trust'] == dict(P=approx(0.06839391474329416),
                                                                        s=approx([0.13057680273413985]), s1=False,
                                                                        admit=True)
    assert not spv.select_path([view(0.05)], dict(sp, tau_keep=0.5), y=[0.3])['admit']
    assert spv.select_path([view(0.7)], sp)['fast'] is not None  # no y (two-image rows): tau_opt
    assert spv.select_path([], sp, y=[]) == dict(fast=None, admit=True, pi=1.0, trust=None)  # no pointer views


def test_pointer_sets_rebuild():
    """The builders reproduce the shipped pointer-calibration questions."""
    for name in pointer_sets.NAMES:
        assert pointer_sets.load(name) == data.load_rows(name), name
    assert pointer_sets.pointer_format('STAR_1011') in pointer_sets.FORMATS
    assert pointer_sets.pointer_box([10, 10, 50, 50], (100, 100), 'q', 'bad') is None  # no room for a wrong box
    assert pointer_sets.pointer_box([-5, 10, 50, 120], (100, 100), 'q', 'ok') == [0, 10, 50, 100]


def test_training_synthesis():
    rng = np.random.default_rng(0)
    q = dict(qid='t', W=2000, H=1500, gt=[1000, 700, 1100, 800], stem='What color is the car?', natural=[])
    for _ in range(50):
        b = tv.corrupt(q['gt'], 2000, 1500, rng)
        assert b is None or (0 <= b[0] < b[2] <= 2000 and 0 <= b[1] < b[3] <= 1500)
        refs = tv.precise_refs(q, rng, False, ('box', 'point', 'options')[rng.integers(3)])
        assert refs and all(r.precise for r in refs)
        c = [((r.box[0] + r.box[2]) / 2 * 2000, (r.box[1] + r.box[3]) / 2 * 1500) for r in refs]
        assert any(abs(x - 1050) < 25 and abs(y - 750) < 25 for x, y in c), (refs, c)
    assert tv.fit_tau(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 1]))[0] == 0.8
    assert tv.fit_tau(np.array([0.9, 0.8]), np.array([0, 0]))[0] == math.inf
    # the gated net learns to trust a matching pointer and to distrust one the visual logit contradicts
    N = 20000
    y = (rng.random(N) < 0.3).astype(np.float32)
    z = (y * 2 - 1) * 1.0 + rng.normal(0, 1.5, N)
    match = np.where(rng.random(N) < 0.25, rng.random(N) < 0.1, y > 0)
    geo = np.zeros((N, pointers.N_GEO), np.float32)
    geo[:, 0], geo[:, 7] = 1, match
    arr = (z.astype(np.float32), geo, np.ones(N, np.float32), y, np.zeros(N, bool))
    state, _, _ = tv.train_one(lambda e: arr, {c: arr for c in ('none', 'correct', 'corrupt', 'mix')}, 16, 3e-3, 0,
                               log=lambda *_: None)
    net = spv.VerifierNet(16)
    net.load_state_dict(state)
    assert auroc(y, tv.logits(net, arr)[0]) > auroc(y, z) + 0.05


def test_train_end_to_end(tmp_path):
    """questions() + train() on a small synthetic crop set: the outputs load as a PointerVerifier."""
    rng = np.random.default_rng(1)
    stems = ('What color is the car?', 'How many ships are in the upper left corner of the image?')
    index, z, manifest = [], [], []
    for i in range(48):
        W, H, s = 2000, 1500, int(rng.integers(60, 300))
        x, y = int(rng.integers(0, W - s)), int(rng.integers(0, H - s))
        gt, qid, split = [x, y, x + s, y + s], f'q{i}', ('train', 'train', 'val', 'calib')[i % 4]
        manifest.append(dict(qid=qid, W=W, H=H, bbox=gt, stem=stems[i % 2], split=split))
        cx, cy, half = x + s / 2, y + s / 2, max(2 * s, 448) / 2
        crops = [('centred', [max(0, cx - half), max(0, cy - half), min(W, cx + half), min(H, cy + half)], 1)]
        crops += [('cell_d1', [c * W / 3, r * H / 3, (c + 1) * W / 3, (r + 1) * H / 3],
                   int(c * W / 3 <= cx < (c + 1) * W / 3 and r * H / 3 <= cy < (r + 1) * H / 3))
                  for r in range(3) for c in range(3)]
        far = [b for k, b, lab in crops if k == 'cell_d1' and not lab and not tv._intersects(b, gt)]
        crops.append(('centred', far[0], 0))
        for j, (kind, box, label) in enumerate(crops):
            index.append(dict(id=qid, split=split, j=j, kind=kind, label=label, bbox=box))
            z.append(2.0 * label - 1 + rng.normal(0, 1))
    np.savez(tmp_path / 'zv.npz', z=np.array(z, np.float32), index=json.dumps(index))
    (tmp_path / 'manifest.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in manifest))
    qs = tv.questions(tmp_path / 'zv.npz', tmp_path / 'manifest.jsonl')
    report = tv.train(qs, tmp_path / 'out', log=lambda *_: None)
    shipped = json.loads((weights_dir(CFG, 'verifier') / 'verifier.json').read_text())
    assert list(report) == list(shipped) and json.loads((tmp_path / 'out' / 'verifier.json').read_text()) == report
    assert len(json.loads((tmp_path / 'out' / 'training_history.json').read_text())[0]['history']) == tv.EPOCHS
    ver = spv.PointerVerifier.load(tmp_path / 'out', report['temperature'])
    p, _ = ver.score([0.0], [[0, 0, 448, 448]], (2000, 1500), [])
    assert 0 < p[0] < 1


def test_cache_zv(tmp_path, monkeypatch):
    """The zv command: the region head on every feature block (numeric shard / block order), crop index alongside."""
    from earthjev.region_head import RegionHead, head_logits
    monkeypatch.setenv('EARTHJEV_OUT', str(tmp_path))
    d = tmp_path / 'features' / 'zwz'
    d.mkdir(parents=True)
    rng, blocks = np.random.default_rng(2), {}
    for shard, k, n in ((10, 0, 3), (2, 1, 4), (2, 0, 5)):
        part = f's{shard}of10.b{k:03d}'
        blocks[shard, k] = rng.normal(0, 1, (n, 4096)).astype(np.float16)
        np.save(d / f'qwen35-9b.{part}.npy', blocks[shard, k])
        (d / f'index.{part}.jsonl').write_text(''.join(json.dumps(dict(id=part, j=j)) + '\n' for j in range(n)))
    tv.main(['zv'])
    f = np.load(tmp_path / 'verifier' / 'qwen35-9b' / 'zv_zwz.npz')
    head = RegionHead.load(weights_dir(CFG, 'region_head'))
    order = [(2, 0), (2, 1), (10, 0)]
    assert np.array_equal(f['z'], np.concatenate([head_logits(head, blocks[key]) for key in order]))
    assert [r['id'] for r in json.loads(str(f['index']))] == [f's{s}of10.b{k:03d}' for s, k in order
                                                             for _ in range(len(blocks[s, k]))]
