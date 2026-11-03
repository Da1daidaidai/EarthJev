"""earthjev.data: the shipped question sets, the frozen question lists, the image list, image paths and shards."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import PurePosixPath

import pytest

from earthjev import data, paths

SIZES = {'mme_rs': 3738, 'lrs_vqa': 7333, 'xlrs_lite': 3080, 'lrs_gro': 9734, 'rshr': 1154, 'gro_train': 3511,
         'lrsrefdev_ok': 200, 'lrsrefdev_bad': 200, 'grorefdev_ok': 2049, 'grorefdev_bad': 2049}
FOLDER = {'mme_rs': 'MME-RealWorld/images/', 'xlrs_lite': 'XLRS-Bench-lite/images/', 'rshr': 'RSHR-Bench/images/',
          'lrs_vqa': 'LRS-VQA/LRS_VQA/image/', 'lrs_gro': 'LRS-GRO/image/'}  # docs/DATA.md
FOLDER.update(dict.fromkeys(('lrsrefdev_ok', 'lrsrefdev_bad'), FOLDER['lrs_vqa']))
FOLDER.update(dict.fromkeys(('gro_train', 'grorefdev_ok', 'grorefdev_bad'), FOLDER['lrs_gro']))
KEYS = {'qid', 'bench', 'question', 'stem', 'options', 'choice_options', 'ground_truth', 'category', 'source',
        'hbox', 'image_size', 'is_mcq', 'is_yes_no'}


def images(row):
    return row['images'] if 'images' in row else [row['image']]


@pytest.mark.parametrize('name', SIZES)
def test_rows(name):
    rows = data.load_rows(name)
    assert len(rows) == len({r['qid'] for r in rows}) == SIZES[name]
    mcq = name in ('mme_rs', 'xlrs_lite', 'rshr')
    for r in rows:
        assert r['bench'] == name and KEYS <= set(r) and r['is_mcq'] == mcq
        assert r['question'].startswith(r['stem']) and r['stem']
        assert all(p.startswith(FOLDER[name]) and '..' not in PurePosixPath(p).parts for p in images(r))
        if mcq:
            assert len(r['options']) == len(r['choice_options']) >= 2
            assert set(r['ground_truth']) <= {o['id'] for o in r['options']}


def test_benchmark_details():
    xlrs = data.load_rows('xlrs_lite')
    assert sum(len(images(r)) == 2 for r in xlrs) == 60 and sum(r['multi_choice'] for r in xlrs) == 100
    assert Counter(r['category'] for r in data.load_rows('mme_rs')) == {'color': 1255, 'count': 1226, 'position': 1257}
    assert Counter(r['source'] for r in data.load_rows('lrs_vqa')) == {'FAIR': 2272, 'GLH-Bridge': 1062, 'STAR': 3999}
    assert Counter(r['source'] for r in data.load_rows('lrs_gro')) == {'global': 2328, 'region': 2296, 'object': 5110}
    assert all(r['hbox'] is None or len(r['hbox']) == 4 for r in data.load_rows('lrs_vqa'))
    # RSHR-Bench's two-image questions in the subset get their first image only (docs/DATA.md)
    two = [r for r in data.load_rows('rshr') if r['category'] == 'future_prediction_two_image']
    assert [r['qid'] for r in two] == [f'RSHR__{i}' for i in (361, 363, 365, 368, 390)]
    assert all('image' in r and 'images' not in r for r in two)
    gro = data.load_rows('lrs_gro') + data.load_rows('gro_train')
    assert all(r['ground_truth'] == r['ground_truth'].strip() for r in gro)  # stripped references (docs/DATA.md)


def test_splits():
    for b in data.BENCHMARKS:
        assert data.load_split(f'{b}_full') == [r['qid'] for r in data.load_rows(b)]
    for b, n_cat, per_cat in (('mme_rs', 3, 30), ('lrs_vqa', 8, 30), ('xlrs_lite', 13, 10)):  # calibration questions
        cat = {r['qid']: r['category'] for r in data.load_rows(b)}
        dev = data.load_split(f'{b}_dev')
        assert Counter(Counter(cat[q] for q in dev).values()) == {per_cat: n_cat}
    for family, base in (('lrsrefdev', set(data.load_split('lrs_vqa_dev'))),  # the pointer calibration sets
                         ('grorefdev', {r['qid'] for r in data.load_rows('gro_train')})):
        ok, bad = data.load_rows(f'{family}_ok'), data.load_rows(f'{family}_bad')
        assert [r['qid'] for r in ok] == [r['qid'] for r in bad] and {r['qid'] for r in ok} <= base
    with pytest.raises(FileNotFoundError):
        data.load_split('mme_rs_missing')


def test_image_paths(monkeypatch, tmp_path):
    monkeypatch.setenv('EARTHJEV_DATA', str(tmp_path))
    row = data.load_rows('mme_rs')[0]
    assert data.image_paths(row) == [tmp_path / row['image']]
    two = next(r for r in data.load_rows('xlrs_lite') if 'images' in r)
    assert data.image_paths(two) == [tmp_path / p for p in two['images']]
    monkeypatch.delenv('EARTHJEV_DATA')
    with pytest.raises(RuntimeError):
        data.image_paths(row)


def test_shard():
    rows = data.load_rows('xlrs_lite')
    key = [(images(r)[0], r['qid']) for r in rows]
    for n in (1, 3, 7):
        parts = [data.shard(rows, i, n) for i in range(1, n + 1)]
        assert [(images(r)[0], r['qid']) for p in parts for r in p] == sorted(key)
        assert max(map(len, parts)) - min(map(len, parts)) <= 1
    for i, n in ((0, 3), (4, 3), (1, 0)):
        with pytest.raises(ValueError):
            data.shard(rows, i, n)


def test_read_runs(tmp_path, capsys):
    """Several run files: independent of their order; an error-free record is kept over one with an error; a question
    in several files is reported with them; records of another configuration are refused."""
    rec = lambda q, error=None, **k: json.dumps(dict(qid=q, bench='mme_rs', error=error, final=k.get('final'),
                                                     **({'config': k['config']} if 'config' in k else {}))) + '\n'
    a, b = tmp_path / 'a_1of2.jsonl', tmp_path / 'b_1of1.jsonl'
    a.write_text(rec('q1', final='A') + rec('q2', 'ValueError: x'))
    b.write_text(rec('q1', 'ValueError: y') + rec('q2', final='B') + rec('q3', final='C', config='earthjev-x'))
    for files in ([a, b], [b, a]):
        runs = data.read_runs(files, 'earthjev-x')['mme_rs']
        assert {q: r['final'] for q, r in runs.items()} == {'q1': 'A', 'q2': 'B', 'q3': 'C'}
        assert f'2 questions have several records, in {a}, {b}' in capsys.readouterr().err
    with pytest.raises(ValueError):
        data.read_runs([a, b], 'earthjev-y')


def manifest():
    with open(data.DATA / 'images.tsv', encoding='ascii', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def test_image_manifest():
    files = manifest()
    # every image file of every question file, once (gro_train's images are not in lrs_gro)
    assert [f['path'] for f in files] == sorted({p for name in SIZES for r in data.load_rows(name) for p in images(r)})
    assert all(len(f['sha256']) == 64 and int(f['bytes']) > 0 for f in files)
    size = {f['path']: [int(f['width']), int(f['height'])] for f in files}
    # MME-RealWorld images above 2^24 pixels are downscaled (docs/DATA.md)
    assert all(w * h <= 2 ** 24 for p, (w, h) in size.items() if p.startswith('MME-RealWorld/'))
    for name in SIZES:  # the image sizes stated by the rows
        assert all(size[r['image']] == r['image_size'] for r in data.load_rows(name) if r.get('image_size'))


@pytest.mark.data
@pytest.mark.skipif(not os.environ.get('EARTHJEV_DATA'), reason='needs EARTHJEV_DATA')
@pytest.mark.parametrize('folder', ['MME-RealWorld', 'LRS-VQA', 'XLRS-Bench-lite', 'LRS-GRO', 'RSHR-Bench'])
def test_images_match_manifest(folder):
    """Every image file of the folder has the size and sha256 listed in data/images.tsv (reads every file)."""
    bad = []
    for f in manifest():
        if f['path'].split('/')[0] != folder:
            continue
        p = paths.data_root() / f['path']
        if not p.is_file():
            bad.append(f"{f['path']}: missing")
        elif p.stat().st_size != int(f['bytes']):
            bad.append(f"{f['path']}: {p.stat().st_size} bytes, expected {f['bytes']} ({f['width']}x{f['height']} px)")
        else:
            with open(p, 'rb') as fh:
                if hashlib.file_digest(fh, 'sha256').hexdigest() != f['sha256']:
                    bad.append(f"{f['path']}: sha256 differs")
    assert not bad, f'{len(bad)} image files differ from data/images.tsv (docs/DATA.md), e.g. ' + '; '.join(bad[:3])
