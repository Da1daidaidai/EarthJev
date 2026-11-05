"""Environment and benchmark conversion tests."""
import base64
import hashlib
import importlib
import importlib.util
import io
import urllib.parse
import pkgutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = script('prepare_data')


def pins(path):
    """name -> version of every requirement in a requirements file."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip() if ' @ ' not in line else line.strip()
        if '==' in line:
            name, version = line.split('==')
            out[name.strip()] = version.strip()
        elif ' @ ' in line:  # name @ URL of a wheel: the version is in the file name
            name, url = line.split(' @ ')
            wheel = urllib.parse.unquote(url.split('#')[0].rsplit('/', 1)[1])
            out[name.strip()] = wheel.split('-')[1]
    return out


def test_everything_imports():
    """Every module of the package and every script imports with the pinned packages (requirements.txt)."""
    import earthjev
    for module in pkgutil.walk_packages(earthjev.__path__, 'earthjev.'):
        importlib.import_module(module.name)
    for path in sorted((REPO / 'scripts').glob('*.py')):
        script(path.stem)


def test_requirement_pins():
    gpu = pins(REPO / 'requirements.txt')
    assert {k: gpu[k].split('+')[0] for k in ('torch', 'torchvision', 'transformers', 'pillow', 'numpy', 'flash-attn',
                                              'causal-conv1d', 'flash-linear-attention')} == {
        'torch': '2.10.0', 'torchvision': '0.25.0', 'transformers': '5.5.0', 'pillow': '12.0.0', 'numpy': '1.26.4',
        'flash-attn': '2.8.1', 'causal-conv1d': '1.6.1', 'flash-linear-attention': '0.4.2'}
    for line in (REPO / 'requirements.txt').read_text().splitlines():  # wheels by URL carry their sha256
        assert ' @ ' not in line or '#sha256=' in line


def test_mme_size_is_smart_resize():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        w, h = (int(x) for x in rng.integers(4097, 30000, 2))
        if w * h > 2 ** 24 and max(w, h) / min(w, h) < 200:
            assert prep.mme_size(w, h) == smart_resize(h, w, 32, 65536, 2 ** 24)[::-1]
    data = jpeg(0)
    assert prep.mme_image(data) is data  # at most 2^24 pixels: the file is written unchanged


def jpeg(seed):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(buf, 'JPEG')
    return buf.getvalue()


def writer(root, files):
    return prep.Writer(root, {rel: len(data) for rel, data in files.items()})


def test_converters(tmp_path):
    pa = pytest.importorskip('pyarrow')  # the data preparation's only extra package
    pq = pytest.importorskip('pyarrow.parquet')
    out = tmp_path / 'out'
    a, b, c = jpeg(1), jpeg(2), jpeg(3)
    image = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    # MME-RealWorld: parquet rows with the base64 image in 'bytes'; the listed index is written
    (tmp_path / 'mme' / 'data').mkdir(parents=True)
    pq.write_table(pa.table({'index': [7, 19871], 'category': ['Remote Sensing/x', 'Remote Sensing/y'],
                             'bytes': [base64.b64encode(x).decode() for x in (a, b)]}),
                   tmp_path / 'mme' / 'data' / 'train-00000-of-00001.parquet')
    prep.convert_mme(tmp_path / 'mme', writer(out, {'MME-RealWorld/images/019871.jpg': b}))
    assert (out / 'MME-RealWorld/images/019871.jpg').read_bytes() == b
    assert not (out / 'MME-RealWorld/images/000007.jpg').exists()
    # XLRS-Bench-lite: Arrow IPC stream, a list of images per question, files named by md5
    (tmp_path / 'xlrs' / 'train').mkdir(parents=True)
    images = pa.array([[{'bytes': a, 'path': None}, {'bytes': c, 'path': None}]], pa.list_(image))
    with pa.OSFile(str(tmp_path / 'xlrs' / 'train' / 'data-00000-of-00001.arrow'), 'wb') as f, \
            pa.ipc.new_stream(f, pa.schema([('image', images.type)])) as s:
        s.write_table(pa.table({'image': images}))
    names = {f'XLRS-Bench-lite/images/{hashlib.md5(x).hexdigest()}.jpg': x for x in (a, c)}
    prep.convert_xlrs(tmp_path / 'xlrs', writer(out, names))
    assert all((out / rel).read_bytes() == x for rel, x in names.items())
    # RSHR-Bench: image_1 of the listed ids, unchanged
    (tmp_path / 'rshr' / 'data').mkdir(parents=True)
    pq.write_table(pa.table({'id': [5, 6], 'image_1': pa.array([{'bytes': a, 'path': 'x'}, {'bytes': c, 'path': 'y'}],
                                                                 image)}),
                   tmp_path / 'rshr' / 'data' / 'test-00000-of-00197.parquet')
    prep.convert_rshr(tmp_path / 'rshr', writer(out, {'RSHR-Bench/images/rshr_6.png': c}))
    assert (out / 'RSHR-Bench/images/rshr_6.png').read_bytes() == c
    assert not (out / 'RSHR-Bench/images/rshr_5.png').exists()
    # LRS-GRO: a relative link to the LRS-VQA images (dangling here, which a second run replaces)
    for _ in range(2):
        prep.convert_lrs_gro(None, writer(out, {}), 'link')
    assert (out / 'LRS-GRO' / 'image').readlink() == Path('../LRS-VQA/LRS_VQA/image')


def test_lrs_vqa_resume(tmp_path, monkeypatch):
    """Before 7-Zip extracts (keeping existing files), listed files with another size, cut short by an interrupted
    run, are deleted."""
    files = {f'LRS-VQA/LRS_VQA/image/{n}.tif': b'x' * 10 for n in 'abc'}
    for rel, size in zip(files, (10, 4, 10)):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b'x' * size)
    calls = []
    monkeypatch.setattr(prep, 'seven_zip', lambda: '7zz')
    monkeypatch.setattr(prep.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    prep.convert_lrs_vqa(tmp_path / 'src', writer(tmp_path, files))
    assert [(tmp_path / rel).exists() for rel in files] == [True, False, True] and calls[0][:3] == ['7zz', 'x', '-aos']


def test_untar_split_archive(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in (('image/a.png', b'a' * 3000), ('image/b.png', b'b' * 5000)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    parts = []
    for i, k in enumerate(range(0, len(raw), 4096)):  # split like images.tar.part-aa, -ab, ...
        parts.append(tmp_path / f'images.tar.part-a{"abcdefgh"[i]}')
        parts[-1].write_bytes(raw[k:k + 4096])
    prep.untar(parts, tmp_path / 'one', limit=1)
    assert sorted(p.name for p in (tmp_path / 'one' / 'image').iterdir()) == ['a.png']
    prep.untar(parts, tmp_path / 'all')
    assert (tmp_path / 'all/image/b.png').read_bytes() == b'b' * 5000
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode='w') as tf:
        info = tarfile.TarInfo('../escape.txt')
        info.size = 1
        tf.addfile(info, io.BytesIO(b'x'))
    (tmp_path / 'evil.tar').write_bytes(evil.getvalue())
    with pytest.raises(tarfile.OutsideDestinationError):
        prep.untar([tmp_path / 'evil.tar'], tmp_path / 'safe')


def test_verify(tmp_path):
    folder = 'XLRS-Bench-lite'
    assert prep.verify(tmp_path, folder, partial=True) and not prep.verify(tmp_path, folder)
    rel, size = next(iter(prep.manifest(folder).items()))
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).touch()
    assert not prep.verify(tmp_path, folder, partial=True)
    with (tmp_path / rel).open('wb') as f:
        f.truncate(size)
    assert prep.verify(tmp_path, folder, partial=True)
