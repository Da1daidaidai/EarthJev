"""Download the benchmark images from their official sources and arrange them under $EARTHJEV_DATA (docs/DATA.md).

    python scripts/prepare_data.py mme_rs xlrs_lite lrs_vqa lrs_gro rshr   # download, convert, verify
    python scripts/prepare_data.py verify                                  # every image vs data/images.tsv
    python scripts/prepare_data.py zwz                                     # training data only (48 GB)

Downloads go to --src (default $EARTHJEV_DATA/sources; they can be deleted afterwards). Each benchmark is fetched
at a pinned revision, converted (MME-RealWorld: images above 2^24 pixels downscaled, see mme_image; XLRS-Bench-lite
and RSHR-Bench: the embedded image bytes; LRS-VQA: the 7z archive, which needs 7-Zip; LRS-GRO: its images are the
LRS-VQA files of the same name, linked by default), and every written file is checked against the size and sha256 in
data/images.tsv. Interrupted runs resume: files already present with the listed size are kept.
--no-download uses what is already in --src; --limit N writes only N files per benchmark (a quick trial).
"""
import argparse
import base64
import csv
import hashlib
import io
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[1]
RSHR_SHARDS = [*range(0, 32), 33, 34, 38, *range(47, 145)]  # the 133 of 197 parquet files that hold our questions
SOURCES = {  # name: (Hugging Face dataset, revision, files used)
    'mme_rs': ('yifanzhang114/MME-RealWorld-Lmms-eval', 'f157390129e8d2b60757eb68d340aade9446b52b', ['data/*.parquet']),
    'xlrs_lite': ('initiacms/XLRS-Bench-lite', 'e540ee2aa745ce9a83784ae76541ddb7f79f03ac', ['train/*']),
    'lrs_vqa': ('ll-13/LRS-VQA', 'f7c0f5f5a28666dddaad5ca2a09b1cec39984d79', ['LRS_VQA.7z.*']),
    'lrs_gro': ('HappyBug/LRS-GRO', '869acdcdd13eb9b6687871c96101fe431d202dec', ['images.tar.part-*']),
    'rshr': ('RL-MIND/RSHR-Bench', '68674294ed5c0bbba0896404c1dec255f6e82e8c',
             [f'data/test-{i:05d}-of-00197.parquet' for i in RSHR_SHARDS]),
    'zwz': ('inclusionAI/ZwZ-RL-VQA', 'fec404cb7bd5a18458e5add2802a5fb183981d14',
            ['train.parquet', 'original_images/*']),
}
FOLDER = {'mme_rs': 'MME-RealWorld', 'xlrs_lite': 'XLRS-Bench-lite', 'lrs_vqa': 'LRS-VQA', 'lrs_gro': 'LRS-GRO',
          'rshr': 'RSHR-Bench'}
ZWZ_SHA256 = {  # not in data/images.tsv: checked as downloaded (the values of earthjev/train/zwz.py)
    'train.parquet': '89879ba2a6f8c370117f6259ff844d382b5e1ac83b63d8dee94cc219313661a3',
    'original_images/original_images.tar.gz00': '64fd86333bbd07ba495dc7253a0a44672e01df28830144553d8279361a4e3ea0',
    'original_images/original_images.tar.gz01': 'bfb62c4266a99ad3a1a3a17b62dfe7d888cac77ddbe1309028ca61134b17bead',
    'original_images/original_images.tar.gz02': '3632252b59d7f4222b89cfa7a85e0d0c525cc084dd412b1edf01d72f90fee82f',
    'original_images/original_images.tar.gz03': 'fbd2cc806a14adce169d7d60cb6b067e005d0c1c4cf4a1c9d2c01fd06a34b59e',
    'original_images/original_images.tar.gz04': '4100d674dcfc34dcd639915bfe132893185113017b0f07c0c25ab0898747f555',
}


def mme_size(w, h):
    """Target size of an MME-RealWorld image above 2^24 pixels: qwen_vl_utils 0.0.14 smart_resize(h, w, factor=32,
    min_pixels=65536, max_pixels=2 ** 24), both sides multiples of 32."""
    nw, nh = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    if nw * nh > 2 ** 24:
        beta = math.sqrt(w * h / 2 ** 24)
        nw, nh = math.floor(w / beta / 32) * 32, math.floor(h / beta / 32) * 32
    return nw, nh


def mme_image(data):
    """The image file for the base64-decoded `bytes` of an MME-RealWorld parquet row: unchanged up to 2^24 pixels,
    else bicubic-downscaled to mme_size and saved as JPEG quality 95 (the files all our runs read)."""
    im = Image.open(io.BytesIO(data))
    if im.width * im.height <= 2 ** 24:
        return data
    out = io.BytesIO()
    im.convert('RGB').resize(mme_size(*im.size), Image.BICUBIC).save(out, 'JPEG', quality=95)
    return out.getvalue()


def manifest(folder=None):
    """data/images.tsv: path -> (bytes, sha256), optionally for one folder."""
    with open(REPO / 'data' / 'images.tsv', newline='') as f:
        return {r['path']: (int(r['bytes']), r['sha256']) for r in csv.DictReader(f, delimiter='\t')
                if folder is None or r['path'].split('/')[0] == folder}


class Writer:
    """Writes the listed files of one folder under the data root; skips files already present with the listed size."""

    def __init__(self, root, want, limit=None):
        self.root, self.want, self.limit, self.n = root, want, limit, 0

    def needed(self, rel):
        p = self.root / rel
        return rel in self.want and not (p.is_file() and p.stat().st_size == self.want[rel][0])

    def full(self):
        return self.limit is not None and self.n >= self.limit

    def write(self, rel, data):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + '.part')
        tmp.write_bytes(data)
        tmp.replace(p)
        self.n += 1


def convert_mme(src, w):
    import pyarrow.parquet as pq
    for f in sorted((src / 'data').glob('*.parquet')):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=8, columns=['index', 'bytes']):
            for row in batch.to_pylist():
                rel = f"MME-RealWorld/images/{row['index']:06d}.jpg"
                if w.needed(rel):
                    w.write(rel, mme_image(base64.b64decode(row['bytes'])))
                if w.full():
                    return


def convert_xlrs(src, w):
    """The datasets dump: Arrow IPC streams whose 'image' column lists the images of a question; one file per
    distinct image, named by the md5 of its bytes, written unchanged."""
    import pyarrow as pa
    for f in sorted((src / 'train').glob('*.arrow')):
        with pa.memory_map(str(f)) as m:
            for batch in pa.ipc.open_stream(m):
                for images in batch.column('image').to_pylist():
                    for im in images:
                        rel = f"XLRS-Bench-lite/images/{hashlib.md5(im['bytes']).hexdigest()}.jpg"
                        if w.needed(rel):
                            w.write(rel, im['bytes'])
                        if w.full():
                            return


def convert_rshr(src, w):
    """The image_1 bytes of each question, written unchanged as rshr_<id>.png or .jpg (the name listed in
    data/images.tsv)."""
    import pyarrow.parquet as pq
    by_id = {int(rel.rsplit('_', 1)[1].split('.')[0]): rel for rel in w.want}
    for f in sorted((src / 'data').glob('*.parquet')):  # a missing shard shows as missing files in verify
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1, columns=['id', 'image_1']):
            for row in batch.to_pylist():
                rel = by_id.get(row['id'])
                if rel and w.needed(rel):
                    w.write(rel, row['image_1']['bytes'])
                if w.full():
                    return


def seven_zip():
    exe = shutil.which('7zz') or shutil.which('7z') or shutil.which('7za')
    if not exe:
        sys.exit('LRS-VQA needs 7-Zip on PATH (7zz, 7z or 7za; docs/DATA.md)')
    return exe


def convert_lrs_vqa(src, w):
    """7z x LRS_VQA.7z.001 into LRS-VQA/ gives LRS-VQA/LRS_VQA/image/<name>; existing files are kept (-aos), after the
    listed ones with another size (cut short by an interrupted run) are deleted."""
    exe, archive = seven_zip(), src / 'LRS_VQA.7z.001'
    for rel, (size, _) in w.want.items():
        p = w.root / rel
        if p.is_file() and p.stat().st_size != size:
            p.unlink()
    names = []
    if w.limit is not None:  # the first files of the solid archive, which extract fastest
        listing = subprocess.run([exe, 'l', '-slt', str(archive)], capture_output=True, text=True, check=True).stdout
        names = [line[7:] for line in listing.splitlines() if line.startswith('Path = LRS_VQA/image/')][:w.limit]
    subprocess.run([exe, 'x', '-aos', f'-o{w.root / "LRS-VQA"}', str(archive), *names], check=True,
                   stdout=subprocess.DEVNULL)


def untar(parts, dest, limit=None):
    """cat part1 part2 ... | tar -x, in Python with the safe 'data' filter (untrusted archives)."""
    cat = subprocess.Popen(['cat', *map(str, parts)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=cat.stdout, mode='r|') as tf:
            n = 0
            for member in tf:
                tf.extract(member, dest, filter='data')
                n += member.isfile()
                if limit is not None and n >= limit:
                    break
    finally:
        cat.kill()
        cat.wait()


def convert_lrs_gro(src, w, mode):
    """LRS-GRO/image: a link to the identical LRS-VQA images (default), or the official archive extracted."""
    dest = w.root / 'LRS-GRO' / 'image'
    if mode == 'link':
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() and not dest.exists():  # dangling: e.g. linked before the LRS-VQA images were there
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(Path('..') / 'LRS-VQA' / 'LRS_VQA' / 'image')
    else:
        untar(sorted(src.glob('images.tar.part-*')), w.root / 'LRS-GRO', w.limit)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(root, folder, partial=False):
    """Size and sha256 of every listed file of a folder. partial: missing files are not errors (--limit)."""
    want = manifest(folder)
    good = missing = 0
    wrong = []
    for rel, (size, digest) in want.items():
        p = root / rel
        if not p.is_file():
            missing += 1
        elif p.stat().st_size != size or sha256(p) != digest:
            wrong.append(rel)
        else:
            good += 1
    print(f'{folder}: {good} of {len(want)} files match data/images.tsv, {missing} missing, {len(wrong)} differ'
          + (f' (e.g. {wrong[:3]})' if wrong else ''), flush=True)
    return not wrong and (partial or not missing)


def download(name, src, offline):
    repo, revision, files = SOURCES[name]
    local = src / repo.split('/')[1]
    if not offline:
        os.environ['HF_HUB_OFFLINE'] = '0'  # the run settings of docs/INSTALL.md set it to 1
        from huggingface_hub import snapshot_download
        print(f'{name}: downloading {repo} at {revision} to {local}', flush=True)
        snapshot_download(repo, repo_type='dataset', revision=revision, local_dir=local, allow_patterns=files)
    elif not local.is_dir():
        sys.exit(f'{local} is missing (run without --no-download)')
    return local


def prepare(name, src, root, limit, lrs_gro_mode):
    if name == 'zwz':
        dest = root / 'ZwZ-RL-VQA'
        dest.mkdir(parents=True, exist_ok=True)
        for rel, digest in ZWZ_SHA256.items():
            if sha256(src / rel) != digest:
                sys.exit(f'{src / rel}: sha256 differs from the pinned revision')
        shutil.copyfile(src / 'train.parquet', dest / 'train.parquet')
        untar(sorted(src.glob('original_images/original_images.tar.gz0*')), dest, limit)
        print(f'zwz: {dest}/train.parquet and {dest}/original_images/ (sha256 of the downloads checked)')
        return True
    folder = FOLDER[name]
    w = Writer(root, manifest(folder), limit)
    if name == 'lrs_gro':
        convert_lrs_gro(src, w, lrs_gro_mode)
    else:
        convert = {'mme_rs': convert_mme, 'xlrs_lite': convert_xlrs, 'rshr': convert_rshr, 'lrs_vqa': convert_lrs_vqa}
        convert[name](src, w)
    return verify(root, folder, partial=limit is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', nargs='+', choices=[*SOURCES, 'verify'])
    ap.add_argument('--src', type=Path, help='downloads (default $EARTHJEV_DATA/sources)')
    ap.add_argument('--no-download', action='store_true', help='use the files already in --src')
    ap.add_argument('--limit', type=int, help='write at most this many files per benchmark (a trial run)')
    ap.add_argument('--lrs-gro', choices=('link', 'extract'), default='link',
                    help='link LRS-GRO/image to the identical LRS-VQA images (default) or extract its archive')
    args = ap.parse_args()
    if not os.environ.get('EARTHJEV_DATA'):
        sys.exit('set EARTHJEV_DATA to the data root (docs/DATA.md)')
    root = Path(os.environ['EARTHJEV_DATA'])
    if args.what == ['verify']:
        sys.exit(0 if all([verify(root, folder) for folder in FOLDER.values()]) else 1)
    src = args.src or root / 'sources'
    ok = True
    for name in (n for n in SOURCES if n in args.what):  # LRS-VQA before LRS-GRO (link mode)
        link = name == 'lrs_gro' and args.lrs_gro == 'link'
        local = None if link else download(name, src, args.no_download)
        ok &= prepare(name, local, root, args.limit, args.lrs_gro)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
