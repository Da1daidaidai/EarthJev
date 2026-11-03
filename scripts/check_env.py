"""Check that this environment matches the one behind the paper's numbers (docs/INSTALL.md, "Verification").

    python scripts/check_env.py                 # packages, GPU and kernels, NLTK WordNet, model files, data files
    python scripts/check_env.py --hash-model    # also the sha256 of every model file (19.3 GB; Qwen3-VL-8B 17.5 GB)
    python scripts/check_env.py --cpu-only      # GPU parts only warn (the default when no CUDA GPU is visible)

Exits 1 when something is wrong. The GPU parts (CUDA build of torch, flash-attn, causal-conv1d, flash-linear-attention,
a tiny kernel call) are errors only when a CUDA GPU is visible and --cpu-only is not given. Missing models or data are
warnings: they are needed only by the steps that read them. Image contents are checked by
`python scripts/prepare_data.py verify` (sha256 of every file, docs/DATA.md).
"""
import argparse
import csv
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = (3, 12, 13)
GPU_ONLY = re.compile(r'^(nvidia-.*|triton|cuda-.*|flash-attn|causal-conv1d|flash-linear-attention|fla-core|einops|'
                      r'ninja)$')
KERNELS = {'flash_attn': 'flash-attn', 'causal_conv1d': 'causal-conv1d', 'fla': 'flash-linear-attention'}
WORDNET_SHA256 = 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59'  # nltk_data wordnet.zip (3.0)
OMW_SHA256 = '3b941e664852f3297b6040236626065796a2aaf7d7f9eec8779a3beaa1096c2d'  # omw-1.4.zip (not needed)
MODEL_DIR = 'Qwen3.5-9B'  # Qwen/Qwen3.5-9B at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a: (bytes, sha256)
MODEL_FILES = {
    '.gitattributes': (1570, '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930'),
    'LICENSE': (11544, 'bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a'),
    'README.md': (77643, 'c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0'),
    'chat_template.jinja': (7756, 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'),
    'config.json': (3126, 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'),
    'merges.txt': (3353259, 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d'),
    'model.safetensors-00001-of-00004.safetensors':
        (5276436216, 'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e'),
    'model.safetensors-00002-of-00004.safetensors':
        (5335161512, '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330'),
    'model.safetensors-00003-of-00004.safetensors':
        (5368717440, '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec'),
    'model.safetensors-00004-of-00004.safetensors':
        (3325995712, 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6'),
    'model.safetensors.index.json': (79657, '26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'),
    'preprocessor_config.json': (390, '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516'),
    'tokenizer.json': (12807982, '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
    'tokenizer_config.json': (16710, '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'),
    'video_preprocessor_config.json': (385, '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13'),
    'vocab.json': (6722759, 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'),
}
VL_MODEL_DIR = 'Qwen3-VL-8B-Instruct'  # Qwen/Qwen3-VL-8B-Instruct at revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
VL_MODEL_FILES = {
    '.gitattributes': (1519, '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361'),
    'README.md': (7133, '6d5d06e0c3f069097002445d30dce9ee107db3afaf15563f09e6df49b8dcb4d7'),
    'chat_template.json': (5499, '5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce'),
    'config.json': (1474, '5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661'),
    'generation_config.json': (269, '8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44'),
    'merges.txt': (1671839, '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'),
    'model-00001-of-00004.safetensors':
        (4902275944, 'd5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa'),
    'model-00002-of-00004.safetensors':
        (4915962496, '8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5'),
    'model-00003-of-00004.safetensors':
        (4999831048, '83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192'),
    'model-00004-of-00004.safetensors':
        (2716270024, '0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5'),
    'model.safetensors.index.json': (67759, '520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070'),
    'preprocessor_config.json': (390, '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516'),
    'tokenizer.json': (7032403, 'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7'),
    'tokenizer_config.json': (10868, 'c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399'),
    'video_preprocessor_config.json': (385, '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13'),
    'vocab.json': (2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
}
MODELS = {MODEL_DIR: ('c202236...', MODEL_FILES, 'qwen35-9b'),  # directory: (revision, files, configuration)
          VL_MODEL_DIR: ('0c351dd...', VL_MODEL_FILES, 'qwen3-vl-8b')}
errors, warnings = [], []
CPU_ONLY = True  # set in main()


def ok(msg):
    print(f'  ok    {msg}')


def bad(msg, gpu=False, warn=False):
    """An error, or a warning when warn is set or a GPU part is checked in CPU-only mode."""
    warn = warn or (gpu and CPU_ONLY)
    (warnings if warn else errors).append(msg)
    print(f'  {"WARN " if warn else "ERROR"} {msg}')


def pins(path):
    """name -> version of every requirement in a requirements file (following -r includes)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip() if ' @ ' not in line else line.strip()
        if line.startswith('-r '):
            out.update(pins(path.parent / line[3:].strip()))
        elif '==' in line:
            name, version = line.split('==')
            out[name.strip()] = version.strip()
        elif ' @ ' in line:  # name @ URL of a wheel: the version is in the file name
            name, url = line.split(' @ ')
            wheel = urllib.parse.unquote(url.split('#')[0].rsplit('/', 1)[1])
            out[name.strip()] = wheel.split('-')[1]
    return out


def public(version):
    return version.split('+')[0]


def check_packages():
    print('packages (requirements/*.txt)')
    v = sys.version_info[:3]
    if v[:2] != PYTHON[:2]:
        bad(f'Python {".".join(map(str, v))}: needs CPython 3.12 (3.12.13)')
    elif v != PYTHON:
        bad(f'Python {".".join(map(str, v))}: the pins were made with 3.12.13', warn=True)
    else:
        ok(f'Python {sys.version.split()[0]} ({sys.executable})')
    want = pins(REPO / 'requirements' / 'gpu.txt')
    n = m = 0
    for name, version in sorted(want.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        gpu = bool(GPU_ONLY.match(name))
        if have is None and gpu and CPU_ONLY:
            continue  # the CPU stack has no GPU packages
        m += 1
        if have is None:
            bad(f'{name} is not installed (want {version})', gpu=gpu)
        elif public(have) != public(version):
            bad(f'{name} {have} installed, want {version}', gpu=gpu)
        else:
            n += 1
    msg = f'{n} of {m} pinned packages at their pinned versions'
    ok(msg) if n == m else print(f'  info  {msg}')


def check_torch():
    print('torch, GPU and kernels')
    try:
        import torch
    except ImportError as e:
        bad(f'cannot import torch ({e}): install requirements/cpu.txt or gpu.txt (docs/INSTALL.md, section 3)')
        return
    cuda = torch.version.cuda
    ok(f'torch {torch.__version__} (CUDA runtime {cuda}), {torch.get_num_threads()} CPU threads '
       f'(OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS", "unset")})')
    if cuda is None and CPU_ONLY:
        print('  info  CPU build of torch: GPU checks skipped (live inference needs requirements/gpu.txt)')
        return
    if cuda != '12.8':
        bad(f'torch was built for CUDA {cuda}; live inference needs the CUDA 12.8 build (requirements/gpu.txt)',
            gpu=True)
    if not torch.cuda.is_available():
        bad('no CUDA GPU visible (fine for the CPU parts: tests, evaluation, CPU training)', gpu=True)
    else:
        p = torch.cuda.get_device_properties(0)
        try:
            driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True).stdout.split('\n')[0] or 'unknown'
        except OSError:  # no nvidia-smi on PATH
            driver = 'unknown'
        ok(f'GPU 0: {p.name}, compute capability {p.major}.{p.minor}, {p.total_memory / 2**30:.1f} GiB, '
           f'driver {driver}')
        if (p.major, p.minor) < (8, 0):
            bad('flash-attn 2 needs compute capability 8.0 or newer (Ampere, Ada, Hopper)', gpu=True)
        if p.total_memory < 32 * 10**9:
            bad('less than 32 GB of GPU memory: EarthJev with Qwen3.5-9B peaks near 26 GB', warn=True)
    for module, dist in KERNELS.items():
        try:
            __import__(module)
            ok(f'import {module} ({dist} {md.version(dist)})')
        except Exception as e:  # ImportError, or an OSError from a kernel library (e.g. GLIBC_2.32 not found)
            bad(f'import {module} failed: {type(e).__name__}: {str(e)[:200]}', gpu=True)
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        if q.is_fast_path_available:
            ok('transformers uses the causal-conv1d / flash-linear-attention fast path of Qwen3.5')
        else:
            bad('transformers falls back to the slower torch path of Qwen3.5 (different numerics)', gpu=True)
    except Exception as e:
        bad(f'transformers Qwen3.5 import failed: {type(e).__name__}: {e}')
    if torch.cuda.is_available() and not CPU_ONLY:
        tiny_kernel_calls(torch)


def tiny_kernel_calls(torch):
    """One flash-attn and one causal-conv1d call on tiny tensors, against torch references."""
    try:
        from causal_conv1d import causal_conv1d_fn
        from flash_attn import flash_attn_func
        g = torch.Generator(device='cuda').manual_seed(0)
        q, k, v = (torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16, generator=g) for _ in range(3))
        out = flash_attn_func(q, k, v, causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)),
                                                               is_causal=True).transpose(1, 2)
        x = torch.randn(1, 8, 32, device='cuda', dtype=torch.bfloat16, generator=g)
        w = torch.randn(8, 4, device='cuda', dtype=torch.bfloat16, generator=g)
        conv = causal_conv1d_fn(x, w, None)
        conv_ref = torch.nn.functional.conv1d(x.float(), w.float()[:, None], padding=3, groups=8)[..., :32]
        torch.cuda.synchronize()
        da, dc = (out - ref).abs().max().item(), (conv.float() - conv_ref).abs().max().item()
        if da < 2e-2 and dc < 1e-1:
            ok(f'flash-attn and causal-conv1d run on the GPU (max diff to torch: {da:.1e}, {dc:.1e})')
        else:
            bad(f'GPU kernels disagree with torch: flash-attn {da:.3g}, causal-conv1d {dc:.3g}', gpu=True)
    except Exception as e:
        bad(f'GPU kernel call failed: {type(e).__name__}: {str(e)[:200]}', gpu=True)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_nltk():
    print('NLTK WordNet (official LRS-VQA and LRS-GRO scorers)')
    try:
        import nltk
    except ImportError as e:
        bad(f'cannot import nltk ({e}): install requirements/cpu.txt (docs/INSTALL.md, section 3)')
        return
    zip_path = lambda name: Path(nltk.data.find(f'corpora/{name}.zip').zipfile.filename)
    try:
        path = zip_path('wordnet')
    except LookupError:
        bad(f'corpora/wordnet.zip not found in {nltk.data.path} (python -m nltk.downloader -d "$NLTK_DATA" wordnet; '
            'a symlinked zip is not found, copy it)')
        return
    if sha256(path) != WORDNET_SHA256:
        bad(f'{path}: sha256 differs from WordNet 3.0 of nltk_data ({WORDNET_SHA256[:12]}...)')
    else:
        from nltk.corpus import wordnet
        ok(f'{path} (WordNet {wordnet.get_version()}, sha256 {WORDNET_SHA256[:12]}...)')
    try:
        omw = zip_path('omw-1.4')
        ok(f'{omw} (sha256 {"as pinned" if sha256(omw) == OMW_SHA256 else "DIFFERS (unused by the scorers)"})')
    except LookupError:
        print('  info  omw-1.4 not installed (not used by the scorers)')


def check_model(hash_files):
    print('answering models')
    root = os.environ.get('EARTHJEV_MODELS')
    if not root:
        bad('EARTHJEV_MODELS is not set: model tests and live runs need $EARTHJEV_MODELS/Qwen3.5-9B', warn=True)
        return
    for name, (revision, files, config) in MODELS.items():
        d = Path(root) / name
        if not d.is_dir():
            if name == MODEL_DIR:
                bad(f'{d} does not exist (docs/INSTALL.md, "6. The answering model")', warn=True)
            else:  # needed only for the configuration that uses it
                print(f'  info  {d} does not exist (needed only for configs/{config}.json)')
            continue
        extra = sorted(p.name for p in d.iterdir() if p.name not in files and p.name != '.cache')
        if extra:  # e.g. a generation_config.json would change the decoding defaults
            bad(f'{d}: files not in revision {revision}: {extra}')
        good = 0
        tests_only = name != MODEL_DIR and not any(d.glob('*.safetensors'))  # the second model's processor files only
        for fname, (size, digest) in files.items():
            p = d / fname
            if not p.is_file():  # no weights are needed for CPU-only work (docs/INSTALL.md, "6. The answering model")
                weights = fname.endswith('.safetensors')
                bad(f'{p} is missing', gpu=weights, warn=weights and tests_only)
            elif p.stat().st_size != size:
                bad(f'{p}: {p.stat().st_size} bytes, want {size}')
            elif hash_files and sha256(p) != digest:
                bad(f'{p}: sha256 differs from revision {revision}')
            else:
                good += 1
        msg = f'{d}: {good} of {len(files)} files of revision {revision} match by size' + (
            ' and sha256' if hash_files else '')
        ok(msg) if good == len(files) else print(f'  info  {msg}')


def check_data():
    print('benchmark images (data/images.tsv)')
    root = os.environ.get('EARTHJEV_DATA')
    if not root:
        bad('EARTHJEV_DATA is not set: data tests and live runs need the images (docs/DATA.md)', warn=True)
        return
    count = {}
    with open(REPO / 'data' / 'images.tsv', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            folder = row['path'].split('/')[0]
            p = Path(root) / row['path']
            n = count.setdefault(folder, [0, 0, 0])
            n[0] += 1
            n[1] += p.is_file() and p.stat().st_size == int(row['bytes'])
            n[2] += p.is_file() and p.stat().st_size != int(row['bytes'])
    for folder, (total, good, wrong) in count.items():
        if good == total:
            ok(f'{folder}: {total} files with the listed sizes')
        else:
            bad(f'{folder}: {good} of {total} files present with the listed size, {wrong} with another size '
                '(docs/DATA.md)', warn=True)


def main():
    global CPU_ONLY
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cpu-only', action='store_true', help='GPU parts only warn')
    ap.add_argument('--hash-model', action='store_true', help='sha256 of every model file')
    args = ap.parse_args()
    try:
        import torch
        CPU_ONLY = args.cpu_only or not torch.cuda.is_available()
    except ImportError:
        CPU_ONLY = True
    print(f'EarthJev environment check ({"CPU only: GPU parts warn" if CPU_ONLY else "GPU"})')
    for k in ('EARTHJEV_DATA', 'EARTHJEV_MODELS', 'EARTHJEV_OUT', 'NLTK_DATA', 'HF_HOME', 'HF_HUB_OFFLINE',
              'CUDA_VISIBLE_DEVICES'):
        print(f'  {k}={os.environ.get(k, "")}')
    check_packages()
    check_torch()
    check_nltk()
    check_model(args.hash_model)
    check_data()
    print(f'{len(errors)} error(s), {len(warnings)} warning(s)')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
