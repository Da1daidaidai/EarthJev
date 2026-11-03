# Installation

This guide installs the exact software stack behind the paper's numbers on a Linux server, step by step. There are
two stacks:

- **CPU stack** (`requirements/cpu.txt`): the tests, the evaluation (official scorers), the CPU parts of training
  (fitting the decision parameters, the pointer verifier, the region head on cached features) and the data
  preparation. No GPU, no CUDA.
- **GPU stack** (`requirements/gpu.txt`): the CPU stack with the CUDA 12.8 build of torch and the kernels of
  Qwen3.5, for inference and feature extraction.

Both pin every package to the version of the environment that produced the paper's numbers (CPython 3.12.13,
torch 2.10.0, transformers 5.5.0, flash-attn 2.8.1, ...), closed over everything this repository imports.

We followed this guide in fresh virtual environments on a Linux server (glibc 2.34, x86-64 CPU with AVX-512, NVIDIA
H100 GPU): both stacks, `scripts/check_env.py` with a kernel call on the GPU, the CPU and model
tests, WordNet, the model download without the weight files (their sha256 values come from a copy of the same
revision and equal Hugging Face's own checksums) and the data preparation on copies of the official files. Not run
by us in this form: the source build of section 3.4 and the GPU check of section 8, step 5.

## 1. Requirements

| | CPU stack | GPU stack |
|---|---|---|
| OS | Linux x86_64, glibc >= 2.28 (`ldd --version`) | the same, glibc >= 2.32 for the prebuilt flash-attn and causal-conv1d wheels |
| GPU | none | NVIDIA, compute capability >= 8.0 (Ampere, Ada or Hopper: flash-attn 2); the paper's runs used one NVIDIA H100 |
| GPU memory | | >= 32 GB recommended for EarthJev with Qwen3.5-9B (weights 19.3 GB, measured peak about 26 GB) |
| NVIDIA driver | | one that supports CUDA 12.8 (R570 or newer, `nvidia-smi`); no CUDA toolkit is needed unless you build the kernels yourself (section 3.4) |
| Host RAM | 8 GB (the tests peaked at 1.3 GB, the MME-RealWorld conversion at about 6 GB) | 32 GB or more (not measured; one decoded scene reaches 2.4 GB) |
| Disk | 2.5 GB (venv 1.2 GB, uv cache 1.2 GB) | 16 GB (venv 7.8 GB, uv cache 7.8 GB), plus the model (19.3 GB) and the data (below) |
| Tools | `git`, `curl` | the same; 7-Zip for the LRS-VQA archive (docs/DATA.md) |

Data: the benchmark images take 303 GiB, or 236 GiB when LRS-GRO links to the identical LRS-VQA images (the
default of `scripts/prepare_data.py`). The official downloads they are made from take 273 GiB more (they can be
deleted afterwards). docs/DATA.md has the details.

## 2. Code, uv and Python

```bash
git clone <repository URL> EarthJev
cd EarthJev
```

Install uv 0.11.26 (the version used here), then CPython 3.12.13, and create the virtual environment `.venv` in the
repository:

```bash
curl -LsSf https://astral.sh/uv/0.11.26/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version                      # uv 0.11.26
uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate
python --version                  # Python 3.12.13
```

The installer puts `uv` in `~/.local/bin` and adds that directory to your shell profile (`UV_INSTALL_DIR=<dir>`
installs elsewhere, `UV_NO_MODIFY_PATH=1` leaves the profile alone). uv keeps its cache in `~/.cache/uv`; set
`UV_CACHE_DIR` to put it on a larger disk. Every later section assumes the environment is active
(`source .venv/bin/activate`) and the repository is the working directory.

## 3. Packages

Install one of the two stacks into the fresh `.venv`.

### 3.1 CPU stack

```bash
uv pip install --torch-backend cpu -r requirements/cpu.txt
```

`--torch-backend cpu` takes torch and torchvision from the PyTorch CPU index (`2.10.0+cpu`, `0.25.0+cpu`, about
200 MB instead of 3 GB of CUDA libraries); every other package comes from PyPI. Without the flag, uv installs the
CUDA build from PyPI, which also works on a CPU-only machine.

### 3.2 GPU stack

```bash
uv pip install -r requirements/gpu.txt
```

This installs torch 2.10.0 from PyPI (built with the CUDA 12.8 runtime, which comes as `nvidia-*` packages), the
Qwen3.5 kernels `flash-linear-attention` / `fla-core` 0.4.2, and the official prebuilt wheels of flash-attn 2.8.1 and
causal-conv1d 1.6.1 for torch 2.10, CUDA 12 and CPython 3.12, downloaded from the projects' GitHub releases and
checked against the sha256 in the file. These are the same files the paper's environment used.

### 3.3 The repository itself

```bash
uv pip install --no-deps -e .
uv pip check                      # All installed packages are compatible
```

This makes `earthjev` importable from anywhere (an editable install: edits in the repository take effect directly).
It installs nothing else.

### 3.4 Building flash-attn and causal-conv1d from source

Only needed if the prebuilt wheels do not load, e.g. `ImportError: ... GLIBC_2.32 not found` on an older system
(`python scripts/check_env.py` reports it). The build needs the CUDA 12.8 toolkit (`nvcc --version`) and a C++
compiler, and takes from about half an hour (one GPU architecture, many cores) to several hours. Each parallel
compile job of flash-attn needs up to 9 GB of RAM; `MAX_JOBS` limits them. `FLASH_ATTN_CUDA_ARCHS` selects the GPU
architectures (80 = A100, 86/89 = RTX/L40 class are covered by 80, 90 = H100; default `80;90;100;120`).

```bash
export CUDA_HOME=/usr/local/cuda-12.8           # your CUDA 12.8 toolkit
uv pip install wheel==0.45.1                     # imported by both setup.py files
MAX_JOBS=8 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv pip install --no-build-isolation --no-deps --reinstall-package flash-attn --reinstall-package causal-conv1d \
    flash-attn==2.8.1 causal-conv1d==1.6.1
```

We did not run this build for the release. Self-built kernels are different binaries: expect small numerical
differences from the paper's runs, of the size of other GPU or driver changes.

### 3.5 Without the kernels

- **flash-attn** is required: the answerer is loaded with `attn_implementation: flash_attention_2`
  (`configs/qwen35-9b.json`, `answerer`), and transformers stops with an error when flash-attn is missing. A copy
  of that file with `"attn_implementation": "sdpa"`, passed to `scripts/run_earthjev.py` with `--config <file>`,
  runs on any GPU, but its logits differ (by up to 0.25 on a test prompt), and so can the answers. Such runs are not
  the paper's configuration. Of the scripts, only `run_earthjev.py` takes such a copy: `scripts/evaluate.py` and
  `scripts/check_live.py` take the name of the configuration it was copied from (`--config qwen35-9b`); the copy
  keeps that configuration's `name`, which its records carry.
- **causal-conv1d and flash-linear-attention** serve the linear-attention layers of Qwen3.5. When either is missing,
  transformers silently uses a slower torch implementation with different numerics; it logs once "The fast path is
  not available because one of the required library is not installed". `scripts/check_env.py` reports this as an
  error on a GPU machine.

## 4. Environment variables

Choose where the data, the model and the outputs live, and put these lines in a file you source before working (or
in `~/.bashrc`):

```bash
export EARTHJEV_DATA=/data/earthjev/datasets        # benchmark images (docs/DATA.md)
export EARTHJEV_MODELS=/data/earthjev/models        # holds Qwen3.5-9B/ (and Qwen3-VL-8B-Instruct/)
export EARTHJEV_OUT=$HOME/earthjev-outputs          # run outputs; default <repository>/outputs
export NLTK_DATA=/data/earthjev/nltk_data           # WordNet for the official scorers
export HF_HUB_OFFLINE=1                             # runs read only local files
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NUMPY_MADVISE_HUGEPAGE=0                     # no stalls on transparent huge pages (see the table)
```

| variable | used by | notes |
|---|---|---|
| `EARTHJEV_DATA` | inference, `pytest -m data`, `scripts/prepare_data.py` | layout in docs/DATA.md |
| `EARTHJEV_MODELS` | inference; `pytest -m model` and `-m data` (processor files only) | `$EARTHJEV_MODELS/Qwen3.5-9B`, `$EARTHJEV_MODELS/Qwen3-VL-8B-Instruct` |
| `EARTHJEV_OUT` | every script that writes | default `outputs/` in the repository (git-ignored) |
| `NLTK_DATA` | the LRS-VQA and LRS-GRO scorers | NLTK also searches `~/nltk_data` and a few system paths |
| `HF_HOME` | downloads (`hf download`, `scripts/prepare_data.py`) | Hugging Face cache, default `~/.cache/huggingface`; set it if your home is small |
| `HF_HUB_OFFLINE` | everything that loads the model | `1` for runs; the download commands below override it |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | CPU numerics and speed | the paper's runs used 1; the shipped CPU results were checked with 1 thread |
| `NUMPY_MADVISE_HUGEPAGE` | NumPy memory allocation | `0` stops NumPy from asking for transparent huge pages; on our server the kernel's memory compaction for them made one image test take 37 s instead of 10 s. No effect on results; the paper's runs used `0` |
| `CUDA_VISIBLE_DEVICES` | inference | the scripts use the first visible GPU |
| `TRITON_CACHE_DIR` | flash-linear-attention | Triton kernels compile on first use and are cached, default `~/.triton/cache` |

## 5. WordNet for the scorers

The official LRS-VQA and LRS-GRO scorers match answers through WordNet 3.0 (NLTK's `wordnet` package). The LRS-GRO
scorer would silently score synonyms as wrong without it, so `earthjev.scoring` checks for WordNet 3.0 and raises
an error if it is missing.

```bash
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4
sha256sum "$NLTK_DATA"/corpora/wordnet.zip "$NLTK_DATA"/corpora/omw-1.4.zip
```

```
cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59  .../corpora/wordnet.zip
3b941e664852f3297b6040236626065796a2aaf7d7f9eec8779a3beaa1096c2d  .../corpora/omw-1.4.zip
```

These are the files of the paper's environment. `omw-1.4` is not used by the scorers; it is listed because that
environment had it. The `RuntimeWarning` that `python -m nltk.downloader` prints is harmless. Keep the zip files as
they are: NLTK reads them without unpacking, and it does not find a symlinked `wordnet.zip` (copy it instead).

## 6. The answering model

EarthJev uses Qwen3.5-9B at a fixed revision (`configs/qwen35-9b.json`, `answerer`). Download it (19.3 GB):

```bash
HF_HUB_OFFLINE=0 hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --local-dir "$EARTHJEV_MODELS/Qwen3.5-9B"
python scripts/check_env.py --hash-model
```

The check compares every file with the snapshot the paper used (16 files; the sha256 values are in
`scripts/check_env.py`, `MODEL_FILES`) and fails on extra files: this revision has no `generation_config.json`, and
one would change the decoding defaults. The `.cache` folder that `hf download` writes inside the directory is
ignored. The tests (`pytest -m model` and `-m data`) need only the tokenizer and processor files, so for CPU-only
work this is enough:

```bash
HF_HUB_OFFLINE=0 hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --local-dir "$EARTHJEV_MODELS/Qwen3.5-9B" --exclude "*.safetensors"
```

The second configuration, `configs/qwen3-vl-8b.json` (EarthJev with Qwen3-VL-8B, Table IV of the paper), uses
Qwen3-VL-8B-Instruct at a fixed revision (17.5 GB):

```bash
HF_HUB_OFFLINE=0 hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --local-dir "$EARTHJEV_MODELS/Qwen3-VL-8B-Instruct"     # --exclude "*.safetensors" for CPU-only work
python scripts/check_env.py --hash-model
```

`check_env.py` checks its 16 files too (`VL_MODEL_FILES`). This revision has a `generation_config.json`, which belongs
to it: it makes generation stop at `<|im_end|>`, as in the paper's runs. A missing Qwen3-VL-8B directory is only
reported, since only that configuration needs it.

The trained weights of EarthJev itself (the region head and the pointer verifier of each configuration, a few MB)
are part of the repository (`weights/`).

## 7. Benchmark data

See docs/DATA.md: `python scripts/prepare_data.py mme_rs xlrs_lite lrs_vqa lrs_gro rshr` downloads every benchmark
at a pinned revision, converts it to the layout the question files expect, and checks every image file against
`data/images.tsv`. Steps 1-3 of section 8 do not need the images.

## 8. Verification

Run these in order; each step needs a little more of the setup.

```bash
python scripts/check_env.py                      # 1. versions, GPU and kernels, WordNet, model, data
python -m pytest -q -m "not data and not model"  # 2. CPU tests: repository files only
python -m pytest -q -m model                     # 3. needs the models' processor files (no weights)
python -m pytest -q -m data                      # 4. needs $EARTHJEV_DATA (and step 3's files)
```

1. `check_env.py` exits with status 1 on any error. On a machine without a visible CUDA GPU (or with
   `--cpu-only`), the GPU parts only warn; missing models or data are always warnings.
2. The CPU tests check the method's components (pointer verification, region head, search, evidence views and
   presentations, prompts, answer selection) against values recorded from the code behind the paper, the official
   scorers, the data files, and the refit of the decision parameters from `data/calibration`. They take about
   2 minutes on one CPU core with the settings of section 4 (1.4 GB of memory).
3. Runs the answerer, the region head and the whole pipeline on a tiny random model with the real Qwen3.5
   processor and tokenizer, against values recorded from the code behind the paper; the same for Qwen3-VL-8B when
   its tokenizer and processor files are in `$EARTHJEV_MODELS/Qwen3-VL-8B-Instruct` (skipped otherwise).
4. Reads every image file (303 GiB) and runs the pipeline simulation on real benchmark rows;
   `python scripts/prepare_data.py verify` makes the same image check without pytest.

Step 3 skips its tests when the processor files are missing. Step 4 checks the images only when `EARTHJEV_DATA` is
set, and then every missing or different image is a failure: prepare the data first (docs/DATA.md).

5. On the GPU, run EarthJev on a shard of the reference records (`reference/live/<config>`: the original pipeline's
   answers to 1,612 questions on one H100) and compare them field by field:

```bash
python scripts/run_earthjev.py --bench mme_rs --shard 1/12
python scripts/check_live.py "$EARTHJEV_OUT/earthjev_qwen35-9b/mme_rs_full_1of12.jsonl"
```

   The MME-RS shard has 311 questions (about 8 minutes on an H100); `--limit 20` makes a quick trial (then
   continue the shard with `--resume`).
   reference/README.md gives the commands for the other four shards and what agreement to expect on the same and on
   other GPUs.

## 9. What is identical, and where

- **CPU results** (the scorers, the refit of the decision parameters, the components' recorded values) are
  deterministic. On an x86-64 CPU with AVX-512 the refit reproduces `configs/qwen35-9b.json` bitwise. Without
  AVX-512, NumPy's vectorized `exp` and `log` round differently, and 8 refitted values differ in their last 1-3 bits
  (e.g. `2.0995217418433283` against `2.099521741843329`); `python -m earthjev.train.fit_decisions` and
  `tests/test_fit.py` accept differences up to a relative 1e-12 and say which case applies. We checked this with the
  AVX-512 code paths of NumPy and torch switched off:
  `NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL" ATEN_CPU_CAPABILITY=avx2`.
- **GPU results** depend on the GPU, the driver and the kernels: bf16 kernels round differently on other GPUs,
  and a small change of a score can change a decision. Expect most, not all, answers to be identical to the
  reference records; reference/README.md gives the measured agreement.

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"` (or open a new shell) |
| `ImportError: ... GLIBC_2.32 not found` when importing `flash_attn` or `causal_conv1d` | the system is older than the prebuilt wheels: build them (section 3.4) |
| the server cannot reach github.com (the two kernel wheels) | download the two wheels of `requirements/gpu.txt` elsewhere, put them in a directory on the server and replace their `https://...` URLs in that file by `file:///<directory>/<wheel file name>`; the `#sha256=` check stays |
| `FlashAttention2 has been toggled on, but it cannot be used` | flash-attn is missing or does not import: `python -c "import flash_attn"` shows why |
| `The fast path is not available because one of the required library is not installed` | causal-conv1d or flash-linear-attention is missing or broken (section 3.5) |
| `torch.cuda.is_available()` is `False` with a GPU present | the driver is older than CUDA 12.8 needs (`nvidia-smi` shows the highest CUDA version it supports), or `CUDA_VISIBLE_DEVICES` hides the GPU |
| `CUDA out of memory` | EarthJev with Qwen3.5-9B needs about 26 GB of free GPU memory. Check other processes with `nvidia-smi` |
| `LookupError: Resource wordnet not found` | `NLTK_DATA` does not point to the directory of section 5, or `wordnet.zip` is a symlink |
| `RuntimeError: set the environment variable EARTHJEV_DATA` (or `EARTHJEV_MODELS`) | section 4 |
| `hf download` fails with HTTP 429 or a timeout | rerun the same command: it resumes. A Hugging Face token (`hf auth login`) raises the rate limit |
| the uv cache, the Hugging Face cache or the Triton cache fill a small home directory | `UV_CACHE_DIR`, `HF_HOME`, `TRITON_CACHE_DIR` (section 4); `uv cache clean` frees the uv cache after installing |
