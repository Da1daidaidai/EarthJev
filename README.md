<div align="center">

<h1 align="center">EarthJev: Efficient Visual Evidence Acquisition and Decision for UHR Remote Sensing Understanding</h1>

<p align="center">
  <strong><em>
    Pengyu Dai<sup>1,3,*</sup>, Yao Zhang<sup>2,*</sup>, Jian Song<sup>1</sup>, Junjue Wang<sup>3</sup>, Yafei Ou<sup>1</sup>,<br>
    Yuzeng Chen<sup>3</sup>, Junshi Xia<sup>1</sup>, Hongruixuan Chen<sup>1</sup>, Naoto Yokoya<sup>1,3†</sup>
  </em></strong>
</p>

<p align="center">
  <sup>1</sup> RIKEN AIP &nbsp; <sup>2</sup> Wuhan University &nbsp; <sup>3</sup> The University of Tokyo<br>
  <sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding authors
</p>

[![Paper](https://img.shields.io/badge/Paper-under%20review-lightgrey.svg)](#reference) [![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](#quick-start) [![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-EarthJev-yellow.svg)](https://huggingface.co/daidaidaidai1dai/EarthJev) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C.svg)](requirements.txt) [![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[**Overview**](#overview) | [**Get Started**](#quick-start) | [**Results**](#main-results) | [**Common Issues**](#common-issues) | [**Reference**](#reference) | [**Acknowledgments**](#acknowledgments)

</div>

## Overview

EarthJev answers questions about ultra-high-resolution remote sensing images with a frozen multimodal model. It verifies spatial cues, scores candidate views during hierarchical search, and selects between answers generated from memory and canvas using calibrated confidence.

![EarthJev framework](figures/framework.png)

## Main results

Accuracy (%) on the five full benchmark sets, as reported in the paper. Mean gives equal weight to each benchmark.

| Answering model | MME-RS | LRS-VQA | XLRS-lite | LRS-GRO | RSHR-Bench | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 63.16 | 34.79 | 55.81 | 59.27 | 52.25 | **53.06** |
| Qwen3-VL-8B | 56.39 | 34.97 | 54.38 | 59.40 | 49.48 | **50.92** |

## Quick start

From the repository root on Linux x86_64, use Python 3.12 and an Ampere or newer NVIDIA GPU with at least 32 GB of memory and a CUDA 12.8-compatible driver.

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps -e .

export EARTHJEV_DATA="$PWD/datasets" EARTHJEV_MODELS="$PWD/models" EARTHJEV_OUT="$PWD/outputs"
hf download daidaidaidai1dai/EarthJev --local-dir "$EARTHJEV_MODELS"
python scripts/prepare_data.py mme_rs
mkdir -p "$EARTHJEV_OUT"
python scripts/run_earthjev.py --bench mme_rs --limit 5 --out "$EARTHJEV_OUT/demo.jsonl" --resume
python scripts/evaluate.py "$EARTHJEV_OUT/demo.jsonl"
```

For Qwen3-VL-8B, download `Qwen/Qwen3-VL-8B-Instruct` into `$EARTHJEV_MODELS/Qwen3-VL-8B-Instruct` and pass `--config qwen3-vl-8b` to inference and evaluation.

For CPU-only evaluation, replace the first install command with `sed '/^# GPU inference/,$d' requirements.txt | uv pip install --torch-backend cpu -r -`.

## Training

Prepare ZwZ-RL-VQA with `python scripts/prepare_data.py zwz`. The [training modules](earthjev/train/) build region crops, extract features, train the region head and pointer verifier, and fit the decision parameters. Released configurations are in [configs/](configs/).

## Common issues

Install WordNet for LRS-VQA and LRS-GRO scoring with `python -m nltk.downloader wordnet omw-1.4`. Install 7-Zip before preparing LRS-VQA. In the default linked layout, prepare LRS-VQA before LRS-GRO.

## Reference

The paper is under review. A BibTeX entry will be added after review.

## Acknowledgments

We thank the Qwen, Vision-Jev, VISTA, ZoomSearch, and benchmark teams. See [LICENSE](LICENSE) for EarthJev's license and [NOTICE](NOTICE) for third-party attribution and data terms.
