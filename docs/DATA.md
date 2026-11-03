# Data

The question files, the frozen question lists and the list of image files are part of this repository. The images
are not: `scripts/prepare_data.py` downloads them from the official sources, at pinned revisions, and arranges
them under one directory, `$EARTHJEV_DATA`. Every image path in the question files is relative to it.

Inference and `pytest -m data` need the images; the other tests and the refit of the decision parameters do not.

## Preparing the images

With the environment of docs/INSTALL.md active and `EARTHJEV_DATA` set:

```bash
python scripts/prepare_data.py mme_rs xlrs_lite lrs_vqa lrs_gro rshr
```

For each benchmark this downloads the official files (to `$EARTHJEV_DATA/sources`, or `--src <dir>`), converts them
to the layout below, and checks every image file against its size and sha256 in `data/images.tsv`. It exits with
status 1 if a file differs. Interrupted runs resume where they stopped. `--limit 3` makes a quick trial (three files
per benchmark), `--no-download` uses files already in the source directory, and
`python scripts/prepare_data.py verify` checks all five folders again (the same check as
`python -m pytest -m data tests/test_data.py`).

| benchmark | download | images | conversion |
|---|---:|---:|---|
| `mme_rs` | 28.8 GiB | 20.1 GiB, 3,738 files | the parquet's JPEG files; those above 2^24 pixels downscaled (below). About an hour on one CPU core |
| `xlrs_lite` | 35.6 GiB | 16.7 GiB, 1,493 files | the embedded image bytes, unchanged |
| `lrs_vqa` | 53.0 GiB | 67.0 GiB, 1,228 files | the 7z archive, extracted; needs 7-Zip (`7zz`, `7z` or `7za` on `PATH`) |
| `lrs_gro` | none | 66.8 GiB, 1,224 files | a link to the identical LRS-VQA images (default); `--lrs-gro extract` downloads (67.0 GiB) and extracts the official archive instead |
| `rshr` | 155.8 GiB | 132.4 GiB, 1,154 files | the embedded image bytes, unchanged; only the 133 of 197 parquet files that hold the subset are downloaded |
| `zwz` (training only) | 44.9 GiB | 45 GiB | `train.parquet` and the original images, extracted; see "ZwZ-RL-VQA" |

7-Zip: on Debian or Ubuntu `apt install 7zip` (older releases: `p7zip-full`), on RHEL or Fedora
`dnf install p7zip p7zip-plugins` (EPEL); without root, the static `7zz` binary in `7z<version>-linux-x64.tar.xz` of
the official releases (github.com/ip7z/7zip/releases; we used 26.04) works from any directory on `PATH`. The
downloads can be deleted once `verify` passes.

### Download links

All sources are Hugging Face datasets, used at the pinned revisions below (the links open the file list at that
revision). `scripts/prepare_data.py` downloads them itself; the table and the commands are for downloading by hand,
e.g. on a machine with internet access before copying `$EARTHJEV_DATA/sources` to the server.

| benchmark | Hugging Face dataset | revision | files used |
|---|---|---|---|
| MME-RealWorld (MME-RS) | [yifanzhang114/MME-RealWorld-Lmms-eval](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lmms-eval/tree/f157390129e8d2b60757eb68d340aade9446b52b) | `f157390129e8` | `data/*.parquet` |
| XLRS-Bench-lite | [initiacms/XLRS-Bench-lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite/tree/e540ee2aa745ce9a83784ae76541ddb7f79f03ac) | `e540ee2aa745` | `train/*` |
| LRS-VQA | [ll-13/LRS-VQA](https://huggingface.co/datasets/ll-13/LRS-VQA/tree/f7c0f5f5a28666dddaad5ca2a09b1cec39984d79) | `f7c0f5f5a286` | `LRS_VQA.7z.*` |
| LRS-GRO | [HappyBug/LRS-GRO](https://huggingface.co/datasets/HappyBug/LRS-GRO/tree/869acdcdd13eb9b6687871c96101fe431d202dec) | `869acdcdd13e` | `images.tar.part-*` (only for `--lrs-gro extract`; by default the identical LRS-VQA images are linked) |
| RSHR-Bench | [RL-MIND/RSHR-Bench](https://huggingface.co/datasets/RL-MIND/RSHR-Bench/tree/68674294ed5c0bbba0896404c1dec255f6e82e8c) | `68674294ed5c` | `data/test-*-of-00197.parquet` (the script fetches the 133 shards that hold the subset) |
| ZwZ-RL-VQA (training only) | [inclusionAI/ZwZ-RL-VQA](https://huggingface.co/datasets/inclusionAI/ZwZ-RL-VQA/tree/fec404cb7bd5a18458e5add2802a5fb183981d14) | `fec404cb7bd5` | `train.parquet`, `original_images/*` |

```bash
export HF_HUB_OFFLINE=0 S=$EARTHJEV_DATA/sources
hf download yifanzhang114/MME-RealWorld-Lmms-eval --repo-type dataset \
    --revision f157390129e8d2b60757eb68d340aade9446b52b --include "data/*.parquet" --local-dir $S/MME-RealWorld-Lmms-eval
hf download initiacms/XLRS-Bench-lite --repo-type dataset \
    --revision e540ee2aa745ce9a83784ae76541ddb7f79f03ac --include "train/*" --local-dir $S/XLRS-Bench-lite
hf download ll-13/LRS-VQA --repo-type dataset \
    --revision f7c0f5f5a28666dddaad5ca2a09b1cec39984d79 --include "LRS_VQA.7z.*" --local-dir $S/LRS-VQA
hf download RL-MIND/RSHR-Bench --repo-type dataset \
    --revision 68674294ed5c0bbba0896404c1dec255f6e82e8c --include "data/*.parquet" --local-dir $S/RSHR-Bench
hf download inclusionAI/ZwZ-RL-VQA --repo-type dataset --revision fec404cb7bd5a18458e5add2802a5fb183981d14 \
    --include "train.parquet" --include "original_images/*" --local-dir $S/ZwZ-RL-VQA     # training only
python scripts/prepare_data.py mme_rs xlrs_lite lrs_vqa lrs_gro rshr --no-download
```

The RSHR-Bench command fetches all 197 shards (210 GiB instead of 155.8 GiB); the conversion reads only the questions
of the subset.

## Question files: `data/benchmarks/<set>.jsonl`

One JSON object per line, in a fixed order. The order matters for `earthjev.data.shard`.

| set | questions | content |
|---|---:|---|
| `mme_rs` | 3,738 | MME-RealWorld, remote-sensing part (MME-RS), single-choice A-E |
| `lrs_vqa` | 7,333 | LRS-VQA, open-ended |
| `xlrs_lite` | 3,080 | XLRS-Bench-lite, multiple-choice in 13 L3 tasks (100 multi-choice, 60 two-image) |
| `lrs_gro` | 9,734 | LRS-GRO test split, open-ended |
| `rshr` | 1,154 | RSHR-Bench, a subset of its single-turn four-option multiple-choice questions (see below) |
| `gro_train` | 3,511 | LRS-GRO official training split (`rl` 2,500 + `sft` 1,011), calibration only |
| `lrsrefdev_ok`, `lrsrefdev_bad` | 200 each | pointer sets built from the LRS-VQA calibration questions: a correct / wrong pointer inserted |
| `grorefdev_ok`, `grorefdev_bad` | 2,049 each | pointer sets built from `gro_train` |

`earthjev/train/pointer_sets.py` builds the pointer sets from the `lrs_vqa` and `gro_train` rows.

Fields:

| field | meaning |
|---|---|
| `qid`, `bench` | question id and set name |
| `image` | the image path relative to `$EARTHJEV_DATA` |
| `images` | used instead of `image` by the 60 two-image questions of XLRS-Bench-lite |
| `question` | the full prompt given to the answering model |
| `stem` | the question without its options and answer-format line |
| `options`, `choice_options` | `[{id, text}]` and the `(A) text` strings (multiple-choice only, else `[]`) |
| `ground_truth`, `category`, `source` | the reference answer and the grouping fields (see below) |
| `is_mcq`, `is_yes_no` | answer type |
| `multi_choice` | XLRS-Bench-lite only: the question may have several correct options |
| `hbox`, `image_size` | evidence box and image size in pixels (LRS-VQA and the pointer sets; else `null`) |
| `ref_format`, `ref_box` | pointer sets: the format and box of the inserted pointer |
| `gro_box` | `gro_train` and `grorefdev_*`: LRS-GRO box in its 0-1024 frame (pixels = value x max(W, H) / 1024) |

The score groups of `earthjev.scoring.benchmark_scores` use `source` for LRS-VQA (`FAIR`, `GLH-Bridge`, `STAR`)
and LRS-GRO (`global`, `region`, `object`). They use `category` for MME-RS (`color`, `count`, `position`),
XLRS-Bench-lite (the L3 tasks) and RSHR-Bench (task categories, spelled as in the benchmark).

## Frozen question lists: `data/splits/<name>.json`

| name | questions | use |
|---|---:|---|
| `<bench>_full` | all | the evaluated sets, in question-file order |
| `mme_rs_dev`, `lrs_vqa_dev`, `xlrs_lite_dev` | 90 / 240 / 130 | calibration questions (30 / 30 / 10 per category) |

The calibration questions and `gro_train` are used to fit the per-benchmark answer-confidence maps. The
`lrsrefdev_*` and `grorefdev_*` pointer sets are used to fit the pointer-trust map and the route thresholds. The
calibration questions are part of the full sets on which the paper reports accuracy.

## Calibration runs: `data/calibration/<config>`

One folder per configuration: `qwen35-9b` (21 gzip files, 4.6 MB) and `qwen3-vl-8b` (23 gzip files, 4.5 MB; the
same runs, split into files differently). Each holds the logged runs of EarthJev with that configuration on
the calibration sets above, from which `python -m earthjev.train.fit_decisions [--config qwen3-vl-8b]` refits the
fitted decision parameters of `configs/<config>.json` on CPU, bitwise (`tests/test_fit.py`):

| file / directory | questions | used for |
|---|---|---|
| `mme_rs`, `xlrs_lite`, `lrs_vqa` | 90, 130, 240 (the `*_dev` splits) | answer maps; XLRS-lite also for the route thresholds |
| `gro_train` | 3,511 | LRS-GRO answer maps |
| `lrsrefdev_ok`, `lrsrefdev_bad` | 200 + 200 | pointer-trust map and route thresholds |
| `grorefdev_ok`, `grorefdev_bad` | 2,049 + 2,049 | pointer-trust map and route thresholds |
| `presence_probe_dev.jsonl.gz` | 3,045 crops of 210 LRS-VQA dev questions | presence map |

Each question directory holds the evidence search with its answers (`search_*.jsonl.gz`) and the answer pass with
the log-probabilities and the pointer probabilities (`answers_*.jsonl.gz`). The probe file holds the presence-map
fitting data: the crops around the evidence boxes of the LRS-VQA dev questions, each with the cached logit of the
configuration's region head (the same crops for both configurations). Like the runs, these are shipped data produced
by the original pipeline; no script of this repository recomputes them. `MANIFEST.tsv` lists the size and sha256 of
every file (decompressed), and the folder's README.md describes the fields. Refitting needs no images and no model.

## Image layout under `$EARTHJEV_DATA`

```
$EARTHJEV_DATA/
  MME-RealWorld/images/<index>.jpg          e.g. 019871.jpg
  LRS-VQA/LRS_VQA/image/<name>              e.g. 15565.tif
  XLRS-Bench-lite/images/<md5>.jpg
  LRS-GRO/image/<name>                      e.g. 6270.tif (a link to ../LRS-VQA/LRS_VQA/image by default)
  RSHR-Bench/images/rshr_<id>.png|.jpg
  ZwZ-RL-VQA/train.parquet, original_images/<name>   training only
  sources/                                  the downloads (scripts/prepare_data.py --src)
```

`data/images.tsv` lists every image file named by a question file (8,837 files, 303 GiB). For each file it gives the
path, the size in bytes, the width, the height and the sha256. The files are not unique by content: 4,748 distinct
sha256 values for 8,837 files (every LRS-GRO file equals the LRS-VQA file of the same name, and some benchmarks share
source images).

## The benchmarks

### MME-RealWorld (MME-RS)

- Source: Hugging Face `yifanzhang114/MME-RealWorld-Lmms-eval` at revision
  `f157390129e8d2b60757eb68d340aade9446b52b` (parquet). The questions are the rows whose category contains
  "Remote Sensing". The official question ids come from `yifanzhang114/MME-RealWorld`. The project is at
  github.com/MME-Benchmarks/MME-RealWorld.
- License as stated by the source:
  - the `yifanzhang114/MME-RealWorld` dataset card says `apache-2.0`;
  - the GitHub README says "MME-RealWorld is only used for academic research. Commercial use in any form is
    prohibited.";
  - the Lmms-eval card states no license.
- Images: `MME-RealWorld/images/<index>.jpg` (the `index` zero-padded to six digits) is made from the base64-decoded
  `bytes` field of the parquet row with that `index`. The question id is `MMERS__<index>`.
- Prompt: the official question, options and instruction, followed by `\nThe best answer is:`.

The parquet holds the JPEG files of the Lmms-eval release, not the original PNG files. An image with at most
2^24 = 16,777,216 pixels is written unchanged. A larger image is downscaled to at most 2^24 pixels, with both sides
multiples of 32 (Qwen `smart_resize` with factor 32), using PIL bicubic resampling, and saved as JPEG quality 95
(`mme_image` and `mme_size` in `scripts/prepare_data.py`). This is the case for 2,454 of the 3,738 MME-RS images.
All our runs read these files, and the boxes in the released logs are in their pixel frame, so full-resolution
copies do not reproduce our results. The re-encoded bytes depend on the JPEG library. With Pillow 12.0.0
(libjpeg-turbo 3.1.2, the version of `requirements/cpu.txt`) and with Pillow 12.3.0 (libjpeg-turbo 3.1.4), the rule
reproduces all 3,738 files byte for byte; other versions may differ, and `verify` names the files that do.

### LRS-VQA

- Source: Hugging Face `ll-13/LRS-VQA` at revision `f7c0f5f5a28666dddaad5ca2a09b1cec39984d79`. It contains
  `LRS_VQA_merged.jsonl` (the questions) and the image archive `LRS_VQA.7z.001`-`.013`. The code and the
  official scorer are at github.com/VisionXLab/LRS-VQA.
- License as stated by the source: CC-BY-NC-4.0 (dataset card). In the GitHub repository, `LRS-VQA-Code/` is
  Apache-2.0; no license covers `Script/`, which holds the official scorer.
  The images come from FAIR1M, GLH-Bridge and STAR.
- Images: the archive extracted into `$EARTHJEV_DATA/LRS-VQA` (`7z x LRS_VQA.7z.001`) gives
  `LRS-VQA/LRS_VQA/image/`, which matches the `image` field of the official questions.

### XLRS-Bench-lite

- Source: Hugging Face `initiacms/XLRS-Bench-lite` at revision `e540ee2aa745ce9a83784ae76541ddb7f79f03ac`
  (a `datasets` dump with embedded images). The benchmark is at github.com/AI9Stars/XLRS-Bench.
- License as stated by the source: the annotations are CC-BY-NC-SA-4.0. The images keep the terms of their
  sources:
  - DOTA: academic use only, and Google Earth terms apply;
  - ITCVD: CC-BY-NC-SA-4.0;
  - MiniFrance and HRSCD: IGN "licence ouverte";
  - ISPRS Toronto and Potsdam: research use only, no redistribution.
- Images: `XLRS-Bench-lite/images/<md5>.jpg` is one file per distinct image of the release, named by the md5 of
  its bytes and written unchanged. Change-detection questions have two images.
- Prompt: the lmms-eval `xlrs_doc_to_text` prompt.

### LRS-GRO

- Source: Hugging Face `HappyBug/LRS-GRO` at revision `869acdcdd13eb9b6687871c96101fe431d202dec`. The test
  questions are `data/test-00000-of-00001.jsonl.jsonl`, and the training split is `data/rl-00000-of-00001.jsonl` +
  `data/sft-00000-of-00001.jsonl.jsonl` (checksums below). The images are in the archive
  `images.tar.part-aa`-`an`. The project and its official scorer (`src/eval/eval.py`) are at
  github.com/earth-insights/ZoomEarth.
- License as stated by the source:
  - the dataset card states no license;
  - the ZoomEarth GitHub repository (commit `d5174519`) ships the questions (`LRS-GRO.jsonl`), and its README says
    "(c) 2025 ZoomEarth Project. Released under the Apache 2.0 License." The LRS-GRO question files of this
    repository are redistributed on the basis of that statement.
- Images: in our copies, all 1,224 LRS-GRO images that the questions use are byte-identical to the LRS-VQA images
  of the same name, so `prepare_data.py` links `LRS-GRO/image` to `../LRS-VQA/LRS_VQA/image` instead of keeping a
  second copy. `--lrs-gro extract` unpacks the official archive (`cat images.tar.part-* | tar -x`, giving
  `LRS-GRO/image/<image_name>`); we checked the archive's layout, not its 67 GiB of content.
- Questions: the stripped question followed by `\n Answer the question using a single word or phrase.`, with the
  stripped reference answer. The question id is `LRSGRO__<question_id>`.
- Reference answers: two cases differ from running the official `evaluate_dataset` on the raw files. All reported
  LRS-GRO scores use the question files as shipped.
  - Four upstream references have surrounding spaces: `LRS-GRO-4639` (`'straight line '`), `LRS-GRO-7683`
    (`'trees '`), `LRS-GRO-12832` (`'spring '`) and, in `rl-00000-of-00001.jsonl`, `LRS-GRO-1858`
    (`' a single large structure'`). The question files strip them. The official script lowercases the reference
    but strips only the answer, so on the raw files no answer to these four questions can be correct.
  - `LRS-GRO-11315` has an empty reference. The official rule is kept, so an empty or blank answer to it is correct
    (two empty strings match). A missing answer (`None`) is wrong.

### RSHR-Bench

- Source: Hugging Face `RL-MIND/RSHR-Bench` at revision `68674294ed5c0bbba0896404c1dec255f6e82e8c` (parquet with
  embedded images; arXiv:2512.17319; github.com/Yunkaidang/RSHR).
- License as stated by the source: Apache-2.0 (dataset card).
- Images: `RSHR-Bench/images/rshr_<id>.png` (or `.jpg` when the bytes are JPEG) is the `image_1` bytes of the
  parquet row with that `id`, written unchanged. The question id is `RSHR__<id>`.
- Questions: the 1,154 questions listed in `data/splits/rshr_full.json`. Each prompt is the question, an
  `Options:` block (A-D) and `Answer with the option letter only (A, B, C, or D).`
- Two-image questions: five of the questions are `future_prediction_two_image` questions that have two images in
  the benchmark (`image_count` 2): `RSHR__361`, `RSHR__363`, `RSHR__365`, `RSHR__368` and `RSHR__390`. As in our
  experiments, they get only `image_1`, which is Image A (the earlier year), although each question compares Image A
  with Image B (the later year). Reasoning is an unweighted mean over its three categories, so these five questions
  carry one third of the Reasoning score.
- Groups: Perception has 9 categories and 1,074 questions: `color`, `detection`, `object_clasfication`,
  `object_counting`, `object_grounding`, `regional_grouding`, `regionla_counting`, `relation` and `shape`.
  Reasoning has 3 categories and 80 questions: `anomaly_detection` (39), `object_state_judgement` (36) and
  `future_prediction_two_image` (5).

### ZwZ-RL-VQA (training only)

It is needed only to retrain the region head and the pointer verifier, not to evaluate.

- Source: Hugging Face `inclusionAI/ZwZ-RL-VQA` at revision `fec404cb7bd5a18458e5add2802a5fb183981d14`:
  `train.parquet` and the original images without drawn boxes, `original_images/original_images.tar.gz00`-`04`
  (a plain tar split in five parts despite the name).
- License as stated by the source: apache-2.0 (dataset card). The card names SA-1B, LAION, Visual Genome and CC12M
  as the image sources.
- `python scripts/prepare_data.py zwz` downloads these six files, checks their sha256 (also listed in
  `earthjev/train/zwz.py`), and writes `ZwZ-RL-VQA/train.parquet` and `ZwZ-RL-VQA/original_images/<name>`.
  `earthjev/train/zwz.py` builds the training examples from them.

## Checksums of the question sources we used (sha256)

```
16935bd04b29fdbebe86b378999d742ebcdafb9b9d2a8139491cb0b4b6454f44  LRS-VQA LRS_VQA_merged.jsonl
f63d01e170c9b22d8790976553f779d8ff605fb5f4b2b2fb3c1a956e0e2bd691  LRS-GRO data/test-00000-of-00001.jsonl.jsonl
63da71ab6c6ad431a92ae0764bc82b2cdcf6f2c5f5b12efe4320063aea0d033e  LRS-GRO data/rl-00000-of-00001.jsonl
50ea136ed46d8fcb3e7fa29511cc720c77e03e4bf2b848b513b83124252b06a6  LRS-GRO data/sft-00000-of-00001.jsonl.jsonl
```

## Terms

The repository's Apache-2.0 license covers its code and `configs/` (and the trained modules), not the data derived
from the benchmarks. Those files keep the terms of their sources and are released under them:

| files | derived from | license | changes |
|---|---|---|---|
| `data/benchmarks/mme_rs.jsonl`, `data/splits/mme_rs_*.json` | MME-RealWorld (`yifanzhang114/MME-RealWorld-Lmms-eval`, ids of `yifanzhang114/MME-RealWorld`) | the source's terms: apache-2.0 on the dataset card (https://huggingface.co/datasets/yifanzhang114/MME-RealWorld), academic research only and no commercial use per the README (https://github.com/MME-Benchmarks/MME-RealWorld) | the remote-sensing rows; unified fields; `\nThe best answer is:` appended to the prompt |
| `data/benchmarks/lrs_vqa.jsonl`, `lrsrefdev_ok.jsonl`, `lrsrefdev_bad.jsonl`, `data/splits/lrs_vqa_*.json` | LRS-VQA (`ll-13/LRS-VQA`, `LRS_VQA_merged.jsonl`; the LRS-VQA authors, github.com/VisionXLab/LRS-VQA) | CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), with attribution to the LRS-VQA authors | unified fields, image sizes; `lrsrefdev_*`: a sentence with a correct or a wrong box appended to the question |
| `data/benchmarks/xlrs_lite.jsonl`, `data/splits/xlrs_lite_*.json` | XLRS-Bench-lite (`initiacms/XLRS-Bench-lite`) | CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), ShareAlike: these files and the XLRS-lite rows in `data/calibration`, `reference/live` and `tests/fixtures` | the lmms-eval prompt; image files named by the md5 of their bytes; unified fields |
| `data/benchmarks/lrs_gro.jsonl`, `gro_train.jsonl`, `grorefdev_ok.jsonl`, `grorefdev_bad.jsonl`, `data/splits/lrs_gro_full.json` | LRS-GRO (`HappyBug/LRS-GRO`; the ZoomEarth project) | Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0), on the basis of the ZoomEarth statement above | questions and references stripped, the answer-format line appended; `grorefdev_*`: a sentence with a correct or a wrong box appended |
| `data/benchmarks/rshr.jsonl`, `data/splits/rshr_full.json` | RSHR-Bench (`RL-MIND/RSHR-Bench`) | Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0) | a subset of 1,154 questions; the prompt with an `Options:` block |
| `data/calibration/<config>/` | runs of EarthJev on calibration questions of MME-RS, LRS-VQA, XLRS-lite and LRS-GRO (and the pointer sets) | per question, the license of its question file above: CC BY-NC-SA 4.0 for the XLRS-lite rows, CC BY-NC 4.0 for the LRS-VQA rows | the model's outputs added to the question fields |
| `reference/live/<config>/` | EarthJev's records on 1,612 benchmark questions | per file, the license of its benchmark above | converted from the original run logs |
| benchmark rows in `tests/fixtures/` (`cas_decisions`, `pipeline_real_*`, `pipeline_synthetic_*`, `pointers_golden`, `prompts_golden`, `scoring_final_answers`, `spv_golden`, `spv_pointer_replay`, `tiny_golden`) | questions of the five benchmarks and EarthJev's outputs on them | per row, the license of its benchmark above | selected rows and fields |

`data/images.tsv` lists file names, sizes and checksums only. The images themselves are not redistributed; they keep
the terms of their sources (above), and `scripts/prepare_data.py` downloads them.

## Scoring data

The LRS-VQA and LRS-GRO scorers need the NLTK WordNet 3.0 corpus: docs/INSTALL.md, section 5.
`earthjev.scoring` raises an error if it is missing.
