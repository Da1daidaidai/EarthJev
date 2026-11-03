# Reference results

## `live/<config>/<bench>.jsonl.gz`: 1,612 questions answered live, per configuration

One record per question, written by the live pipeline that produced the paper's runs: EarthJev with the Qwen3.5-9B
answerer (`live/qwen35-9b`, `configs/qwen35-9b.json`) and with the Qwen3-VL-8B answerer (`live/qwen3-vl-8b`,
`configs/qwen3-vl-8b.json`), each with its shipped region head and pointer verifier, on one NVIDIA H100, bf16,
flash-attention 2, the GPU stack pinned in `requirements/gpu.txt`. The questions are the first shard of each benchmark
in the order of `scripts/run_earthjev.py` (questions sorted by image, then id), the same for both configurations:

| benchmark | file | questions | shard |
|---|---|---|---|
| MME-RealWorld-RS | `mme_rs.jsonl.gz` | 311 | 1/12 |
| LRS-VQA | `lrs_vqa.jsonl.gz` | 305 | 1/24 |
| XLRS-Bench-lite | `xlrs_lite.jsonl.gz` | 308 | 1/10 |
| LRS-GRO | `lrs_gro.jsonl.gz` | 304 | 1/32 |
| RSHR-Bench | `rshr.jsonl.gz` | 384 | 1/3 |

The format is the record of `earthjev/pipeline.py` (route and pointer trust per image, views, both presentations'
answers with their generation log-probs, the chosen presentation, its calibrated P, the veto, the final answer, stage
timings), converted from the original run logs (the original key names mapped to these, XLRS-lite letter sets sorted,
timings rounded to 0.1 ms). They hold no error records, and no `config` field: the records of `run_earthjev.py` name
their configuration, and `evaluate.py` and `check_live.py` accept records without it.

Reproduce and compare (one GPU; each run writes `$EARTHJEV_OUT/earthjev_<config>/<bench>_full_1of<n>.jsonl`).
Qwen3.5-9B:

```
python scripts/run_earthjev.py --bench mme_rs --shard 1/12
python scripts/run_earthjev.py --bench lrs_vqa --shard 1/24
python scripts/run_earthjev.py --bench xlrs_lite --shard 1/10
python scripts/run_earthjev.py --bench lrs_gro --shard 1/32
python scripts/run_earthjev.py --bench rshr --shard 1/3
O=$EARTHJEV_OUT/earthjev_qwen35-9b
python scripts/check_live.py $O/mme_rs_full_1of12.jsonl $O/lrs_vqa_full_1of24.jsonl $O/xlrs_lite_full_1of10.jsonl \
    $O/lrs_gro_full_1of32.jsonl $O/rshr_full_1of3.jsonl
```

Qwen3-VL-8B: the runs take `--config qwen3-vl-8b`, and both the directory and `--config` of `check_live.py` change:

```
python scripts/run_earthjev.py --bench mme_rs --shard 1/12 --config qwen3-vl-8b
python scripts/run_earthjev.py --bench lrs_vqa --shard 1/24 --config qwen3-vl-8b
python scripts/run_earthjev.py --bench xlrs_lite --shard 1/10 --config qwen3-vl-8b
python scripts/run_earthjev.py --bench lrs_gro --shard 1/32 --config qwen3-vl-8b
python scripts/run_earthjev.py --bench rshr --shard 1/3 --config qwen3-vl-8b
O=$EARTHJEV_OUT/earthjev_qwen3-vl-8b
python scripts/check_live.py $O/mme_rs_full_1of12.jsonl $O/lrs_vqa_full_1of24.jsonl $O/xlrs_lite_full_1of10.jsonl \
    $O/lrs_gro_full_1of32.jsonl $O/rshr_full_1of3.jsonl --config qwen3-vl-8b
```

Name the files of these shards rather than a pattern such as `*_full_1of*.jsonl`, which also matches other runs
(e.g. `mme_rs_full_1of1.jsonl`); a question found in several files is reported.

What to expect:

- The answers are greedy, so on the same GPU model and software stack the same inputs give the same records, except
  `P` after a presence veto in the Qwen3.5-9B records (4 MME-RS questions): they hold the vetoed answer's P, the
  pipeline the P of the replacement letter (the Qwen3-VL-8B records already hold the latter). `check_live.py` compares
  P on the questions without a veto.
- On another GPU configuration the bf16 kernels round differently, and the differences grow through the model's 32
  layers: region scores, view scores and log-probs move (by a few hundredths), the search ranks regions by those
  scores, so some views change, and some answers change with them. We ran this release (Qwen3.5-9B) on all 1,612
  questions on a MIG partition of an H100 (same software; on that partition the code behind the paper's results gives
  records bit for bit identical to this release's). Against these records:

  | | MME-RS | LRS-VQA | XLRS-lite | LRS-GRO | RSHR | all |
  |---|---|---|---|---|---|---|
  | same route | 100% | 100% | 100% | 100% | 97.7% | 99.5% |
  | same view boxes | 55.0% | 50.5% | 73.7% | 29.3% | 56.8% | 53.3% |
  | same final answer | 87.1% | 84.9% | 91.9% | 83.9% | 91.1% | 88.0% |
  | same correctness | 94.2% | 95.4% | 97.1% | 95.1% | 93.2% | 94.9% |
  | accuracy, these records | 69.77 | 37.70 | 54.22 | 56.25 | 59.64 | 55.77 |
  | accuracy, MIG run | 69.77 | 38.36 | 53.90 | 55.26 | 61.72 | 56.14 |

  The answers that change are about as often right as wrong (38 questions turned wrong, 44 right), so benchmark
  accuracies agree to within the noise of a few hundred questions. On a GPU other than the H100, compare accuracies,
  not records; to reproduce the records themselves, use a full H100 with the pinned stack. `check_live.py` compares
  `views` with their scores, which differ across GPUs even where the boxes agree.
- Qwen3-VL-8B on the same MIG partition, 53 of these questions (the first 10 of each benchmark's shard, 2 vetoed
  MME-RS questions and a two-image XLRS-lite question; again bit for bit identical to the original code on that
  partition): the same route, view boxes, canvas and memory answers, chosen presentation and final answer on all 53;
  view scores and P differ in their last digits (|dP| at most 0.045). Qwen3-VL-8B has no linear-attention layers,
  whose kernels are the main source of the Qwen3.5-9B differences.
- These live records and the paper's tables answer the same questions with the same algorithm, but the tables come from
  the logged runs of the full sets (`scripts/evaluate.py` prints them next to a run; `paper_scores.json`). On these
  1,612 questions the correctness of the live final answers equals that of the paper's answers on 99.48-100% of the
  questions per benchmark (MME-RS 99.68%, LRS-VQA 99.67%, XLRS-lite 100%, LRS-GRO 99.67%, RSHR 99.48%); with
  Qwen3-VL-8B on 99.01-100% (MME-RS 99.68%, LRS-VQA 99.67%, XLRS-lite 100%, LRS-GRO 99.01%, RSHR 99.74%).

## `paper_scores.json`

Per configuration: EarthJev with Qwen3.5-9B in Tables I-II of the paper (accuracy (%) per benchmark and group) and
with Qwen3-VL-8B in Table IV (accuracy per benchmark), on the full question sets, unrounded, and their mean
(`overall`: 53.06 and 50.92). `scripts/evaluate.py --config <config>` prints them next to the scores of a run.
