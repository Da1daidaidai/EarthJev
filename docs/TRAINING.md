# Training

EarthJev trains two small modules on top of a frozen answering model, the region head (EDM) and the pointer verifier
(SPV), and fits its calibration maps and thresholds. The released weights (`region_head/` and `verifier/` under
`$EARTHJEV_MODELS`, README section B) and fitted parameters (`configs/`) are the reference; this page describes how
they were produced and how to reproduce them.

## Steps

```bash
python -m earthjev.train.zwz export          # ZwZ-RL-VQA train.parquet -> $EARTHJEV_OUT/zwz/train_rows.jsonl
python -m earthjev.train.zwz build           # -> $EARTHJEV_OUT/zwz/manifest.jsonl (train / val / calib split by image)
python -m earthjev.train.extract_features --set zwz              # GPU: answering-model features of every crop
python -m earthjev.train.train_region_head                       # region head f_phi
python -m earthjev.train.train_verifier zv                       # region-head logits z_v of every crop
python -m earthjev.train.train_verifier train                    # pointer verifier g_psi, T_g and tau_opt
python -m earthjev.train.fit_decisions                           # calibration maps and routing thresholds (CPU)
```

| step | reads | writes |
|---|---|---|
| `zwz export` / `build` | `$EARTHJEV_DATA/ZwZ-RL-VQA/` (docs/DATA.md) | `$EARTHJEV_OUT/zwz/` |
| `extract_features` | the ZwZ manifest and images, the answering model | `$EARTHJEV_OUT/features/zwz/<config>.s<i>of<n>.b<k>.npy` and their index files |
| `train_region_head` | the features | `$EARTHJEV_OUT/region_head/<config>/{head.safetensors, head.json, training_history.json}` |
| `train_verifier zv` | the features and the configuration's region head | `$EARTHJEV_OUT/verifier/<config>/zv_zwz.npz` |
| `train_verifier train` | the z_v cache and the ZwZ manifest | `$EARTHJEV_OUT/verifier/<config>/{verifier.safetensors, verifier.json, training_history.json}` |
| `fit_decisions` | `data/calibration/<config>/` and the configuration's verifier | prints the refitted values (`--out fit.json` writes them) |

Every step except `zwz` (its examples are the same for every configuration) takes `--config qwen3-vl-8b` for the
Qwen3-VL-8B configuration; outputs carry the configuration's name. `extract_features` takes `--shard i/n` to split
the work over GPUs.

## What `fit_decisions` refits

`fit_decisions` runs on CPU and refits five entries of `configs/<config>.json`, bitwise for both configurations:
`spv.trust_platt`, `spv.tau_fast`, `spv.tau_keep`, `cas.answer_platt` and `cas.presence_platt`. It reads
`data/calibration/<config>`: the method's logged searches and answers on the calibration questions, and the presence
probe crops (LRS-VQA dev crops with their region-head logits). Both are shipped data produced by the original
pipeline. The other entries are not refitted there: `spv.temperature` (T_g) and `spv.tau_opt` come from the verifier
training (`verifier.json`, keys `temperature` and `tau`), and `cas.veto_threshold` is fixed at 0.5.

## Chaining retrained weights

Each step reads the released artifacts of its configuration: `train_verifier zv` scores the crops with the released
region head, and `fit_decisions` (and inference) use the released verifier. To chain retrained weights, copy the
configuration file under the same name into another directory and set its `weights` entries to absolute paths (they
are joined to `$EARTHJEV_MODELS`, which an absolute path replaces; `<out>` below is `$EARTHJEV_OUT` written out), then
pass that file to the later steps:

```bash
C=$EARTHJEV_OUT/retrained/qwen35-9b.json
mkdir -p $EARTHJEV_OUT/retrained && cp configs/qwen35-9b.json $C
# edit $C: "weights": {"region_head": "<out>/region_head/qwen35-9b", "verifier": "<out>/verifier/qwen35-9b"}
python -m earthjev.train.train_verifier zv --config $C      # the crops scored by the retrained head
python -m earthjev.train.train_verifier train --config $C
# edit $C: "spv": {"temperature": ..., "tau_opt": ...} from <out>/verifier/qwen35-9b/verifier.json ("temperature", "tau")
python -m earthjev.train.fit_decisions --config $C --out fit.json
```

The file name keeps the configuration's feature, output and calibration names. Copy the new `temperature` and
`tau_opt` into the copy before `fit_decisions`: the presence map is fitted on verifier probabilities at that
temperature. `fit_decisions` then exits with status 1 because its values differ from the copy's; copy them from
`fit.json` before running inference with `scripts/run_earthjev.py --config <copy> --out <file>` (its default output
directory is named after the file). Only the presence map depends on the retrained weights there, and it scores the
shipped region-head logits of the probe crops; the other four entries are fitted on the logged runs of the original
pipeline.

## What is reproduced exactly

- `fit_decisions` reproduces the fitted parameters of both configurations bitwise on an x86-64 CPU with AVX-512;
  without AVX-512, eight values differ in their last bits (docs/INSTALL.md, section 9).
- The verifier retrains bitwise on one CPU thread from the original region-head logits of the ZwZ crops, a cache
  that is not shipped.
- Training the head is seeded, but bf16 feature extraction and float sums differ across GPUs and thread counts, so
  re-extracted features give a head (and a verifier trained on its logits) close to, not identical with, the released
  weights. The released weights are the reference.
