# Calibration runs (Qwen3.5-9B)

The logged runs of EarthJev on the calibration questions, from which the decision parameters of
`configs/qwen35-9b.json` are fitted (`spv.trust_platt`, `spv.tau_fast`, `spv.tau_keep`, `cas.answer_platt`,
`cas.presence_platt`). Refit them on CPU with

    python -m earthjev.train.fit_decisions

which reproduces the configuration bitwise (`tests/test_fit.py`).

| file / directory | questions | used for |
|---|---|---|
| `mme_rs`, `xlrs_lite`, `lrs_vqa` | 90, 130, 240 (the `*_dev` splits) | answer maps; XLRS-lite also for the route thresholds |
| `gro_train` | 3,511 (LRS-GRO training split) | LRS-GRO answer maps |
| `lrsrefdev_ok`, `lrsrefdev_bad` | 200 + 200 (LRS-VQA dev questions with a correct / wrong stated box) | pointer trust map and route thresholds |
| `grorefdev_ok`, `grorefdev_bad` | 2,049 + 2,049 (LRS-GRO training questions with a correct / wrong stated box) | pointer trust map and route thresholds |
| `presence_probe_dev.jsonl.gz` | 3,045 crops of 210 LRS-VQA dev questions | presence map |

Each question directory has two row files, read by `earthjev.train.fit_decisions.load_units`: `search_*.jsonl.gz` (the
evidence search with its own route and answers) and `answers_*.jsonl.gz` (per route the evidence views, both
presentations' answers with their teacher-forced log-probs, and the verifier and Yes/No probabilities of the stated
pointers). The `grorefdev_*` answers rows carry no log-probs (only the pointer trust and the route accuracy are fitted
on them). The `gro_train` answers rows hold the teacher-forced log-probs of the search's own answers under the official
answer prompt, from a separate scoring pass. The rows keep the key names of the original runs (route keys `s1`, `s2`
and `s2x` for the fast path, the search and the search without the pointer views; the pointer record of a search is
`last_search.spv`, whose `path` is `S1` for the fast path).

The probe file has one row per crop: `qid`, `kind`, `bbox` (pixels), `label` (1 for a crop centred on the question's
evidence box or the 3x3 grid cell holding its centre, 0 for a crop disjoint from it) and `z`, the cached raw
region-head logit (float32). `earthjev.train.fit_decisions.presence_probe` turns them into the verifier's region
probabilities with each question's directional phrase; the presence map is the Platt map of those probabilities.

`MANIFEST.tsv` lists the sha256 and size of every file (decompressed).
