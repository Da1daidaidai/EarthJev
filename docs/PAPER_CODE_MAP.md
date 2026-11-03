# Paper to code

Where each part of the paper is implemented, and the exact rules the code applies where the paper leaves a detail open.
Fitted values live in `configs/qwen35-9b.json`; trained weights in `region_head/` and `verifier/` under
`$EARTHJEV_MODELS` ([Hugging Face](https://huggingface.co/daidaidaidai1dai/EarthJev), README section B). A second
configuration, `configs/qwen3-vl-8b.json` (Table IV: Qwen3-VL-8B as the answering model), has its own region head,
verifier, calibration runs and fitted values; the code is the same (`--config qwen3-vl-8b`).

## Method (Sec. III)

| Paper | Code | Parameters |
|---|---|---|
| III-A: frozen answering model, evidence set V (at most K views), two presentations, calibrated choice | `earthjev/pipeline.py` `EarthJev.run_row` (`find_evidence`, `answer`, `decide`) | whole configuration |
| III-B: spatial cues R(q): boxes in several formats, points, option boxes, directional phrases; coordinates removed from every prompt | `earthjev/pointers.py` `parse`, `strip`, `phrase_axes` | — |
| III-B: pointer views t_j | `earthjev/spv.py` `pointer_proposals` | square of side max(2 x long side, 448 px) around the pointer, at most 8, a view with IoU > 0.5 to an earlier one dropped |
| III-B, Eq. (spv): pointer verifier g_psi | `spv.PointerVerifier.score`; geometry e_j: `pointers.geometry` (22-d); directional prior b_pos: `pointers.rule_prior` | `verifier/qwen35-9b`; T_g = `spv.temperature` |
| III-B: verify-then-propagate mixture p_mix, pi = max_j p_j | `spv.PointerVerifier.mixture` | — |
| III-B: pointer trust P_ptr and routing | `spv.pointer_trust`, `spv.route`, `spv.select_path`; y_j: `answerer.Answerer.yes_probability` | `spv.trust_platt` (a_ptr, b_ptr), `tau_fast`, `tau_keep`, `tau_opt` |
| III-C: region decision head f_phi on the frozen answerer's hidden state (views of at most 256 visual tokens, coordinate-free evidence prompt) | `earthjev/region_head.py` `RegionScorer.logits`, `RegionHead` | `region_head/qwen35-9b` |
| III-C: hierarchical 3x3 best-first search, at most 10 expansions | `earthjev/search.py` `hierarchical_search`, `grid_cells` | `MAX_EXPANSIONS = 10`, `CELL = 224` |
| III-C: evidence retention | `search.select_by_sigmoid` | `KEEP_THRESHOLD = 0.6`, `KEEP_MAX = 6` |
| III-C: evidence set V | `earthjev/evidence.py` `select_views` | each view grown 1.5x to at least 512 px per side, merged at IoU > 0.5, top K = 4 by score |
| III-C, Eq. (pres): presence probability P_pres | `earthjev/cas.py` `presence_probability` | `cas.presence_platt` |
| III-D: memory presentation (global view + local views with their positions in the text) | `evidence.render_memory_views`, `prompts.memory_answer` | each view shown at 1,024 px on its long side; global image at most 2^20 px |
| III-D: canvas presentation (views re-assembled by spatial layout) | `evidence.compose_canvas`, `prompts.canvas_answer`, `prompts.position_hint` | canvas at most 2^20 px |
| III-D: confidence z_r, per-benchmark Platt maps, choice of the presentation | `cas.confidence_feature`, `cas.platt`, `cas.select_presentation`, `cas.decide` | `cas.answer_platt` |
| III-D: evidence presence veto | `cas.veto` | `cas.veto_threshold = 0.5` |
| Answering model (greedy, thinking off, generation-time log-probs) | `earthjev/answerer.py` `Answerer.generate` | `answerer` block of the configuration |

## Training and fitting (Sec. III, IV-A)

| Paper | Code | Output |
|---|---|---|
| Region head: crops with annotated evidence regions (ZwZ-RL-VQA), BCE, the frozen answerer's features | `earthjev/train/zwz.py`, `train/extract_features.py` (GPU), `train/train_region_head.py` | `region_head/qwen35-9b` |
| Pointer verifier: annotation-derived pointers and corrupted copies (shifted, rescaled, relocated), cross-entropy, temperature T_g and tau_opt on held-out data | `earthjev/train/train_verifier.py`, `train/pointer_sets.py` | `verifier/qwen35-9b` |
| Calibration maps (P_ptr, answers, presence) and the routing thresholds | `earthjev/train/fit_decisions.py`, `earthjev/calibration.py` (Platt scaling) | `configs/qwen35-9b.json` (fitted on `data/calibration/qwen35-9b`) |

## Evaluation (Sec. IV)

| Paper | Code |
|---|---|
| IV-A3: accuracy per benchmark with the official matchers; Overall = mean over the five benchmarks | `earthjev/scoring.py` `score`, `benchmark_scores`, `overall`; `scripts/evaluate.py` |
| Inference on a benchmark | `scripts/run_earthjev.py` |
| Checking a reproduction against the original pipeline's outputs | `scripts/check_live.py`, `reference/` |

## Implementation details

The code applies these exact rules (all of them were in effect for every number in the paper):

- **Pointer views and routing.** P_ptr is computed over the pointers stated in the question stem; y_j is the
  answering model's Yes probability on the pointer view at 256 visual tokens. The fast path answers from the view of the
  single stem pointer with the highest p_j. Below tau_fast, the stem pointer views join the search's pool iff
  P_ptr >= tau_keep: with `qwen35-9b` tau_keep = 0, so they always do; with `qwen3-vl-8b` tau_keep = tau_fast = 0.5,
  so a question with stem pointers either takes the fast path or searches without its stem pointer views. Option-box
  pointers (and two-image questions) take the fast path iff max_j p_j >= tau_opt; option-box views always join the pool.
- **Scores of the search.** Region scores use the verifier temperature T_g (`spv.temperature`), also for the search's
  children; the head's own fitted temperature (`region_head/qwen35-9b/head.json`) is not used at inference.
  The verifier's pointer term carries a bias and a validity mask (`spv.PointerVerifier`).
- **Retention.** A node c is kept iff sigmoid(5 (conf(c) - mean conf)) > 0.6 with conf = 2p - 1, i.e.
  sigmoid(10 (p - mean p)) > 0.6; at most 6 nodes, else the single best one. The same rule decides which children of an
  expansion enter the open set, and which of the final candidates (all popped leaves plus every node of the deepest
  level reached) are kept.
- **End of the search.** At most 10 pops; the search also ends when the open set is empty, and by a size-dependent
  early-stop rule (see the docstring of `earthjev/search.py`). Cells smaller than 224 px are not scored; an image too
  narrow to split keeps the whole image as its only view.
- **Confidence features.** For open-ended answers z_r is the mean log-probability over the generated tokens up to and
  including the end-of-turn token. For multiple-choice answers z_r is the logit of the first-position probability of
  the extracted letter; multi-answer XLRS-lite rows and outputs without a letter use the whole-output probability.
  Exact ties in P go to the memory presentation for multiple-choice and to the canvas for open-ended questions.
- **Calibration sets.** RSHR-Bench reuses the answer maps fitted on the MME-RS and XLRS-lite calibration questions
  (no RSHR question is used for fitting). The calibration questions of MME-RS (90), LRS-VQA (240) and XLRS-lite (130)
  are listed in `data/splits/*_dev.json`.
- **Veto.** The absence option is detected by the regular expression `cas.NOT_FEATURED` ("does not feature", MME-RS
  option E).
- **Two-image questions** (XLRS-lite change detection): the search runs on each image, pointer trust is not used, and
  both presentations show the views of both images.
