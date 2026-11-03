# Third-party notices

The code of this repository is licensed under the Apache License 2.0 (LICENSE). Parts of it are adapted from other
projects or reproduce official evaluation code, listed below with their licenses. The benchmark data, the answering
models and the Python packages they run with keep their own terms; none of the images, answering-model weights or
packages is redistributed here.

## Code adapted from other projects

Code of the method is taken over only where its license allows it (Vision-Jev: Apache-2.0; VISTA: MIT). ZoomSearch's
search is part of the method and its repository has no license, so it is rewritten: no file or function is copied (apart from two
prompt strings, below), and the code reproduces its behaviour. Evaluation code is the opposite case: it is copied verbatim, so that the official
metrics are reproduced exactly (below).

### Vision-Jev (Apache License 2.0)

- Source: https://github.com/LMDHQ-0420/Vision-Jev, commit `39b0e7ba798901ff4f948891972d4feca5aa159e`
  (`vision_jev/model/heads.py`, `vision_jev/train/sft.py`). Copyright 2026 Vision-JEV Contributors. The license text
  is that of LICENSE; the project ships no NOTICE file.
- Used in `earthjev/region_head.py`: the architecture of the region head (`NoulHead`) and the 'noul' prompt
  (`SYSTEM_PROMPT`, `question_text`, `conversation`). Modified: reduced to the 'noul' task and inlined; the head
  returns only its logit. The prompt text, including its first sentence "You are Vision-Jev.", is kept verbatim,
  because the shipped region head and pointer verifier were trained with it.

### VISTA (MIT License)

- Source: https://github.com/joshhhhhan/VISTA, commit `c97c354f0a8948256545a3e8850a1848f8e5b8c1`. The VISTA authors
  ask to cite arXiv:2610.02200.
- Used in `earthjev/evidence.py` (`render_memory_views`, the memory presentation): the inspection limits of
  `src/vista/core/tools.py` (`MAX_INSPECTION_VIEWS = 16`, `MAX_INSPECTION_QUESTION_CHARS = 1024`), the errors
  "Invalid arguments." and "View {i}: invalid ..." and the rule that fits a view's longer side to the display size,
  enlarging with NEAREST resampling, of `src/vista/core/session.py` (`_inspect`, `_inspection_view`). New here: views
  larger than the display size are box-filtered and downscaled with LANCZOS (VISTA resizes with NEAREST both ways).

```
MIT License

Copyright (c) 2026 VISTA contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ZoomSearch (no license; rewritten)

- Zhou et al., arXiv:2511.20460, https://github.com/kiki-zyq/ZoomSearch (commit
  `d2158264f4dc4e80674788e5be5126e1399efdb1`). The repository has no license file. Its search is part of the method,
  so it is rewritten here: no file or function is copied.
- `earthjev/search.py` (decision-only hierarchical search and node selection), `earthjev/evidence.py` (canvas
  layout) and `earthjev/answerer.py` (Yes/No answer confidence) reproduce its behaviour; the tests replay traces
  recorded from the original.
- Two short prompt strings that the method sends to the model are ZoomSearch's, kept identical because the fitted
  parameters depend on them: the evidence question and the position hint of the canvas presentation
  (`earthjev/prompts.py`).

## Official evaluation code (`earthjev/scoring.py`, blocks marked VERBATIM)

The paper reports each benchmark's official metric, so these functions are copied verbatim, to reproduce the official
metrics exactly (renamed where two had the same name).

| function | source | license |
|---|---|---|
| `extract_characters_regex` | MME-RealWorld, `evaluation/eval_your_results.py`, https://github.com/MME-Benchmarks/MME-RealWorld | not stated: the repository has no license file; its README says "MME-RealWorld is only used for academic research. Commercial use in any form is prohibited." |
| `are_synonyms` | LRS-VQA, `Script/evaluation_LRSVQA.py`, https://github.com/VisionXLab/LRS-VQA (commit `1b3b6cd4`) | not stated for `Script/`: the repository's `LRS-VQA-Code/LICENSE` (Apache-2.0) covers only that directory |
| `_lemmatize`, `gro_are_synonyms` (upstream `are_synonyms`) | ZoomEarth (the LRS-GRO scorer), `src/eval/eval.py`, https://github.com/earth-insights/ZoomEarth (commit `d5174519`) | the README says "(c) 2025 ZoomEarth Project. Released under the Apache 2.0 License."; there is no license file at the root or in `src/eval/` (the `LICENSE` files under `src/train/RL/` cover that bundled training code) |
| `extract_characters_regex_xlrs` (upstream `extract_characters_regex`) | lmms-eval, `lmms_eval/tasks/xlrs/mcq_utils.py`, https://github.com/EvolvingLMMs-Lab/lmms-eval | Apache License 2.0 (the repository licenses `lmms_eval/tasks` under Apache-2.0 and its core under MIT), Copyright (c) 2025 LMMs-Lab. Modified: renamed |

Where no license is stated upstream (the MME-RealWorld and LRS-VQA functions), the code is not covered by this
repository's LICENSE: it remains its authors' and is included unchanged only to compute the official metric. The block
headers in `earthjev/scoring.py` say the same.

`scripts/prepare_data.py` (`mme_size`) restates, without copying it, the size rule of `smart_resize` in the
qwen-vl-utils package (Apache-2.0). `earthjev/answerer.py` imports `smart_resize` from transformers (Apache-2.0).

## Answering models

Downloaded by the user (docs/INSTALL.md), each under the Apache License 2.0:

- Qwen3.5-9B (https://huggingface.co/Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), the
  answering model of `configs/qwen35-9b.json`;
- Qwen3-VL-8B-Instruct (https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`), the answering model of `configs/qwen3-vl-8b.json`.

The trained weights in `weights/region_head/<config>` and `weights/verifier/<config>` were trained on the hidden states of the
configuration's answering model for crops of ZwZ-RL-VQA images (apache-2.0, see below; the images keep the terms of
their sources). They are released with the code under Apache-2.0.

## Benchmark data

The question files (`data/benchmarks/`), question lists (`data/splits/`), the image list (`data/images.tsv`), the
calibration runs (`data/calibration/`, model outputs on calibration questions), the reference records
(`reference/`, model outputs on benchmark questions) and the test fixtures made from benchmark questions are derived
from the datasets below. Their use is subject to the datasets' terms, not to this repository's license; the table
"Terms" in docs/DATA.md gives, per file, the source, the license and the changes made (the XLRS-Bench-lite files and
their derivatives under CC BY-NC-SA 4.0, the LRS-VQA derivatives under CC BY-NC 4.0 with attribution, the LRS-GRO
question files under Apache-2.0). The images are not redistributed: `scripts/prepare_data.py` downloads them from the
official sources (docs/DATA.md).

| dataset | terms stated by the source |
|---|---|
| MME-RealWorld (Hugging Face `yifanzhang114/MME-RealWorld-Lmms-eval`) | dataset card of `yifanzhang114/MME-RealWorld`: apache-2.0; GitHub README: academic research only, no commercial use; the Lmms-eval card states none |
| LRS-VQA (`ll-13/LRS-VQA`) | CC-BY-NC-4.0; images from FAIR1M, GLH-Bridge and STAR |
| XLRS-Bench-lite (`initiacms/XLRS-Bench-lite`) | annotations CC-BY-NC-SA-4.0; images under the terms of their sources (DOTA: academic use, Google Earth terms; ITCVD: CC-BY-NC-SA-4.0; MiniFrance, HRSCD: IGN licence ouverte; ISPRS Toronto, Potsdam: research use, no redistribution) |
| LRS-GRO (`HappyBug/LRS-GRO`) | none stated on the dataset card; the ZoomEarth repository (commit `d5174519`) ships the questions (`LRS-GRO.jsonl`) and its README says "(c) 2025 ZoomEarth Project. Released under the Apache 2.0 License.": the LRS-GRO question files are redistributed on the basis of that statement |
| RSHR-Bench (`RL-MIND/RSHR-Bench`) | Apache-2.0 |
| ZwZ-RL-VQA (`inclusionAI/ZwZ-RL-VQA`, training only) | apache-2.0; images from SA-1B, LAION, Visual Genome and CC12M |
| WordNet 3.0 (NLTK data package `wordnet`, for the scorers) | WordNet 3.0 license, Copyright 2006 Princeton University |

## Python packages

The packages of `requirements/` are installed by the user from PyPI, the PyTorch package index and the GitHub
releases of flash-attn and causal-conv1d, each under its own license (for example torch, flash-attn and
causal-conv1d: BSD-3-Clause; transformers, NLTK and pyarrow: Apache-2.0; flash-linear-attention: MIT).
