"""The answerer, the region head and the pipeline on a tiny random model of each configuration's answerer, against
golden values recorded from the code behind the paper's results (fixtures/tiny_golden.json).

The tiny models (fixtures/tiny_model.py) have the transformers classes of Qwen3.5-9B (Qwen3_5ForConditionalGeneration:
linear- and full-attention layers, the vision tower) and of Qwen3-VL-8B (Qwen3VLForConditionalGeneration: deepstack
vision features, its generation configuration) with shrunk sizes and numpy-seeded weights, float32 on the CPU; the
processor and tokenizer are the real ones (files under $EARTHJEV_MODELS, no weights). Checked: greedy generation
and its log-probs (Answerer.generate, text exact), the Yes/No evidence probability, region-head logits of 17 views of
different shapes (two right-padded batches), and complete pipeline records (EarthJev.run_row) on synthetic images
with real question rows. On the reference machine (x86-64, AVX-512) these equal the original code
bitwise, with 1 or 4 threads and with the AVX2 kernels. Floats are compared to a relative 1e-4: other float32 kernels
move them (4 threads: 1.7e-7 relative; the AVX2 kernels: 8.3e-6 on the region-head logits), a logic error by far more.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import transformers

from earthjev import pipeline, prompts
from earthjev.answerer import Answerer
from earthjev.paths import load_config
from earthjev.region_head import RegionScorer, decision_prompt

FIXTURES = Path(__file__).parent / 'fixtures'
MODELS = Path(os.environ.get('EARTHJEV_MODELS', '/nonexistent'))
CONFIGS = ('qwen35-9b', 'qwen3-vl-8b')
pytestmark = pytest.mark.model
GOLDEN = json.loads((FIXTURES / 'tiny_golden.json').read_text())
CASES = GOLDEN['cases']
REL = 1e-4


def _module(name):
    spec = importlib.util.spec_from_file_location(name, FIXTURES / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fake, tiny_model = _module('pipeline_fake'), _module('tiny_model')


@pytest.fixture(scope='module', params=CONFIGS)
def tiny(request, tmp_path_factory):
    """(answerer, configuration with the tiny region head, golden values) around the tiny model of a configuration."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5
    cfg = copy.deepcopy(load_config(request.param))
    model_dir = MODELS / cfg['answerer']['dir']
    if not (model_dir / 'preprocessor_config.json').is_file():
        pytest.skip(f'needs the {cfg["answerer"]["dir"]} processor files under $EARTHJEV_MODELS')
    with pytest.MonkeyPatch.context() as mp:
        for name, value in tiny_model.PURE_TORCH.items():
            mp.setattr(qwen3_5, name, value)
        model = tiny_model.tiny_model(model_dir)
        mp.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', lambda path, **kw: model)
        cfg['weights']['region_head'] = str(tiny_model.tiny_head(tmp_path_factory.mktemp('head')))
        a = cfg['answerer']
        yield Answerer(model_dir, a['dtype'], a['attn_implementation'], a['max_pixels']), cfg, GOLDEN[request.param]


def same(want, got):
    assert not (m := fake.mismatches(want, got, REL)), m[:5]


def test_generate(tiny):
    answerer, _, golden = tiny
    for (row, images, text, n), want in zip(CASES['generate'], golden['research']['generate']):
        got = answerer.generate([(fake.synthetic(p), px) for p, px in images], text, n,
                                system=prompts.answer_system(row))
        assert got['output'] == want['output'] and got['ended'] == want['ended']
        same(want, got)


def test_yes_probability(tiny):
    answerer, _, golden = tiny
    got = [answerer.yes_probability(fake.synthetic(name), q) for name, q in CASES['yes']]
    same(golden['research']['yes'], got)


def test_region_head_logits(tiny):
    answerer, cfg, golden = tiny
    case = CASES['region']
    z = RegionScorer(answerer, cfg['weights']['region_head']).logits([fake.synthetic(p) for p in case['crops']],
                                                                     *decision_prompt(case['stem']))
    same(golden['research']['region'], [float(x) for x in z])


def test_pipeline_records(tiny, tmp_path, monkeypatch):
    answerer, cfg, golden = tiny
    rows = CASES['rows']
    for name in {n for r in rows for n in r.get('images') or [r['image']]}:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        fake.synthetic(name).save(tmp_path / name, compress_level=1)
    monkeypatch.setenv('EARTHJEV_DATA', str(tmp_path))
    method = pipeline.EarthJev.load(cfg, answerer)
    for row, want in zip(rows, golden['records']):
        rec = method.run_row(row)
        got = {k: rec.get(k) for k in ('error', 'final', 'render', 'P', 'outputs', 'gen', 'views', 'decision')}
        got['spv'] = [dict(route=s['route'], trust=s['trust']) for s in rec['spv']]
        same(want, got)
