"""The live pipeline (earthjev.pipeline) on a fake answering network, against traces recorded from the code behind the
paper's results.

Only the network is fake (fixtures/pipeline_fake.py: every output is a function of the exact model inputs); the
answerer's processor and tokenizer (files under $EARTHJEV_MODELS, no weights), image decoding, crops, renders, prompts,
region head, pointer verifier, search, CAS and veto run for real. A trace holds every model call in order (input
digest, flags, outputs), per image the SPV / EDM record (route, trust, pointer views with p, search result, pool,
views, canvas digest) and the question's record (answers, log-probs, decision, final answer).
- fixtures/pipeline_synthetic_<config>.json.gz: real question rows on seeded synthetic images made here: a 224 px
  image (the root is a leaf), a 200 x 100 px image (the search fails: an error record), 224 px leaves, nodes without
  children, deep leaves and early stops, 25:1 and 115:1 strips, stem- and option-pointer rows, a two-image question;
- fixtures/pipeline_real_<config>.json.gz (pytest -m data): real benchmark rows, images under $EARTHJEV_DATA.
Both configurations (Qwen3.5-9B and Qwen3-VL-8B: their own processors, chat templates and tokenizers) are checked.
The traces are exact on the reference machine (x86-64 with AVX-512); floats are compared to 1e-6 (relative), because
other CPUs may round the region head's and the verifier's float32 products differently.
"""
import gzip
import importlib.util
import json
import os
from pathlib import Path

import pytest
import transformers

from earthjev import data, pipeline
from earthjev.paths import load_config

FIXTURES = Path(__file__).parent / 'fixtures'
_spec = importlib.util.spec_from_file_location('pipeline_fake', FIXTURES / 'pipeline_fake.py')
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)
CONFIGS = ('qwen35-9b', 'qwen3-vl-8b')


def needs_processor(config):
    """Skip unless the processor files of the configuration's answerer are under $EARTHJEV_MODELS."""
    d = Path(os.environ.get('EARTHJEV_MODELS', '/nonexistent')) / load_config(config)['answerer']['dir']
    if not (d / 'preprocessor_config.json').is_file():
        pytest.skip(f'needs the {d.name} processor files under $EARTHJEV_MODELS')


def load(name):
    with gzip.open(FIXTURES / name, 'rt') as f:
        return json.load(f)


def image_record(f):
    keys = ('bbox', 'crop', 'option', 'coord', 'conf', 'note', 'p', 'gate', 'confidence')
    s = f['search']
    return dict(route=f['route'], trust=f['trust'], pi=f['pi'], proposals=[{k: x[k] for k in keys}
                                                                         for x in f['pointer_views']],
                search=None if s is None else dict(
                    final_selected=[{k: n[k] for k in ('bbox', 'depth', 'confidence', 'full_path')}
                                    for n in s['final_selected']],
                    **{k: s[k] for k in ('n_expansions', 'n_children_scored', 'iterations')}),
                pool=f['pool'], views=f['views'], canvas=fake.image_digest(f['canvas']))


def record(rec):
    """The record's fields compared (decision and spv only when the question got that far)."""
    out = {k: rec.get(k) for k in ('error', 'final', 'render', 'P', 'outputs', 'gen', 'views')}
    if 'spv' in rec:
        out['spv'] = [dict(route=s['route'], trust=s['trust']) for s in rec['spv']]
    if 'decision' in rec:
        out['decision'] = rec['decision']
    return out


def run(rows, monkeypatch, config):
    """Traces of the release pipeline on rows with the fake network."""
    model = {}

    def from_pretrained(path, **kw):
        return model.setdefault('fake', fake.FakeModel(transformers.AutoTokenizer.from_pretrained(path)))

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', from_pretrained)
    method = pipeline.EarthJev.load(load_config(config))
    found, find = [], pipeline.EarthJev.find_evidence
    monkeypatch.setattr(pipeline.EarthJev, 'find_evidence', lambda *a, **k: found.append(find(*a, **k)) or found[-1])
    traces = []
    for row in rows:
        model['fake'].calls.clear()
        found.clear()
        rec = method.run_row(row)
        assert rec['config'] == method.cfg['name']
        traces.append(dict(calls=list(model['fake'].calls), images=[image_record(f) for f in found],
                           record=record(rec)))
    return traces


def check(golden, traces):
    bad = {}
    for row, want, got in zip(golden['rows'], golden['traces'], traces):
        if m := fake.mismatches(want, got, 1e-6):
            bad[row['qid']] = m[:5]
    assert not bad, bad


@pytest.mark.model
@pytest.mark.parametrize('config', CONFIGS)
def test_synthetic_images(tmp_path, monkeypatch, config):
    needs_processor(config)
    golden = load(f'pipeline_synthetic_{config}.json.gz')
    for name in {n for r in golden['rows'] for n in r.get('images') or [r['image']]}:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        fake.synthetic(name).save(tmp_path / name, compress_level=1)
    monkeypatch.setenv('EARTHJEV_DATA', str(tmp_path))
    traces = run(golden['rows'], monkeypatch, config)
    assert traces[1]['record']['error'] == 'ValueError: math domain error'  # a 200 x 100 px image cannot be searched
    check(golden, traces)


@pytest.mark.data
@pytest.mark.model
@pytest.mark.parametrize('config', CONFIGS)
def test_real_rows(monkeypatch, config):
    needs_processor(config)
    golden = load(f'pipeline_real_{config}.json.gz')
    if not os.environ.get('EARTHJEV_DATA') or not all(p.is_file() for r in golden['rows'] for p in data.image_paths(r)):
        pytest.skip('needs the benchmark images under $EARTHJEV_DATA')
    check(golden, run(golden['rows'], monkeypatch, config))


def test_run_rows_resume(tmp_path):
    """--resume keeps the error-free records (dropping error records and a torn last line) and runs the rest; without
    --resume or --overwrite a non-empty output is refused before anything runs."""
    out = tmp_path / 'run.jsonl'
    rows = [dict(qid=f'q{i}', bench='b') for i in range(4)]
    out.write_text(json.dumps(dict(qid='q0', bench='b', error=None)) + '\n'
                   + json.dumps(dict(qid='q1', bench='b', error='ValueError: x')) + '\n{"qid": "q2", "bench"')
    ran = []
    pipeline.run_rows(rows, lambda r: ran.append(r['qid']) or dict(r, error=None), out, resume=True)
    assert ran == ['q1', 'q2', 'q3']
    assert [json.loads(line)['qid'] for line in out.read_text().splitlines()] == ['q0', 'q1', 'q2', 'q3']
    with pytest.raises(FileExistsError):
        pipeline.run_rows(rows, lambda r: ran.append(r['qid']), out)
    assert len(ran) == 3 and len(out.read_text().splitlines()) == 4
    pipeline.run_rows(rows[:2], lambda r: dict(r, error=None), out, overwrite=True)  # a new file
    assert len(out.read_text().splitlines()) == 2
