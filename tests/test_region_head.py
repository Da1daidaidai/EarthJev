"""Region head: prompt text and processor batch (golden values of the original code), head forward on cached
features, and deterministic training on synthetic features."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from earthjev import paths
from earthjev.calibration import auroc
from earthjev.region_head import RegionHead, decision_prompt, head_logits, prompt_batch, prompt_text, view
from earthjev.train.train_region_head import fit_temperature, train_one

FIXTURES = Path(__file__).parent / 'fixtures'
WEIGHTS = paths.REPO / 'weights' / 'region_head' / 'qwen35-9b'
GOLDEN = json.loads((FIXTURES / 'region_head_golden.json').read_text())


def test_prompt_text():
    assert prompt_text(*decision_prompt(GOLDEN['stem'])) == GOLDEN['text']
    assert GOLDEN['text'].startswith('Task: noul\n\nVisible state: The image is a zoomed-in crop of a much larger ')
    assert prompt_text('q', '') == 'Task: noul\n\nQuestion: q\n\nReturn exactly {"noul":true} or {"noul":false}.'


def test_head_on_cached_features():
    head = RegionHead.load(WEIGHTS)
    assert sorted(head.state_dict()) == sorted(load_file(str(WEIGHTS / 'head.safetensors')))
    f = np.load(FIXTURES / 'region_head_features.npz')
    np.testing.assert_allclose(head_logits(head, f['features']), f['z'], rtol=0, atol=1e-5)  # bitwise on the
    # reference CPU; other BLAS builds may differ in the last bits


def _synthetic_crops():
    y, x = np.mgrid[0:500, 0:700]
    a = np.stack([(x * 3 + y) % 256, (x + y * 5) % 256, (x * y) % 256], -1).astype(np.uint8)
    y, x = np.mgrid[0:900, 0:300]
    b = np.stack([((x // 30 + y // 30) % 2) * 255, x % 256, y % 256], -1).astype(np.uint8)
    return [Image.fromarray(a), Image.fromarray(b)]


@pytest.mark.model
def test_prompt_batch_matches_original():
    model_dir = Path(os.environ.get('EARTHJEV_MODELS', '/nonexistent')) / 'Qwen3.5-9B'
    if not (model_dir / 'preprocessor_config.json').exists():
        pytest.skip('needs the Qwen3.5-9B processor files under $EARTHJEV_MODELS')
    from transformers import AutoProcessor
    views = [view(c) for c in _synthetic_crops()]
    assert [list(v.size) for v in views] == GOLDEN['view_sizes']
    batch = prompt_batch(AutoProcessor.from_pretrained(model_dir), views, *decision_prompt(GOLDEN['stem']))
    sha = lambda t: hashlib.sha256(f'{t.dtype}{tuple(t.shape)}'.encode() + t.contiguous().numpy().tobytes()).hexdigest()
    assert {k: sha(v) for k, v in batch.items()} == GOLDEN['tensors']
    assert batch['attention_mask'][1].tolist() == sorted(batch['attention_mask'][1].tolist(), reverse=True)  # right pad


def test_training_is_deterministic_and_learns():
    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.normal(size=(20000, 16)).astype(np.float16))
    y = (X[:, 0].float().numpy() > 0.5).astype(np.float32)
    rows = np.arange(len(y))
    ev = lambda head: dict(val_auroc=auroc(y, head_logits(head, X.numpy())))
    runs = [train_one(X, y, rows, ev, 16, 'cpu', log=lambda *_: None) for _ in range(2)]
    assert all(torch.equal(runs[0][0][k], runs[1][0][k]) for k in runs[0][0]) and runs[0][1:] == runs[1][1:]
    assert runs[0][2][-1]['val_auroc'] > 0.97


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    assert abs(fit_temperature(np.log(4) * np.ones(4000) * 2, (rng.random(4000) < 0.8)) - 2.0) < 0.15
    assert auroc([0, 1, 0, 1], [0.1, 0.9, 0.5, 0.5]) == 0.875 and auroc([1, 1], [0.2, 0.3]) is None
