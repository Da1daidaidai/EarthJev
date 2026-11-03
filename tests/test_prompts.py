"""Prompt text (earthjev.prompts) and the answerer's model inputs, letter tokens and generation bookkeeping
(earthjev.answerer). The golden values in fixtures/prompts_golden.json were produced by the code behind the paper's
results. The answerer tests run the real Qwen3.5 processor (files under $EARTHJEV_MODELS, no weights) on a fake
model."""
import hashlib
import json
import os
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from earthjev import prompts
from earthjev.paths import load_config
from earthjev.region_head import decision_prompt

GOLDEN = json.loads((Path(__file__).parent / 'fixtures' / 'prompts_golden.json').read_text())
MODEL_DIR = Path(os.environ.get('EARTHJEV_MODELS', '/nonexistent')) / load_config()['answerer']['dir']
needs_processor = pytest.mark.skipif(not (MODEL_DIR / 'preprocessor_config.json').is_file(),
                                     reason='needs the Qwen3.5 processor files under $EARTHJEV_MODELS')


def pattern(w, h):
    """Deterministic high-frequency test image."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    arr = np.stack([(x * 7 + y * 13 + (x * y) % 251) % 256, (x * 3 + y * 5) % 256, (x ^ y) % 256], -1)
    return Image.fromarray(arr.astype(np.uint8))


def tensor_digest(t):
    return f'{t.dtype}|{list(t.shape)}|' + hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


class FakeModel:
    """Stands in for the Qwen3.5 model on CPU. Each call records its flags and input digests; logits are seeded by the
    input ids. generate() follows the argmax for max_new_tokens steps, with an <|im_end|> at a seeded step (or none)
    and text tokens around it, as the real model may continue after <|im_end|>."""
    device = torch.device('cpu')

    def __init__(self, vocab, im_end, text_ids):
        self.vocab, self.im_end, self.text_ids, self.calls = vocab, im_end, text_ids, []

    def eval(self):
        return self

    def _record(self, kind, kw):
        self.calls.append(dict(kind=kind, flags={k: repr(v) for k, v in sorted(kw.items()) if not torch.is_tensor(v)},
                               tensors={k: tensor_digest(v) for k, v in sorted(kw.items()) if torch.is_tensor(v)}))
        return zlib.crc32(kw['input_ids'].numpy().tobytes())

    def __call__(self, **kw):
        g = torch.Generator().manual_seed(self._record('forward', kw))
        return SimpleNamespace(logits=(torch.randn(1, 1, self.vocab, generator=g) * 3).to(torch.bfloat16))

    def generate(self, **kw):
        seed = self._record('generate', kw)
        g, stop, steps = torch.Generator().manual_seed(seed), seed % (kw['max_new_tokens'] + 8), []
        for t in range(kw['max_new_tokens']):
            logits = torch.randn(1, self.vocab, generator=g) * 3
            logits[0, self.im_end if t == stop else self.text_ids[(seed + 3 * t) % len(self.text_ids)]] += 50
            steps.append(logits.to(torch.bfloat16).float())
        new = torch.tensor([[int(x[0].argmax()) for x in steps]])
        return SimpleNamespace(sequences=torch.cat([kw['input_ids'], new], 1), logits=tuple(steps))


def fake_model(tokenizer):
    words = ['A', 'B', 'C', ' D', '(E', 'Yes', 'no', ' road', ' ship', ')', ' 3', 'harbor']
    return FakeModel(len(tokenizer), tokenizer.convert_tokens_to_ids('<|im_end|>'),
                     [tokenizer.encode(w, add_special_tokens=False)[0] for w in words])


def test_prompt_strings():
    hint = prompts.position_hint()
    for case in GOLDEN['rows']:
        row = case['row']
        assert prompts.answer_system(row) == case['system']
        assert prompts.canvas_answer(row, hint) == case['canvas']
        assert prompts.search_query(row) == case['query']
        assert prompts.evidence_question(prompts.search_query(row)) == case['evidence']
        assert decision_prompt(row['stem'])[0] == case['decision']  # the region head's question
        assert prompts.memory_answer(row, tuple(case['size']), case['boxes'], case['notes']) == case['memory']
        sizes = [tuple(s) for s in case['sizes']]
        assert prompts.memory_answer_multi(row, sizes, case['boxes_multi']) == case['memory_multi']
    assert decision_prompt('')[1] == GOLDEN['decision_state']


@pytest.mark.model
@needs_processor
def test_answerer_on_a_fake_model(monkeypatch):
    import transformers

    from earthjev.answerer import Answerer

    fake = fake_model(transformers.AutoTokenizer.from_pretrained(MODEL_DIR))
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', lambda *a, **k: fake)
    ans = Answerer(MODEL_DIR)
    assert ans.letter_token_ids == GOLDEN['letter_token_ids'] and [ans.yes_id, ans.no_id] == GOLDEN['yes_no']
    for case in GOLDEN['generate']:
        out = ans.generate([(pattern(*size), px) for size, px in case['images']], case['text'],
                           case['max_new_tokens'], system=case['system'])
        assert fake.calls[-1] == case['call'] and out == case['out']
        assert len(out['letter_lp']) == 5 and '<|im_end|>' not in out['output']
        assert out['ended'] or len(out['tok_lp']) == case['max_new_tokens']
    for case in GOLDEN['yes']:
        y = ans.yes_probability(pattern(*case['size']), case['query'])
        assert fake.calls[-1] == case['call'] and y == case['y'] and 0 <= y <= 1
