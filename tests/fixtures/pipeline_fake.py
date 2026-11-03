"""Fake answering network of the pipeline simulation tests (tests/test_pipeline.py).

FakeModel stands in for the Qwen3.5 model and nothing else: the processor, tokenizer, image decoding, crops, renders,
prompts, region head, verifier, search, CAS and veto all run for real. Every output is a deterministic function of the
exact model inputs (sha256 of the input tensors, max_new_tokens and do_sample), so two implementations that make the
same model calls get the same outputs, and any difference in a model input changes everything after it.

model.model(**batch)          last-layer hidden states (bf16, B x L x 4096): per row a seeded Gaussian vector plus a
                              seeded drift along the positions, so z_v depends on the row and on the token that is read
model(**inputs, logits_to_keep=1)
                              next-token logits (bf16, 1 x 1 x V): seeded Yes and No logits, zeros elsewhere
model.generate(**inputs, ...) a seeded answer among plausible ones for the answer format the system turn asks for
                              (letters and letter forms, letter sets, words and numbers, empty or long answers), then
                              <|im_end|> and <|endoftext|>, text after <|im_end|>, or text without <|im_end|> up to
                              the cap; per step float32 logits (seeded noise, the generated token on top; rarely 400
                              below the rest, so that calibrated probabilities underflow to exact ties)
mode 'tie': generation is seeded by the last TAIL prompt tokens only (the question and the generation prompt, the same
in both presentations), and absence options are preferred: both presentations get the same answer and log-probs.
Every call is appended to .calls as (kind, input digest, flags, output summary).
synthetic(name) draws the test image of a name 'synthetic/<w>x<h>_<seed>.png' (seeded numpy; the same pixels on every
platform), e.g. for rows whose image is replaced by a synthetic one (the shared question rows of the tests).
mismatches(a, b, rel) lists the differences of two JSON-like values, floats compared to a relative tolerance.
"""
import hashlib
import math
import re
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

HIDDEN = 4096
TAIL = 24
HASHED = ('max_new_tokens', 'do_sample', 'logits_to_keep')  # the non-tensor arguments that can change an output
ANSWERS = {
    'mcq': ['A', 'B', 'C', 'D', 'E', '(B)', 'C.', 'D) the fourth one', 'The answer is A', 'E', 'B', '', 'none of them',
            'The image shows a long row of small buildings next to a wide road and a river bank'],
    'multi': ['A', 'B C', 'A, D', '(C)', 'B D E', 'C', 'A B C D E', ''],
    'open': ['yes', 'no', '2', '3', '12', 'road', 'ship', 'white', 'left', 'airplane', 'A', '',
             'a large number of small residential buildings are located along the river bank near the bridge'],
}
TAILS = ('\nThe answer above is final.', ' and more', '\n\n')


def synthetic(name):
    """The RGB test image of a name like 'synthetic/2016x1500_3.png': smooth gradients, blocks and seeded noise."""
    w, h, seed = map(int, re.fullmatch(r'(?:.*/)?(\d+)x(\d+)_(\d+)\.png', name).groups())
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // max(1, w - 1)), (y * 255 // max(1, h - 1)), ((x // 37 + y // 23) % 7) * 36], -1)
    noise = rng.integers(0, 48, (h, w, 3))
    return Image.fromarray(((base + noise) % 256).astype(np.uint8))


def mismatches(a, b, rel, path=''):
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=rel, abs_tol=rel * 1e-3) else [f'{path}: {a!r} != {b!r}']
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [m for k in a for m in mismatches(a[k], b[k], rel, f'{path}.{k}')]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in mismatches(x, y, rel, f'{path}[{i}]')]
    return [] if a == b else [f'{path}: {str(a)[:200]} != {str(b)[:200]}']


def image_digest(im):
    return f'{im.mode}|{list(im.size)}|' + hashlib.sha256(im.tobytes()).hexdigest()


def tensor_digest(t):
    t = t.detach().cpu().contiguous()
    return f'{t.dtype}|{list(t.shape)}|' + hashlib.sha256(t.view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def call_digest(kind, kw):
    h = hashlib.sha256(kind.encode())
    for k in sorted(kw):
        if torch.is_tensor(kw[k]):
            h.update(f'{k}|{tensor_digest(kw[k])}'.encode())
        elif k in HASHED:
            h.update(f'{k}={kw[k]!r}'.encode())
    return h.hexdigest()


def rng_of(hexdigest):
    return np.random.default_rng([int(hexdigest[i:i + 8], 16) for i in range(0, 64, 8)])


class _Inner:
    def __init__(self, outer):
        self.outer = outer

    def __call__(self, **kw):
        return self.outer.hidden(kw)


class FakeModel(torch.nn.Module):
    """The fake network; tokenizer: the real Qwen3.5 tokenizer (for the answer tokens)."""

    def __init__(self, tokenizer, mode='hash'):
        super().__init__()
        self.tok, self.mode, self.calls = tokenizer, mode, []
        self.vocab = len(tokenizer)
        self.im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        self.yes, self.no = (tokenizer.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'No'))
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # device / parameters() of the model
        self.lm_head = torch.nn.Linear(1, 1, bias=False)
        self.model = _Inner(self)

    @property
    def device(self):
        return self.anchor.device

    def _flags(self, kw):
        return {k: repr(v) for k, v in sorted(kw.items()) if not torch.is_tensor(v)}

    def hidden(self, kw):
        d = call_digest('hidden', kw)
        b, n = kw['input_ids'].shape
        rng = rng_of(d)
        base = rng.standard_normal((b, 1, HIDDEN), dtype=np.float32)
        drift = rng.standard_normal((1, 1, HIDDEN), dtype=np.float32)
        h = base + (np.arange(n, dtype=np.float32)[None, :, None] / n) * drift
        self.calls.append(dict(kind='hidden', digest=d, flags=self._flags(kw), rows=b))
        return SimpleNamespace(last_hidden_state=torch.from_numpy(h).to(torch.bfloat16))

    def forward(self, **kw):
        d = call_digest('forward', kw)
        yes, no = (float(x) for x in rng_of(d).normal(0, 2.5, 2))
        logits = torch.zeros(1, 1, self.vocab, dtype=torch.bfloat16)
        logits[0, 0, self.yes], logits[0, 0, self.no] = yes, no
        self.calls.append(dict(kind='forward', digest=d, flags=self._flags(kw), out=[float(logits[0, 0, self.yes]),
                                                                                       float(logits[0, 0, self.no])]))
        return SimpleNamespace(logits=logits)

    def _answer(self, rng, prompt):
        head = prompt[:300]
        fmt = 'multi' if 'letter(s)' in head else 'mcq' if 'letter of the correct option' in head else 'open'
        if self.mode == 'tie' and fmt == 'mcq' and "feature" in prompt and rng.random() < 0.6:
            return 'E'
        answers = ANSWERS[fmt]
        return answers[int(rng.integers(len(answers)))]

    def generate(self, **kw):
        ids = kw['input_ids'][0].tolist()
        d = call_digest('generate', kw)
        if self.mode == 'tie':
            d = call_digest('generate-tail', dict(input_ids=torch.tensor(ids[-TAIL:]),
                                                  **{k: kw[k] for k in HASHED if k in kw}))
        rng, n = rng_of(d), kw['max_new_tokens']
        new = self.tok.encode(self._answer(rng, self.tok.decode(ids)), add_special_tokens=False)
        r = rng.random()
        if r < 0.55:
            new += [self.im_end, self.eos]
        elif r < 0.7:
            new += [self.im_end] + self.tok.encode(TAILS[int(rng.integers(len(TAILS)))], add_special_tokens=False)
            new += [self.eos]
        elif r < 0.85:
            new += [self.im_end] + [int(x) for x in rng.integers(1000, 100000, n)]
        else:  # no <|im_end|>: text up to the cap (a real model generates at least one token)
            new += [int(x) for x in rng.integers(1000, 100000, n)]
        new = new[:n]
        if self.eos in new:
            new = new[:new.index(self.eos) + 1]
        saturate = rng.random() < 0.08
        steps = []
        for t in new:
            x = rng.standard_normal(self.vocab, dtype=np.float32) * 2
            x[t] = x.min() - 400 if saturate else x.max() + rng.uniform(0.5, 6)
            steps.append(torch.from_numpy(x)[None])
        self.calls.append(dict(kind='generate', digest=d, flags=self._flags(kw), out=new))
        sequences = torch.cat([kw['input_ids'], torch.tensor([new], dtype=kw['input_ids'].dtype)], 1)
        if kw.get('return_dict_in_generate'):
            return SimpleNamespace(sequences=sequences, logits=tuple(steps) if kw.get('output_logits') else None)
        return sequences
