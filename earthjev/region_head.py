# The head architecture (NoulHead, vision_jev/model/heads.py) and the 'noul' prompt (SYSTEM_PROMPT, question_text and
# conversation, vision_jev/train/sft.py) are adapted from Vision-Jev (https://github.com/LMDHQ-0420/Vision-Jev, commit
# 39b0e7b), Copyright 2026 Vision-JEV Contributors, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modified: reduced to the 'noul' task and inlined; the head returns
# only its logit.
"""Region decision head f_phi of the EDM (paper Sec. III-C, region score p(c | q)).

z_v(c, q) = f_phi(h_theta(c, q)): the frozen answerer reads a candidate view c (at most 256 visual tokens) under the
evidence-querying prompt below, and the two-layer head maps the hidden state of the last prompt token to a logit.
The prompt is the Vision-Jev 'noul' prompt the head was trained with, including its 'You are Vision-Jev.' system
line: the shipped head, the pointer verifier and every fitted map depend on these exact tokens, so the text must not
change. At inference the question stem has its pointers removed (earthjev.pointers.strip) before decision_prompt.

No temperature is applied here: the search and the pointer views divide z_v by the verifier temperature T_g
(earthjev.spv); the head's own temperature T_f in head.json is recorded only.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from earthjev.answerer import DECISION_PIXELS, MIN_PIXELS, prepare_image

SYSTEM_PROMPT = 'You are Vision-Jev. Read the visual evidence and return only the requested compact JSON.'
QUESTION = 'Does this image show the object or area that this question asks about: '
STATE = 'The image is a zoomed-in crop of a much larger high-resolution image.'
BATCH = 16  # views per forward at inference (right-padded)


def decision_prompt(stem):
    """(question, state) of the region head for a question stem."""
    return QUESTION + stem, STATE


def prompt_text(question, state=STATE):
    """The user text of Vision-Jev's 'noul' task: sections joined by blank lines."""
    sections = ['Task: noul']
    if state.strip():
        sections.append(f'Visible state: {state.strip()}')
    sections += [f'Question: {question}', 'Return exactly {"noul":true} or {"noul":false}.']
    return '\n\n'.join(sections)


def view(crop):
    """The head's view of a crop: at most 256 visual tokens (DECISION_PIXELS), at least the processor floor."""
    return prepare_image(crop, DECISION_PIXELS, MIN_PIXELS)


def prompt_batch(processor, views, question, state=STATE):
    """Right-padded processor batch of one 'noul' conversation per view (views already at their 256-token size)."""
    text = prompt_text(question, state)
    conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
                      {'role': 'user', 'content': [{'type': 'image', 'image': v}, {'type': 'text', 'text': text}]}]
                     for v in views]
    # thinking off only for chat templates that have the switch (Qwen3.5, not Qwen3-VL): transformers routes a keyword the
    # template does not use into processor_kwargs, replacing the padding and image budget below
    think = {'enable_thinking': False} if 'enable_thinking' in str(processor.chat_template) else {}
    return processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, **think, return_dict=True, return_tensors='pt',
        processor_kwargs={'padding': True, 'padding_side': 'right', 'images_kwargs': {
            'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': DECISION_PIXELS}}})


@torch.no_grad()
def last_hidden(model, batch):
    """h_theta: the last-layer hidden state at the last non-padding token of each row of a right-padded batch."""
    batch = batch.to(model.device)
    out = model.model(**batch, return_dict=True, use_cache=False)
    last = batch['attention_mask'].sum(dim=1) - 1
    return out.last_hidden_state[torch.arange(len(last), device=last.device), last]


class RegionHead(nn.Module):
    """f_phi: LayerNorm -> Linear(hidden, width) -> GELU -> Linear(width, 1), returning the logit z_v."""

    def __init__(self, hidden_size=4096, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(),
                                 nn.Linear(width, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)

    @classmethod
    def load(cls, weights_dir, device='cpu'):
        """The trained head of a weights directory (head.json, head.safetensors), in eval mode."""
        weights_dir = Path(weights_dir)
        meta = json.loads((weights_dir / 'head.json').read_text())
        head = cls(meta['hidden_size'], meta['width'])
        head.load_state_dict(load_file(str(weights_dir / 'head.safetensors')))
        return head.to(device).eval()


@torch.no_grad()
def head_logits(head, features, rows=None, chunk=4096):
    """z_v of cached features (numpy [n, hidden], float16) or of the given rows of them, `chunk` rows per forward on
    the head's device. The released z_v caches ran it on the CPU per feature file with chunks of 4,096 rows, head
    training uses 8,192; another row grouping can change the last bits of a few rows (the matmul tail)."""
    idx = np.arange(len(features)) if rows is None else np.asarray(rows)
    device = next(head.parameters()).device
    out = [head(torch.as_tensor(features[idx[s:s + chunk]], device=device).float()).cpu()
           for s in range(0, len(idx), chunk)]
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


class RegionScorer:
    """The region head on the answerer's own model: raw logits z_v of crops for one question."""

    def __init__(self, answerer, weights_dir):
        self.model, self.processor = answerer.model, answerer.processor
        self.head = RegionHead.load(weights_dir, next(self.model.parameters()).device)

    @torch.no_grad()
    def logits(self, images, question, state=STATE):
        """z_v (float32 numpy) of crops (PIL) for a decision question (decision_prompt): each crop at its 256-token
        view, BATCH views per right-padded forward."""
        views = [view(im) for im in images]
        h = torch.cat([last_hidden(self.model, prompt_batch(self.processor, views[i:i + BATCH], question, state))
                       for i in range(0, len(views), BATCH)])
        return self.head(h.float()).cpu().numpy()
