"""A tiny, randomly initialized model of the same transformers classes as an answerer (tests/test_answerer_tiny.py).

tiny_model(model_dir) builds the model class of model_dir's configuration from that configuration with shrunk sizes,
float32 on the CPU with SDPA attention:
- Qwen3.5 (Qwen3_5ForConditionalGeneration): text 4 layers (3 linear-attention + 1 full-attention), hidden 64, the full
  vocabulary; vision 2 blocks, hidden 32; the PyTorch linear-attention kernels (PURE_TORCH);
- Qwen3-VL (Qwen3VLForConditionalGeneration): text 4 layers, hidden 64, the full vocabulary; vision 2 blocks, hidden
  32, both feeding the text model (deepstack); the snapshot's generation configuration (its end-of-turn tokens).
Every parameter is drawn from numpy's PCG64 (seeded, in sorted name order), so the weights do not depend on torch's
CPU random kernels. tiny_head(dir) writes a random region head (head.json, head.safetensors) of matching hidden size.
"""
import json
from pathlib import Path

import numpy as np
import torch

SEED = 20261102
HIDDEN, HEAD_WIDTH = 64, 16


# names in transformers.models.qwen3_5.modeling_qwen3_5 to replace before building the model, so that it uses its
# PyTorch fallbacks: the kernels of flash-linear-attention / causal-conv1d (when installed) need CUDA tensors
PURE_TORCH = dict(causal_conv1d_fn=None, causal_conv1d_update=None, chunk_gated_delta_rule=None,
                  fused_recurrent_gated_delta_rule=None, FusedRMSNormGated=None, is_fast_path_available=False)


def tiny_config(model_dir):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    text = dict(hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=32)
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN)
    if cfg.model_type == 'qwen3_5':
        text.update(layer_types=['linear_attention'] * 3 + ['full_attention'], linear_key_head_dim=16,
                    linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0)
        section = [2, 1, 1]  # rotary dims: head_dim 32 x partial_rotary_factor 0.25 / 2
    elif cfg.model_type == 'qwen3_vl':
        vision.update(deepstack_visual_indexes=[0, 1])
        section = [6, 5, 5]  # head_dim 32 / 2
    else:
        raise ValueError(f'no tiny variant of {cfg.model_type}')
    cfg.text_config.update(text)
    cfg.text_config.rope_parameters = dict(cfg.text_config.rope_parameters, mrope_section=section)
    cfg.vision_config.update(vision)
    cfg._attn_implementation = 'sdpa'
    cfg.text_config._attn_implementation = cfg.vision_config._attn_implementation = 'sdpa'
    return cfg


def _fill(module, rng, scale):
    for _, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        p.data = torch.from_numpy(rng.normal(0, scale, tuple(p.shape)).astype(np.float32))


def tiny_model(model_dir, seed=SEED):
    """The tiny model in eval mode (float32, CPU); build a Qwen3.5 with PURE_TORCH in place."""
    import transformers
    cfg = tiny_config(model_dir)
    with torch.device('cpu'):
        model = getattr(transformers, cfg.architectures[0])(cfg)
    _fill(model, np.random.default_rng(seed), 0.08)
    if (Path(model_dir) / 'generation_config.json').is_file():  # as from_pretrained loads it
        model.generation_config = transformers.GenerationConfig.from_pretrained(model_dir)
    return model.eval()


def tiny_head(out_dir, seed=SEED + 1):
    """A random region head for the tiny model's hidden size, written as head.json + head.safetensors."""
    from safetensors.torch import save_file
    rng = np.random.default_rng(seed)
    shapes = {'net.0.weight': (HIDDEN,), 'net.0.bias': (HIDDEN,), 'net.1.weight': (HEAD_WIDTH, HIDDEN),
              'net.1.bias': (HEAD_WIDTH,), 'net.3.weight': (1, HEAD_WIDTH), 'net.3.bias': (1,)}
    state = {k: torch.from_numpy(rng.normal(0, 0.5, s).astype(np.float32)) for k, s in sorted(shapes.items())}
    state['net.0.weight'] += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / 'head.safetensors'))
    (out_dir / 'head.json').write_text(json.dumps(dict(hidden_size=HIDDEN, width=HEAD_WIDTH, temperature=1.0)))
    return out_dir
