"""Extract frozen visual-language features for region-head training."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from earthjev import paths
from earthjev.answerer import Answerer
from earthjev.region_head import decision_prompt, last_hidden, prompt_batch, view
from earthjev.train import zwz

BLOCK = 500  # questions per output block
ROWS_PER_FORWARD = 32


def items():
    """[{id, image, stem, split, crops}] of the ZwZ questions in manifest order."""
    return [dict(id=q['qid'], image=str(zwz.image_root() / q['image']), stem=q['stem'], split=q['split'],
                 crops=zwz.crops(q)) for q in zwz.load_manifest()]


class _Questions(torch.utils.data.Dataset):
    """Worker side: read the image once, crop, make the views and the question's prompt batch (CPU)."""

    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = dict(self.items[i])
        image = Image.open(it['image'])
        image.load()
        image = image if image.mode == 'RGB' else image.convert('RGB')
        views = [view(image.crop(tuple(c['bbox']))) for c in it['crops']]
        del image
        return it, prompt_batch(self.processor, views, *decision_prompt(it['stem']))


def chunks(batch, n):
    """Split a right-padded prompt batch into chunks of at most n rows (pixel rows follow image_grid_thw), dropping
    the columns that are padding in every row of a chunk."""
    rows = len(batch['input_ids'])
    if rows <= n:
        yield batch
        return
    offsets = np.cumsum([0] + batch['image_grid_thw'].prod(dim=1).tolist())
    for s in range(0, rows, n):
        e = min(rows, s + n)
        sub = {k: v[s:e] for k, v in batch.items() if k not in ('pixel_values', 'image_grid_thw')}
        sub['image_grid_thw'] = batch['image_grid_thw'][s:e]
        sub['pixel_values'] = batch['pixel_values'][offsets[s]:offsets[e]]
        width = int(sub['attention_mask'].sum(dim=1).max())
        for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
            if key in sub:
                sub[key] = sub[key][:, :width]
        yield type(batch)(sub)


def _file(out_dir, tag, shard, k):
    part = f's{shard[0]}of{shard[1]}.b{k:03d}'
    return out_dir / f'{tag}.{part}.npy', out_dir / f'index.{part}.jsonl'


def extract(config, set_name, shard=(1, 1), workers=6):
    """Write the feature blocks of shard i/n of a set (see the module docstring)."""
    cfg, tag = paths.load_config(config), Path(config).stem
    i, n = shard
    if not 1 <= i <= n:
        raise ValueError(f'shard {i}/{n}: needs 1 <= i <= n')
    todo_items = items()
    todo_items = todo_items[(i - 1) * len(todo_items) // n:i * len(todo_items) // n]
    out_dir = paths.out_root() / 'features' / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [range(s, min(s + BLOCK, len(todo_items))) for s in range(0, len(todo_items), BLOCK)]
    todo = [k for k in range(len(blocks)) if not _file(out_dir, tag, shard, k)[0].exists()]
    print(f'{set_name} shard {i}/{n}: {len(todo_items)} questions, {len(blocks)} blocks, {len(todo)} to do', flush=True)
    if not todo:
        return
    a = cfg['answerer']
    answerer = Answerer(paths.models_root() / a['dir'], a['dtype'], a['attn_implementation'], a['max_pixels'])
    t0, n_done = time.perf_counter(), 0
    for k in todo:
        loader = torch.utils.data.DataLoader(_Questions([todo_items[j] for j in blocks[k]], answerer.processor),
                                             batch_size=None, num_workers=workers,
                                             prefetch_factor=4 if workers else None)
        feats, index = [], []
        for it, batch in loader:
            h = torch.cat([last_hidden(answerer.model, sub) for sub in chunks(batch, ROWS_PER_FORWARD)])
            feats.append(h.float().cpu().numpy().astype(np.float16))
            index += [dict(id=it['id'], split=it['split'], j=j, **c) for j, c in enumerate(it['crops'])]
            n_done += 1
        npy, jsonl = _file(out_dir, tag, shard, k)
        arr = np.concatenate(feats)
        assert len(arr) == len(index)
        part = jsonl.with_name(jsonl.name + '.part')
        part.write_text(''.join(json.dumps(r) + '\n' for r in index))
        part.replace(jsonl)
        part = npy.with_name(npy.name + '.part')
        with open(part, 'wb') as f:
            np.save(f, arr)
        part.replace(npy)
        rate = n_done / (time.perf_counter() - t0)
        print(f'block {k + 1}/{len(blocks)}: {len(index)} crops; {rate:.2f} questions/s', flush=True)


def load_features(features_dir, tag):
    """(features float16 [N, hidden], index rows) of a feature set over all shards and blocks, in shard/block order."""
    files = sorted(Path(features_dir).glob(f'{tag}.s*of*.b*.npy'),
                   key=lambda p: tuple(map(int, re.search(r'\.s(\d+)of(\d+)\.b(\d+)\.npy$', p.name).group(1, 3))))
    if not files:
        raise FileNotFoundError(f'no {tag}.s*of*.b*.npy features in {features_dir}')
    feats, index = [], []
    for p in files:
        part = re.search(r'\.(s\d+of\d+\.b\d+)\.npy$', p.name).group(1)
        feats.append(np.load(p))
        index += [json.loads(line) for line in open(p.parent / f'index.{part}.jsonl')]
    return np.concatenate(feats), index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='zwz', choices=['zwz'])
    ap.add_argument('--config', default='qwen35-9b')
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args(argv)
    extract(a.config, a.set, tuple(map(int, a.shard.split('/'))), a.workers)


if __name__ == '__main__':
    main()
