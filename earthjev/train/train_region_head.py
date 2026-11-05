"""Train the region head on cached features and select it on validation data."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from earthjev import paths
from earthjev.calibration import auroc
from earthjev.region_head import RegionHead, head_logits
from earthjev.train.extract_features import load_features

SEED = 20261009
LR, EPOCHS, BATCH, WEIGHT_DECAY, WARMUP = 1e-3, 8, 512, 0.01, 0.03
WIDTH = 256
SELECT_KINDS = ('centred', 'cell_d1', 'cell_d2')
T_GRID = np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))
EVAL_ROWS = 8192  # rows per forward when scoring cached features


def fit_temperature(logits, labels):
    """T minimising the NLL of sigmoid(z / T) over T_GRID."""
    z, y = torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(labels, dtype=torch.float64)
    nll = [float(torch.nn.functional.binary_cross_entropy_with_logits(z / t, y)) for t in T_GRID]
    return float(T_GRID[int(np.argmin(nll))])


def train_one(X, y, train_rows, val_eval, hidden, device, lr=LR, seed=SEED, log=print):
    """Train a head on X[train_rows] (X: float16 features on `device`); returns (state of the best epoch by
    val_eval(head)['val_auroc'] (the first on ties), that epoch, per-epoch history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = RegionHead(hidden, WIDTH).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_rows) / BATCH)
    total = steps_per_epoch * EPOCHS
    warm = max(1, round(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm,
                                                                 max(0.0, (total - s) / (total - warm))))
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    history, best = [], (-1.0, None, None)
    for epoch in range(1, EPOCHS + 1):
        order = torch.as_tensor(rng.permutation(train_rows), device=device)
        for s in range(steps_per_epoch):
            b = order[s * BATCH:(s + 1) * BATCH]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(X[b].float()), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
        metrics = val_eval(head)
        history.append(dict(epoch=epoch, train_loss=loss.item(), **metrics))
        log(f'epoch {epoch}: ' + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, float)))
        if metrics['val_auroc'] > best[0]:
            best = (metrics['val_auroc'], epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    return best[2], best[1], history


def train(features_dir, tag, out_dir, device):
    """Train the head on the features of features_dir/<tag>.*, pick the epoch, fit T_f, write out_dir/*."""
    X_np, index = load_features(features_dir, tag)
    hidden = X_np.shape[1]
    X = torch.as_tensor(X_np, device=device)
    y = np.array([r['label'] for r in index], np.float32)
    split = np.array([r['split'] for r in index])
    rows = {s: np.flatnonzero(split == s) for s in ('train', 'val', 'calib')}
    sel = rows['val'][np.isin([index[i]['kind'] for i in rows['val']], SELECT_KINDS)]
    kinds = np.array([index[i]['kind'] for i in sel])
    print(f'{tag}: hidden {hidden}; crops ' + ', '.join(f'{s} {len(v)}' for s, v in rows.items())
          + f'; selection crops {len(sel)}', flush=True)

    def val_eval(head):
        z = head_logits(head, X_np, sel, EVAL_ROWS)
        return dict(val_auroc=auroc(y[sel], z), **{f'val_auroc_{k}': auroc(y[sel][kinds == k], z[kinds == k])
                                                    for k in SELECT_KINDS if (kinds == k).any()})

    state, epoch, history = train_one(X, y, rows['train'], val_eval, hidden, device)
    head = RegionHead(hidden, WIDTH).to(device)
    head.load_state_dict(state)
    T = fit_temperature(head_logits(head, X_np, rows['calib'], EVAL_ROWS), y[rows['calib']])
    report = dict(hidden_size=hidden, width=WIDTH, temperature=T, lr=LR, epoch=epoch,
                  val_auroc_select=max(h['val_auroc'] for h in history),
                  val_auroc_all=auroc(y[rows['val']], head_logits(head, X_np, rows['val'], EVAL_ROWS)),
                  n_crops={s: int(len(v)) for s, v in rows.items()},
                  n_questions={s: len({index[i]['id'] for i in v}) for s, v in rows.items()},
                  budget_tokens=256, selection='ZwZ val pooled AUROC over ' + '/'.join(SELECT_KINDS),
                  prompt='earthjev.region_head (Vision-Jev "noul" prompt)')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / 'head.safetensors'))
    (out_dir / 'head.json').write_text(json.dumps(report, indent=1))
    (out_dir / 'training_history.json').write_text(json.dumps(
        [dict(lr=LR, best_epoch=epoch, best_val_auroc=report['val_auroc_select'], history=history)], indent=1))
    print(json.dumps(report, indent=1), flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='qwen35-9b')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    config = Path(a.config).stem
    train(paths.out_root() / 'features' / 'zwz', config, paths.out_root() / 'region_head' / config, a.device)


if __name__ == '__main__':
    main()
