"""Train the spatial pointer verifier using region-head logits."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from earthjev import pointers
from earthjev.calibration import auroc
from earthjev.paths import load_config, out_root, weights_dir
from earthjev.spv import PointerVerifier, VerifierNet

SEED = 20261013
RHO = 0.25  # corruption probability of a synthesized pointer (paper: 25% incorrect spatial cues)
P_NONE, P_PRECISE, P_PHRASE = 0.30, 0.70, 0.30
P_KIND = (('box', 0.65), ('point', 0.20), ('options', 0.15))
BOX_FMTS = ('px_res', 'px', 'n1000', 'n1', 'qwen')
POINT_FMTS = ('point_px', 'point_n1')
WIDTH, LR = 256, 1e-3
EPOCHS, BATCH, WD, WARMUP = 8, 512, 0.01, 0.03  # AdamW weight decay; linear warmup over 3% of the steps
SELECT_KINDS = ('centred', 'cell_d1', 'cell_d2')  # crop kinds of the ZwZ-val selection sets
T_GRID = np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))
TARGET_PRECISION = 0.90
TAU_DRAWS = 4  # seeded pointer-view draws per ZwZ-val question for tau
SELECTION = 'mean ZwZ-val AUROC (none, correct, corrupt references; centred/cell_d1/cell_d2 + proposal-mode)'


# ------------------------------------------------------------------ questions and synthesis

def questions(zv_path, manifest_path):
    """[{qid, split, W, H, gt (px box), stem, natural (phrase refs), crops px [N, 4], labels, kinds, z}]."""
    f = np.load(zv_path)
    z, index = f['z'], json.loads(str(f['index']))
    with open(manifest_path) as fm:
        meta = {r['qid']: dict(W=r['W'], H=r['H'], gt=r['bbox'], stem=r['stem']) for r in map(json.loads, fm)}
    by = {}
    for i, r in enumerate(index):
        by.setdefault(r['id'], []).append(i)
    out = []
    for qid, rows in by.items():
        m = meta[qid]
        out.append(dict(qid=qid, split=index[rows[0]]['split'], **m,
                        natural=[r for r in pointers.parse(m['stem']) if r.kind == 'phrase'],
                        crops=np.array([index[i]['bbox'] for i in rows], np.float64),
                        labels=np.array([index[i]['label'] for i in rows], np.float32),
                        kinds=np.array([index[i]['kind'] for i in rows]), z=z[rows]))
    return out


def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256('|'.join(map(str, (SEED,) + parts)).encode()).hexdigest()[:15], 16))


def _clip(b, W, H):
    x1, y1, x2, y2 = b
    w, h = min(x2 - x1, W), min(y2 - y1, H)
    x1, y1 = min(max(0, x1), W - w), min(max(0, y1), H - h)
    return [x1, y1, x1 + w, y1 + h]


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def wrong_region(gt, W, H, rng, tries=50):
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    for _ in range(tries):
        x, y = rng.uniform(0, max(1e-6, W - w)), rng.uniform(0, max(1e-6, H - h))
        b = [x, y, x + w, y + h]
        if not _intersects(b, gt):
            return b
    return None


def corrupt(gt, W, H, rng):
    """A wrong reference: shifted / wrong region / wrong scale (one third each); None if impossible."""
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    kind = rng.integers(3)
    if kind == 0:
        for _ in range(20):
            t, d = rng.uniform(0, 2 * math.pi), rng.uniform(1.5, 4) * max(w, h)
            nx, ny = cx + d * math.cos(t), cy + d * math.sin(t)
            b = _clip([nx - w / 2, ny - h / 2, nx + w / 2, ny + h / 2], W, H)
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (gt[0] <= bx < gt[2] and gt[1] <= by < gt[3]):
                return b
        kind = 1
    if kind == 1:
        return wrong_region(gt, W, H, rng)
    s = rng.uniform(0.25, 0.5) if rng.random() < 0.5 else rng.uniform(3, 6)
    nx, ny = cx + rng.uniform(-0.5, 0.5) * w, cy + rng.uniform(-0.5, 0.5) * h
    return _clip([nx - s * w / 2, ny - s * h / 2, nx + s * w / 2, ny + s * h / 2], W, H)


def jitter(gt, W, H, rng):
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    s = rng.uniform(0.9, 1.1)
    cx, cy = (gt[0] + gt[2]) / 2 + rng.uniform(-0.05, 0.05) * w, (gt[1] + gt[3]) / 2 + rng.uniform(-0.05, 0.05) * h
    return _clip([cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2], W, H)


def precise_refs(q, rng, wrong, kind):
    """Render one precise reference (or an option set) as text and parse it back."""
    W, H, gt = q['W'], q['H'], q['gt']
    if kind == 'options':
        boxes = [gt] + [b for b in (wrong_region(gt, W, H, rng) for _ in range(rng.integers(1, 4))) if b]
        order = rng.permutation(len(boxes))
        opts = [('ABCD'[i], '[{}, {}, {}, {}]'.format(*(round(v) for v in boxes[j]))) for i, j in enumerate(order)]
        return [r for r in pointers.parse(q['stem'] + f' resolution: {W} x {H}', opts, (W, H)) if r.precise]
    box = corrupt(gt, W, H, rng) if wrong else jitter(gt, W, H, rng)
    if box is None:
        return []
    fmts = POINT_FMTS if kind == 'point' else BOX_FMTS
    fmt = fmts[rng.integers(len(fmts))]
    res = None
    if fmt == 'px_res' and rng.random() < 0.2:  # a stated resolution that differs from the file
        s = (0.5, 2.0)[rng.integers(2)]
        res = (round(W * s), round(H * s))
    text = q['stem'] + ' ' + pointers.render(fmt, box, (W, H), res=res)
    return [r for r in pointers.parse(text, (), (W, H)) if r.precise]


def phrase_ref(q, rng, wrong):
    """A synthetic directional phrase (structure only) naming the third that holds the evidence centre (or another)."""
    W, H, gt = q['W'], q['H'], q['gt']
    c = {'u': (gt[0] + gt[2]) / 2 / W, 'v': (gt[1] + gt[3]) / 2 / H}
    u = rng.random()
    axes_used = ('u', 'v') if u < 0.4 else ('u',) if u < 0.7 else ('v',)
    axes = {}
    for a in axes_used:
        t = min(2, int(c[a] * 3)) / 2
        if wrong:
            t = rng.choice([x for x in (0.0, 0.5, 1.0) if x != t])
        st = (pointers.SUPER if t in (0.0, 1.0) and rng.random() < 0.3
              else pointers.REGION if rng.random() < 0.5 else pointers.PREP)
        axes[a] = (float(t), st)
    return pointers.Ref('phrase', pointers.phrase_band(axes), 'phrase', None, 1.0, None, axes)


def draw(q, rng, cond=None):
    """References of one question: the train mixture (cond None) or a fixed condition ('none', 'correct' or
    'corrupt')."""
    refs = list(q['natural'])
    if cond == 'none':
        return refs
    if cond is None:
        if rng.random() < P_NONE:
            return refs
        has_p, has_ph = rng.random() < P_PRECISE, rng.random() < P_PHRASE and not q['natural']
        has_p = has_p or not has_ph
        kind = ('box', 'point', 'options')[rng.choice(3, p=[p for _, p in P_KIND])]
    else:
        has_p, has_ph = True, False
        kind = 'point' if rng.random() < 0.25 else 'box'
    if has_p:
        wrong = (rng.random() < RHO) if cond is None else cond == 'corrupt'
        refs += precise_refs(q, rng, wrong and kind != 'options', kind)
    if has_ph:
        refs.append(phrase_ref(q, rng, rng.random() < RHO))
    return refs


def proposal_example(q, rng, cond=None):
    """(crop px, label, z, refs): a centred crop scored against a reference centred on it; the reference is the
    evidence (positive crop) w.p. 1 - rho, else a same-size box at the same relative place in a negative crop."""
    pos = np.flatnonzero((q['kinds'] == 'centred') & (q['labels'] == 1))
    neg = np.flatnonzero((q['kinds'] == 'centred') & (q['labels'] == 0))
    if not len(pos) or not len(neg):
        return None
    wrong = (rng.random() < RHO) if cond is None else cond == 'corrupt'
    i, gt = pos[0], q['gt']
    if not wrong:
        refs = precise_refs(q, rng, False, 'point' if rng.random() < 0.25 else 'box')
        return q['crops'][i], 1.0, q['z'][i], refs + list(q['natural'])
    j = neg[rng.integers(len(neg))]
    pc, nc = q['crops'][i], q['crops'][j]
    off = np.array(gt) - np.array([pc[0], pc[1], pc[0], pc[1]])
    fake = dict(q, gt=[float(v) for v in np.array([nc[0], nc[1], nc[0], nc[1]]) + off])
    refs = precise_refs(fake, rng, False, 'point' if rng.random() < 0.25 else 'box')
    return nc, 0.0, q['z'][j], refs + list(q['natural'])


def build(qs, seed, cond=None, kinds=None):
    """Arrays (z, geo, valid, label, is_proposal) for questions qs under one synthesis."""
    Z, G, V, Y, P = [], [], [], [], []
    for q in qs:
        rng = _rng(seed, cond, q['qid'])
        refs = draw(q, rng, cond)
        keep = np.ones(len(q['labels']), bool) if kinds is None else np.isin(q['kinds'], kinds)
        norm = q['crops'][keep] / [q['W'], q['H'], q['W'], q['H']]
        g, v = pointers.geometry(norm, refs)
        Z.append(q['z'][keep])
        G.append(g)
        V.append(v)
        Y.append(q['labels'][keep])
        P.append(np.zeros(keep.sum(), bool))
        if cond != 'none':
            ex = proposal_example(q, rng, cond)
            if ex is not None:
                crop, y, z, prefs = ex
                g, v = pointers.geometry(np.asarray(crop)[None] / [q['W'], q['H'], q['W'], q['H']], prefs)
                Z.append([z])
                G.append(g)
                V.append(v)
                Y.append([y])
                P.append([True])
    return (np.concatenate(Z).astype(np.float32), np.concatenate(G), np.concatenate(V).astype(np.float32),
            np.concatenate(Y).astype(np.float32), np.concatenate(P))


def tau_examples(qs):
    """Pointer-view examples (q, crop, label, z, refs) on ZwZ val, TAU_DRAWS seeded draws per question."""
    out = []
    for d in range(TAU_DRAWS):
        for q in qs:
            ex = proposal_example(q, _rng('val', 'tau', d, q['qid']))
            if ex is not None:
                out.append((q, *ex))
    return out


# ------------------------------------------------------------------ training and fitting

@torch.enable_grad()
def train_one(data, val, width, lr, seed, log=print):
    """data: callable epoch -> arrays; val: {cond: arrays}. Returns (best state, best epoch, history)."""
    torch.manual_seed(seed)
    net = VerifierNet(width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    first = data(0)
    steps = math.ceil(len(first[0]) / BATCH)
    total, warm = steps * EPOCHS, max(1, round(WARMUP * steps * EPOCHS))
    lr_factor = lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm)))  # warmup, linear decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    best, hist = (-1.0, None, None), []
    for epoch in range(EPOCHS):
        Z, G, V, Y, _ = first if epoch == 0 else data(epoch)
        Z, G, V, Y = (torch.as_tensor(a) for a in (Z, G, V, Y))
        order = torch.as_tensor(np.random.default_rng([seed, epoch]).permutation(len(Z)))
        net.train()
        for s in range(0, len(Z), BATCH):
            b = order[s:s + BATCH]
            z, _ = net(Z[b], G[b], V[b])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        m = evaluate(net, val)
        hist.append(dict(epoch=epoch + 1, loss=loss.item(), **m))
        log(f'  width={width} lr={lr} epoch {epoch + 1}: ' + ' '.join(f'{k}={v:.4f}' for k, v in m.items()
                                                                      if v is not None))
        if m['select'] > best[0]:
            best = (m['select'], epoch + 1, {k: v.clone() for k, v in net.state_dict().items()})
    return best[2], best[1], hist


@torch.no_grad()
def logits(net, arrays):
    Z, G, V = (torch.as_tensor(a) for a in arrays[:3])
    net.eval()
    z, g = net(Z, G, V)
    return z.numpy(), g.numpy()


def evaluate(net, val):
    m = {}
    for cond, arr in val.items():
        z, _ = logits(net, arr)
        m[f'auroc_{cond}'] = auroc(arr[3], z)
        pr = arr[4]
        if pr.any() and cond == 'mix':
            m['auroc_proposal_mix'] = auroc(arr[3][pr], z[pr])
    m['select'] = float(np.mean([m['auroc_none'], m['auroc_correct'], m['auroc_corrupt']]))
    return m


def fit_temperature(z, y):
    """argmin over T_GRID of the NLL of p = sigmoid(z / T)."""
    zt, yt = torch.as_tensor(z, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    nll = [float(torch.nn.functional.binary_cross_entropy_with_logits(zt / t, yt)) for t in T_GRID]
    return float(T_GRID[int(np.argmin(nll))])


def fit_tau(p, y, target=TARGET_PRECISION):
    """(tau, coverage): the smallest threshold whose precision (positives among p >= tau) reaches the target;
    (inf, 0) if none does."""
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(prec >= target)
    if not len(ok):
        return math.inf, 0.0
    k = ok[-1]
    k = np.flatnonzero(ps >= ps[k])[-1]  # include ties at the threshold
    return float(ps[k]), float((p >= ps[k]).mean())


def train(qs, out_dir, log=print):
    """Train, select, calibrate and write the verifier of the questions qs (questions())."""
    tr = [q for q in qs if q['split'] == 'train']
    va = [q for q in qs if q['split'] == 'val']
    ca = [q for q in qs if q['split'] == 'calib']
    log(f'questions train {len(tr)} val {len(va)} calib {len(ca)}; rho {RHO}')
    val = {c: build(va, 'val', cond=c, kinds=SELECT_KINDS) for c in ('none', 'correct', 'corrupt')}
    val['mix'] = build(va, 'val-mix', kinds=SELECT_KINDS)
    state, epoch, hist = train_one(lambda e: build(tr, ('train', e)), val, WIDTH, LR, SEED, log)
    net = VerifierNet(WIDTH)
    net.load_state_dict(state)
    cal = build(ca, 'calib')
    T = fit_temperature(logits(net, cal)[0], cal[3])
    ex = tau_examples(va)
    geo = [pointers.geometry(np.asarray(crop)[None] / [q['W'], q['H'], q['W'], q['H']], refs)
           for q, crop, _, _, refs in ex]
    ts = (np.array([z for *_, z, _ in ex], np.float32), np.array([g[0] for g, _ in geo]),
          np.array([v[0] for _, v in geo], np.float32), np.array([y for _, _, y, _, _ in ex], np.float32))
    z_ts, _ = logits(net, ts)
    tau_learned, _ = fit_tau(1 / (1 + np.exp(-z_ts / T)), ts[3])
    gates = {c: float(logits(net, val[c])[1][val[c][2] > 0].mean()) for c in ('correct', 'corrupt')}
    verifier = PointerVerifier(net, T)  # phrases as b_pos, as at inference; one example per call
    p = np.array([verifier.score([z], np.asarray(crop)[None], (q['W'], q['H']), refs)[0][0]
                  for q, crop, _, z, refs in ex])
    tau, coverage = fit_tau(p, ts[3])
    select = max(h['select'] for h in hist)
    report = dict(rho=RHO, width=WIDTH, lr=LR, epoch=epoch, select=select, temperature=T, tau=tau,
                  tau_target_precision=TARGET_PRECISION, tau_val_coverage=coverage, tau_val_auroc=auroc(ts[3], z_ts),
                  mean_gate_val=gates, val=hist[epoch - 1], n_geo=pointers.N_GEO, selection=SELECTION,
                  tau_learned=tau_learned, phrase='rule')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, out_dir / 'verifier.safetensors')
    with open(out_dir / 'verifier.json', 'w') as f:
        json.dump(report, f, indent=1)
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump([dict(lr=LR, width=WIDTH, best_epoch=epoch, select=select, history=hist)], f, indent=1)
    log(json.dumps(report, indent=1))
    return report


def cache_zv(features_dir, tag, head_dir, out_path):
    """z_v of every cached crop feature (earthjev.train.extract_features files <tag>.s<i>of<n>.b<k>.npy), the region
    head on the CPU one feature block at a time, as the shipped cache was made (the float sums of a batch depend on its
    rows) -> out_path."""
    from earthjev.region_head import RegionHead, head_logits
    head = RegionHead.load(head_dir)
    files = sorted(Path(features_dir).glob(f'{tag}.s*of*.b*.npy'),
                   key=lambda p: tuple(map(int, re.search(r'\.s(\d+)of\d+\.b(\d+)\.npy$', p.name).groups())))
    if not files:
        raise FileNotFoundError(f'no {tag}.s*of*.b*.npy features in {features_dir}')
    z, index = [], []
    for p in files:
        z.append(head_logits(head, np.load(p)))
        part = re.search(r'\.(s\d+of\d+\.b\d+)\.npy$', p.name).group(1)
        with open(p.parent / f'index.{part}.jsonl') as f:
            index += [json.loads(line) for line in f]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, z=np.concatenate(z).astype(np.float32), index=json.dumps(index))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['zv', 'train'])
    ap.add_argument('--config', default='qwen35-9b', help='the configuration (region head, feature files, outputs)')
    ap.add_argument('--zv', help='z_v cache of the ZwZ crops (default $EARTHJEV_OUT/verifier/<config>/zv_zwz.npz)')
    ap.add_argument('--manifest', help='train: the ZwZ manifest (default: the one earthjev.train.zwz builds)')
    ap.add_argument('--out', help='train: output directory (default $EARTHJEV_OUT/verifier/<config>)')
    a = ap.parse_args(argv)
    torch.set_num_threads(1)  # the shipped weights were trained on one thread; more threads change the float sums
    config = Path(a.config).stem
    zv = a.zv or out_root() / 'verifier' / config / 'zv_zwz.npz'
    if a.cmd == 'zv':
        cache_zv(out_root() / 'features' / 'zwz', config, weights_dir(load_config(a.config), 'region_head'), zv)
        return
    if a.manifest is None:
        from earthjev.train.zwz import manifest_path
        a.manifest = manifest_path()
    train(questions(zv, a.manifest), a.out or out_root() / 'verifier' / config, log=lambda s: print(s, flush=True))


if __name__ == '__main__':
    main()
