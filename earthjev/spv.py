"""Verify spatial cues and choose between direct answering and further search."""
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from earthjev.calibration import logit, sigmoid
from earthjev.evidence import MERGE_IOU, iou, proposal_crop
from earthjev.pointers import N_GEO, geometry, rule_prior

MAX_POINTER_VIEWS = 8  # pointer views t_j per image; a view with IoU > MERGE_IOU to an earlier one is dropped


def pointer_proposals(refs, image_size):
    """[{bbox, crop, option, coord, conf, note}] for the precise references, in parse order: bbox = the pointer in
    pixels (at least 1 px per side), crop = its pointer view t_j, note = its provenance (shown in the memory prompt)."""
    W, H = image_size
    props = []
    for r in refs:
        if not r.precise:
            continue
        if len(props) == MAX_POINTER_VIEWS:
            break
        x1, y1 = round(r.box[0] * W), round(r.box[1] * H)
        box = [x1, y1, max(round(r.box[2] * W), x1 + 1), max(round(r.box[3] * H), y1 + 1)]
        crop = proposal_crop(box, image_size)
        if any(iou(crop, x['crop']) > MERGE_IOU for x in props):
            continue
        note = f'the region of option {r.option}' if r.option else 'the region referenced in the question'
        props.append(dict(bbox=box, crop=crop, option=r.option, coord=r.coord, conf=r.conf, note=note))
    return props


class VerifierNet(torch.nn.Module):
    """z = z_v + valid * gamma * (w . e + c), gamma = sigmoid(w_g [e; z_v] + b_g), e = MLP(geometry); -> (z, gamma)."""

    def __init__(self, width, n_geo=N_GEO):
        super().__init__()
        self.geo = torch.nn.Sequential(torch.nn.Linear(n_geo, width), torch.nn.GELU(), torch.nn.Linear(width, width),
                                       torch.nn.GELU())
        self.zg = torch.nn.Linear(width, 1)
        self.gate = torch.nn.Linear(width + 1, 1)
        torch.nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, z, geo, valid):
        e = self.geo(geo)
        g = torch.sigmoid(self.gate(torch.cat([e, z[:, None]], 1))).squeeze(1)
        return z + valid * g * self.zg(e).squeeze(1), g


class PointerVerifier:
    """g_psi on CPU float32. Directional phrases enter as the prior b_pos on the calibrated logit; the network sees
    the precise references only."""

    def __init__(self, net, temperature):
        self.net, self.T = net.eval(), float(temperature)

    @classmethod
    def load(cls, weights_dir, temperature):
        state = load_file(Path(weights_dir) / 'verifier.safetensors')
        net = VerifierNet(state['geo.0.weight'].shape[0])
        net.load_state_dict(state)
        return cls(net, temperature)

    @torch.no_grad()
    def score(self, z_v, crop_boxes_px, image_size, refs):
        """(p, gate) per crop (pixel boxes of an image of image_size); gate is NaN where no precise reference exists."""
        W, H = image_size
        crops = np.asarray(crop_boxes_px, float).reshape(-1, 4) / [W, H, W, H]
        prior = rule_prior(crops, [r for r in refs if r.kind == 'phrase'])
        geo, valid = geometry(crops, [r for r in refs if r.precise])
        z, g = self.net(torch.as_tensor(np.asarray(z_v, np.float32)), torch.as_tensor(geo),
                        torch.as_tensor(valid.astype(np.float32)))
        p = 1 / (1 + np.exp(-(np.asarray(z.numpy(), np.float64) / self.T + prior)))
        return p, np.where(valid, g.numpy(), np.nan)

    def mixture(self, z_v, child_boxes_px, image_size, refs, pi):
        """p_mix of the search's children: pi p(c | refs) + (1 - pi) p(c | refs without the precise ones)."""
        p_ref, _ = self.score(z_v, child_boxes_px, image_size, refs)
        if pi >= 1 or not any(r.precise for r in refs):
            return p_ref
        p_free, _ = self.score(z_v, child_boxes_px, image_size, [r for r in refs if not r.precise])
        return pi * p_ref + (1 - pi) * p_free


def pointer_trust(p, y, trust_platt):
    """(P_ptr, [s_j]) of the stem pointers: s_j = sigmoid((logit p_j + logit y_j) / 2), P_ptr = sigmoid(a logit(max_j
    s_j) + b) with trust_platt = (a_ptr, b_ptr). Every logit clips its probability to [1e-6, 1 - 1e-6]
    (earthjev.calibration.EPS), so P_ptr saturates where |l_j| > 13.8. One y_j per p_j (ValueError otherwise)."""
    s = [float(sigmoid(0.5 * (logit(pj) + logit(yj)))) for pj, yj in zip(p, y, strict=True)]
    a, b = trust_platt
    return float(sigmoid(a * logit(max(s)) + b)), s


def route(P_ptr, tau_fast, tau_keep):
    """'fast' iff P_ptr >= tau_fast; else 'search' with the stem pointer views iff P_ptr >= tau_keep, otherwise
    'search_without_pointers'."""
    return 'fast' if P_ptr >= tau_fast else ('search' if P_ptr >= tau_keep else 'search_without_pointers')


def select_path(props, spv_cfg, y=None):
    """SPV decision for one image from its scored pointer views (props from pointer_proposals, each with 'p').

    spv_cfg: the configuration's 'spv' section (trust_platt, tau_fast, tau_keep, tau_opt). y: the answerer's Yes
    probabilities y_j on the stem pointer views (props without an option, in order; ValueError on a length mismatch),
    or None for multi-image questions. Returns dict(fast, admit, pi, trust): the view the fast path answers from
    (None: run the search), whether the stem pointer views join the search pool (option-box views always do),
    pi = max_j p_j for the mixture (1 without pointer views), and the trust record {P, s, s1, admit} (None when P_ptr
    is not computed).
    With stem pointers and y the route follows P_ptr and the fast path uses the stem pointer with the highest p;
    otherwise the fast path is taken iff max_j p_j >= tau_opt."""
    if not props:
        return dict(fast=None, admit=True, pi=1.0, trust=None)
    best = max(props, key=lambda x: x['p'])
    out = dict(fast=best if best['p'] >= spv_cfg['tau_opt'] else None, admit=True, pi=best['p'], trust=None)
    stem = [x for x in props if x['option'] is None]
    if y is not None and stem:
        P, s = pointer_trust([x['p'] for x in stem], y, spv_cfg['trust_platt'])
        fast = route(P, spv_cfg['tau_fast'], spv_cfg['tau_keep']) == 'fast'
        admit = P >= spv_cfg['tau_keep']
        out.update(fast=max(stem, key=lambda x: x['p']) if fast else None, admit=admit,
                   trust=dict(P=P, s=s, s1=fast, admit=admit))
    return out
