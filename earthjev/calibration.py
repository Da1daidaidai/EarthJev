"""Platt scaling of the decision scores (paper Sec. III-D) and the AUROC that selects the trained heads."""
import numpy as np

EPS = 1e-6  # probabilities are clipped to [EPS, 1 - EPS] before a logit


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_fit(scores, labels, iters=100):
    """(a, b) of the Platt map P = sigmoid(a s + b), by maximum likelihood (Newton steps; a 1e-6 ridge keeps
    separable data finite)."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    X = np.c_[s, np.ones_like(s)]
    w = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (y - p) - 1e-6 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def auroc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties; None unless both classes are present."""
    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(s, kind='stable')
    _, first, count = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(first + (count + 1) / 2, count)  # 1-based average rank of each tie group
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
