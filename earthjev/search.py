"""Best-first evidence search over 3-by-3 subdivisions, following ZoomSearch."""
import math

import numpy as np

CELL = 224  # px: a node is a leaf at this longer side; cells with a shorter side are never scored
# the paper's name ("a maximum of 10 search expansions", Sec. IV); it counts pops of the open set, while the search
# result's n_expansions counts the nodes that were split
MAX_EXPANSIONS = 10
KEEP_THRESHOLD, KEEP_MAX = 0.6, 6  # retention rule (Sec. III-C)
GRID_NAMES = ('top-left', 'top-center', 'top-right', 'middle-left', 'middle-center', 'middle-right',
              'bottom-left', 'bottom-center', 'bottom-right')


def grid_cells(box, min_side=CELL):
    """The 3x3 cells of a pixel box as (cell, name) pairs in row-major order: cell sides are side // 3, the last row
    and column take the remainder, and cells with a side below min_side are dropped."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) // 3, (y2 - y1) // 3
    cells = []
    for i in range(3):
        for j in range(3):
            cell = (x1 + j * w, y1 + i * h, x2 if j == 2 else x1 + (j + 1) * w, y2 if i == 2 else y1 + (i + 1) * h)
            if cell[2] - cell[0] >= min_side and cell[3] - cell[1] >= min_side:
                cells.append((cell, GRID_NAMES[3 * i + j]))
    return cells


def select_by_sigmoid(nodes, threshold=KEEP_THRESHOLD, max_nodes=KEEP_MAX):
    """Retention rule: the nodes with sigmoid(5 (c - mean c)) > threshold, in their order, or the max_nodes of them
    with the highest sigmoid (in decreasing order) when there are more; the most confident node when none passes.
    Fewer than two nodes are returned as they are. With c = 2p - 1 the rule reads sigma(10 (p - mean p)) > 0.6."""
    if len(nodes) < 2:
        return list(nodes)
    c = np.array([n['confidence'] for n in nodes])
    s = 1.0 / (1.0 + np.exp(-np.clip((c - np.mean(c)) * 5, -500, 500)))
    keep = np.flatnonzero(s > threshold)
    if len(keep) == 0:
        return [nodes[int(np.argmax(c))]]
    if len(keep) > max_nodes:
        keep = keep[np.argsort(s[keep])[::-1][:max_nodes]]
    return [nodes[i] for i in keep]


def hierarchical_search(image_size, score_children, max_expansions=MAX_EXPANSIONS):
    """Best-first search over an image of image_size = (W, H) px (see the module docstring).

    score_children(boxes) -> the confidences 2p - 1 of the given cells [(x1, y1, x2, y2)]; one call per split node.
    Returns dict(final_selected=[{bbox, depth, confidence, full_path}] (the root has depth 1 and full_path []),
    n_expansions (nodes split), n_children_scored, iterations (pops))."""
    width, height = image_size
    root = dict(bbox=[0, 0, width, height], depth=1, confidence=0.0, full_path=[])
    bound, next_drop = 1.0, int(math.log(max(width, height) // CELL, 4) * 5)
    open_set, leaves, deepest, deepest_depth = [root], [], [], 1
    leaf_seen, pops, n_expansions, n_scored = False, 0, 0, 0

    def reach(nodes, depth):  # the deepest level reached so far and all its nodes
        nonlocal deepest, deepest_depth
        if depth > deepest_depth:
            deepest, deepest_depth = list(nodes), depth
        elif depth == deepest_depth:
            deepest += nodes

    while open_set and pops < max_expansions:
        node = open_set.pop(int(np.argmax([n['confidence'] for n in open_set])))
        pops += 1
        x1, y1, x2, y2 = node['bbox']
        if max(x2 - x1, y2 - y1) <= CELL:
            leaf_seen = True
            leaves.append(node)
            reach([node], node['depth'])
            continue
        if leaf_seen and node['confidence'] > bound:
            break
        if pops >= next_drop:
            bound -= 0.1
            next_drop += 5
        cells = grid_cells(node['bbox'])
        if not cells:
            continue
        scores = score_children([cell for cell, _ in cells])
        if len(scores) != len(cells):
            raise ValueError(f'score_children returned {len(scores)} scores for {len(cells)} cells')
        n_expansions += 1
        n_scored += len(cells)
        children = [dict(bbox=list(cell), depth=node['depth'] + 1, confidence=float(score),
                         full_path=node['full_path'] + [name]) for (cell, name), score in zip(cells, scores)]
        reach(children, node['depth'] + 1)
        open_set += select_by_sigmoid(children)

    candidates = list({id(n): n for n in leaves + deepest}.values())  # each node once, first occurrence first
    final = select_by_sigmoid(candidates) if candidates else [root]
    return dict(final_selected=final, n_expansions=n_expansions, n_children_scored=n_scored, iterations=pops)
