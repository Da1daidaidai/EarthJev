"""Evidence views and their two presentations (paper Sec. III-C and III-D).

select_views          pooled boxes (search selection and pointer views) -> at most K = 4 views: each box grows to
                      1.5 times its size and at least 512 px per side, and a view overlapping a higher-ranked one
                      (IoU > 0.5) merges into it.
compose_canvas        canvas presentation: the views' crops laid out by the 3 x 3 cell of the image their centres fall
                      in. This is the layout of ZoomSearch (Zhou et al., arXiv:2511.20460), re-implemented here with
                      identical output.
render_memory_views   memory presentation: each view rendered from the native pixels with its longer side 1024 px.
"""
# render_memory_views follows VISTA (https://github.com/joshhhhhan/VISTA, commit
# c97c354f0a8948256545a3e8850a1848f8e5b8c1; MIT License, Copyright (c) 2026 VISTA contributors): the inspection limits
# of src/vista/core/tools.py (MAX_INSPECTION_VIEWS = 16, MAX_INSPECTION_QUESTION_CHARS = 1024), the errors
# "Invalid arguments." and "View {i}: invalid ..." and the rule that fits a view's longer side to the display size,
# enlarging with NEAREST, of src/vista/core/session.py (_inspect, _inspection_view). The box-filtered LANCZOS
# downscaling of views larger than the display size is new.
import operator

from PIL import Image

K_VIEWS = 4  # local views per image (Sec. III-C)
MIN_VIEW_SIDE, VIEW_CONTEXT = 512, 1.5  # a view: 1.5 x its box, at least 512 px per side (Sec. III-C)
MERGE_IOU = 0.5  # views overlapping above this IoU merge
POINTER_VIEW_MIN_SIDE = 448  # a pointer view: square of side max(2 x long side, 448 px) (Sec. III-B)
GRID_NAMES = ('top-left', 'top-center', 'top-right', 'middle-left', 'middle-center', 'middle-right', 'bottom-left',
              'bottom-center', 'bottom-right')  # the 3 x 3 cells of an image, row-major
GRAY = (128, 128, 128)  # canvas background
DISPLAY_SIZE = 1024  # longer side of a memory view
MAX_QUESTION_CHARS, MAX_VIEWS = 1024, 16  # VISTA's inspection limits


def proposal_crop(box, size):
    """The pointer view of a pixel box: a square of side max(2 x long side, 448 px) centred on it, clipped."""
    W, H = size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(2 * max(box[2] - box[0], box[3] - box[1]), POINTER_VIEW_MIN_SIDE) / 2
    return [max(0, round(cx - half)), max(0, round(cy - half)), min(W, round(cx + half)), min(H, round(cy + half))]


def grid_name(box, size):
    """Name of the 3 x 3 cell of an image of `size` that holds the centre of a box."""
    W, H = size
    col = min(2, int(3 * (box[0] + box[2]) / 2 / W))
    row = min(2, int(3 * (box[1] + box[3]) / 2 / H))
    return GRID_NAMES[3 * row + col]


def grow(box, width, height):
    """The context box of a view: each side max(512, 1.5 x side) around the centre, shifted inside the image and
    clipped to it."""
    out = []
    for lo, hi, n in ((box[0], box[2], width), (box[1], box[3], height)):
        side = min(n, max(MIN_VIEW_SIDE, round(VIEW_CONTEXT * (hi - lo))))
        start = min(max(0, (lo + hi - side) // 2), n - side)
        out += [int(start), int(start + side)]
    return [out[0], out[2], out[1], out[3]]


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def select_views(pool, size, k=K_VIEWS):
    """pool [{bbox, confidence, note?}] -> the top-k views [{bbox, score, note?}] by confidence (stable order). A grown
    box with IoU > 0.5 to an earlier view merges into it (union box, the earlier view's score); notes are kept on the
    view they land in."""
    views = []
    for node in sorted(pool, key=lambda n: -n['confidence']):
        box = grow(node['bbox'], *size)
        hit = next((v for v in views if iou(v['bbox'], box) > MERGE_IOU), None)
        if hit is None:
            views.append({'bbox': box, 'score': float(node['confidence'])})
            hit = views[-1]
        else:
            b = hit['bbox']
            hit['bbox'] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
        if node.get('note') and node['note'] not in hit.get('note', ''):
            hit['note'] = '; '.join(filter(None, (hit.get('note'), node['note'])))
    return views[:k]


def _stack(tiles):
    """Tiles one above the other, each scaled (LANCZOS) to the widest one's width."""
    width = max(t.width for t in tiles)
    tiles = [t if t.width == width else t.resize((width, int(t.height * width / t.width)), Image.LANCZOS)
             for t in tiles]
    out = Image.new('RGB', (width, sum(t.height for t in tiles)), GRAY)
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height
    return out


def _cell_tile(image, name, boxes):
    """The tile of one grid cell: its only crop, or its crops placed by their centres in the cell's own 3 x 3 grid,
    keeping only the used sub-rows and sub-columns; crops sharing a sub-cell are stacked, and every sub-cell is
    scaled (LANCZOS) to its column's widest and its row's tallest member."""
    crops = [image.crop(tuple(b)) for b in boxes]
    if len(crops) == 1:
        return crops[0]
    W, H = image.size
    row, col = divmod(GRID_NAMES.index(name), 3)
    x0, y0 = col * (W // 3), row * (H // 3)
    cell_w = ((W if col == 2 else x0 + W // 3) - x0) / 3
    cell_h = ((H if row == 2 else y0 + H // 3) - y0) / 3
    pos = [(min(max(int(((b[1] + b[3]) / 2 - y0) / cell_h), 0), 2),
            min(max(int(((b[0] + b[2]) / 2 - x0) / cell_w), 0), 2)) for b in boxes]
    rows, cols = sorted({r for r, _ in pos}), sorted({c for _, c in pos})
    groups = {}
    for (r, c), crop in zip(pos, crops):
        groups.setdefault((rows.index(r), cols.index(c)), []).append(crop)
    tiles = {rc: g[0] if len(g) == 1 else _stack(g) for rc, g in groups.items()}
    heights = [max(t.height for (r, _), t in tiles.items() if r == i) for i in range(len(rows))]
    widths = [max(t.width for (_, c), t in tiles.items() if c == j) for j in range(len(cols))]
    out = Image.new('RGB', (sum(widths), sum(heights)), GRAY)
    for (r, c), t in tiles.items():
        out.paste(t.resize((widths[c], heights[r]), Image.LANCZOS), (sum(widths[:c]), sum(heights[:r])))
    return out


def compose_canvas(image, views):
    """Canvas presentation (Sec. III-D): one tile per grid cell holding view centres, all tiles scaled (LANCZOS) to the
    largest tile size and placed at their cells of a 3 x 3 grid on gray."""
    if not views:
        return image
    cells = {}
    for v in views:
        cells.setdefault(grid_name(v['bbox'], image.size), []).append(v['bbox'])
    tiles = {name: _cell_tile(image, name, boxes) for name, boxes in cells.items()}
    w, h = max(t.width for t in tiles.values()), max(t.height for t in tiles.values())
    canvas = Image.new('RGB', (3 * w, 3 * h), GRAY)
    for name, t in tiles.items():
        row, col = divmod(GRID_NAMES.index(name), 3)
        canvas.paste(t if t.size == (w, h) else t.resize((w, h), Image.LANCZOS), (col * w, row * h))
    return canvas


def _native_box(bbox, size):
    """bbox as four integers (numpy integers too, floats not) of a non-empty box inside the image, else None."""
    try:
        x1, y1, x2, y2 = map(operator.index, bbox)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if 0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1] else None


def _render(image, box):
    """A native crop with its longer side scaled to DISPLAY_SIZE: big boxes are box-filtered by an integer factor that
    keeps at least 3x for the final LANCZOS step, small ones are enlarged with NEAREST."""
    w, h = box[2] - box[0], box[3] - box[1]
    longest = max(w, h)
    size = max(1, w * DISPLAY_SIZE // longest), max(1, h * DISPLAY_SIZE // longest)
    if longest > DISPLAY_SIZE:
        factor = longest // (3 * DISPLAY_SIZE)
        if factor > 1:
            return image.reduce(factor, box=box).resize(size, Image.LANCZOS, box=(0, 0, w / factor, h / factor))
        return image.crop(box).resize(size, Image.LANCZOS)
    crop = image.crop(box)
    return crop if crop.size == size else crop.resize(size, Image.NEAREST)


def render_memory_views(image, views, stem):
    """Memory presentation (Sec. III-D): the rendered views [{bbox, ...}] of an image, for the question stem. Raises
    ValueError, failing the question, on an empty stem, no or more than 16 views, or a box that is not four integers
    of a non-empty region inside the image."""
    if not stem[:MAX_QUESTION_CHARS].strip() or not 1 <= len(views) <= MAX_VIEWS:
        raise ValueError('Invalid arguments.')
    image = image if image.mode == 'RGB' else image.convert('RGB')
    renders = []
    for i, view in enumerate(views, start=1):
        box = _native_box(view['bbox'], image.size)
        if box is None:
            raise ValueError(f'View {i}: invalid native_bbox.')
        renders.append(_render(image, box))
    return renders
