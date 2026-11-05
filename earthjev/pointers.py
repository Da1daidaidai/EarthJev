"""Parse spatial cues and compute their geometric features."""
import re
from dataclasses import dataclass, field

import numpy as np

SUPER, REGION, PREP, ADJ = 2.0, 1.5, 1.0, 1.0  # directional prior strengths s_a by phrase type
POINT_HALF = 0.002  # a point is a box of side 0.004 (normalized) around it


@dataclass
class Ref:
    kind: str                   # 'box' | 'point' | 'phrase'
    box: tuple                  # normalized (x1, y1, x2, y2); phrase: the band/cell it names
    coord: str = ''             # 'px_res' | 'px' | 'n1' | 'n1000' | 'xywh_*' | 'phrase'
    option: str = None          # answer option letter holding the reference
    conf: float = 1.0           # parser confidence
    span: tuple = None          # (start, end) in the stem (None for option references)
    axes: dict = field(default_factory=dict)  # phrase: {'u': (target, strength), 'v': (target, strength)}

    @property
    def precise(self):
        return self.kind in ('box', 'point')


# ------------------------------------------------------------------ coordinates

_N = r'(\d+(?:\.\d+)?)'
_SEP = r'\s*[,，]\s*'
_RES = re.compile(r'(?:image\s+)?resolution\s*(?:is|of|:|：|=)?\s*(\d{2,6})\s*[x×*]\s*(\d{2,6})'
                  r'|(\d{2,6})\s*[x×]\s*(\d{2,6})\s*(?:pixels|px)\b', re.I)
_BOX = re.compile(r'(<\|box_start\|>|<box>)?\s*(?:'
                  rf'[\[(]\s*{_N}{_SEP}{_N}{_SEP}{_N}{_SEP}{_N}\s*[\])]'  # [x1, y1, x2, y2]
                  rf'|\(\s*{_N}{_SEP}{_N}\s*\)\s*[,，]?\s*\(\s*{_N}{_SEP}{_N}\s*\)'  # (x1,y1),(x2,y2)
                  r')\s*(<\|box_end\|>|</box>)?')
_POINT = re.compile(rf'(<point>)?\s*[\[(]\s*{_N}{_SEP}{_N}\s*[\])]\s*(</point>)?')
_POINT_CUE = re.compile(r'\b(point|pixel|location|coordinates?|position|at|centred|centered|located)\b'
                        r'[^.?!\n]{0,25}$', re.I)
_NORM_CUE = re.compile(r'normali[sz]|0\s*(?:-|–|~|to)\s*1000|relative|per[- ]?mille|‰|box_start', re.I)
_PX_CUE = re.compile(r'\bpixels?\b|\bpx\b', re.I)


def _resolution(text):
    m = _RES.search(text)
    if not m:
        return None
    w, h = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return int(w), int(h)


def _sentence(text, start, end):
    a = max(text.rfind(c, 0, start) for c in '.?!\n')
    b = min([i for i in (text.find(c, end) for c in '?!\n') if i >= 0] + [len(text)])
    return text[a + 1:b]


def _normalize(vals, text, start, end, res, size):
    """Coordinates -> (normalized values, coord system, confidence), or None when they fit no system."""
    sent = _sentence(text, start, end)
    if all(v <= 1.0 for v in vals) and any('.' in s for s in vals.raw):
        return [v for v in vals], 'n1', 1.0
    if _NORM_CUE.search(sent) or _NORM_CUE.search(text[max(0, start - 40):end + 40]):
        return [v / 1000 for v in vals] if max(vals) <= 1000 else None, 'n1000', 1.0
    xs, ys = vals[0::2], vals[1::2]
    if res:
        return [v / res[i % 2] for i, v in enumerate(vals)], 'px_res', 1.0
    if size:
        W, H = size
        if max(xs) <= W * 1.02 and max(ys) <= H * 1.02:
            ambiguous = max(vals) <= 1000 < max(W, H) and not _PX_CUE.search(sent)
            return [v / size[i % 2] for i, v in enumerate(vals)], 'px', 0.5 if ambiguous else 1.0
    if max(vals) <= 1000:
        return [v / 1000 for v in vals], 'n1000', 0.5
    return None


class _Vals(list):
    """Floats that remember their source strings (decimal points decide 0-1 coordinates)."""

    def __init__(self, raw):
        super().__init__(float(x) for x in raw)
        self.raw = raw


def _clip_box(b):
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in b)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _coord_refs(text, res, size, option=None, offset=True):
    """Box and point references in `text` (one stem or one option)."""
    out, taken = [], []
    for m in _BOX.finditer(text):
        raw = [g for g in m.groups()[1:-1] if g is not None]
        if len(raw) != 4:
            continue
        vals = _Vals(raw)
        tag = m.group(1) or ''
        norm = _normalize(vals, text + (' 0-1000' if 'box_start' in tag else ''), m.start(), m.end(), res, size)
        if norm is None or norm[0] is None:
            continue
        v, coord, conf = norm
        if v[2] <= v[0] or v[3] <= v[1]:  # [x, y, w, h]
            v, coord, conf = [v[0], v[1], v[0] + v[2], v[1] + v[3]], 'xywh_' + coord, 0.5
        box = _clip_box(v)
        if box:
            out.append(Ref('box', box, coord, option, conf, (m.start(), m.end()) if offset else None))
            taken.append((m.start(), m.end()))
    for m in _POINT.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        if not (m.group(1) or _POINT_CUE.search(text[max(0, m.start() - 40):m.start()])):
            continue
        vals = _Vals([m.group(2), m.group(3)])
        norm = _normalize(vals, text, m.start(), m.end(), res, size)
        if norm is None or norm[0] is None:
            continue
        (x, y), coord, conf = norm
        if 0 <= x <= 1 and 0 <= y <= 1:
            box = (max(0.0, x - POINT_HALF), max(0.0, y - POINT_HALF),
                   min(1.0, x + POINT_HALF), min(1.0, y + POINT_HALF))
            out.append(Ref('point', box, coord, option, conf, (m.start(), m.end()) if offset else None))
    return out


# ------------------------------------------------------------------ directional phrases

_V = {'top': 0.0, 'upper': 0.0, 'north': 0.0, 'northern': 0.0, 'bottom': 1.0, 'lower': 1.0, 'south': 1.0,
      'southern': 1.0}
_H = {'left': 0.0, 'west': 0.0, 'western': 0.0, 'right': 1.0, 'east': 1.0, 'eastern': 1.0}
_MID = {'middle', 'center', 'centre', 'central'}
_IMG = r'(?:image|picture|scene|photo|photograph|map|view|frame)'
_SUPER = re.compile(r'\b(top|bottom|left|right|upper|lower|north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?)'
                    r'[- ]?most\b')
_REGION = re.compile(r'((?:[\w-]+\s+){1,6}?)of\s+(?:the|this|the\s+whole|the\s+entire)\s+' + _IMG + r'\b')
_POS = (r'(?:far\s+)?(?:(?:top|bottom|upper|lower|north|south)(?:[- ]?(?:left|right|east|west|center|centre|middle))?'
        r'|left|right|east|west|center|centre|middle)')
_PREP = re.compile(r'(?<!\bto )\b(?:in|at|on|near|towards?|toward)\s+the\s+((?>' + _POS + r'))\b'  # atomic: no
                   # backtracking from 'top-left side of X' to 'top'
                   r'(?![- ](?:hand|arm|leg|foot|eye|ear|wing|shoulder|lane)\b)'  # body / object parts
                   r'(?!(?:[- ](?:corner|part|area|region|section|side|half|portion|edge|end|zone))?\s+of\b)')
_ADJ = re.compile(r'\bthe\s+((?:top|bottom|upper|lower|north|south)[- ]?(?:left|right|east|west|center|centre|middle)'
                  r'|(?:left|right|top|bottom)[- ]side|northern|southern|eastern|western|central|centre|center)'
                  r'(?=\s+[\w-])(?!(?:\s+(?!of\b)[\w-]+){0,4}\s+of\b)')  # a noun follows, no 'of' within 4 words


def _axes_of(words):
    words = [w for w in re.split(r'[\s-]+', words.lower()) if w]
    joined = []
    for w in words:  # 'northeast' / 'southwest' as one word
        m = re.fullmatch(r'(north|south)(east|west)', w)
        joined += [m.group(1), m.group(2)] if m else [w]
    v = next((_V[w] for w in joined if w in _V), None)
    h = next((_H[w] for w in joined if w in _H), None)
    mid = any(w in _MID for w in joined)
    if v is None and mid and h is not None:
        v = 0.5
    if h is None and mid and v is not None:
        h = 0.5
    if v is None and h is None and mid:
        v = h = 0.5
    return v, h


def phrase_axes(stem):
    """{'u': (target u*, strength s_a), 'v': (target v*, strength s_a)} of the stem's directional phrases; the first
    phrase per axis wins (superlatives, then image regions, prepositional and adjectival phrases)."""
    s = stem.lower()
    out = {}

    def put(v, h, st):
        if v is not None:
            out.setdefault('v', (v, st))
        if h is not None:
            out.setdefault('u', (h, st))

    for m in _SUPER.finditer(s):
        w = m.group(1)
        put(_V.get(w), _H.get(w), SUPER)
    for m in _REGION.finditer(s):
        put(*_axes_of(m.group(1)), REGION)
    for m in _PREP.finditer(s):
        put(*_axes_of(m.group(1)), PREP)
    for m in _ADJ.finditer(s):  # 'the southern end of the terminal' never matches (object-relative)
        put(*_axes_of(m.group(1)), ADJ)
    return out


def phrase_band(axes):
    """Normalized box a phrase names: the third of each constrained axis, [0, 1] elsewhere."""
    lo = lambda t: min(2, int(t * 3)) / 3
    u, v = axes.get('u'), axes.get('v')
    x1, x2 = (lo(u[0]), lo(u[0]) + 1 / 3) if u else (0.0, 1.0)
    y1, y2 = (lo(v[0]), lo(v[0]) + 1 / 3) if v else (0.0, 1.0)
    return (x1, y1, x2, y2)


# ------------------------------------------------------------------ parse / strip / render

def parse(stem, options=(), size=None):
    """All references of a question. options: [(letter, text)]; size: the image file's (W, H)."""
    texts = stem + ' ' + ' '.join(t for _, t in options)
    res = _resolution(texts)
    refs = _coord_refs(stem, res, size)
    for letter, text in options:
        refs += _coord_refs(text, res, size, option=letter, offset=False)
    axes = phrase_axes(stem)
    if axes:
        refs.append(Ref('phrase', phrase_band(axes), 'phrase', None, 1.0, None, axes))
    return refs


_REF_VOCAB = set('''image resolution bounding box boxes the a an target object objects is are in at within region
regions area located location of interest look pixel pixels coordinates coordinate normalized normalised point given
reference x to this relative see focus on inside here its and with size absolute range from px it centered centred
mille per answer question refer refers referenced queried shown marked'''.split())


def _spans(text):
    spans = [m.span() for m in _RES.finditer(text)]
    spans += [r.span for r in _coord_refs(text, _resolution(text), None) if r.span]
    spans += [r.span for r in _coord_refs(text, _resolution(text), (10 ** 9, 10 ** 9)) if r.span]
    return sorted(set(spans))


def strip(stem):
    """The stem without coordinate references; a sentence that only states a reference is dropped."""
    spans = _spans(stem)
    if not spans:
        return stem
    out = []
    for m in re.finditer(r'[^.?!\n]*(?:[.?!]+(?=\s|$)|\n|$)', stem):
        a, b = m.span()
        sent = stem[a:b]
        if not sent.strip():
            continue
        inside = [(max(s, a) - a, min(e, b) - a) for s, e in spans if s < b and e > a]
        if not inside:
            out.append(sent.strip())
            continue
        rest = sent
        for s, e in sorted(inside, reverse=True):
            rest = rest[:s] + ' ' + rest[e:]
        words = re.findall(r'[a-zA-Z]+', rest.lower())
        if all(w in _REF_VOCAB for w in words):
            continue
        rest = re.sub(r'\(\s*(?:pixel|normali[sz]ed)[^)]*\)', ' ', rest)
        rest = re.sub(r'(?:bounding box|box|point|coordinates?)\s*:\s*(?=[.?!,，]|$)', ' ', rest, flags=re.I)
        rest = re.sub(r'\s*[,，:]\s*(?=[.?!]|$)', '', rest)
        out.append(re.sub(r'\s+([.?!,])', r'\1', re.sub(r'\s{2,}', ' ', rest)).strip())
    # a dot at a decimal-free sentence end can split '... 1000. Bounding' correctly; rejoin with spaces
    return re.sub(r'\s{2,}', ' ', ' '.join(out)).strip()


FORMATS = ('px_res', 'px', 'n1000', 'n1', 'qwen', 'point_px', 'point_n1')


def render(fmt, box, size, res=None):
    """One sentence stating `box` (pixel [x1, y1, x2, y2] in an image of `size`) in format `fmt`.
    res: the resolution to state for 'px_res' (default size)."""
    W, H = size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    n = lambda v, d: v / d
    if fmt == 'px_res':
        rw, rh = res or size
        sx, sy = rw / W, rh / H
        return (f'Image resolution: {rw} x {rh}. Bounding box: '
                f'[{round(x1 * sx)}, {round(y1 * sy)}, {round(x2 * sx)}, {round(y2 * sy)}].')
    if fmt == 'px':
        return f'The target is in the region [{round(x1)}, {round(y1)}, {round(x2)}, {round(y2)}] (pixel coordinates).'
    if fmt == 'n1000':
        q = [round(1000 * n(x1, W)), round(1000 * n(y1, H)), round(1000 * n(x2, W)), round(1000 * n(y2, H))]
        return f'Look at the box [{q[0]}, {q[1]}, {q[2]}, {q[3]}] (normalized 0-1000 coordinates).'
    if fmt == 'n1':
        return (f'Region of interest: [{n(x1, W):.3f}, {n(y1, H):.3f}, {n(x2, W):.3f}, {n(y2, H):.3f}] '
                '(normalized coordinates).')
    if fmt == 'qwen':
        q = [round(1000 * n(x1, W)), round(1000 * n(y1, H)), round(1000 * n(x2, W)), round(1000 * n(y2, H))]
        return f'The target is <|box_start|>({q[0]},{q[1]}),({q[2]},{q[3]})<|box_end|>.'
    if fmt == 'point_px':
        return f'The object is at pixel ({round(cx)}, {round(cy)}).'
    if fmt == 'point_n1':
        return f'The object is at point ({n(cx, W):.3f}, {n(cy, H):.3f}) in normalized coordinates.'
    raise ValueError(fmt)


# ------------------------------------------------------------------ verifier inputs

N_GEO = 22


def geometry(crops, refs):
    """crops: [N, 4] normalized boxes -> [N, N_GEO] float32 features e_j and the valid flag [N] (any reference).
    Precise slot (13): valid, box, point, option, IoU, |c&r|/|r|, |c&r|/|c|, r-centre in c, c-centre in r,
    (r-centre - c-centre) / c-size (x, y; clipped to +-4), log(area r / area c) (clipped -8..4), 1/#options.
    The precise reference of a crop is the one it covers most (then IoU). Phrase slot (9): valid, and per axis
    (u, v): constrained, |c-centre - target|, c-centre in the named third, strength / 2. At inference the verifier
    sees precise references only (phrases enter through rule_prior), so the phrase slot is zero; training fills it."""
    c = np.asarray(crops, np.float64).reshape(-1, 4)
    N = len(c)
    f = np.zeros((N, N_GEO), np.float32)
    precise = [r for r in refs if r.precise]
    phrase = next((r for r in refs if r.kind == 'phrase'), None)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    ccx, ccy = (c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2
    if precise:
        R = np.array([r.box for r in precise])
        ix = np.clip(np.minimum(c[:, None, 2], R[None, :, 2]) - np.maximum(c[:, None, 0], R[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(c[:, None, 3], R[None, :, 3]) - np.maximum(c[:, None, 1], R[None, :, 1]), 0, None)
        inter = ix * iy
        ar = (R[:, 2] - R[:, 0]) * (R[:, 3] - R[:, 1])
        ac = cw * ch
        cov_r = inter / ar[None]
        iou = inter / (ac[:, None] + ar[None] - inter)
        best = np.lexsort((-iou, -cov_r), axis=1)[:, 0] if len(precise) > 1 else np.zeros(N, int)
        k = np.arange(N)
        rb = R[best]
        rcx, rcy = (rb[:, 0] + rb[:, 2]) / 2, (rb[:, 1] + rb[:, 3]) / 2
        n_opt = sum(r.option is not None for r in precise)
        kinds = [precise[i] for i in best]
        f[:, 0] = 1
        f[:, 1] = [r.kind == 'box' for r in kinds]
        f[:, 2] = [r.kind == 'point' for r in kinds]
        f[:, 3] = [r.option is not None for r in kinds]
        f[:, 4] = iou[k, best]
        f[:, 5] = cov_r[k, best]
        f[:, 6] = inter[k, best] / ac
        f[:, 7] = (c[:, 0] <= rcx) & (rcx < c[:, 2]) & (c[:, 1] <= rcy) & (rcy < c[:, 3])
        f[:, 8] = (rb[:, 0] <= ccx) & (ccx < rb[:, 2]) & (rb[:, 1] <= ccy) & (ccy < rb[:, 3])
        f[:, 9] = np.clip((rcx - ccx) / cw, -4, 4)
        f[:, 10] = np.clip((rcy - ccy) / ch, -4, 4)
        f[:, 11] = np.clip(np.log(ar[best] / ac), -8, 4)
        f[:, 12] = [1 / n_opt if r.option is not None else 1.0 for r in kinds]
    if phrase is not None:
        f[:, 13] = 1
        for j, (axis, cc) in enumerate((('u', ccx), ('v', ccy))):
            if axis in phrase.axes:
                t, st = phrase.axes[axis]
                lo = min(2, int(t * 3)) / 3
                f[:, 14 + 4 * j] = 1
                f[:, 15 + 4 * j] = np.abs(cc - t)
                f[:, 16 + 4 * j] = (lo <= cc) & (cc <= lo + 1 / 3)
                f[:, 17 + 4 * j] = st / 2
    return f, (f[:, 0] + f[:, 13]) > 0


def rule_prior(crops, phrase_refs):
    """Directional prior b_pos(c) = -sum_a s_a |u_a(c) - u_a*| over the axes the phrases constrain (crop centre vs
    target, normalized coordinates); added to the calibrated logit. Precise references are ignored."""
    c = np.asarray(crops, np.float64).reshape(-1, 4)
    ccx, ccy = (c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2
    prior = np.zeros(len(c))
    for r in phrase_refs:
        if r.kind == 'phrase':
            for axis, cc in (('u', ccx), ('v', ccy)):
                if axis in r.axes:
                    t, st = r.axes[axis]
                    prior -= st * np.abs(cc - t)
    return prior
