"""The EarthJev pipeline, live: one question in, its answer and a log record out (paper Sec. III, Fig. 2).

Per image of the question (two images for the XLRS-lite change-detection questions):
1. SPV (Sec. III-B): the precise pointers the question states become pointer views t_j, scored in one batch by the
   region head and the pointer verifier (p_j). For stem pointers on a single image, the answerer's Yes/No probability
   y_j on each view gives the pointer trust P_ptr, which routes the question to the fast path (the view of the stem
   pointer with the highest p_j alone) or to the search; option-box pointers and two-image questions take the fast
   path iff max_j p_j >= tau_opt.
2. EDM (Sec. III-C): the hierarchical search scores 3x3 cells with the verify-then-propagate mixture (pi = max_j p_j);
   its final selection and the pointer views form the pool, from which select_views keeps at most 4 views.
3. Both presentations of the same views (Sec. III-D): the canvas (laid out with the search, as in the runs behind the
   paper) and the memory views; the answerer answers each greedily with the log-probs of its own generation.
4. CAS and the evidence presence veto choose the final answer (Sec. III-D).

Record of one question (one JSON line; see run_row): qid, bench, config (the configuration's name, e.g.
earthjev-qwen35-9b), timings (CUDA-synchronized seconds: load_s, search_s, canvas_answer_s, memory_answer_s, decide_s,
total_s), error (None, or 'Type: message' of the exception that failed the question; the stages finished before it are
kept), spv (per image: route 'fast' | 'search' | 'search_without_pointers', pointers = the number of pointer views,
trust = {P, s, s1, admit} or None, pointer_s), final, render (the chosen presentation), P (its calibrated probability of
being correct), decision (letter: the option letter read from each presentation's answer; veto: the replacement letter
of the presence veto or None), outputs and gen ({tok_lp, letter_lp, ended}) per presentation, views ([{bbox, score,
note?}], a list per image for two-image questions).
"""
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from earthjev import cas, data, evidence, paths, pointers, prompts, scoring, search, spv
from earthjev.answerer import Answerer
from earthjev.region_head import RegionScorer, decision_prompt

PRESENTATIONS = ('canvas', 'memory')  # answered in this order


def now():
    """perf_counter after the GPU has finished its queued work (the stage timings)."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.perf_counter()


class Images:
    """Image decoding: the file decoded once (converted to RGB if needed) and kept while consecutive questions
    share it; the previous image is released before the next one is decoded."""

    def __init__(self):
        self.path = self.image = None

    def __call__(self, path):
        if path != self.path:
            self.path = self.image = None
            image = Image.open(path)
            image.load()
            self.image = image if image.mode == 'RGB' else image.convert('RGB')
            self.path = path
        return self.image


class EarthJev:
    """The method with one configuration (configs/<name>.json): the answerer, its region head and the pointer
    verifier. run_row(row) answers a question (a row of earthjev.data)."""

    def __init__(self, answerer, scorer, verifier, cfg):
        self.answerer, self.scorer, self.verifier, self.cfg = answerer, scorer, verifier, cfg
        self.images = Images()

    @classmethod
    def load(cls, cfg, answerer=None):
        """The models of a configuration (the answerer on the GPU), warmed up; answerer: share a loaded one."""
        a = cfg['answerer']
        answerer = answerer or Answerer(paths.models_root() / a['dir'], a['dtype'], a['attn_implementation'],
                                        a['max_pixels'])
        verifier = spv.PointerVerifier.load(paths.weights_dir(cfg, 'verifier'), cfg['spv']['temperature'])
        method = cls(answerer, RegionScorer(answerer, paths.weights_dir(cfg, 'region_head')), verifier, cfg)
        method.warm()
        return method

    def warm(self):
        """Run every model call once, so that the timings of the first question exclude kernel compilation."""
        image = Image.new('RGB', (448, 448), (90, 120, 90))
        self.scorer.logits([image, image], *decision_prompt('Is there a road?'))
        self.answerer.yes_probability(image, 'Is there a road?')
        self.answerer.generate([(image, self.answerer.max_pixels)], 'Describe.', 2,
                               system=prompts.answer_system({'is_mcq': False}))

    def find_evidence(self, row, image, trust):
        """SPV and EDM on one image of a question (trust: compute P_ptr for stem pointers, single-image questions).
        Returns dict(route, pointer_views, trust, pi, search (the hierarchical_search result, None on the fast path),
        pool, views, canvas, pointer_s)."""
        t0 = now()
        size = image.size
        refs = pointers.parse(row['stem'], [(o['id'], o['text']) for o in row.get('options') or []], size)
        stem = pointers.strip(row['stem'])
        question, state = decision_prompt(stem)
        props = spv.pointer_proposals(refs, size)
        if props:  # the pointer views, scored in one batch
            z = self.scorer.logits([image.crop(tuple(x['crop'])) for x in props], question, state)
            p, gate = self.verifier.score(z, [x['crop'] for x in props], size, refs)
            for x, pj, gj in zip(props, p, gate):
                x.update(p=float(pj), gate=None if gj != gj else float(gj), confidence=float(2 * pj - 1))
        y = None
        if trust:  # the answerer's Yes/No on each stem pointer view, for the reference-free question
            query = prompts.search_query({**row, 'stem': stem})
            y = [self.answerer.yes_probability(image.crop(tuple(x['crop'])), query) for x in props
                 if x['option'] is None]
        d = spv.select_path(props, self.cfg['spv'], y)
        pointer_s = now() - t0
        found = None
        if d['fast'] is not None:
            route = 'fast'
            pool = [{k: d['fast'][k] for k in ('bbox', 'confidence', 'note')}]
        else:
            route = 'search' if d['admit'] else 'search_without_pointers'

            def score_children(boxes):  # one batch per split node: confidence 2 p_mix - 1
                z = self.scorer.logits([image.crop(b) for b in boxes], question, state)
                return [float(2 * x - 1) for x in self.verifier.mixture(z, boxes, size, refs, d['pi'])]

            found = search.hierarchical_search(size, score_children)
            pool = [dict(bbox=n['bbox'], confidence=n['confidence']) for n in found['final_selected']]
            pool += [{k: x[k] for k in ('bbox', 'confidence', 'note')} for x in props
                     if d['admit'] or x['option'] is not None]
        views = evidence.select_views(pool, size)
        return dict(route=route, pointer_views=props, trust=d['trust'], pi=d['pi'], search=found, pool=pool,
                    views=views, canvas=evidence.compose_canvas(image, views), pointer_s=pointer_s)

    def answer(self, row, images, found):
        """The two presentations of the views and their answers: {presentation: Answerer.generate output} and the
        seconds of each (rendering + generation)."""
        px = self.cfg['answerer']['max_pixels']
        n = self.cfg['answerer']['max_new_tokens']['multiple_choice' if row['is_mcq'] else 'open_ended']
        seconds = {'canvas': 0.0, 'memory': 0.0}
        t0 = now()
        renders = [r for im, f in zip(images, found) for r in evidence.render_memory_views(im, f['views'], row['stem'])]
        seconds['memory'] += now() - t0
        if len(images) > 1:
            text = prompts.memory_answer_multi(row, [im.size for im in images],
                                               [[v['bbox'] for v in f['views']] for f in found])
        else:
            views = found[0]['views']
            text = prompts.memory_answer(row, images[0].size, [v['bbox'] for v in views],
                                         [v.get('note') for v in views])
        shown = {'canvas': ([(f['canvas'], px) for f in found], prompts.canvas_answer(row, prompts.position_hint())),
                 'memory': ([(im, px) for im in images] + [(r, px) for r in renders], text)}
        gen = {}
        for r in PRESENTATIONS:
            t0 = now()
            gen[r] = self.answerer.generate(*shown[r], n, system=prompts.answer_system(row))
            seconds[r] += now() - t0
        return gen, seconds

    def decide(self, row, gen, views):
        """CAS and the presence veto over the two answers: cas.decide's result and the letter of each answer."""
        cand = {r: dict(output=g['output'], letter=scoring.answer_letter(row, g['output']), letter_lp=g['letter_lp'],
                        tok_lp=g['tok_lp']) for r, g in gen.items()}
        presence = (cas.presence_probability([v['score'] for v in views], self.cfg['cas']['presence_platt'])
                    if row['is_mcq'] else None)
        return cas.decide(row, cand, self.cfg, presence), {r: c['letter'] for r, c in cand.items()}

    def run_row(self, row):
        """Answer one question; returns its record (module docstring)."""
        out = dict(qid=row['qid'], bench=row['bench'], config=self.cfg['name'], timings={}, error=None)
        t = out['timings']
        try:
            t0 = now()
            images = [self.images(p) for p in data.image_paths(row)]
            t['load_s'] = now() - t0
            t0 = now()
            found = [self.find_evidence(row, im, trust=len(images) == 1) for im in images]
            t['search_s'] = now() - t0
            out['spv'] = [dict(route=f['route'], pointers=len(f['pointer_views']), trust=f['trust'],
                               pointer_s=f['pointer_s']) for f in found]
            gen, seconds = self.answer(row, images, found)
            for r in PRESENTATIONS:
                t[f'{r}_answer_s'] = seconds[r]
            t0 = now()
            d, letters = self.decide(row, gen, [v for f in found for v in f['views']])
            t['decide_s'] = now() - t0
            out.update(final=d['out'], render=d['render'], P=d['P'], decision=dict(letter=letters, veto=d['veto']),
                       outputs={r: g['output'] for r, g in gen.items()},
                       gen={r: {k: g[k] for k in ('tok_lp', 'letter_lp', 'ended')} for r, g in gen.items()},
                       views=[f['views'] for f in found] if len(found) > 1 else found[0]['views'])
            t['total_s'] = t['load_s'] + t['search_s'] + sum(t[f'{r}_answer_s'] for r in PRESENTATIONS) + t['decide_s']
        except Exception as e:  # noqa: BLE001 -- one bad question must not end a run: recorded, the run goes on
            traceback.print_exc()
            out['error'] = f'{type(e).__name__}: {e}'
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        return out


def _finished(path):
    """(bench, qid) of the error-free records of a previous run; the file is rewritten without the other records
    (and a torn last line)."""
    keep = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('error') is None:
                keep.append((line if line.endswith('\n') else line + '\n', (r['bench'], r['qid'])))
    tmp = Path(f'{path}.tmp')
    tmp.write_text(''.join(line for line, _ in keep))
    tmp.replace(path)
    return {k for _, k in keep}


def run_rows(rows, run, out, resume=False, overwrite=False):
    """Write run(row) for every row as one JSON line of the file out (flushed per question). resume: keep the
    error-free records already in out and run the other rows; overwrite: replace out. A non-empty out is refused
    (FileExistsError) unless one of the two is given."""
    out = Path(out)
    if not (resume or overwrite) and out.exists() and out.stat().st_size:
        raise FileExistsError(f'{out} is not empty: --resume keeps its finished questions, --overwrite replaces it')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _finished(out) if resume and out.exists() else set()
    todo = [r for r in rows if (r['bench'], r['qid']) not in done]
    print(f'{len(rows)} questions, {len(todo)} to run -> {out}', flush=True)
    if not todo:
        return
    t0, errors = time.perf_counter(), 0
    with open(out, 'a' if done else 'w') as f:
        for k, row in enumerate(todo, 1):
            res = run(row)
            errors += res['error'] is not None
            f.write(json.dumps(res) + '\n')
            f.flush()
            if k % 10 == 0 or k == len(todo):
                rate = k / (time.perf_counter() - t0)
                print(f'{k}/{len(todo)}  {rate:.3f} q/s  ETA {(len(todo) - k) / rate / 60:.1f} min  errors {errors}',
                      flush=True)
