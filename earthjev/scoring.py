"""Score predictions using the benchmark-specific answer matching rules."""
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn

from earthjev.data import BENCHMARKS

LRS_SETS = ('lrs_vqa', 'lrsrefdev_ok', 'lrsrefdev_bad')  # LRS-VQA questions
GRO_SETS = ('lrs_gro', 'gro_train', 'grorefdev_ok', 'grorefdev_bad')  # LRS-GRO questions


# ---- VERBATIM from MME-RealWorld evaluation/eval_your_results.py (github.com/MME-Benchmarks/MME-RealWorld) ----
# ---- No license stated upstream; not covered by this repository's LICENSE; included unchanged to compute the
# ---- official metric.
def extract_characters_regex(s, choices):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ""
    return matches[0]
# ---- end verbatim ----


# ---- VERBATIM from LRS-VQA Script/evaluation_LRSVQA.py (github.com/VisionXLab/LRS-VQA) ----
# ---- No license stated upstream for Script/ (the repository's LRS-VQA-Code/LICENSE, Apache-2.0, covers only that
# ---- directory); not covered by this repository's LICENSE; included unchanged to compute the official metric.
def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.path_similarity(synset2) is not None and synset1.path_similarity(synset2) > 0.8:
                return True
    return False
# ---- end verbatim ----


# ---- VERBATIM from ZoomEarth src/eval/eval.py (github.com/earth-insights/ZoomEarth, the LRS-GRO scorer;
# ---- (c) 2025 ZoomEarth Project, Apache-2.0 per its README); are_synonyms renamed gro_are_synonyms.
def _lemmatize(word:str) -> str:
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(word)
    except Exception:
        return word

def gro_are_synonyms(word1:str, word2:str) -> bool:
    w1, w2 = _lemmatize(word1.lower()), _lemmatize(word2.lower())
    try:
        synsets1 = wn.synsets(w1)
        synsets2 = wn.synsets(w2)
    except Exception:
        return False
    best = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.path_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best >= 0.8
# ---- end verbatim ----


# ---- VERBATIM from lmms-eval lmms_eval/tasks/xlrs/mcq_utils.py (github.com/EvolvingLMMs-Lab/lmms-eval, Apache-2.0,
# ---- Copyright (c) 2025 LMMs-Lab; the scorer the XLRS-Bench README points to); extract_characters_regex renamed
# ---- extract_characters_regex_xlrs. The joined letter order follows Python's set order (callers sort it).
def extract_characters_regex_xlrs(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)"]):
    if type(s) is dict:
        s = ""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if not re.search("[ABCDE]", s):
        return ""
    matches = re.findall(r"\(([a-eA-E])\)", s)
    if len(matches) == 0:
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", s)
    if len(matches) == 0:
        matches = re.findall(r"[a-eA-E]", s)
    if len(matches) == 0:
        return ""
    else:
        matches = set(mat.upper() for mat in matches)
        return "".join(matches)
# ---- end verbatim ----


def mme_letter(output, choices):
    """The official extraction, except that an empty answer gives no letter (the verbatim function returns 'A' for it,
    because '' is a substring of every option)."""
    if not (output or "").strip():
        return ""
    return extract_characters_regex(output, choices)


def xlrs_correct(output, gt):
    """(letters, correct): correct iff the extracted letter set equals the reference set ('BCD' on the multi-answer
    land-use task); no answer extracts no letter."""
    letters = extract_characters_regex_xlrs(output or "")
    return letters, set(letters) == set(re.sub(r"[^A-E]", "", gt.upper()))


def rshr_letter(output):
    """The leading option letter of an answer ('B', '(B)', 'B.', 'B) ...'; not the first letter of a word), else None.
    A letter is read only from answers of at most 3 characters or whose second character is not a letter, so
    '(B) text' gives None."""
    raw = (output or "").strip()
    m = re.match(r"^\(?([A-Da-d])[\).:\s]?", raw)
    return m.group(1).upper() if m and (len(raw) <= 3 or not raw[1:2].isalpha()) else None


def rshr_correct(output, gt, options):
    """Correct iff the leading option letter is the reference, or the answer is the reference option's full text
    (case-insensitive)."""
    raw, gt = (output or "").strip(), str(gt).strip().upper()
    if rshr_letter(raw) == gt:
        return True
    opts = [str(o).strip().lower() for o in options]
    return raw.lower() in opts and "ABCD"[opts.index(raw.lower())] == gt


@lru_cache(maxsize=None)
def lrs_correct(gt, answer):
    """LRS-VQA: lowercase both (no stripping); correct iff equal or WordNet synonyms."""
    gt, answer = gt.lower(), answer.lower()
    return gt == answer or are_synonyms(gt, answer)


@lru_cache(maxsize=None)
def gro_correct(gt, answer):
    """LRS-GRO: lowercase both and strip the answer; correct iff equal or (lemmatized) WordNet synonyms."""
    gt, answer = gt.lower(), answer.lower().strip()
    return gt == answer or gro_are_synonyms(gt, answer)


@lru_cache(maxsize=None)
def _check_wordnet():
    """The WordNet rules need WordNet 3.0 (the LRS-GRO rule would silently score every synonym as wrong without it).
    Raises LookupError (with NLTK's download instructions) when the corpus is missing."""
    if wn.get_version() != '3.0':
        raise RuntimeError(f'the official scorers use WordNet 3.0, found WordNet {wn.get_version()}')


def score(row, output):
    """Correctness of an answer to a question (a row of earthjev.data) under its benchmark's scorer; None (no answer)
    is wrong."""
    bench, gt = row['bench'], str(row['ground_truth'])
    if bench == 'mme_rs':
        return mme_letter(output, row['choice_options']) == gt.strip().upper()
    if bench == 'xlrs_lite':
        return xlrs_correct(output, gt)[1]
    if bench == 'rshr':
        return rshr_correct(output, gt, [o['text'] for o in row['options']])
    if bench not in LRS_SETS + GRO_SETS:
        raise ValueError(f'no scorer for {bench!r}')
    _check_wordnet()
    if output is None:
        return False
    return gro_correct(gt, output) if bench in GRO_SETS else lrs_correct(gt, output)


def answer_letter(row, output):
    """The option letter(s) the benchmark's extraction reads from an answer ('' if none; XLRS-lite letter sets in
    alphabetical order); None for open-ended questions."""
    if row['bench'] == 'mme_rs':
        return mme_letter(output, row['choice_options'])
    if row['bench'] == 'xlrs_lite':
        return ''.join(sorted(extract_characters_regex_xlrs(output or '')))
    if row['bench'] == 'rshr':
        return rshr_letter(output) or ''
    return None


# ------------------------------------------------------------------ the scores of Tables I-II

RSHR_PERCEPTION = {'color', 'detection', 'object_clasfication', 'object_counting', 'object_grounding',
                   'regional_grouding', 'regionla_counting', 'relation', 'shape'}  # RSHR-Bench's category names
RSHR_REASONING = {'anomaly_detection', 'object_state_judgement', 'future_prediction_two_image'}
XLRS_L3 = [('AD', 'Complex reasoning/Anomaly Detection and Interpretation'),
           ('CCR', 'Counting/Counting with complex reasoning'),
           ('ECR', 'Complex reasoning/Environmental condition reasoning'), ('OC', 'Counting/Overall counting'),
           ('OCC', 'Object properties/Object classification'), ('OCL', 'Object properties/Object color'),
           ('OLUC', 'Land use classification/Overall Land use classification'),
           ('OMS', 'Object properties/Object motion state'),
           ('OSR', 'Object spatial relationship/Object spatial relationship'), ('RC', 'Counting/Regional counting'),
           ('RCCD', 'Counting/Counting with changing detection'),
           ('RLUC', 'Land use classification/Regional Land use classification'),
           ('RP', 'Complex reasoning/Route planning')]
GROUPS = {  # benchmark -> [(column, question predicate)]; every benchmark also has 'avg' (all questions)
    'lrs_vqa': [('fair', lambda r: r['source'] == 'FAIR'), ('bridge', lambda r: r['source'] == 'GLH-Bridge'),
                ('star', lambda r: r['source'] == 'STAR')],
    'mme_rs': [(c, lambda r, c=c: r['category'] == c) for c in ('color', 'count', 'position')],
    'lrs_gro': [(c, lambda r, c=c: r['source'] == c) for c in ('global', 'region', 'object')],
    'rshr': [('perception', lambda r: r['category'] in RSHR_PERCEPTION),
             ('reasoning', lambda r: r['category'] in RSHR_REASONING)],
    'xlrs_lite': [(a.lower(), lambda r, c=c: r['category'] == c) for a, c in XLRS_L3],
}


def _accuracy(rows, ok):
    return 100 * (sum(ok[r['qid']] for r in rows) / len(rows))


def _category_mean(rows, ok):
    """Unweighted mean over the task categories (in sorted order) of their accuracies."""
    by = defaultdict(list)
    for r in rows:
        by[r['category']].append(ok[r['qid']])
    return 100 * float(np.mean([sum(v) / len(v) for _, v in sorted(by.items(), key=lambda kv: str(kv[0]))]))


def benchmark_scores(bench, rows, outputs):
    """{'avg', <group columns>} of one benchmark in % (unrounded), for the answers outputs {qid: text} to the questions
    rows; a question without an answer is wrong, an empty group gives None."""
    ok = {r['qid']: score(r, outputs.get(r['qid'])) for r in rows}
    out = {}
    for col, keep in GROUPS.get(bench, []):
        group = [r for r in rows if keep(r)]
        out[col] = ((_category_mean if bench == 'rshr' else _accuracy)(group, ok)) if group else None
    out['avg'] = _accuracy(rows, ok)
    return out


def overall(avgs):
    """Overall Avg.: the unweighted mean of the five benchmark averages {bench: avg}."""
    return sum(avgs[b] for b in BENCHMARKS) / len(BENCHMARKS)
