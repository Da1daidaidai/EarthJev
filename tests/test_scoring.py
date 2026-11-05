"""earthjev.scoring: the official answer matching (quirks included) and the Tables I-II scores. Needs WordNet 3.0."""
import gzip
import json
from pathlib import Path

import pytest
from nltk.corpus import wordnet as wn

from earthjev import data, scoring as S

CH = ['(A) Yellow', '(B) White', '(C) Blue', '(D) Green', '(E) The image does not feature the color.']
FIXTURE = Path(__file__).parent / 'fixtures' / 'scoring_final_answers.json.gz'  # final answers of the paper's method
PAPER = {  # Tables I-II, Qwen3.5-9B row of the method (unrounded)
    'lrs_vqa': {'fair': 30.633802816901408, 'bridge': 36.53483992467043, 'star': 36.68417104276069,
                'avg': 34.78794490658667},
    'mme_rs': {'color': 72.03187250996017, 'count': 39.8858075040783, 'position': 77.00875099443118,
               'avg': 63.16211878009631},
    'lrs_gro': {'global': 77.10481099656357, 'region': 52.743902439024396, 'object': 54.070450097847356,
                'avg': 59.266488596671465},
    'rshr': {'perception': 53.50327356001565, 'reasoning': 60.52706552706554, 'avg': 52.25303292894281},
    'xlrs_lite': {'ad': 76.0, 'ccr': 51.0, 'ecr': 77.0, 'oc': 26.666666666666668, 'occ': 61.875,
                  'ocl': 59.12500000000001, 'oluc': 11.0, 'oms': 80.0, 'osr': 37.6, 'rc': 54.0,
                  'rccd': 41.66666666666667, 'rluc': 88.0, 'rp': 28.999999999999996, 'avg': 55.811688311688314},
}
OVERALL = 53.056254704797105


def row(bench, gt, **kw):
    return dict(qid='q', bench=bench, ground_truth=gt, **kw)


def test_wordnet_version():
    assert wn.get_version() == '3.0'


def test_mme_quirks():
    letter = lambda s: S.mme_letter(s, CH)
    assert letter('The best answer is (B).') == 'B' and letter('(D) Green') == 'D'
    assert letter('Based on the image, the roof is C') == 'B'  # first capital A-E anywhere
    assert letter('Every roof is green') == 'E'  # <= 10 words: the 'E' of 'Every'
    assert letter('Best answer: (C)') == 'B'  # 'Best answer:' is never stripped (a missing comma upstream)
    assert letter('white') == 'B'  # no A-E: option-text fallback
    assert letter('looking at the lower right area of the image there is a square building') == ''  # > 10 words
    assert letter('') == letter('  ') == letter(None) == ''  # empty answers score 0 (verbatim says 'A')
    assert S.extract_characters_regex('', CH) == 'A'
    r = row('mme_rs', 'B', choice_options=CH)
    assert S.score(r, 'B') and S.score(r, 'white') and not S.score(r, '') and not S.score(r, None)
    assert S.answer_letter(r, 'The best answer is (B).') == 'B'


def test_xlrs_letter_sets():
    letters = lambda s: S.xlrs_correct(s, 'A')[0]
    assert letters('B') == 'B' and letters('(C) Red') == 'C' and letters('The answer is D') == 'D'
    assert letters('b') == '' and letters('') == letters(None) == ''  # needs an upper-case A-E somewhere
    assert set(letters('Because')) == {'B', 'C', 'A', 'E'}  # stray letters of verbose answers (official quirk)
    r = row('xlrs_lite', 'BCD')
    assert S.score(r, 'D, C, B') and not S.score(r, 'B C') and not S.score(r, None)
    assert S.answer_letter(r, 'D C B') == 'BCD' and S.answer_letter(r, 'nothing') == ''  # sorted: deterministic


def test_rshr_rule():
    opts = ['Red roofs', 'Blue roofs', 'Grey roofs', 'Bare soil']
    r = row('rshr', 'B', options=[{'id': k, 'text': t} for k, t in zip('ABCD', opts)])
    assert all(S.score(r, s) for s in ('B', '(B)', 'B.', 'b', 'B) Blue roofs', ' blue roofs '))
    assert not S.score(r, 'Blue') and not S.score(r, 'Because B') and not S.score(r, None)
    assert S.answer_letter(r, 'C) Grey roofs') == 'C' and S.answer_letter(r, 'Blue') == ''
    assert S.answer_letter(r, '(C) Grey roofs') == S.answer_letter(r, '(C).') == ''  # 2nd character is a letter
    assert S.score(row('rshr', 'D', options=r['options']), 'bare soil')  # full option text


def test_open_ended_rules():
    lrs, gro = (lambda gt, a: S.score(row('lrs_vqa', gt), a)), (lambda gt, a: S.score(row('lrs_gro', gt), a))
    assert lrs('Yes', 'yes') and lrs('car', 'automobile') and lrs('2', 'two')  # WordNet synonyms
    assert not lrs('2', '2.') and not lrs('2', '2 ') and not lrs('car', 'truck')  # LRS-VQA: no stripping
    assert gro('ship', ' Ships ') and gro('2', '2 ') and not gro('car', 'truck')  # LRS-GRO: stripped, lemmatized
    assert not lrs('car', None) and not gro('car', None)
    # an empty reference (LRS-GRO-11315): the official rule matches an empty or blank answer
    empty = next(r for r in data.load_rows('lrs_gro') if r['qid'] == 'LRSGRO__LRS-GRO-11315')
    assert empty['ground_truth'] == '' and S.score(empty, '') and S.score(empty, ' ') and not S.score(empty, None)
    assert S.score(row('lrsrefdev_bad', 'car'), 'automobile') and S.score(row('grorefdev_ok', 'ship'), 'ships')
    assert S.answer_letter(row('lrs_vqa', 'car'), 'car') is None
    with pytest.raises(ValueError):
        S.score(row('unknown', 'car'), 'car')


def test_group_scores():
    rows = [dict(row('rshr', 'A', category=c), qid=f'q{i}', options=[{'id': k, 'text': k} for k in 'ABCD'])
            for i, c in enumerate(['color', 'color', 'color', 'shape', 'anomaly_detection'])]
    got = S.benchmark_scores('rshr', rows, {'q0': 'A', 'q1': 'A', 'q2': 'B', 'q3': 'A'})  # q4: no answer
    assert got['avg'] == 60.0 and got['reasoning'] == 0.0
    assert got['perception'] == pytest.approx(100 * (2 / 3 + 1) / 2)  # category mean, not 3 / 4
    assert S.overall(dict.fromkeys(data.BENCHMARKS, 50.0)) == 50.0


def test_paper_tables():
    answers = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    got = {b: S.benchmark_scores(b, data.load_rows(b), answers[b]) for b in data.BENCHMARKS}
    assert got == PAPER  # exact floats
    assert S.overall({b: got[b]['avg'] for b in data.BENCHMARKS}) == OVERALL
