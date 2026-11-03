"""The decision parameters refit from the calibration runs (earthjev.train.fit_decisions on data/calibration/<config>)
equal the shipped configuration bitwise, and the calibration runs are the files their manifest lists."""
import gzip
import hashlib

import pytest

from earthjev.data import DATA
from earthjev.paths import load_config
from earthjev.train import fit_decisions

# configuration: (number of calibration files, the chosen (tau_fast, tau_keep), its J rounded to 4 decimals)
CONFIGS = {'qwen35-9b': (21, (0.6, 0.0), 50.5128), 'qwen3-vl-8b': (23, (0.5, 0.5), 49.3968)}


@pytest.mark.parametrize('config', CONFIGS)
def test_calibration_runs_match_manifest(config):
    runs = DATA / 'calibration' / config
    rows = (runs / 'MANIFEST.tsv').read_text().splitlines()[1:]
    for line in rows:
        name, sha, n = line.split('\t')
        raw = gzip.decompress((runs / f'{name}.gz').read_bytes())
        assert (hashlib.sha256(raw).hexdigest(), len(raw)) == (sha, int(n)), name
    files = {str(p.relative_to(runs)) for p in runs.rglob('*.gz')}
    assert len(rows) == CONFIGS[config][0] and files == {line.split('\t')[0] + '.gz' for line in rows}


@pytest.mark.parametrize('config', CONFIGS)
def test_refit_equals_configuration(config):
    cfg = load_config(config)
    fit, J = fit_decisions.refit(cfg, DATA / 'calibration' / config)
    assert fit_decisions.compare(fit, cfg) == []  # bitwise on AVX-512 CPUs; a few ulps elsewhere (REL_TOL)
    best = max(J, key=lambda g: (round(J[g], 9), -g[1], g[0]))
    assert best == CONFIGS[config][1] and round(J[best], 4) == CONFIGS[config][2]


def test_load_units_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit_decisions.load_units('mme_rs', tmp_path)
