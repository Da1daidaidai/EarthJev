"""Resolve model, dataset and output paths."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _env(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f'set the environment variable {name} (README.md)')
    return Path(value)


def data_root():
    return _env('EARTHJEV_DATA')


def models_root():
    return _env('EARTHJEV_MODELS')


def out_root():
    return _env('EARTHJEV_OUT', str(REPO / 'outputs'))


def load_config(name='qwen35-9b'):
    """The method configuration: a name under configs/ (without .json) or a path to a JSON file."""
    path = Path(name)
    if not path.suffix:
        path = REPO / 'configs' / f'{name}.json'
    with open(path) as f:
        return json.load(f)


def weights_dir(cfg, part):
    """Directory of the trained weights of one part ('region_head' or 'verifier') of a configuration: its entry in the
    configuration, under $EARTHJEV_MODELS unless it is an absolute path."""
    d = models_root() / cfg['weights'][part]
    if not d.is_dir():
        raise FileNotFoundError(f'{d} does not exist: download the weights (README, "B. Download the Weights")')
    return d
