"""IGT experiment metadata spec (loaded from YAML)."""
from __future__ import annotations

from pathlib import Path

from p0ly_eeg.metadata.core import ExperimentSpec
from p0ly_eeg.metadata.parser import parse_metadata

_SPEC_PATH = Path(__file__).parent / "specs" / "igt.yaml"
spec = ExperimentSpec.from_yaml(_SPEC_PATH)


def get_metadata(df, f=None, sel_trials=False):
    return parse_metadata(spec, df, csv_path=f, expand_trials=sel_trials)
