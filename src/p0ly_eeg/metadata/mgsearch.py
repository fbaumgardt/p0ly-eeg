"""Memory Guided Search experiment metadata spec (loaded from YAML)."""
from __future__ import annotations

from pathlib import Path

from p0ly_eeg.metadata.core import ExperimentSpec
from p0ly_eeg.metadata.parser import parse_metadata

_SPEC_PATH = Path(__file__).parent / "specs" / "mgsearch.yaml"
spec = ExperimentSpec.from_yaml(_SPEC_PATH)


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
