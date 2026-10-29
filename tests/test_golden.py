from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from p0ly_eeg.metadata import dmss, dotprobe, intwm, mgsearch, simonfb

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = DATA_DIR / "expected"

GOLDEN_CONFIGS = [
    ("dmss",      dmss,     DATA_DIR / "dmss_ann.csv",     None,                        EXPECTED_DIR / "dmss.csv"),
    ("dotprobe",  dotprobe, DATA_DIR / "dotprobe_ann.csv",  None,                        EXPECTED_DIR / "dotprobe.csv"),
    ("mgsearch",  mgsearch, DATA_DIR / "mgsearch_ann.csv",  None,                        EXPECTED_DIR / "mgsearch.csv"),
    ("simonfb",   simonfb,  DATA_DIR / "simonfb_ann.csv",   None,                        EXPECTED_DIR / "simonfb.csv"),
    ("intwm",     intwm,    DATA_DIR / "intwm_ann.csv",     None,                        EXPECTED_DIR / "intwm.csv"),
    ("intwm_ptb", intwm,    DATA_DIR / "intwm_ann.csv",     DATA_DIR / "intwm_ptb.csv",  EXPECTED_DIR / "intwm_ptb.csv"),
]


@pytest.mark.parametrize("name,mod,ann_path,csv_path,golden_path", GOLDEN_CONFIGS, ids=[c[0] for c in GOLDEN_CONFIGS])
def test_matches_golden(name, mod, ann_path, csv_path, golden_path):
    ann = pd.read_csv(ann_path, index_col=0)
    result = mod.get_metadata(ann, f=csv_path)
    expected = pd.read_csv(golden_path)
    # CodeLookup returns "" for no-match; CSV round-trip converts "" -> NaN.
    # Align by filling NaN with "" in object (string) columns only.
    for col in expected.select_dtypes(include="object").columns:
        expected[col] = expected[col].fillna("")
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_dtype=False,
    )
