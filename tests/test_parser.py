from __future__ import annotations

import io
import math

import mne
import numpy as np
import pandas as pd
import pytest

from p0ly_eeg.metadata import dmss, dotprobe, igt, intwm, mgsearch, simonfb
from p0ly_eeg.metadata.parser import events_from_raw, parse_metadata


def _events(rows: list[tuple[float, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["onset", "description"])


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Two complete dmss trials wrapped in one block
DMSS_EVENTS = _events([
    (0.0, "Stim/S  3"),   # block start
    (0.1, "Stim/S  5"),   # trial 1 start
    (0.2, "Stim/S 11"),   # Size=1
    (0.5, "Stim/S 57"),   # RT start
    (1.0, "Stim/S 64"),   # Correct
    (1.2, "Stim/S  6"),   # trial 1 end
    (1.3, "Stim/S  5"),   # trial 2 start
    (1.4, "Stim/S 25"),   # Size=2
    (1.5, "Stim/S 57"),
    (1.8, "Stim/S 60"),
    (1.9, "Stim/S  6"),
    (2.0, "Stim/S  4"),   # block end
])

# Two dotprobe trials in two blocks
DOTPROBE_EVENTS = _events([
    (0.0, "Stim/S  9"),    # trial 1 start
    (0.1, "Stim/S 11"),
    (0.2, "Stim/S 16"),
    (0.5, "Stim/S 14"),    # RT start
    (1.0, "Stim/S 37"),
    (1.2, "Stim/S 34"),
    (1.3, "Stim/S 30"),    # RT end
    (1.4, "Stim/S 10"),    # trial 1 end marker (part of spec for filtering)
    (1.5, "Stim/S103"),    # block boundary
    (1.6, "Stim/S  9"),    # trial 2 start
    (1.7, "Stim/S 13"),
    (1.8, "Stim/S 17"),
    (1.9, "Stim/S 14"),
    (2.2, "Stim/S 38"),
    (2.3, "Stim/S 34"),
    (2.4, "Stim/S 30"),
    (2.5, "Stim/S 10"),
])

# Two intwm trials (no block codes)
INTWM_EVENTS = _events([
    (0.0, "Stim/S 81"),   # trial 1 start
    (0.1, "Stim/S201"),
    (0.2, "Stim/S251"),   # correct
    (0.5, "Stim/S 82"),   # trial 2 start
    (0.6, "Stim/S202"),
    (0.7, "Stim/S250"),   # incorrect
])

# Two simonfb trials in one block
SIMONFB_EVENTS = _events([
    (0.0, "Stim/S  7"),   # block start
    (0.1, "Stim/S 17"),   # trial 1 start
    (0.2, "Stim/S 16"),
    (0.3, "Stim/S 15"),
    (0.4, "Stim/S 12"),
    (0.8, "Stim/S 37"),
    (0.9, "Stim/S 34"),
    (1.0, "Stim/S 30"),
    (1.1, "Stim/S 18"),   # trial 1 end
    (1.2, "Stim/S 17"),   # trial 2 start
    (1.3, "Stim/S 19"),
    (1.4, "Stim/S 14"),
    (1.5, "Stim/S 12"),
    (1.9, "Stim/S 38"),
    (2.0, "Stim/S 31"),
    (2.1, "Stim/S 30"),
    (2.2, "Stim/S 18"),   # trial 2 end
    (2.3, "Stim/S  8"),   # block end
])

# One igt trial (uses trial_expander)
IGT_EVENTS = _events([
    (0.0, "Stim/S 20"),   # block start
    (0.1, "Stim/S 30"),   # trial start
    (0.2, "Stim/S 40"),   # select
    (0.3, "Stim/S 41"),   # card A
    (0.4, "Stim/S 45"),   # deck 1
    (0.8, "Stim/S 50"),   # submit
    (1.0, "Stim/S 61"),   # win
    (1.1, "Stim/S 31"),   # trial end
    (1.2, "Stim/S 21"),   # block end
])


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_dmss_columns(self):
        r = dmss.get_metadata(DMSS_EVENTS)
        assert set(r.columns) >= {"Block", "Trial", "Size", "Correct", "RT"}

    def test_dotprobe_columns(self):
        r = dotprobe.get_metadata(DOTPROBE_EVENTS)
        assert set(r.columns) >= {"Block", "Trial", "Cue_type", "Correct", "RT"}

    def test_intwm_columns(self):
        r = intwm.get_metadata(INTWM_EVENTS)
        assert set(r.columns) >= {"Block", "Trial", "Correct"}

    def test_simonfb_columns(self):
        r = simonfb.get_metadata(SIMONFB_EVENTS)
        assert set(r.columns) >= {"Block", "Trial", "Color", "Side", "Correct", "RT", "Congruent"}

    def test_igt_columns(self):
        r = igt.get_metadata(IGT_EVENTS)
        assert set(r.columns) >= {"Block", "Trial", "RT_Submit", "Card", "Deck", "Result"}


# ---------------------------------------------------------------------------
# Trial and block monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_trial_monotone_in_block(self):
        r = dmss.get_metadata(DMSS_EVENTS)
        for _, blk in r.groupby("Block"):
            trials = blk["Trial"].tolist()
            assert trials == sorted(trials)

    def test_block_monotone(self):
        r = dotprobe.get_metadata(DOTPROBE_EVENTS)
        blocks = r["Block"].tolist()
        assert blocks == sorted(blocks)

    def test_two_trials_numbered_1_and_2(self):
        r = dmss.get_metadata(DMSS_EVENTS)
        assert list(r["Trial"]) == [1, 2]

    def test_two_blocks_numbered_1_and_2(self):
        r = dotprobe.get_metadata(DOTPROBE_EVENTS)
        assert sorted(r["Block"].unique()) == [1, 2]


# ---------------------------------------------------------------------------
# Non-spec events ignored
# ---------------------------------------------------------------------------

class TestNonSpecEventsIgnored:
    def _inject_noise(self, events: pd.DataFrame, noise_code: str) -> pd.DataFrame:
        noise = pd.DataFrame({
            "onset": [0.05, 0.35, 0.95],
            "description": [noise_code] * 3,
        })
        return pd.concat([events, noise]).sort_values("onset").reset_index(drop=True)

    def test_extra_codes_do_not_change_dmss_output(self):
        clean = dmss.get_metadata(DMSS_EVENTS)
        noisy = dmss.get_metadata(self._inject_noise(DMSS_EVENTS, "Stim/S  1"))
        pd.testing.assert_frame_equal(clean.reset_index(drop=True), noisy.reset_index(drop=True))

    def test_extra_codes_do_not_change_intwm_output(self):
        clean = intwm.get_metadata(INTWM_EVENTS)
        noisy = intwm.get_metadata(self._inject_noise(INTWM_EVENTS, "Stim/NOISE"))
        pd.testing.assert_frame_equal(clean.reset_index(drop=True), noisy.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    @pytest.mark.parametrize("mod", [dmss, dotprobe, mgsearch, simonfb, intwm, igt])
    def test_empty_dataframe_returns_empty(self, mod):
        empty = pd.DataFrame(columns=["onset", "description"])
        result = mod.get_metadata(empty)
        assert result.empty

    @pytest.mark.parametrize("mod", [dmss, dotprobe, mgsearch, simonfb, intwm, igt])
    def test_empty_does_not_raise(self, mod):
        empty = pd.DataFrame(columns=["onset", "description"])
        mod.get_metadata(empty)  # must not raise

    @pytest.mark.parametrize("mod", [dmss, dotprobe, mgsearch, simonfb, intwm, igt])
    def test_empty_result_keeps_schema(self, mod):
        """An event-less run must stay typed, not collapse to shape (0, 0)."""
        result = mod.get_metadata(pd.DataFrame(columns=["onset", "description"]))
        assert len(result) == 0
        assert {"Block", "Trial", "Onset"}.issubset(result.columns)

    @pytest.mark.parametrize(
        ("mod", "events"),
        [
            (dmss, DMSS_EVENTS),
            (dotprobe, DOTPROBE_EVENTS),
            (intwm, INTWM_EVENTS),
            (simonfb, SIMONFB_EVENTS),
            (igt, IGT_EVENTS),
        ],
    )
    def test_empty_columns_match_populated(self, mod, events):
        """Empty and populated runs must agree on columns *and order* (SCHEMA §2)."""
        populated = mod.get_metadata(events)
        empty = mod.get_metadata(pd.DataFrame(columns=["onset", "description"]))
        assert len(populated) > 0
        assert list(empty.columns) == list(populated.columns)

    @pytest.mark.parametrize("mod", [dmss, dotprobe, mgsearch, simonfb, intwm, igt])
    def test_empty_result_tsv_roundtrip(self, mod):
        """Regression: a headerless TSV made ``read_csv`` raise ``EmptyDataError``."""
        empty = mod.get_metadata(pd.DataFrame(columns=["onset", "description"]))
        buf = io.StringIO()
        empty.to_csv(buf, sep="\t", index=False)
        assert buf.getvalue().strip(), "empty metadata must still write a header"
        back = pd.read_csv(io.StringIO(buf.getvalue()), sep="\t")
        assert back.empty
        assert list(back.columns) == list(empty.columns)


# ---------------------------------------------------------------------------
# Stimulus prefix normalization
# ---------------------------------------------------------------------------

class TestPrefixNormalization:
    def test_stimulus_prefix_same_as_stim(self):
        stim_events = DMSS_EVENTS.copy()
        stim_events["description"] = stim_events["description"].str.replace("Stim/", "Stimulus/", regex=False)
        result_stim = dmss.get_metadata(stim_events)
        result_clean = dmss.get_metadata(DMSS_EVENTS)
        pd.testing.assert_frame_equal(
            result_stim.reset_index(drop=True),
            result_clean.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# CSV merge
# ---------------------------------------------------------------------------

class TestCsvMerge:
    def test_csv_values_appended_by_position(self, tmp_path):
        csv_path = tmp_path / "intwm.csv"
        csv_path.write_text(
            "condArray,tarLoci,Ori_1,Ori_2,tarOri,startTime4int,chgRT\n"
            "1,1,10.0,20.0,30.0,0.5,1.2\n"
            "3,2,11.0,21.0,31.0,0.6,1.3\n"
        )
        result = intwm.get_metadata(INTWM_EVENTS, f=str(csv_path))
        assert list(result["Condition"]) == ["static", "constant"]
        assert list(result["Target"]) == ["left", "right"]
        assert list(result["OrientL"]) == [10.0, 11.0]

    def test_csv_shorter_than_trials_pads_with_nan(self, tmp_path):
        csv_path = tmp_path / "intwm_short.csv"
        csv_path.write_text(
            "condArray,tarLoci,Ori_1,Ori_2,tarOri,startTime4int,chgRT\n"
            "2,1,10.0,20.0,30.0,0.5,1.2\n"
        )
        result = intwm.get_metadata(INTWM_EVENTS, f=str(csv_path))
        assert result["Condition"].iloc[0] == "ignore"
        assert pd.isna(result["Condition"].iloc[1])


# ---------------------------------------------------------------------------
# Block boundary codes
# ---------------------------------------------------------------------------

class TestBlockBoundaries:
    def test_block_code_increments_block(self):
        result = dotprobe.get_metadata(DOTPROBE_EVENTS)
        assert 1 in result["Block"].values
        assert 2 in result["Block"].values

    def test_no_block_code_all_trials_in_block_1(self):
        result = intwm.get_metadata(INTWM_EVENTS)
        assert list(result["Block"].unique()) == [1]


# ---------------------------------------------------------------------------
# infer_block_from
# ---------------------------------------------------------------------------

class TestInferBlockFrom:
    def test_mgsearch_infers_block_from_cue_side_change(self):
        events = _events([
            (0.0,  "Stim/S182"),  # trial 1 start
            (0.1,  "Stim/S 10"),  # cue left
            (0.3,  "Stim/S110"),
            (0.6,  "Stim/S189"),
            (0.7,  "Stim/S183"),  # trial 1 end
            (0.8,  "Stim/S182"),  # trial 2 start
            (0.9,  "Stim/S 11"),  # cue right -> new block
            (1.1,  "Stim/S110"),
            (1.4,  "Stim/S189"),
            (1.5,  "Stim/S183"),  # trial 2 end
        ])
        result = mgsearch.get_metadata(events)
        assert list(result["Block"]) == [1, 2]


# ---------------------------------------------------------------------------
# Trial expander
# ---------------------------------------------------------------------------

class TestTrialExpander:
    def test_expand_false_one_row_per_trial(self):
        result = igt.get_metadata(IGT_EVENTS, sel_trials=False)
        assert len(result) == 1

    def test_expand_true_one_row_per_selection(self):
        multi_sel_events = _events([
            (0.0, "Stim/S 20"),
            (0.1, "Stim/S 30"),
            (0.2, "Stim/S 40"),   # first select
            (0.3, "Stim/S 41"),
            (0.4, "Stim/S 45"),
            (0.5, "Stim/S 40"),   # second select
            (0.6, "Stim/S 42"),
            (0.7, "Stim/S 46"),
            (0.9, "Stim/S 50"),
            (1.0, "Stim/S 61"),
            (1.1, "Stim/S 31"),
            (1.2, "Stim/S 21"),
        ])
        result = igt.get_metadata(multi_sel_events, sel_trials=True)
        assert len(result) == 2

    def test_rt_select_is_positive(self):
        result = igt.get_metadata(IGT_EVENTS)
        assert result["RT_Select"].iloc[0] > 0

    def test_total_sel_count(self):
        result = igt.get_metadata(IGT_EVENTS)
        assert result["Total_Sel"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Negative RT behavior
# ---------------------------------------------------------------------------

class TestNegativeRT:
    def test_dotprobe_negative_rt_becomes_nan(self):
        events = _events([
            (0.0, "Stim/S  9"),
            (0.1, "Stim/S 30"),   # response before cue
            (0.5, "Stim/S 14"),   # cue
            (0.6, "Stim/S 11"),
        ])
        result = dotprobe.get_metadata(events)
        assert math.isnan(result["RT"].iloc[0])

    def test_dmss_allows_negative_rt(self):
        events = _events([
            (0.0, "Stim/S  3"),
            (0.1, "Stim/S  5"),
            (0.3, "Stim/S 60"),   # response before probe
            (1.0, "Stim/S 57"),   # probe
            (1.2, "Stim/S  6"),
            (1.3, "Stim/S  4"),
        ])
        result = dmss.get_metadata(events)
        assert result["RT"].iloc[0] < 0


# ---------------------------------------------------------------------------
# events_from_raw: raw.annotations -> parser event frame (US-016)
# ---------------------------------------------------------------------------

def _raw_with_annotations(rows: list[tuple[float, str]]) -> mne.io.Raw:
    """Build a tiny Raw carrying Psychtoolbox-style marker annotations."""
    sfreq = 100.0
    onsets = [t for t, _ in rows]
    descriptions = [d for _, d in rows]
    info = mne.create_info(["Cz"], sfreq, ch_types=["eeg"])
    # 2 s of zeros, long enough to host the test markers (max ~2.0 s).
    data = np.zeros((1, int(sfreq * 2.0)))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[0.0] * len(rows), description=descriptions))
    return raw


class TestEventsFromRaw:
    def test_extracts_description_and_onset_columns(self):
        raw = _raw_with_annotations([(0.1, "Stim/S  3"), (0.5, "Stim/S 11")])
        events = events_from_raw(raw)
        assert list(events.columns) == ["description", "onset"]
        assert list(events["description"]) == ["Stim/S  3", "Stim/S 11"]
        assert events["onset"].tolist() == pytest.approx([0.1, 0.5])

    def test_empty_raw_returns_empty_frame_with_columns(self):
        raw = _raw_with_annotations([])
        events = events_from_raw(raw)
        assert events.empty
        assert list(events.columns) == ["description", "onset"]

    def test_feeds_parse_metadata_for_core_and_extended_columns(self):
        # Two dmss trials wrapped in one block (mirrors DMSS_EVENTS) annotated
        # onto a Raw; the full raw -> metadata path must reproduce the column
        # shape parse_metadata expects.
        raw = _raw_with_annotations([
            (0.0, "Stim/S  3"),   # block start
            (0.1, "Stim/S  5"),   # trial 1 start
            (0.2, "Stim/S 11"),   # Size=1
            (0.5, "Stim/S 57"),   # RT start
            (1.0, "Stim/S 64"),   # Correct
            (1.2, "Stim/S  6"),   # trial 1 end
            (1.3, "Stim/S  5"),   # trial 2 start
            (1.4, "Stim/S 25"),   # Size=2
            (1.5, "Stim/S 57"),
            (1.8, "Stim/S 60"),
            (1.9, "Stim/S  6"),
        ])
        events = events_from_raw(raw)
        result = parse_metadata(dmss.spec, events)

        # AC: core columns + at least one extended column.
        assert {"Block", "Trial", "Onset"}.issubset(result.columns)
        assert {"RT", "Correct", "Size"}.issubset(result.columns)
        assert len(result) == 2
        assert list(result["Trial"]) == [1, 2]
        assert list(result["Block"]) == [1, 1]
