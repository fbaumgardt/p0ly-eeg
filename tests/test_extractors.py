from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from p0ly_eeg.metadata.core import (
    BoolPresence,
    CodeLookup,
    DerivedColumn,
    ExpandOnEvent,
    ExperimentSpec,
    IntSum,
    ListCollect,
    RTMeasure,
)


def _group(*descriptions: str, onsets: list[float] | None = None) -> pd.DataFrame:
    if onsets is None:
        onsets = list(range(len(descriptions)))
    return pd.DataFrame({"description": list(descriptions), "onset": onsets})


# ---------------------------------------------------------------------------
# CodeLookup
# ---------------------------------------------------------------------------

class TestCodeLookup:
    def test_join_mode_concatenates_matches(self):
        ext = CodeLookup({"left": "S1", "right": "S2"})
        assert ext.extract(_group("S1", "S3", "S2")) == "leftright"

    def test_join_mode_empty_group_returns_empty(self):
        ext = CodeLookup({"left": "S1"})
        assert ext.extract(_group("S9")) == ""

    def test_first_mode_returns_first_match(self):
        ext = CodeLookup({"left": "S1", "right": "S2"}, agg="first")
        assert ext.extract(_group("S9", "S2", "S1")) == "right"

    def test_first_mode_no_match_returns_empty(self):
        ext = CodeLookup({"left": "S1"}, agg="first")
        assert ext.extract(_group("S9", "S8")) == ""

    def test_codes_returns_values(self):
        ext = CodeLookup({"left": "S1", "right": "S2"})
        assert ext.codes() == {"S1", "S2"}


# ---------------------------------------------------------------------------
# BoolPresence
# ---------------------------------------------------------------------------

class TestBoolPresence:
    def test_true_when_code_present(self):
        ext = BoolPresence("S34")
        assert ext.extract(_group("S12", "S34")) is True

    def test_false_when_code_absent(self):
        ext = BoolPresence("S34")
        assert ext.extract(_group("S12", "S35")) is False

    def test_false_on_empty_group(self):
        ext = BoolPresence("S34")
        assert ext.extract(pd.DataFrame({"description": [], "onset": []})) is False

    def test_codes_returns_singleton(self):
        ext = BoolPresence("S34")
        assert ext.codes() == {"S34"}


# ---------------------------------------------------------------------------
# IntSum
# ---------------------------------------------------------------------------

class TestIntSum:
    def test_sums_mapped_codes(self):
        ext = IntSum({"S64": 1, "S25": 2})
        assert ext.extract(_group("S64", "S25", "S64")) == 4

    def test_ignores_unmapped_codes(self):
        ext = IntSum({"S64": 1})
        assert ext.extract(_group("S99", "S00")) == 0

    def test_zero_value_codes_contribute_nothing(self):
        ext = IntSum({"S250": 0, "S251": 1})
        assert ext.extract(_group("S250", "S250", "S251")) == 1

    def test_codes_returns_keys(self):
        ext = IntSum({"S64": 1, "S65": 0})
        assert ext.codes() == {"S64", "S65"}


# ---------------------------------------------------------------------------
# ListCollect
# ---------------------------------------------------------------------------

class TestListCollect:
    def test_collects_values_in_order(self):
        ext = ListCollect({"A": "S41", "B": "S42"})
        assert ext.extract(_group("S41", "S99", "S42", "S41")) == ["A", "B", "A"]

    def test_empty_list_when_no_matches(self):
        ext = ListCollect({"A": "S41"})
        assert ext.extract(_group("S99")) == []

    def test_codes_returns_values(self):
        ext = ListCollect({"A": "S41", "B": "S42"})
        assert ext.codes() == {"S41", "S42"}


# ---------------------------------------------------------------------------
# DerivedColumn
# ---------------------------------------------------------------------------

class TestDerivedColumn:
    def test_extract_raises(self):
        ext = DerivedColumn(depends_on=["a"], fn=lambda row: row["a"])
        with pytest.raises(NotImplementedError):
            ext.extract(_group("S1"))

    def test_derive_applies_fn(self):
        ext = DerivedColumn(depends_on=["x", "y"], fn=lambda row: row["x"] + row["y"])
        assert ext.derive({"x": 3, "y": 4, "z": 99}) == 7

    def test_codes_returns_empty(self):
        ext = DerivedColumn(depends_on=[], fn=lambda row: None)
        assert ext.codes() == set()


# ---------------------------------------------------------------------------
# RTMeasure
# ---------------------------------------------------------------------------

class TestRTMeasure:
    def test_computes_onset_difference(self):
        rt = RTMeasure("RT", start=["S1"], end=["S2"])
        group = _group("S1", "S2", onsets=[1.0, 1.5])
        assert rt.extract(group) == pytest.approx(0.5)

    def test_nan_if_negative_true(self):
        rt = RTMeasure("RT", start=["S1"], end=["S2"], nan_if_negative=True)
        group = _group("S2", "S1", onsets=[1.0, 1.5])
        assert math.isnan(rt.extract(group))

    def test_nan_if_negative_false_allows_negative(self):
        rt = RTMeasure("RT", start=["S1"], end=["S2"], nan_if_negative=False)
        group = _group("S2", "S1", onsets=[1.0, 1.5])
        assert rt.extract(group) < 0

    def test_zero_when_start_and_end_missing(self):
        rt = RTMeasure("RT", start=["S1"], end=["S2"])
        group = _group("S9", onsets=[5.0])
        assert rt.extract(group) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ExperimentSpec.column_codes() and event_codes()
# ---------------------------------------------------------------------------

class TestExperimentSpec:
    def _make_spec(self, **kwargs) -> ExperimentSpec:
        defaults: dict[str, Any] = dict(
            name="test",
            timelocks={},
            trial_codes=["T1"],
            columns={},
            block_codes=[],
        )
        defaults.update(kwargs)
        return ExperimentSpec(**defaults)

    def test_column_codes_from_columns(self):
        spec = self._make_spec(columns={"C": IntSum({"S1": 1, "S2": 2})})
        assert spec.column_codes() == {"S1", "S2"}

    def test_column_codes_from_rt_defs(self):
        spec = self._make_spec(rt_defs=[RTMeasure("RT", start=["S3"], end=["S4"])])
        assert spec.column_codes() == {"S3", "S4"}

    def test_column_codes_from_timelocks(self):
        spec = self._make_spec(timelocks={"stim": {"all": "S5"}})
        assert "S5" in spec.column_codes()

    def test_column_codes_from_expander(self):
        expander = ExpandOnEvent("S10", {"Card": ListCollect({"A": "S11"})})
        spec = self._make_spec(trial_expander=expander)
        assert {"S10", "S11"} <= spec.column_codes()

    def test_event_codes_includes_trial_and_block_codes(self):
        spec = self._make_spec(
            trial_codes=["T1", "T2"],
            block_codes=["B1"],
            columns={"C": BoolPresence("S9")},
        )
        ec = spec.event_codes()
        assert {"T1", "T2", "B1", "S9"} <= ec

    def test_event_codes_is_union(self):
        spec = self._make_spec(
            trial_codes=["T1"],
            block_codes=["B1"],
            columns={"C": IntSum({"S1": 1})},
        )
        assert spec.event_codes() == spec.column_codes() | {"T1", "B1"}

    def test_derived_column_contributes_no_codes(self):
        spec = self._make_spec(
            columns={"D": DerivedColumn(depends_on=[], fn=lambda r: 0)}
        )
        assert spec.column_codes() == set()
