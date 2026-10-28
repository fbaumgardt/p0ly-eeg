"""Tests for p0ly_utils.analysis.regression (US-014).

All data is simulated in memory (no fixture files, no ``@pytest.mark.slow``):
synthetic voltage built as ``intercept + RT * slope + is_correct * shift``
with per-channel, per-time profiles so a transposed reshaping cannot pass,
plus a small Morlet power grid for the 4-D power path. Cohort tests use
subjects with different trial counts *and* different coefficient magnitudes
to prove the aggregation is the arithmetic subject mean, not a pooled-trial
estimate.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info
from numpy.typing import NDArray

from p0ly_utils.analysis import (
    GroupRegressionResult,
    RegressionResult,
    freq_grid,
    group_regression,
    regression_epochs,
    tfr_epochs,
    tfr_power_epochs,
)

_SFREQ = 100.0
_N_TIMES = 100  # 1 s epochs
_N_CHANNELS = 3
_CH_NAMES = ("Fp1", "Fz", "Pz")
_N_EPOCHS = 40
_SEED = 20261020
_TIME_PROFILE = 1.0 + np.arange(_N_TIMES) / _SFREQ  # catches time-axis flips
_INTERCEPT = np.array([1.0, 2.0, 3.0])[:, None] * _TIME_PROFILE
_RT_SLOPE = np.array([0.5, 1.0, 1.5])[:, None] * _TIME_PROFILE
_CORRECT_SHIFT = np.array([0.25, 0.5, 0.75])[:, None] * _TIME_PROFILE
_TFR_SETTINGS: dict[str, object] = {
    "method": "morlet",
    "fmin": 4.0,
    "fmax": 20.0,
    "step": 4.0,
    "n_cycles": None,
    "time_bandwidth": 4.0,
    "decim": 2,
}


def _info() -> Any:
    return create_info(list(_CH_NAMES), _SFREQ, "eeg")


def _metadata(n_epochs: int = _N_EPOCHS, seed: int = _SEED) -> pd.DataFrame:
    """RT + a two-level categorical predictor, both spread across trials."""
    rng = np.random.default_rng(seed)
    correct = np.arange(n_epochs) % 2 == 0  # balanced False/True
    return pd.DataFrame(
        {
            "RT": track(np.linspace(0.4, 1.4, n_epochs) + rng.normal(0, 0.01, n_epochs)),
            "Correct": np.where(correct, "True", "False"),
        }
    )


def track(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Identity helper keeping array dtypes explicit for mypy."""
    return np.asarray(array, dtype=np.float64)


def _voltage_data(meta: pd.DataFrame, scale: float = 1.0) -> NDArray[np.float64]:
    """Exact ``intercept + RT*slope + is_correct*shift`` voltage response."""
    rt = meta["RT"].to_numpy(dtype=np.float64)
    is_correct = (meta["Correct"] == "True").to_numpy(dtype=np.float64)
    return (
        scale * _INTERCEPT[None]
        + scale * rt[:, None, None] * _RT_SLOPE[None]
        + scale * is_correct[:, None, None] * _CORRECT_SHIFT[None]
    )


def _epochs(
    meta: pd.DataFrame | None = None, scale: float = 1.0
) -> EpochsArray:
    meta = _metadata() if meta is None else meta
    return EpochsArray(
        _voltage_data(meta, scale), _info(), metadata=meta, tmin=0.0, baseline=None
    )


class _StubEpochs:
    """Minimal duck-typed epochs with a mismatched response axis (guard test)."""

    metadata: pd.DataFrame
    ch_names: tuple[str, ...] = _CH_NAMES
    times: NDArray[np.float64]
    _data: NDArray[np.float64]

    def __init__(self, metadata: pd.DataFrame, n_trials: int) -> None:
        self.metadata = metadata
        self.times = np.arange(_N_TIMES) / _SFREQ
        self._data = np.zeros((n_trials, _N_CHANNELS, _N_TIMES))

    def __len__(self) -> int:
        return len(self.metadata)

    def get_data(self) -> NDArray[np.float64]:
        return self._data


class TestVoltageRecovery:
    """§3.2 design semantics on synthetic voltage with known coefficients."""

    def test_recovers_intercept_continuous_and_categorical(self) -> None:
        result = regression_epochs(
            _epochs(), continuous=["RT"], categorical=["Correct"]
        )
        assert isinstance(result, RegressionResult)
        assert set(result.beta_maps) == {"intercept", "RT", "Correct[True]"}
        np.testing.assert_allclose(result.beta_maps["intercept"], _INTERCEPT, atol=1e-9)
        np.testing.assert_allclose(result.beta_maps["RT"], _RT_SLOPE, atol=1e-9)
        np.testing.assert_allclose(
            result.beta_maps["Correct[True]"], _CORRECT_SHIFT, atol=1e-9
        )
        assert result.domain == "voltage"
        assert result.freqs is None
        assert result.ch_names == _CH_NAMES
        np.testing.assert_allclose(result.times, np.arange(_N_TIMES) / _SFREQ)
        for term, beta in result.beta_maps.items():
            assert beta.shape == (_N_CHANNELS, _N_TIMES), term
            assert beta.dtype == np.float64
        assert (result.n_input, result.n_used, result.n_excluded) == (_N_EPOCHS, _N_EPOCHS, 0)
        assert result.excluded_rows.size == 0

    def test_term_order_and_reference_coding_two_categoricals(self) -> None:
        meta = _metadata().assign(Block=["1" if i % 3 else "2" for i in range(_N_EPOCHS)])
        result = regression_epochs(
            _epochs(meta), continuous=["RT"], categorical=["Correct", "Block"]
        )
        # declared order, lexicographic levels, reference level omitted
        assert list(result.beta_maps) == [
            "intercept",
            "RT",
            "Correct[True]",
            "Block[2]",
        ]
        assert result.reference_levels == {"Correct": "False", "Block": "1"}
        # Block[2] is a block-2 vs block-1 difference: absent from the data
        np.testing.assert_allclose(result.beta_maps["Block[2]"], 0.0, atol=1e-9)

    def test_categorical_continuous_only_and_declared_order(self) -> None:
        categorical_only = regression_epochs(
            _epochs(), continuous=[], categorical=["Correct"]
        )
        assert list(categorical_only.beta_maps) == ["intercept", "Correct[True]"]
        continuous_only = regression_epochs(
            _epochs(), continuous=["RT"], categorical=[]
        )
        assert list(continuous_only.beta_maps) == ["intercept", "RT"]

    def test_deterministic_across_calls(self) -> None:
        first = regression_epochs(_epochs(), continuous=["RT"], categorical=["Correct"])
        second = regression_epochs(_epochs(), continuous=["RT"], categorical=["Correct"])
        for term in first.beta_maps:
            np.testing.assert_array_equal(first.beta_maps[term], second.beta_maps[term])
        np.testing.assert_array_equal(first.excluded_rows, second.excluded_rows)


class TestPowerRecovery:
    """4-D power-domain maps through the public TFR path."""

    @pytest.fixture(scope="class")
    def power_result(self) -> RegressionResult:
        return regression_epochs(
            _epochs(),
            continuous=["RT"],
            categorical=["Correct"],
            domain="power",
            tfr_settings=_TFR_SETTINGS,
        )

    def test_maps_are_four_dimensional(self, power_result: RegressionResult) -> None:
        freqs = freq_grid(4.0, 20.0, 4.0)
        for beta in power_result.beta_maps.values():
            assert beta.shape == (_N_CHANNELS, freqs.size, _N_TIMES // 2)
        assert power_result.domain == "power"
        assert power_result.freqs is not None
        np.testing.assert_allclose(power_result.freqs, freqs)

    def test_matches_manual_lstsq_on_tfr_power(self, power_result: RegressionResult) -> None:
        epochs = _epochs()
        tfr = tfr_epochs(
            epochs,
            freq_grid(4.0, 20.0, 4.0),
            None,
            method="morlet",
            time_bandwidth=4.0,
            decim=2,
        )
        power = tfr_power_epochs(tfr)
        np.testing.assert_allclose(power_result.times, tfr.times)
        meta = epochs.metadata
        assert meta is not None
        design = np.column_stack(
            [
                np.ones(len(epochs)),
                meta["RT"].to_numpy(dtype=np.float64),
                (meta["Correct"] == "True").to_numpy(dtype=np.float64),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(
            design, power.reshape(len(epochs), -1), rcond=None
        )
        expected = coefficients.reshape((3, *power.shape[1:]))
        for index, term in enumerate(("intercept", "RT", "Correct[True]")):
            np.testing.assert_allclose(power_result.beta_maps[term], expected[index])


class TestExclusions:
    """Null/non-finite predictors are excluded and counted, never silent."""

    def test_null_and_inf_predictors_are_excluded(self) -> None:
        meta = _metadata()
        meta.loc[3, "RT"] = np.nan
        meta.loc[7, "RT"] = np.inf
        result = regression_epochs(_epochs(meta), continuous=["RT"], categorical=[])
        assert result.n_input == _N_EPOCHS
        assert result.n_used == _N_EPOCHS - 2
        assert result.n_excluded == 2
        assert result.n_input == result.n_used + result.n_excluded
        np.testing.assert_array_equal(result.excluded_rows, [3, 7])
        assert result.excluded_rows.dtype == np.int64
        # betas equal a fit on the retained rows only
        clean = meta.drop(index=[3, 7])
        reference = regression_epochs(
            _epochs(clean), continuous=["RT"], categorical=[]
        )
        for term in ("intercept", "RT"):
            np.testing.assert_allclose(result.beta_maps[term], reference.beta_maps[term])

    def test_null_categorical_row_is_excluded(self) -> None:
        meta = _metadata()
        meta.loc[0:2, "Correct"] = None
        result = regression_epochs(
            _epochs(meta), continuous=[], categorical=["Correct"]
        )
        np.testing.assert_array_equal(result.excluded_rows, [0, 1, 2])
        assert result.n_excluded == 3


class TestDesignGuards:
    """Every invalid design fails loud with a diagnosable message."""

    def test_no_metadata(self) -> None:
        epochs = EpochsArray(
            _voltage_data(_metadata()), _info(), tmin=0.0, baseline=None
        )
        with pytest.raises(ValueError, match="no trial metadata"):
            regression_epochs(epochs, continuous=["RT"], categorical=[])

    def test_no_declared_predictor(self) -> None:
        with pytest.raises(ValueError, match="at least one predictor"):
            regression_epochs(_epochs(), continuous=[], categorical=[])

    def test_duplicate_declaration(self) -> None:
        with pytest.raises(ValueError, match="declared twice"):
            regression_epochs(_epochs(), continuous=["RT", "RT"], categorical=[])

    def test_continuous_and_categorical_overlap(self) -> None:
        with pytest.raises(ValueError, match="both continuous and categorical"):
            regression_epochs(
                _epochs(), continuous=["RT"], categorical=["RT"]
            )

    def test_unknown_column(self) -> None:
        with pytest.raises(ValueError, match="not in the metadata"):
            regression_epochs(_epochs(), continuous=["ReactionTime"], categorical=[])

    def test_term_name_collisions_fail_loud(self) -> None:
        """A predictor must not overwrite another term's beta map."""
        meta = _metadata()
        meta["intercept"] = track(np.linspace(0.0, 1.0, _N_EPOCHS))
        meta["Correct[True]"] = track(np.linspace(1.0, 0.0, _N_EPOCHS))
        with pytest.raises(ValueError, match="term name collision"):
            regression_epochs(_epochs(meta), continuous=["intercept"], categorical=[])
        with pytest.raises(ValueError, match="term name collision"):
            regression_epochs(
                _epochs(meta), continuous=["Correct[True]"], categorical=["Correct"]
            )

    def test_nonnumeric_continuous_column(self) -> None:
        epochs = _epochs()
        epochs.metadata = _metadata().assign(RT=["slow"] * _N_EPOCHS)
        with pytest.raises(ValueError, match="non-numeric"):
            regression_epochs(epochs, continuous=["RT"], categorical=[])

    def test_metadata_length_mismatch(self) -> None:
        epochs = _epochs()
        epochs._metadata = _metadata().iloc[:-1]  # bypass MNE's setter guard
        with pytest.raises(ValueError, match="metadata has 39 rows"):
            regression_epochs(epochs, continuous=["RT"], categorical=[])

    def test_response_length_mismatch(self) -> None:
        stub = _StubEpochs(_metadata(10), n_trials=11)
        with pytest.raises(ValueError, match="response carries 11 trial"):
            regression_epochs(stub, continuous=["RT"], categorical=[])  # type: ignore[arg-type]

    def test_zero_retained_rows(self) -> None:
        meta = _metadata().assign(RT=[np.nan] * _N_EPOCHS)
        with pytest.raises(ValueError, match="usable row"):
            regression_epochs(_epochs(meta), continuous=["RT"], categorical=[])

    def test_one_level_categorical(self) -> None:
        meta = _metadata().assign(Correct="True")
        with pytest.raises(ValueError, match="at least two"):
            regression_epochs(_epochs(meta), continuous=[], categorical=["Correct"])

    def test_rank_deficient_design(self) -> None:
        meta = _metadata().assign(RT=[1.0] * _N_EPOCHS)
        with pytest.raises(ValueError, match="rank-deficient"):
            regression_epochs(_epochs(meta), continuous=["RT"], categorical=[])

    def test_duplicated_predictor_is_rank_deficient(self) -> None:
        meta = _metadata().assign(RT_copy=_metadata()["RT"])
        with pytest.raises(ValueError, match="rank-deficient"):
            regression_epochs(
                _epochs(meta), continuous=["RT", "RT_copy"], categorical=[]
            )

    def test_too_few_rows(self) -> None:
        meta = _metadata(2)
        with pytest.raises(ValueError, match="usable row"):
            regression_epochs(_epochs(meta), continuous=["RT"], categorical=["Correct"])

    def test_nonfinite_response(self) -> None:
        epochs = _epochs()
        epochs._data[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            regression_epochs(epochs, continuous=["RT"], categorical=[])

    def test_invalid_domain(self) -> None:
        with pytest.raises(ValueError, match="unknown signal domain"):
            regression_epochs(
                _epochs(), continuous=["RT"], categorical=[], domain="phase"
            )

    def test_power_domain_requires_tfr_settings(self) -> None:
        with pytest.raises(ValueError, match="needs tfr_settings"):
            regression_epochs(
                _epochs(), continuous=["RT"], categorical=[], domain="power"
            )

    def test_power_domain_requires_frequency_grid(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            regression_epochs(
                _epochs(),
                continuous=["RT"],
                categorical=[],
                domain="power",
                tfr_settings={"fmin": 4.0},
            )


class TestGroupRegression:
    """Equal-subject-weight cohort aggregation (§3.2 / Stage 2)."""

    @staticmethod
    def _result(n_epochs: int, scale: float) -> RegressionResult:
        meta = _metadata(n_epochs, seed=_SEED + n_epochs)
        return regression_epochs(
            _epochs(meta, scale=scale), continuous=["RT"], categorical=["Correct"]
        )

    def test_arithmetic_subject_mean_not_pooled_trials(self) -> None:
        short, long = self._result(20, 1.0), self._result(30, 3.0)
        grouped = group_regression([short, long])
        assert isinstance(grouped, GroupRegressionResult)
        assert grouped.n_subjects == 2
        assert grouped.n_trials_used_total == 50
        assert grouped.ch_names == _CH_NAMES
        assert grouped.reference_levels == {"Correct": "False"}
        assert grouped.domain == "voltage"
        assert grouped.freqs is None
        for term in ("intercept", "RT", "Correct[True]"):
            subject_mean = (short.beta_maps[term] + long.beta_maps[term]) / 2.0
            np.testing.assert_allclose(grouped.beta_maps[term], subject_mean)
            pooled = (
                20 * short.beta_maps[term] + 30 * long.beta_maps[term]
            ) / 50  # trial-weighted alternative
            assert not np.allclose(grouped.beta_maps[term], pooled)

    def test_single_subject_identity(self) -> None:
        only = self._result(_N_EPOCHS, 2.0)
        grouped = group_regression([only])
        assert grouped.n_subjects == 1
        for term, beta in only.beta_maps.items():
            np.testing.assert_array_equal(grouped.beta_maps[term], beta)

    def test_power_domain_grouping_keeps_frequency_axis(self) -> None:
        power = regression_epochs(
            _epochs(),
            continuous=["RT"],
            categorical=["Correct"],
            domain="power",
            tfr_settings=_TFR_SETTINGS,
        )
        grouped = group_regression([power, power])
        assert grouped.freqs is not None
        assert power.freqs is not None
        np.testing.assert_allclose(grouped.freqs, power.freqs)
        assert grouped.domain == "power"

    def test_no_input(self) -> None:
        with pytest.raises(ValueError, match="no results"):
            group_regression([])

    def test_domain_mismatch(self) -> None:
        voltage = self._result(_N_EPOCHS, 1.0)
        power = regression_epochs(
            _epochs(),
            continuous=["RT"],
            categorical=["Correct"],
            domain="power",
            tfr_settings=_TFR_SETTINGS,
        )
        with pytest.raises(ValueError, match="signal domain mismatch"):
            group_regression([voltage, power])

    def test_term_order_mismatch(self) -> None:
        meta = _metadata(20, seed=7)
        reordered = regression_epochs(
            _epochs(meta), continuous=["RT"], categorical=["Correct"]
        )
        shuffled = RegressionResult(
            beta_maps={
                term: reordered.beta_maps[term]
                for term in ("RT", "intercept", "Correct[True]")
            },
            ch_names=reordered.ch_names,
            times=reordered.times,
            freqs=reordered.freqs,
            domain=reordered.domain,
            n_input=reordered.n_input,
            n_used=reordered.n_used,
            n_excluded=reordered.n_excluded,
            excluded_rows=reordered.excluded_rows,
            reference_levels=reordered.reference_levels,
        )
        with pytest.raises(ValueError, match="term order mismatch"):
            group_regression([reordered, shuffled])

    def test_reference_level_mismatch(self) -> None:
        base = self._result(20, 1.0)
        other = replace(base, reference_levels={"Correct": "True"})
        with pytest.raises(ValueError, match="reference-level mismatch"):
            group_regression([base, other])

    def test_channel_mismatch(self) -> None:
        base = self._result(20, 1.0)
        other = replace(base, ch_names=("Fp1", "Fz"))
        with pytest.raises(ValueError, match="channel-set mismatch"):
            group_regression([base, other])

    def test_time_grid_mismatch(self) -> None:
        base = self._result(20, 1.0)
        other = replace(base, times=base.times[:-1])
        with pytest.raises(ValueError, match="time-grid mismatch"):
            group_regression([base, other])

    def test_frequency_grid_mismatch(self) -> None:
        base = self._result(20, 1.0)
        other = replace(base, freqs=np.array([4.0, 8.0]))
        with pytest.raises(ValueError, match="frequency-axis mismatch"):
            group_regression([base, other])

    def test_beta_shape_mismatch(self) -> None:
        base = self._result(20, 1.0)
        truncated = replace(
            base,
            beta_maps={
                term: beta[:, 1:] for term, beta in base.beta_maps.items()
            },
        )
        with pytest.raises(ValueError, match="beta-shape mismatch"):
            group_regression([base, truncated])
