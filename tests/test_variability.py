"""Tests for p0ly_utils.analysis.variability (US-029).

All data is simulated in memory with exactly known trial values — no fixture
files, no ``@pytest.mark.slow``. Voltage fixtures are literal arrays so every
SD/ratio expectation is hand-computable; power fixtures go through the real
TFR path (``tfr_epochs`` + ``tfr_power_epochs``), and the expected maps are
recomputed from that same single-trial power grid with plain NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info
from numpy.typing import NDArray

from p0ly_utils.analysis import (
    GroupVariabilityResult,
    VariabilityResult,
    group_variability,
    variability_epochs,
)
from p0ly_utils.analysis.tfr import freq_grid, tfr_epochs, tfr_power_epochs

_SFREQ = 100.0
_N_TIMES = 200  # 2 s epochs
_CH_NAMES = ("Cz", "Oz")
_ALPHA_HZ = 10.0
_ALPHA_AMP = 1e-6  # V
_SIGMA = 5e-7  # V
_TFR_SETTINGS: dict[str, Any] = {
    "method": "morlet",
    "fmin": 8.0,
    "fmax": 12.0,
    "step": 2.0,
    "n_cycles": None,
    "time_bandwidth": 4.0,
    "decim": 5,
}
_FREQS = freq_grid(8.0, 12.0, 2.0)  # 3 bins: 8, 10, 12 Hz

# (3 epochs, 2 channels, 3 times) — signed voltages, hand-checkable.
_VOLTAGE = np.array(
    [
        [[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]],
        [[3.0, 4.0, 5.0], [2.0, 1.0, 0.0]],
        [[5.0, 8.0, 7.0], [-4.0, 3.0, 2.0]],
    ]
)


def _voltage_epochs(data: NDArray[np.float64] | None = None) -> EpochsArray:
    """Voltage epochs from the literal fixture (or a caller-supplied array)."""
    array = _VOLTAGE if data is None else data
    info = create_info(list(_CH_NAMES[: array.shape[1]]), _SFREQ, "eeg")
    return EpochsArray(array, info, tmin=0.0, verbose=False)


def _power_epochs(n_epochs: int = 4, seed: int = 20261028) -> EpochsArray:
    """2-channel epochs: Cz = 10 Hz sinusoid + noise, Oz = exact zeros.

    The zero channel is the zero-mean probe: TFR power is 0 everywhere, so
    SD is defined (0) while Fano/CV must be NaN with a false mask.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(_N_TIMES) / _SFREQ
    data = np.zeros((n_epochs, len(_CH_NAMES), _N_TIMES))
    data[:, 0] = rng.normal(0.0, _SIGMA, (n_epochs, _N_TIMES))
    data[:, 0] += _ALPHA_AMP * np.cos(2 * np.pi * _ALPHA_HZ * t)
    return EpochsArray(data, create_info(list(_CH_NAMES), _SFREQ, "eeg"), tmin=0.0, verbose=False)


def _metadata_epochs(n_epochs: int = 3) -> EpochsArray:
    """Voltage epochs carrying a by-trial ``Condition`` column."""
    data = np.repeat(_VOLTAGE, 2, axis=0)
    info = create_info(list(_CH_NAMES), _SFREQ, "eeg")
    epochs = EpochsArray(data, info, tmin=0.0, verbose=False)
    epochs.metadata = pd.DataFrame(
        {"Condition": ["a", "b", "a", "b", "a", "b"][: len(data)]}
    )
    return epochs


def _expected_power_maps(
    epochs: EpochsArray,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Independent NumPy reduction of the real single-trial power grid."""
    grid = freq_grid(
        float(_TFR_SETTINGS["fmin"]),
        float(_TFR_SETTINGS["fmax"]),
        float(_TFR_SETTINGS["step"]),
    )
    tfr = tfr_epochs(epochs, grid, decim=int(_TFR_SETTINGS["decim"]))
    power = tfr_power_epochs(tfr)
    mean = power.mean(axis=0)
    var = power.var(axis=0, ddof=1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        fano = np.where(mean > 0, var / mean, np.nan)
        cv = np.where(mean > 0, sd / mean, np.nan)
    return sd, fano, cv, np.asarray(tfr.freqs)


# --------------------------------------------------------------------------- #
# Voltage domain
# --------------------------------------------------------------------------- #
class TestVoltage:
    def test_sd_matches_numpy_sample_sd(self) -> None:
        result = variability_epochs(_voltage_epochs(), domain="voltage", measures=["sd"])
        assert result.domain == "voltage"
        assert result.n_trials == 3
        assert result.ch_names == _CH_NAMES
        assert result.freqs is None
        assert result.positive_mean is None
        assert list(result.maps) == ["sd"]
        np.testing.assert_allclose(
            result.maps["sd"], np.std(_VOLTAGE, axis=0, ddof=1)
        )
        assert result.maps["sd"].shape == (2, 3)
        np.testing.assert_allclose(result.times, np.arange(3) / _SFREQ)

    def test_sd_is_unbiased_variance(self) -> None:
        """ddof=1: the squared SD equals the hand-computed sample variance."""
        result = variability_epochs(_voltage_epochs(), domain="voltage", measures=["sd"])
        np.testing.assert_allclose(result.maps["sd"] ** 2, _VOLTAGE.var(axis=0, ddof=1))

    def test_ratio_measure_on_voltage_fails_loud(self) -> None:
        for measure in ("fano", "cv"):
            with pytest.raises(ValueError, match="undefined for signed voltage"):
                variability_epochs(
                    _voltage_epochs(), domain="voltage", measures=["sd", measure]
                )

    def test_condition_query_selects_subset(self) -> None:
        epochs = _metadata_epochs()
        result = variability_epochs(
            epochs,
            domain="voltage",
            measures=["sd"],
            condition_query="Condition == 'a'",
        )
        assert result.n_trials == 3
        np.testing.assert_allclose(
            result.maps["sd"], np.std(epochs.get_data()[::2], axis=0, ddof=1)
        )

    def test_input_epochs_not_mutated(self) -> None:
        epochs = _metadata_epochs()
        before = epochs.get_data().copy()
        variability_epochs(
            epochs, domain="voltage", measures=["sd"], condition_query="Condition == 'a'"
        )
        np.testing.assert_array_equal(epochs.get_data(), before)


# --------------------------------------------------------------------------- #
# Power domain
# --------------------------------------------------------------------------- #
class TestPower:
    def test_shapes_and_axes(self) -> None:
        epochs = _power_epochs()
        sd, fano, cv, freqs = _expected_power_maps(epochs)
        result = variability_epochs(
            epochs, domain="power", measures=["sd", "fano", "cv"], tfr_settings=_TFR_SETTINGS
        )
        assert result.domain == "power"
        assert result.n_trials == 4
        assert list(result.maps) == ["sd", "fano", "cv"]
        for name in ("sd", "fano", "cv"):
            assert result.maps[name].shape == (2, _FREQS.size, epochs.times[::5].size)
        assert result.positive_mean is not None
        assert result.freqs is not None
        assert result.positive_mean.shape == result.maps["sd"].shape
        np.testing.assert_allclose(result.freqs, freqs)
        np.testing.assert_allclose(result.freqs, _FREQS)
        # decim reaches the existing TFR path: times = epoch times[::decim]
        np.testing.assert_allclose(result.times, epochs.times[::5])

    def test_measures_match_hand_computed_values(self) -> None:
        epochs = _power_epochs()
        expected_sd, expected_fano, expected_cv, _ = _expected_power_maps(epochs)
        result = variability_epochs(
            epochs, domain="power", measures=["sd", "fano", "cv"], tfr_settings=_TFR_SETTINGS
        )
        np.testing.assert_allclose(result.maps["sd"], expected_sd)
        np.testing.assert_allclose(result.maps["fano"], expected_fano, equal_nan=True)
        np.testing.assert_allclose(result.maps["cv"], expected_cv, equal_nan=True)
        # a real 10 Hz signal carries power, so its ratios are defined
        alpha_row = result.maps["sd"][0, 1, :]
        assert np.all(alpha_row > 0)

    def test_zero_mean_cells_are_nan_with_false_mask(self) -> None:
        result = variability_epochs(
            _power_epochs(), domain="power", measures=["sd", "fano", "cv"], tfr_settings=_TFR_SETTINGS
        )
        assert result.positive_mean is not None
        zero_channel = result.ch_names.index("Oz")
        assert not result.positive_mean[zero_channel].any()
        # SD stays defined (0) at zero mean; ratios stay undefined
        np.testing.assert_allclose(result.maps["sd"][zero_channel], 0.0)
        assert np.isnan(result.maps["fano"][zero_channel]).all()
        assert np.isnan(result.maps["cv"][zero_channel]).all()
        assert result.positive_mean[result.ch_names.index("Cz")].all()

    def test_positive_mean_present_for_sd_only(self) -> None:
        result = variability_epochs(
            _power_epochs(), domain="power", measures=["sd"], tfr_settings=_TFR_SETTINGS
        )
        assert list(result.maps) == ["sd"]
        assert result.positive_mean is not None
        assert result.positive_mean.shape == result.maps["sd"].shape

    def test_multitaper_settings_reach_tfr(self) -> None:
        settings = {**_TFR_SETTINGS, "method": "multitaper", "time_bandwidth": 4.0}
        result = variability_epochs(
            _power_epochs(), domain="power", measures=["sd"], tfr_settings=settings
        )
        assert result.maps["sd"].shape == (2, _FREQS.size, 40)

    def test_missing_tfr_settings_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="needs tfr_settings"):
            variability_epochs(_power_epochs(), domain="power", measures=["sd"])

    @pytest.mark.parametrize(
        ("value", "match"),
        [(-1.0, "negative"), (np.nan, "non-finite"), (np.inf, "non-finite")],
    )
    def test_invalid_power_rejected(
        self, value: float, match: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "p0ly_utils.analysis.variability.tfr_power_epochs",
            lambda tfr: np.full(tfr.data.shape, value),
        )
        with pytest.raises(ValueError, match=match):
            variability_epochs(
                _power_epochs(), domain="power", measures=["sd"], tfr_settings=_TFR_SETTINGS
            )


# --------------------------------------------------------------------------- #
# Selection and validation guards
# --------------------------------------------------------------------------- #
class TestGuards:
    def test_unknown_domain_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="unknown domain"):
            variability_epochs(_voltage_epochs(), domain="phase", measures=["sd"])

    def test_unknown_measure_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="unknown measure"):
            variability_epochs(_voltage_epochs(), domain="voltage", measures=["iqr"])

    def test_empty_measures_fail_loud(self) -> None:
        with pytest.raises(ValueError, match="no measures requested"):
            variability_epochs(_voltage_epochs(), domain="voltage", measures=[])

    def test_single_trial_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=">= 2 trials"):
            variability_epochs(
                _voltage_epochs(_VOLTAGE[:1]), domain="voltage", measures=["sd"]
            )
        with pytest.raises(ValueError, match=">= 2 trials"):
            variability_epochs(
                _power_epochs(n_epochs=1),
                domain="power",
                measures=["sd"],
                tfr_settings=_TFR_SETTINGS,
            )

    def test_zero_condition_matches_fail_loud(self) -> None:
        with pytest.raises(ValueError, match="matched 0 of"):
            variability_epochs(
                _metadata_epochs(),
                domain="voltage",
                measures=["sd"],
                condition_query="Condition == 'zzz'",
            )

    def test_query_without_metadata_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="no metadata"):
            variability_epochs(
                _voltage_epochs(),
                domain="voltage",
                measures=["sd"],
                condition_query="Condition == 'a'",
            )

    def test_empty_epochs_fail_loud(self) -> None:
        epochs = _voltage_epochs()
        with pytest.raises(ValueError, match="0 epochs"):
            variability_epochs(
                epochs[np.array([], dtype=int)], domain="voltage", measures=["sd"]
            )


# --------------------------------------------------------------------------- #
# Cohort aggregation
# --------------------------------------------------------------------------- #
def _result(
    maps: dict[str, NDArray[np.float64]],
    *,
    domain: str = "voltage",
    positive_mean: NDArray[np.bool_] | None = None,
    ch_names: tuple[str, ...] = ("Cz",),
    times: tuple[float, ...] = (0.0, 1.0),
    freqs: NDArray[np.float64] | None = None,
    n_trials: int = 4,
) -> VariabilityResult:
    return VariabilityResult(
        maps=maps,
        ch_names=ch_names,
        times=np.asarray(times, dtype=np.float64),
        freqs=freqs,
        domain=domain,
        n_trials=n_trials,
        positive_mean=positive_mean,
    )


def _power_result(
    sd: list[list[float]],
    positive_mean: list[list[bool]],
    ratio: list[list[float]] | None = None,
    *,
    n_trials: int = 4,
) -> VariabilityResult:
    """Power-domain result with SD, (optional) Fano and a positive-mean mask."""
    maps: dict[str, NDArray[np.float64]] = {"sd": np.asarray(sd, dtype=float)}
    if ratio is not None:
        maps["fano"] = np.asarray(ratio, dtype=float)
    return _result(
        maps,
        domain="power",
        positive_mean=np.asarray(positive_mean, dtype=bool),
        times=(0.0, 0.5),
        freqs=np.asarray([10.0, 20.0]),
        n_trials=n_trials,
    )


class TestGroup:
    def test_equal_weight_mean_ignores_trial_counts(self) -> None:
        first = _result({"sd": np.array([[1.0, 2.0]])}, n_trials=2)
        second = _result({"sd": np.array([[3.0, 4.0]])}, n_trials=100)
        group = group_variability([first, second])
        assert isinstance(group, GroupVariabilityResult)
        np.testing.assert_allclose(group.maps["sd"], [[2.0, 3.0]])
        assert group.maps["sd"].shape == (1, 2)
        np.testing.assert_array_equal(group.n_contributing["sd"], [[2, 2]])
        assert group.n_contributing["sd"].dtype == np.int64
        assert group.n_subjects == 2
        assert group.domain == "voltage"
        assert group.freqs is None
        # a trial-weighted mean (1*2 + 3*100) / 102 would be ~2.96, not 2.0
        assert not np.isclose(group.maps["sd"][0, 0], (1.0 * 2 + 3.0 * 100) / 102)

    def test_single_subject_preserves_values(self) -> None:
        only = _result({"sd": np.array([[1.5, -2.0]])})
        group = group_variability([only])
        np.testing.assert_allclose(group.maps["sd"], only.maps["sd"])
        np.testing.assert_array_equal(group.n_contributing["sd"], [[1, 1]])
        assert group.n_subjects == 1

    def test_mixed_positive_mean_averages_only_defined_subjects(self) -> None:
        undefined = _power_result(
            sd=[[0.0, 4.0]], positive_mean=[[False, True]], ratio=[[np.nan, 2.0]]
        )
        defined = _power_result(
            sd=[[2.0, 6.0]], positive_mean=[[True, False]], ratio=[[5.0, np.nan]], n_trials=12
        )
        group = group_variability([undefined, defined])
        # ratio cells take the single defined subject's value, never 0 or NaN-fill
        np.testing.assert_allclose(group.maps["fano"][0, 0], 5.0)
        np.testing.assert_allclose(group.maps["fano"][0, 1], 2.0)
        np.testing.assert_array_equal(group.n_contributing["fano"], [[1, 1]])
        # SD averages every subject, equal weight, and records the full cohort
        np.testing.assert_allclose(group.maps["sd"], [[1.0, 5.0]])
        np.testing.assert_array_equal(group.n_contributing["sd"], [[2, 2]])
        assert group.n_subjects == 2
        assert group.freqs is not None
        np.testing.assert_allclose(group.freqs, [10.0, 20.0])

    def test_all_undefined_ratio_cells_stay_nan(self) -> None:
        first = _power_result(
            sd=[[1.0, 2.0]], positive_mean=[[False, False]], ratio=[[np.nan, np.nan]]
        )
        second = _power_result(
            sd=[[3.0, 4.0]], positive_mean=[[False, False]], ratio=[[np.nan, np.nan]]
        )
        group = group_variability([first, second])
        assert np.isnan(group.maps["fano"]).all()
        np.testing.assert_array_equal(group.n_contributing["fano"], [[0, 0]])
        np.testing.assert_allclose(group.maps["sd"], [[2.0, 3.0]])
        assert group.n_subjects == 2

    def test_two_trial_counts_and_power_integration(self) -> None:
        """End-to-end: unequal trial counts, zero channel stays undefined."""
        short = variability_epochs(
            _power_epochs(n_epochs=4), domain="power", measures=["sd", "fano", "cv"],
            tfr_settings=_TFR_SETTINGS,
        )
        long = variability_epochs(
            _power_epochs(n_epochs=6, seed=7), domain="power", measures=["sd", "fano", "cv"],
            tfr_settings=_TFR_SETTINGS,
        )
        group = group_variability([short, long])
        assert group.n_subjects == 2
        np.testing.assert_allclose(
            group.maps["sd"], (short.maps["sd"] + long.maps["sd"]) / 2.0
        )
        np.testing.assert_array_equal(
            group.n_contributing["sd"], np.full(short.maps["sd"].shape, 2)
        )
        zero_channel = short.ch_names.index("Oz")
        assert np.isnan(group.maps["cv"][zero_channel]).all()
        np.testing.assert_array_equal(group.n_contributing["cv"][zero_channel], 0)

    def test_empty_input_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="no results"):
            group_variability([])

    def test_domain_mismatch_fails_loud(self) -> None:
        voltage = _result({"sd": np.zeros((1, 2))})
        power = _power_result(sd=[[0.0, 0.0]], positive_mean=[[False, False]])
        power.maps = {"sd": np.zeros((1, 2))}
        with pytest.raises(ValueError, match="domain mismatch"):
            group_variability([voltage, power])

    def test_channel_mismatch_fails_loud(self) -> None:
        first = _result({"sd": np.zeros((1, 2))})
        second = _result({"sd": np.zeros((2, 2))}, ch_names=("Cz", "Pz"))
        with pytest.raises(ValueError, match="channel mismatch"):
            group_variability([first, second])

    def test_measure_mismatch_fails_loud(self) -> None:
        first = _result({"sd": np.zeros((1, 2))})
        second = _result({"sd": np.zeros((1, 2)), "cv": np.zeros((1, 2))})
        with pytest.raises(ValueError, match="measure mismatch"):
            group_variability([first, second])

    def test_shape_mismatch_fails_loud(self) -> None:
        first = _result({"sd": np.zeros((1, 2))})
        second = _result({"sd": np.zeros((1, 3))})
        with pytest.raises(ValueError, match="shape mismatch"):
            group_variability([first, second])

    def test_frequency_grid_mismatch_fails_loud(self) -> None:
        first = _power_result(sd=[[0.0, 0.0]], positive_mean=[[True, True]])
        second = _power_result(sd=[[0.0, 0.0]], positive_mean=[[True, True]])
        second.freqs = np.asarray([9.0, 19.0])
        with pytest.raises(ValueError, match="frequency grids"):
            group_variability([first, second])

    def test_mixed_domain_freqs_fails_loud(self) -> None:
        first = _power_result(sd=[[0.0, 0.0]], positive_mean=[[True, True]])
        second = _power_result(sd=[[0.0, 0.0]], positive_mean=[[True, True]])
        second.freqs = None
        with pytest.raises(ValueError, match="mix"):
            group_variability([first, second])

    def test_time_grid_mismatch_fails_loud(self) -> None:
        first = _result({"sd": np.zeros((1, 2))})
        second = _result({"sd": np.zeros((1, 2))}, times=(0.0, 2.0))
        with pytest.raises(ValueError, match="time grids"):
            group_variability([first, second])

    def test_mask_shape_mismatch_fails_loud(self) -> None:
        first = _power_result(
            sd=[[0.0, 0.0]], positive_mean=[[True, True]], ratio=[[1.0, 1.0]]
        )
        second = _power_result(
            sd=[[0.0, 0.0]], positive_mean=[[True, True]], ratio=[[1.0, 1.0]]
        )
        second.positive_mean = np.ones((1, 3), dtype=bool)
        with pytest.raises(ValueError, match="mask shape mismatch"):
            group_variability([first, second])

    def test_missing_mask_for_ratio_fails_loud(self) -> None:
        first = _power_result(
            sd=[[0.0, 0.0]], positive_mean=[[True, True]], ratio=[[1.0, 1.0]]
        )
        second = _power_result(
            sd=[[0.0, 0.0]], positive_mean=[[True, True]], ratio=[[1.0, 1.0]]
        )
        second.positive_mean = None
        with pytest.raises(ValueError, match="positive_mean"):
            group_variability([first, second])

    def test_power_result_without_mask_fails_loud(self) -> None:
        only = _power_result(sd=[[0.0]], positive_mean=[[True]], ratio=[[1.0]])
        only.positive_mean = None
        with pytest.raises(ValueError, match="positive_mean"):
            group_variability([only])
