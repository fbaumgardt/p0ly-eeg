"""Tests for p0ly_utils.analysis.tfr (US-012).

All data is simulated in memory per the AC "unit tests with simulated
oscillatory epochs" — no fixture files, no ``@pytest.mark.slow``.

Synthetic layout: 8 EEG channels at 100 Hz, 200-sample (2 s) epochs of white
noise (sigma 5e-7 V) with optional 10 Hz sinusoids (1e-6 V amplitude):
phase-locked (fixed cosine phase), jittered (per-epoch uniform phase), and
absent. ITPC expectations: white-noise-only epochs give ``E[ITPC] ≈
1/√n_epochs`` (small-n bias, uncorrected); phase-locked epochs give ITPC ≈ 1
and a power peak in the 10 Hz bin; jittered phases give an intermediate ITPC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info
from mne.time_frequency import AverageTFR, EpochsTFR

from p0ly_utils.analysis import (
    average_power,
    freq_grid,
    group_tfr,
    itpc,
    tfr_difference,
    tfr_epochs,
    tfr_power_epochs,
)

_SFREQ = 100.0
_N_TIMES = 200  # 2 s epochs
_N_CHANNELS = 8
_N_EPOCHS = 30
_CH_NAMES = ("Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "C3", "C4")
_FMIN, _FMAX = 4.0, 30.0
_FREQS = freq_grid(_FMIN, _FMAX, 2.0)  # 14 bins, 10 Hz included
_ALPHA_HZ = 10.0
_ALPHA_AMP = 1e-6  # V
_SIGMA = 5e-7  # V
_SEED = 20261020


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


@pytest.fixture(scope="module")
def info() -> object:
    """Info for 8 EEG channels at 100 Hz (no montage — TFR needs no positions)."""
    return create_info(list(_CH_NAMES), _SFREQ, "eeg")


def _epoch_data(
    condition: str, n_epochs: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_epochs, n_channels, n_times) white noise + optional 10 Hz sinusoids.

    Locked-phase epochs use a cosine fixed at zero phase (drives ITPC → 1 at
    10 Hz); jittered-phase epochs use a per-epoch uniform phase flip.
    """
    phase_options: dict[str, float | np.ndarray | None] = {
        "locked": 0.0,
        "jittered": rng.uniform(0.0, 2.0 * np.pi, n_epochs)[:, None, None],
        "none": None,
    }
    phase = phase_options[condition]
    t = np.arange(_N_TIMES) / _SFREQ
    data = rng.normal(0.0, _SIGMA, (n_epochs, _N_CHANNELS, _N_TIMES))
    if phase is not None:
        data += _ALPHA_AMP * np.cos(2 * np.pi * _ALPHA_HZ * t + phase)
    return data


def _synthetic_epochs(info: object, condition: str) -> EpochsArray:
    """Metadata-carrying epochs of one oscillatory condition."""
    rng = _rng()
    data = _epoch_data(condition, _N_EPOCHS, rng)
    meta = pd.DataFrame({"Trial": np.arange(_N_EPOCHS)})
    return EpochsArray(data, info, metadata=meta, tmin=0.0, baseline=None)


@pytest.fixture(scope="module")
def noise_tfr(info: object) -> EpochsTFR:
    """Complex Morlet TFR of pure white-noise epochs."""
    return tfr_epochs(_synthetic_epochs(info, "none"), _FREQS)


@pytest.fixture(scope="module")
def locked_tfr(info: object) -> EpochsTFR:
    """Complex Morlet TFR of phase-locked 10 Hz epochs."""
    return tfr_epochs(_synthetic_epochs(info, "locked"), _FREQS)


@pytest.fixture(scope="module")
def jittered_tfr(info: object) -> EpochsTFR:
    """Complex Morlet TFR of jittered-phase 10 Hz epochs."""
    return tfr_epochs(_synthetic_epochs(info, "jittered"), _FREQS)


def _f10_index(tfr: EpochsTFR) -> int:
    """Index of the 10 Hz bin in ``tfr.freqs``."""
    return int(np.argmin(np.abs(np.asarray(tfr.freqs) - _ALPHA_HZ)))


class TestFreqGrid:
    def test_inclusive_endpoints(self) -> None:
        freqs = freq_grid(4.0, 45.0, 1.0)
        assert freqs[0] == 4.0 and freqs[-1] == 45.0
        assert np.allclose(np.diff(freqs), 1.0)

    def test_last_bin_stays_under_fmax(self) -> None:
        # 4 + 2.5*12 = 34 → four overshoots: grid must end at 34, not 36.5
        freqs = freq_grid(4.0, 34.4, 2.5)
        assert freqs[-1] <= 34.4
        assert freqs[-1] == pytest.approx(34.0, abs=1e-9)

    def test_non_positive_step_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="step"):
            freq_grid(4.0, 30.0, 0.0)

    def test_empty_range_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="fmax"):
            freq_grid(30.0, 4.0, 1.0)


class TestTfrEpochs:
    def test_shape_per_epoch_morlet(self, noise_tfr: EpochsTFR) -> None:
        assert noise_tfr.data.shape == (
            _N_EPOCHS,
            _N_CHANNELS,
            len(_FREQS),
            _N_TIMES,
        )
        assert np.iscomplexobj(noise_tfr.data)

    def test_containers_and_grids(self, noise_tfr: EpochsTFR) -> None:
        assert isinstance(noise_tfr, EpochsTFR)
        np.testing.assert_allclose(noise_tfr.freqs, _FREQS)
        assert noise_tfr.times.size == _N_TIMES
        assert noise_tfr.ch_names == list(_CH_NAMES)

    def test_adaptive_n_cycles_matches_explicit(self, info: object) -> None:
        """n_cycles=None must equal the explicit freqs/2 transform."""
        epochs = _synthetic_epochs(info, "none")
        adaptive = tfr_epochs(epochs, _FREQS, n_cycles=None)
        explicit = tfr_epochs(epochs, _FREQS, n_cycles=_FREQS / 2.0)
        np.testing.assert_allclose(adaptive.data, explicit.data)

    def test_multitaper_method(self, info: object) -> None:
        tfr = tfr_epochs(
            _synthetic_epochs(info, "locked"), _FREQS, method="multitaper"
        )
        assert tfr.data.shape == (
            _N_EPOCHS,
            _N_CHANNELS,
            tfr.weights.shape[0],
            len(_FREQS),
            _N_TIMES,
        )
        assert tfr.weights.shape == (tfr.data.shape[2], len(_FREQS))

    def test_empty_epochs_fail_loud(self, info: object) -> None:
        instances = _synthetic_epochs(info, "none")
        empty = instances[0:0]  # MNE cannot construct a 0-epoch container directly
        with pytest.raises(ValueError, match="0 epochs"):
            tfr_epochs(empty, _FREQS)

    def test_unknown_method_fails_loud(self, info: object) -> None:
        epochs = _synthetic_epochs(info, "none")
        with pytest.raises(ValueError, match="unknown method"):
            tfr_epochs(epochs, _FREQS, method="stft")

    def test_condition_query_filters(self, info: object) -> None:
        rng = _rng()
        data = np.concatenate(
            [
                _epoch_data("locked", 10, rng),
                _epoch_data("none", 10, rng),
            ]
        )
        meta = pd.DataFrame(
            {"Trial": np.arange(20), "AlphaRich": [True] * 10 + [False] * 10}
        )
        epochs = EpochsArray(data, info, metadata=meta, tmin=0.0, baseline=None)
        full = tfr_epochs(epochs, _FREQS, condition_query=None)
        subset = tfr_epochs(epochs, _FREQS, condition_query="AlphaRich")
        locked_full = tfr_epochs(epochs[:10], _FREQS)
        assert full.data.shape[0] == 20 == subset.data.shape[0] * 2
        np.testing.assert_allclose(subset.data, locked_full.data)

    def test_zero_matches_fail_loud(self, info: object) -> None:
        epochs = _synthetic_epochs(info, "none")
        with pytest.raises(ValueError, match="matched 0"):
            tfr_epochs(epochs, _FREQS, condition_query="Trial > 999")

    def test_query_without_metadata_fail_loud(self, info: object) -> None:
        rng = _rng()
        epochs = EpochsArray(
            rng.normal(0.0, _SIGMA, (5, _N_CHANNELS, _N_TIMES)), info
        )
        with pytest.raises(ValueError, match="metadata"):
            tfr_epochs(epochs, _FREQS, condition_query="Trial > 0")


class TestPower:
    def test_shape_reduces_taper_axis(self, noise_tfr: EpochsTFR) -> None:
        power = tfr_power_epochs(noise_tfr)
        assert power.shape == (_N_EPOCHS, _N_CHANNELS, len(_FREQS), _N_TIMES)
        assert power.dtype == np.float64 and np.all(power >= 0)

    def test_multitaper_power_shape(self, info: object) -> None:
        tfr = tfr_epochs(
            _synthetic_epochs(info, "locked"), _FREQS, method="multitaper"
        )
        power = tfr_power_epochs(tfr)
        assert power.shape == (_N_EPOCHS, _N_CHANNELS, len(_FREQS), _N_TIMES)

    def test_epoch_mean_is_average_power(self, locked_tfr: EpochsTFR) -> None:
        avg = average_power(locked_tfr)
        assert isinstance(avg, AverageTFR)
        manual = tfr_power_epochs(locked_tfr).mean(axis=0)
        assert avg.data.shape == (_N_CHANNELS, len(_FREQS), _N_TIMES)
        np.testing.assert_allclose(avg.data, manual)
        assert avg.nave == _N_EPOCHS


class TestITPC:
    def test_white_noise_small_n_bias(self, noise_tfr: EpochsTFR) -> None:
        """Pure noise: E[ITPC] ≈ 1/√n (uncorrected small-n bias)."""
        itc = itpc(noise_tfr)
        assert itc.data.shape == (_N_CHANNELS, len(_FREQS), _N_TIMES)
        assert np.all(itc.data >= 0.0) and np.all(itc.data <= 1.0)
        expected = 1.0 / np.sqrt(_N_EPOCHS)
        # running mean over 2240 (ch, freq, time) points concentrates tightly
        assert itc.data.mean() == pytest.approx(expected, rel=0.3)
        assert itc.data.max() <= 1.0

    def test_phase_locked_hits_one_at_10hz(self, locked_tfr: EpochsTFR) -> None:
        itc = itpc(locked_tfr)
        f10 = _f10_index(locked_tfr)
        assert itc.data[:, f10].mean() > 0.85  # locked: ITPC ≈ 1

    def test_noise_stays_low_adjacent_to_peak(
        self, noise_tfr: EpochsTFR
    ) -> None:
        itc = itpc(noise_tfr)
        f10 = _f10_index(noise_tfr)
        assert itc.data[:, f10].mean() < 2.0 / np.sqrt(_N_EPOCHS)

    def test_power_peaks_at_10hz(self, locked_tfr: EpochsTFR) -> None:
        power = average_power(locked_tfr)
        f10 = _f10_index(locked_tfr)
        strong = power.data[:, :, 100].argmax(axis=1)  # mid-epoch time slice
        assert np.all(strong == f10)

    def test_jittered_is_intermediate(
        self, locked_tfr: EpochsTFR, jittered_tfr: EpochsTFR, noise_tfr: EpochsTFR
    ) -> None:
        f10_locked = _f10_index(locked_tfr)
        locked = itpc(locked_tfr).data[:, f10_locked].mean()
        jittered = itpc(jittered_tfr).data[:, f10_locked].mean()
        noise = itpc(noise_tfr).data[:, f10_locked].mean()
        assert locked > jittered > noise

    def test_multitaper_itpc_detects_locking(self, info: object) -> None:
        tfr = tfr_epochs(
            _synthetic_epochs(info, "locked"), _FREQS, method="multitaper"
        )
        f10 = _f10_index(tfr)
        assert itpc(tfr).data[:, f10].mean() > 0.6

    def test_multitaper_itpc_bounded(self, info: object) -> None:
        tfr = tfr_epochs(
            _synthetic_epochs(info, "jittered"), _FREQS, method="multitaper"
        )
        itc = itpc(tfr).data
        assert np.all(itc >= 0.0) and np.all(itc <= 1.0)


class TestGroupTfr:
    def test_grand_average_is_equal_weight_mean(self, info: object) -> None:
        subject_a = average_power(tfr_epochs(_synthetic_epochs(info, "locked"), _FREQS))
        subject_b = average_power(tfr_epochs(_synthetic_epochs(info, "none"), _FREQS))
        group = group_tfr([subject_a, subject_b])
        manual = (subject_a.data + subject_b.data) / 2.0
        np.testing.assert_allclose(group.data, manual)

    def test_itpc_also_reduces_through_group_tfr(self, info: object) -> None:
        subject = itpc(tfr_epochs(_synthetic_epochs(info, "locked"), _FREQS))
        cast = tfr_difference(subject, subject)  # zero TFR with same grids
        group = group_tfr([subject, cast])
        np.testing.assert_allclose(group.data, (subject.data + cast.data) / 2.0)

    def test_empty_input_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="no TFRs"):
            group_tfr([])

    def test_freq_grid_mismatch_fails_loud(self, info: object) -> None:
        narrow = average_power(
            tfr_epochs(_synthetic_epochs(info, "none"), freq_grid(4.0, 20.0, 2.0))
        )
        wide = average_power(tfr_epochs(_synthetic_epochs(info, "none"), _FREQS))
        with pytest.raises(ValueError, match="frequency grids"):
            group_tfr([narrow, wide])

    def test_channel_mismatch_fails_loud(self, info: object) -> None:
        subset_info = create_info(list(_CH_NAMES[:4]), _SFREQ, "eeg")
        rng = _rng()
        epochs = EpochsArray(
            rng.normal(0.0, _SIGMA, (_N_EPOCHS, 4, _N_TIMES)), subset_info
        )
        narrow = average_power(tfr_epochs(epochs, _FREQS))
        wide = average_power(tfr_epochs(_synthetic_epochs(info, "none"), _FREQS))
        with pytest.raises(ValueError, match="channel"):
            group_tfr([narrow, wide])


class TestTfrDifference:
    def test_point_wise_subtraction(self, info: object) -> None:
        a = average_power(tfr_epochs(_synthetic_epochs(info, "locked"), _FREQS))
        b = average_power(tfr_epochs(_synthetic_epochs(info, "none"), _FREQS))
        diff = tfr_difference(a, b)
        np.testing.assert_allclose(diff.data, a.data - b.data)
        assert diff.data.shape == a.data.shape

    def test_mismatch_fails_loud(self, info: object) -> None:
        narrow = average_power(
            tfr_epochs(_synthetic_epochs(info, "none"), freq_grid(4.0, 20.0, 2.0))
        )
        wide = average_power(tfr_epochs(_synthetic_epochs(info, "locked"), _FREQS))
        with pytest.raises(ValueError, match="frequency grid"):
            tfr_difference(wide, narrow)
