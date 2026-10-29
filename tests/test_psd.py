"""Tests for p0ly_eeg.analysis.psd (US-011).

All data is simulated in memory per the AC "unit tests with synthetic
sinusoidal data" — no fixture files, no ``@pytest.mark.slow``.

Synthetic layout: 8 EEG channels at 100 Hz, 200-sample (2 s) epochs of white
noise (sigma 5e-7 V) with an optional 10 Hz sinusoid (1e-6 V amplitude)
injected into condition 'alpha_rich'. Welch n_fft=100 gives 1 Hz bins over
fmin=1..fmax=45 (45 bins). specparam recovery is checked against an exact
power law ``10 / f`` (offset 1.0, exponent 1.0 in log10 space) plus a Gaussian
peak at 10 Hz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info
from mne.io import RawArray
from mne.time_frequency import Spectrum

from p0ly_eeg.analysis import (
    AperiodicFit,
    average_spectrum,
    band_powers,
    fit_aperiodic,
    fit_aperiodic_channels,
    group_psd,
    psd_by_condition,
    psd_epochs,
    psd_raw,
)

_SFREQ = 100.0
_N_TIMES = 200  # 2 s epochs
_N_CHANNELS = 8
_N_EPOCHS = 20
_CH_NAMES = ("Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "C3", "C4")
_FMIN, _FMAX = 1.0, 45.0
_N_FFT = 100  # 1 Hz bins
_ALPHA_HZ = 10.0
_ALPHA_AMP = 1e-6  # V
_SIGMA = 5e-7  # V
_BUMP_AMP = 50.0
_SEED = 20261020
_FREQS = np.arange(_FMIN, _FMAX + 1.0)  # 45 bins


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


@pytest.fixture(scope="module")
def info() -> object:
    """Info for 8 EEG channels at 100 Hz (no montage — PSD needs no positions)."""
    return create_info(list(_CH_NAMES), _SFREQ, "eeg")


def _synthetic_epoch_data(n_epochs: int, with_alpha: bool, rng: np.random.Generator) -> np.ndarray:
    """(n_epochs, n_channels, n_times) white noise + optional 10 Hz sinusoid."""
    t = np.arange(_N_TIMES) / _SFREQ
    data = rng.normal(0.0, _SIGMA, (n_epochs, _N_CHANNELS, _N_TIMES))
    if with_alpha:
        data += _ALPHA_AMP * np.sin(2 * np.pi * _ALPHA_HZ * t)
    return data


def _alpha_epochs(info: object) -> EpochsArray:
    """20 epochs with metadata: first 10 with an alpha peak, last 10 without."""
    rng = _rng()
    data = np.concatenate(
        [
            _synthetic_epoch_data(_N_EPOCHS // 2, True, rng),
            _synthetic_epoch_data(_N_EPOCHS // 2, False, rng),
        ]
    )
    meta = pd.DataFrame(
        {
            "Trial": np.arange(_N_EPOCHS),
            "Alpha": np.array([True] * (_N_EPOCHS // 2) + [False] * (_N_EPOCHS // 2)),
        }
    )
    return EpochsArray(data, info, metadata=meta, verbose="ERROR")


class TestPsdEpochs:
    def test_shape_per_epoch(self, info: object) -> None:
        spec = psd_epochs(_alpha_epochs(info), fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)
        assert spec.shape == (_N_EPOCHS, _N_CHANNELS, _FREQS.size)
        assert np.allclose(spec.freqs, _FREQS)

    def test_average_reduces_to_channels(self, info: object) -> None:
        spec = average_spectrum(
            psd_epochs(_alpha_epochs(info), fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)
        )
        assert isinstance(spec, Spectrum)
        assert spec.shape == (_N_CHANNELS, _FREQS.size)

    def test_empty_epochs_fail_loud(self, info: object) -> None:
        epochs = _alpha_epochs(info)
        empty = epochs["Alpha == 'nope'"] if epochs.metadata is not None else epochs
        with pytest.raises(ValueError, match="0 epochs"):
            psd_epochs(empty, fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)


class TestPsdRaw:
    def test_shape_continuous(self, info: object) -> None:
        raw = RawArray(_synthetic_epoch_data(1, True, _rng())[0], info, verbose="ERROR")
        spec = psd_raw(raw, fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)
        assert isinstance(spec, Spectrum)
        assert spec.shape == (_N_CHANNELS, _FREQS.size)

    def test_alpha_peak_dominates_band(self, info: object) -> None:
        raw = RawArray(_synthetic_epoch_data(1, True, _rng())[0], info, verbose="ERROR")
        spec = psd_raw(raw, fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)
        powers = band_powers(spec.data, spec.freqs, {"alpha": (8.0, 12.0)}, (_FMIN, _FMAX))
        alpha_abs = powers["alpha"]["absolute"]
        baseline = band_powers(
            spec.data,
            spec.freqs,
            {"beta": (20.0, 30.0)},
            (_FMIN, _FMAX),
        )["beta"]["absolute"]
        assert np.all(alpha_abs > baseline)


class TestPsdByCondition:
    def test_condition_spectra_separate(self, info: object) -> None:
        epochs = _alpha_epochs(info)
        spectra = psd_by_condition(
            epochs,
            {"alpha_rich": "Alpha == True", "alpha_poor": "Alpha == False"},
            fmin=_FMIN,
            fmax=_FMAX,
            n_fft=_N_FFT,
        )
        assert set(spectra) == {"alpha_rich", "alpha_poor"}
        for spec in spectra.values():
            assert spec.shape == (_N_CHANNELS, _FREQS.size)
        powers = {
            name: band_powers(spec.data, spec.freqs, {"alpha": (8.0, 12.0)}, (_FMIN, _FMAX))[
                "alpha"
            ]["relative"]
            for name, spec in spectra.items()
        }
        assert np.all(powers["alpha_rich"] > powers["alpha_poor"])

    def test_all_trials_unfiltered(self, info: object) -> None:
        epochs = _alpha_epochs(info)
        spectra = psd_by_condition(
            epochs, {"all_trials": None}, fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT
        )
        assert spectra["all_trials"].shape == (_N_CHANNELS, _FREQS.size)

    def test_zero_matches_fail_loud(self, info: object) -> None:
        epochs = _alpha_epochs(info)
        with pytest.raises(ValueError, match="matched 0"):
            psd_by_condition(
                epochs,
                {"ghost": "Alpha == 'nope'"},
                fmin=_FMIN,
                fmax=_FMAX,
                n_fft=_N_FFT,
            )

    def test_query_without_metadata_fail_loud(self, info: object) -> None:
        rng = _rng()
        bare = EpochsArray(_synthetic_epoch_data(4, False, rng), info, verbose="ERROR")
        with pytest.raises(ValueError, match="no metadata"):
            psd_by_condition(bare, {"x": "Alpha == True"}, fmin=_FMIN, fmax=_FMAX, n_fft=_N_FFT)


class TestBandPowers:
    def test_flat_spectrum_math(self) -> None:
        psd = np.ones((2, _FREQS.size))
        out = band_powers(
            psd,
            _FREQS,
            {"alpha": (8.0, 12.0), "gamma": (30.0, 40.0)},
            (_FMIN, _FMAX),
        )
        # absolute: mean density == 1.0 everywhere; relative: mass fraction of
        # the 45-bin norm window (alpha = 5 bins, gamma = 30–40 = 11 bins)
        assert np.allclose(out["alpha"]["absolute"], 1.0)
        assert np.allclose(out["gamma"]["absolute"], 1.0)
        assert np.allclose(out["alpha"]["relative"], 5.0 / 45.0)
        assert np.allclose(out["gamma"]["relative"], 11.0 / 45.0)

    def test_relative_scales_with_band_mass(self) -> None:
        psd = np.ones((1, _FREQS.size))
        sel = (_FREQS >= 8.0) & (_FREQS <= 12.0)
        psd[0, sel] = 3.0  # 5 bins of 3.0 inside a 45-bin window
        out = band_powers(psd, _FREQS, {"alpha": (8.0, 12.0)}, (_FMIN, _FMAX))
        assert np.allclose(out["alpha"]["absolute"], 3.0)
        assert np.allclose(out["alpha"]["relative"], 15.0 / 55.0)  # mass fraction

    def test_empty_norm_range_fail_loud(self) -> None:
        psd = np.ones((1, _FREQS.size))
        with pytest.raises(ValueError, match="norm_range"):
            band_powers(psd, _FREQS, {"alpha": (8.0, 12.0)}, (100.0, 110.0))

    def test_empty_band_fail_loud(self) -> None:
        psd = np.ones((1, _FREQS.size))
        with pytest.raises(ValueError, match="overlaps no frequency bins"):
            band_powers(psd, _FREQS, {"sub": (0.2, 0.4)}, (_FMIN, _FMAX))

    def test_freq_mismatch_fail_loud(self) -> None:
        psd = np.ones((1, _FREQS.size + 1))
        with pytest.raises(ValueError, match="does not match freqs"):
            band_powers(psd, _FREQS, {"alpha": (8.0, 12.0)}, (_FMIN, _FMAX))

    def test_non_monotonic_freqs_fail_loud(self) -> None:
        psd = np.ones((1, 4))
        with pytest.raises(ValueError, match="strictly increasing"):
            band_powers(psd, np.array([5.0, 3.0, 7.0, 9.0]), {"a": (5.0, 7.0)}, (5.0, 9.0))


class TestFitAperiodic:
    def _power_law(self) -> tuple[np.ndarray, np.ndarray]:
        psd = 10.0 / _FREQS + 1e-20
        return psd, _FREQS

    def test_recovers_exact_power_law(self) -> None:
        psd, freqs = self._power_law()
        fit = fit_aperiodic(psd, freqs)
        assert 0.85 <= fit.exponent <= 1.15
        assert abs(fit.offset - 1.0) < 0.25
        assert fit.r_squared > 0.99
        assert fit.peak_centers == []

    def test_recovers_injected_peak(self) -> None:
        psd, freqs = self._power_law()
        psd = psd + _BUMP_AMP * np.exp(-0.5 * ((_FREQS - _ALPHA_HZ) / 1.5) ** 2)
        fit = fit_aperiodic(psd, freqs)
        assert 0.85 <= fit.exponent <= 1.15  # 1/f background intact
        assert len(fit.peak_centers) == 1
        assert 8.0 <= fit.peak_centers[0] <= 12.0
        assert fit.peak_powers[0] > 0.0

    def test_freq_range_crops(self) -> None:
        psd, freqs = self._power_law()
        fit = fit_aperiodic(psd, freqs, freq_range=(5.0, 40.0))
        assert fit.freq_range == (5.0, 40.0)

    def test_channels_loop_and_names(self) -> None:
        psd, freqs = self._power_law()
        psd2 = np.vstack([psd, psd * 2.0])  # second channel: same slope, offset +log10(2)
        fits = fit_aperiodic_channels(psd2, freqs, channel_names=["Fz", "Cz"])
        assert len(fits) == 2
        assert [f.channel_name for f in fits] == ["Fz", "Cz"]
        assert all(isinstance(f, AperiodicFit) for f in fits)
        assert np.allclose(fits[1].offset - fits[0].offset, np.log10(2.0), atol=0.1)

    def test_negative_power_fail_loud(self) -> None:
        psd, freqs = self._power_law()
        psd = psd.copy()
        psd[3] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            fit_aperiodic(psd, freqs)

    def test_freq_length_mismatch_fail_loud(self) -> None:
        psd, freqs = self._power_law()
        with pytest.raises(ValueError, match="does not match freqs"):
            fit_aperiodic(np.append(psd, 1.0), freqs)

    def test_channels_dim_mismatch_fail_loud(self) -> None:
        psd, freqs = self._power_law()
        with pytest.raises(ValueError, match="does not match psd rows"):
            fit_aperiodic_channels(np.vstack([psd, psd]), freqs, channel_names=["Fz"])


class TestGroupPsd:
    def test_mean_of_spectra(self, info: object) -> None:
        rng = _rng()
        spec_a = average_spectrum(
            psd_epochs(
                EpochsArray(_synthetic_epoch_data(4, False, rng), info, verbose="ERROR"),
                fmin=_FMIN,
                fmax=_FMAX,
                n_fft=_N_FFT,
            )
        )
        spec_b = average_spectrum(
            psd_epochs(
                EpochsArray(_synthetic_epoch_data(4, True, rng), info, verbose="ERROR"),
                fmin=_FMIN,
                fmax=_FMAX,
                n_fft=_N_FFT,
            )
        )
        grouped = group_psd([spec_a, spec_b])
        assert grouped.shape == (_N_CHANNELS, _FREQS.size)
        assert np.allclose(grouped.data, (spec_a.data + spec_b.data) / 2.0)

    def test_single_spectrum_passthrough(self, info: object) -> None:
        rng = _rng()
        spec = average_spectrum(
            psd_epochs(
                EpochsArray(_synthetic_epoch_data(2, False, rng), info, verbose="ERROR"),
                fmin=_FMIN,
                fmax=_FMAX,
                n_fft=_N_FFT,
            )
        )
        grouped = group_psd([spec])
        assert np.allclose(grouped.data, spec.data)

    def test_freq_grid_mismatch_fail_loud(self, info: object) -> None:
        rng = _rng()
        spec_a = average_spectrum(
            psd_epochs(
                EpochsArray(_synthetic_epoch_data(2, False, rng), info, verbose="ERROR"),
                fmin=_FMIN,
                fmax=_FMAX,
                n_fft=_N_FFT,
            )
        )
        spec_b = average_spectrum(
            psd_epochs(
                EpochsArray(_synthetic_epoch_data(2, False, rng), info, verbose="ERROR"),
                fmin=_FMIN,
                fmax=40.0,  # different grid
                n_fft=_N_FFT,
            )
        )
        with pytest.raises(ValueError, match="different frequency grids"):
            group_psd([spec_a, spec_b])

    def test_empty_input_fail_loud(self) -> None:
        with pytest.raises(ValueError, match="no spectra"):
            group_psd([])
