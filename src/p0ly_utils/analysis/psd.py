"""PSD analysis: absolute/relative band power and aperiodic 1/f
parameterization (US-011).

Design (recorded in ``scrum/01_Backlog/US-011_psd-analysis.md``, locked
2026-10-20):

- Welch spectra are computed **per condition**: epochs are filtered by a
  metadata query (``psd_by_condition``), each subset yields an unaveraged
  :class:`mne.time_frequency.EpochsSpectrum` of shape
  ``(n_epochs, n_channels, n_freqs)``, and :func:`average_spectrum` reduces it
  to ``(n_channels, n_freqs)``. Condition-comparable spectra are the pipeline
  contract (mirrors the ERP topology); continuous ``Raw`` spectra
  (:func:`psd_raw`) are library-only — no DAG rule consumes them today.
- Absolute band power is the **mean power spectral density** (V²/Hz) within a
  band; relative power is the band's **mass fraction** — the sum of density
  within the band divided by the sum across the wider ``norm_range`` (bands
  that partition ``norm_range`` sum to 1). Absolute stays on the density
  scale, relative is unitless.
- Aperiodic parameterization wraps :class:`specparam.SpectralModel` in
  **fixed mode** (offset + exponent; no knee — the config-driven fit range is
  typically 1–45 Hz, too narrow for reliable knee fitting). specparam 2.x
  renamed ``min_peak_threshold`` → ``peak_threshold``; the wrapper pins the
  2.x API and leaves ``peak_threshold`` unset by default — explicitly passing
  it (any value) produced degenerate zero-power peaks on clean spectra in
  2.0.0rc7, while the built-in default recovers exact power laws (0 peaks)
  and injected peaks (1 peak) correctly.
- Group spectra are equal-weight arithmetic means across subjects
  (:func:`mne.grand_average` on ``Spectrum`` objects, verified in MNE 1.12.1).
- All functions are pure: no filesystem access, no I/O, no config reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
from mne.time_frequency import EpochsSpectrum, Spectrum
from numpy.typing import NDArray
from specparam import SpectralModel

__all__ = [
    "AperiodicFit",
    "average_spectrum",
    "band_powers",
    "fit_aperiodic",
    "fit_aperiodic_channels",
    "group_psd",
    "psd_by_condition",
    "psd_epochs",
    "psd_raw",
]

DEFAULT_PEAK_WIDTH_LIMITS: tuple[float, float] = (1.0, 12.0)
DEFAULT_MAX_N_PEAKS: int = 8


@dataclass
class AperiodicFit:
    """specparam fit of one power spectrum (fixed mode).

    Attributes
    ----------
    channel_name : str or None
        Channel the spectrum belongs to (None when the caller did not name it).
    freq_range : tuple of float
        The (fmin, fmax) range the model was fit on, after cropping.
    offset : float
        Aperiodic offset ``b`` (log10 power at 1 Hz under a pure power law).
    exponent : float
        Aperiodic exponent ``a`` (PSD ~ f**-a).
    r_squared : float
        Goodness of fit (``gof_rsquared``).
    error_mae : float
        Mean absolute error of the fit (``error_mae``).
    peak_centers : list of float
        Detected oscillatory peak center frequencies (Hz).
    peak_powers : list of float
        Detected peak heights above the aperiodic fit (log10 power).
    peak_widths : list of float
        Detected peak bandwidths (Hz).
    knee_frequency : float or None
        Always None — fixed mode fits no knee. Kept for API stability when a
        knee-capable specparam algorithm is adopted.
    """

    channel_name: str | None
    freq_range: tuple[float, float]
    offset: float
    exponent: float
    r_squared: float
    error_mae: float
    peak_centers: list[float] = field(default_factory=list)
    peak_powers: list[float] = field(default_factory=list)
    peak_widths: list[float] = field(default_factory=list)
    knee_frequency: float | None = None


def _welch_kwargs(
    fmin: float, fmax: float, n_fft: int, n_per_seg: int | None, n_overlap: int
) -> dict[str, Any]:
    return {
        "method": "welch",
        "fmin": fmin,
        "fmax": fmax,
        "n_fft": n_fft,
        "n_per_seg": n_per_seg,
        "n_overlap": n_overlap,
    }


def _epoch_mask(epochs: mne.Epochs, condition_query: str | None) -> np.ndarray:
    """Boolean selection mask over epochs, mirroring erp.average_epochs."""
    if condition_query is None:
        return np.ones(len(epochs), dtype=bool)
    if epochs.metadata is None:
        raise ValueError(
            "condition_query given but epochs carry no metadata — "
            "cannot filter conditions. (Did the epoch rule run inject_metadata?)"
        )
    mask = epochs.metadata.eval(condition_query)
    if isinstance(mask, bool):  # scalar metadata edge case
        raise ValueError(
            f"condition query {condition_query!r} evaluated to a scalar — "
            "metadata must be a DataFrame with one row per epoch"
        )
    return mask.to_numpy(dtype=bool)


def _validate_freqs(psd: NDArray[np.float64], freqs: NDArray[np.float64]) -> None:
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError(f"freqs must be a 1-D non-empty array, got shape {freqs.shape}")
    if not np.all(np.diff(freqs) > 0):
        raise ValueError("freqs must be strictly increasing")
    if psd.shape[-1] != freqs.size:
        raise ValueError(f"psd last axis ({psd.shape[-1]}) does not match freqs ({freqs.size})")


def psd_epochs(
    epochs: mne.Epochs,
    *,
    fmin: float,
    fmax: float,
    n_fft: int,
    n_per_seg: int | None = None,
    n_overlap: int = 0,
) -> EpochsSpectrum:
    """Welch PSD per epoch, unaveraged.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to transform; must be non-empty.
    fmin, fmax : float
        Frequency bounds (Hz).
    n_fft : int
        FFT length; defines the frequency resolution ``sfreq / n_fft``.
    n_per_seg : int or None
        Welch segment length; defaults to ``n_fft``.
    n_overlap : int
        Welch segment overlap.

    Returns
    -------
    EpochsSpectrum
        Shape ``(n_epochs, n_channels, n_freqs)``.
    """
    if len(epochs) == 0:
        raise ValueError("psd_epochs: received 0 epochs — nothing to compute")
    return epochs.compute_psd(**_welch_kwargs(fmin, fmax, n_fft, n_per_seg, n_overlap))


def average_spectrum(spectrum: EpochsSpectrum) -> Spectrum:
    """Average a per-epoch spectrum over epochs.

    Parameters
    ----------
    spectrum : EpochsSpectrum
        ``(n_epochs, n_channels, n_freqs)`` Welch output of :func:`psd_epochs`.

    Returns
    -------
    Spectrum
        ``(n_channels, n_freqs)``.
    """
    if not isinstance(spectrum, EpochsSpectrum):
        raise TypeError(
            "average_spectrum expects an EpochsSpectrum (from psd_epochs); got "
            f"{type(spectrum).__name__}"
        )
    return spectrum.average()


def psd_raw(
    raw: mne.io.Raw,
    *,
    fmin: float,
    fmax: float,
    n_fft: int,
    n_per_seg: int | None = None,
    n_overlap: int = 0,
) -> Spectrum:
    """Welch PSD of continuous data. Library-only (no DAG rule today).

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording; must be non-empty.
    fmin, fmax : float
        Frequency bounds (Hz).
    n_fft : int
        FFT length.
    n_per_seg : int or None
        Welch segment length; defaults to ``n_fft``.
    n_overlap : int
        Welch segment overlap.

    Returns
    -------
    Spectrum
        Shape ``(n_channels, n_freqs)``.
    """
    if raw.n_times == 0:
        raise ValueError("psd_raw: received an empty raw — nothing to compute")
    return raw.compute_psd(**_welch_kwargs(fmin, fmax, n_fft, n_per_seg, n_overlap))


def psd_by_condition(
    epochs: mne.Epochs,
    queries: Mapping[str, str | None],
    *,
    fmin: float,
    fmax: float,
    n_fft: int,
    n_per_seg: int | None = None,
    n_overlap: int = 0,
) -> dict[str, Spectrum]:
    """Per-condition averaged Welch spectra.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs carrying by-trial metadata (the epoch rule's output).
    queries : Mapping[str, str or None]
        Condition name → pandas query string on ``Epochs.metadata``
        (None = unfiltered, the ``all_trials`` convention). Unknown names or
        zero matches fail loudly.
    fmin, fmax, n_fft, n_per_seg, n_overlap
        Welch parameters, forwarded to :func:`psd_epochs`.

    Returns
    -------
    dict of str to Spectrum
        Condition name → ``(n_channels, n_freqs)`` spectrum.
    """
    spectra: dict[str, Spectrum] = {}
    for name, query in queries.items():
        mask = _epoch_mask(epochs, query)
        n_matches = int(mask.sum())
        if n_matches == 0:
            raise ValueError(
                f"condition {name!r} matched 0 of {len(epochs)} epochs "
                f"(query: {query!r}) — cannot compute an empty spectrum"
            )
        subset = epochs[mask]
        spectra[name] = average_spectrum(
            psd_epochs(
                subset,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                n_per_seg=n_per_seg,
                n_overlap=n_overlap,
            )
        )
    return spectra


def band_powers(
    psd: NDArray[np.float64],
    freqs: NDArray[np.float64],
    bands: Mapping[str, tuple[float, float]],
    norm_range: tuple[float, float],
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Absolute and relative band power from a power spectrum.

    Parameters
    ----------
    psd : ndarray, shape (..., n_freqs)
        Power spectral density (V²/Hz); 1-D (single spectrum) or N-D whose
        last axis is frequency (e.g. ``(n_channels, n_freqs)``).
    freqs : ndarray, shape (n_freqs,)
        Strictly increasing frequency bin centers (Hz).
    bands : Mapping[str, tuple of float]
        Band name → ``(fmin, fmax)`` inclusive bounds (Hz).
    norm_range : tuple of float
        ``(fmin, fmax)`` normalization window; relative power divides the
        band mean by this window's mean. Must overlap the frequencies.

    Returns
    -------
    dict of str to dict of str to ndarray
        ``{band: {"absolute": (...,), "relative": (...)}}`` — power means over
        each band, shape like ``psd`` minus the frequency axis.
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    _validate_freqs(psd, freqs)
    norm = (freqs >= norm_range[0]) & (freqs <= norm_range[1])
    if not norm.any():
        raise ValueError(
            f"norm_range {norm_range} overlaps no frequency bins ({freqs[0]}–{freqs[-1]} Hz)"
        )
    norm_mass = psd[..., norm].sum(axis=-1)
    if not np.all(norm_mass > 0):
        raise ValueError(
            "norm_range total power is zero or negative for at least one "
            "spectrum — relative power undefined"
        )
    out: dict[str, dict[str, NDArray[np.float64]]] = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs <= hi)
        if not sel.any():
            raise ValueError(
                f"band {name!r} ({lo}–{hi} Hz) overlaps no frequency bins "
                f"({freqs[0]}–{freqs[-1]} Hz)"
            )
        absolute = psd[..., sel].mean(axis=-1)
        out[name] = {
            "absolute": absolute,
            "relative": psd[..., sel].sum(axis=-1) / norm_mass,
        }
    return out


def _fit_one(
    psd: NDArray[np.float64],
    freqs: NDArray[np.float64],
    freq_range: tuple[float, float] | None,
    peak_width_limits: tuple[float, float],
    max_n_peaks: int,
    peak_threshold: float | None,
) -> tuple[AperiodicFit, tuple[float, float]]:
    psd = np.asarray(psd, dtype=float).ravel()
    freqs = np.asarray(freqs, dtype=float)
    _validate_freqs(psd, freqs)
    if np.any(psd <= 0):
        raise ValueError(
            "spectral parameterization requires strictly positive power "
            "values (specparam fits in log space)"
        )
    if freq_range is not None:
        sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not sel.any():
            raise ValueError(
                f"freq_range {freq_range} overlaps no frequency bins ({freqs[0]}–{freqs[-1]} Hz)"
            )
        freqs = freqs[sel]
        psd = psd[sel]
    model_kwargs: dict[str, Any] = {
        "peak_width_limits": peak_width_limits,
        "max_n_peaks": max_n_peaks,
    }
    if peak_threshold is not None:
        model_kwargs["peak_threshold"] = peak_threshold
    model = SpectralModel(verbose=False, **model_kwargs)
    model.fit(freqs, psd)
    aperiodic = model.results.params.aperiodic.asdict()["aperiodic_fit"]
    peaks = model.results.params.periodic.asdict()["peak_fit"]
    peaks = np.atleast_2d(np.asarray(peaks, dtype=float))
    metrics = {mm.label: float(mm.result) for mm in model.results.metrics.metrics}
    fit = AperiodicFit(
        channel_name=None,
        freq_range=(float(freqs[0]), float(freqs[-1])),
        offset=float(aperiodic[0]),
        exponent=float(aperiodic[-1]),
        r_squared=metrics["gof_rsquared"],
        error_mae=metrics["error_mae"],
        peak_centers=[float(c) for c in peaks[:, 0]],
        peak_powers=[float(p) for p in peaks[:, 1]],
        peak_widths=[float(w) for w in peaks[:, 2]],
    )
    return fit, (float(freqs[0]), float(freqs[-1]))


def fit_aperiodic(
    psd: NDArray[np.float64],
    freqs: NDArray[np.float64],
    *,
    freq_range: tuple[float, float] | None = None,
    peak_width_limits: tuple[float, float] = DEFAULT_PEAK_WIDTH_LIMITS,
    max_n_peaks: int = DEFAULT_MAX_N_PEAKS,
    peak_threshold: float | None = None,
) -> AperiodicFit:
    """Aperiodic (and oscillatory peak) parameterization of one spectrum.

    Parameters
    ----------
    psd : ndarray, shape (n_freqs,)
        Power spectral density of a single channel (V²/Hz); strictly positive.
    freqs : ndarray, shape (n_freqs,)
        Strictly increasing frequency bin centers (Hz).
    freq_range : tuple of float or None
        Fit window (Hz) applied to ``freqs`` before fitting; None = full range.
    peak_width_limits, max_n_peaks : specparam peak-search settings (2.x names).
    peak_threshold : float or None
        Extra peak-detection threshold; None (default) keeps specparam's
        built-in default — passing an explicit value in 2.0.0rc7 triggers
        degenerate zero-power peaks on clean spectra.

    Returns
    -------
    AperiodicFit
        Fixed-mode fit (offset + exponent; ``knee_frequency`` is None).
    """
    fit, _ = _fit_one(psd, freqs, freq_range, peak_width_limits, max_n_peaks, peak_threshold)
    return fit


def fit_aperiodic_channels(
    psd: NDArray[np.float64],
    freqs: NDArray[np.float64],
    *,
    channel_names: Sequence[str] | None = None,
    freq_range: tuple[float, float] | None = None,
    peak_width_limits: tuple[float, float] = DEFAULT_PEAK_WIDTH_LIMITS,
    max_n_peaks: int = DEFAULT_MAX_N_PEAKS,
    peak_threshold: float | None = None,
) -> list[AperiodicFit]:
    """Row-wise :func:`fit_aperiodic` over a channel spectrum.

    Parameters
    ----------
    psd : ndarray, shape (n_channels, n_freqs)
        Power spectral density, one row per channel.
    freqs : ndarray, shape (n_freqs,)
        Strictly increasing frequency bin centers (Hz).
    channel_names : sequence of str or None
        Names attached to the returned fits (length must match rows).
    freq_range, peak_width_limits, max_n_peaks, peak_threshold
        Forwarded to :func:`fit_aperiodic`.

    Returns
    -------
    list of AperiodicFit
        One fit per channel row.
    """
    psd = np.asarray(psd, dtype=float)
    if psd.ndim != 2:
        raise ValueError(
            f"fit_aperiodic_channels expects a 2-D (n_channels, n_freqs) "
            f"array, got shape {psd.shape}"
        )
    if channel_names is not None and len(channel_names) != psd.shape[0]:
        raise ValueError(
            f"channel_names ({len(channel_names)}) does not match psd rows ({psd.shape[0]})"
        )
    names = list(channel_names) if channel_names is not None else [None] * psd.shape[0]
    fits: list[AperiodicFit] = []
    for i, row in enumerate(psd):
        fit, _ = _fit_one(row, freqs, freq_range, peak_width_limits, max_n_peaks, peak_threshold)
        fit.channel_name = names[i]
        fits.append(fit)
    return fits


def group_psd(subject_spectra: Sequence[Spectrum]) -> Spectrum:
    """Equal-weight grand average of per-subject spectra.

    Parameters
    ----------
    subject_spectra : sequence of Spectrum
        Per-subject ``(n_channels, n_freqs)`` spectra with identical frequency
        grids and channel sets.

    Returns
    -------
    Spectrum
        ``(n_channels, n_freqs)`` mean spectrum.
    """
    if not subject_spectra:
        raise ValueError("group_psd: received no spectra — nothing to average")
    first = subject_spectra[0]
    for spec in subject_spectra[1:]:
        if len(spec.freqs) != len(first.freqs) or not np.allclose(spec.freqs, first.freqs):
            raise ValueError(
                "group_psd: subjects have different frequency grids — "
                "compute all spectra with identical fmin/fmax/n_fft settings"
            )
        if spec.ch_names != first.ch_names:
            raise ValueError(
                "group_psd: subjects have different channel sets — "
                f"{spec.ch_names} vs {first.ch_names}"
            )
    return mne.grand_average(list(subject_spectra))
