"""Cross-frequency coupling: PAC and n:m phase-phase coupling (US-027).

Delivers within-channel cross-frequency maps with axes ``(n_channels,
n_phase_freqs, n_amp_freqs)`` per subject/condition, reproducible
within-subject surrogate nulls, equal-weight descriptive cohort means, and a
subject-as-unit sign-flip contrast with full-map max-statistic correction.

Stage 0 dependency probe — tensorpac 0.6.5 (verified live 2026-10-28)
---------------------------------------------------------------------
- ``Pac(idpac=(pac_method_id, surrogate_id, normalisation_id))``. The
  Gaussian-copula PAC is ``idpac[0] == 6`` (``gauss_cop_pac``); the
  mean-vector-length PAC is ``idpac[0] == 1``. Surrogate method
  ``idpac[1] == 3`` is the Canolty time-lag shift; normalisation
  ``idpac[2] == 0`` keeps the raw estimator.
- Bands are passed as an explicit ``(n_bands, 2)`` ``[low, high]`` array; the
  wrapper builds ``[f - w/2, f + w/2]`` from the center grid and full width.
- ``Pac.filter(sf, x, ftype)`` takes a **2-D** ``(n_series, n_times)`` array
  and returns ``(n_freqs, n_series, n_times)``. Tensorpac has no channel
  axis, so this wrapper folds channels into the series axis
  (``n_series = n_epochs * n_channels``) and computes every channel in one
  call.
- ``Pac.fit(pha, amp, n_perm=..., random_state=...)`` returns
  ``(n_amp, n_pha, n_series)`` — **per series, no epoch reduction**. The
  wrapper reshapes to ``(n_amp, n_pha, n_epochs, n_channels)``, averages over
  epochs (subject map) and transposes to the public
  ``(n_channels, n_phase_freqs, n_amp_freqs)`` SCHEMA shape.
- ``Pac.surrogates`` is the **full native null map**
  ``(n_perm, n_amp, n_pha, n_series)``; it is reshaped/averaged/transposed
  exactly like the observed map. ``random_state`` makes the draw
  reproducible. ``Pac.pvalues`` is *not* used: the wrapper owns the
  ``(1 + exceedances) / (n_surrogates + 1)`` convention.

Evidence boundaries (see [[SCHEMA]] §1 and US-027):
- Subject maps/null/p-values are **descriptive subject-level evidence**
  (``evidence_level="subject"``); they are uncorrected and must never be
  combined into a cohort claim.
- :func:`group_cfc` returns equal-subject-weight observed means
  (``"cohort_descriptive"``) at any ``n_subjects >= 1``; no p-values.
- :func:`cfc_condition_contrast` is the only cohort inference
  (``"cohort_subject_unit"``): per-subject condition differences, sign flips
  at the subject unit, full-map max-statistic correction per method family.
  It accepts observed :class:`CFCMap` objects only, so subject p-values
  cannot leak into cohort math.

Out of scope: cross-channel/region pairs, between-cohort inference, pooled
trial tests, uncorrected cohort p-values presented as corrected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import gcd

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, sosfiltfilt
from tensorpac import Pac

from p0ly_utils.analysis.psd import _epoch_mask

__all__ = [
    "CFCMap",
    "CFCContrastResult",
    "GroupCFCResult",
    "SubjectCFCResult",
    "cfc_condition_contrast",
    "estimate_cfc",
    "group_cfc",
]

#: tensorpac estimator IDs verified in the Stage 0 probe.
_PAC_IDS = {"gaussian_copula": 6, "mean_vector_length": 1}
#: tensorpac surrogate ID 3 = Canolty time-lag shift.
_PAC_SURROGATE_ID = 3
#: tensorpac normalisation ID 0 = raw estimator (no surrogate normalisation).
_PAC_NORM_ID = 0
#: Public method names accepted by :func:`estimate_cfc`.
_METHODS = ("pac", "phase_phase")
#: Butterworth order for the PPC band-pass (stable SOS form).
_PPC_ORDER = 4
#: Elementwise float tolerance for grid comparisons across subjects/runs.
_GRID_RTOL = 1e-9


@dataclass
class CFCMap:
    """One cross-frequency map of shape ``(ch, phase_freq, amp_freq)``.

    Attributes
    ----------
    values : ndarray, shape (n_channels, n_phase_freqs, n_amp_freqs)
        Coupling strength. PAC is Gaussian-copula PAC; PPC keys are the mean
        resultant length of ``n * phi_low - m * phi_high`` (bounded
        ``[0, 1]``). Off-ratio PPC cells are NaN where ``valid_mask`` is
        False.
    ch_names : tuple of str
        Channel labels, length ``n_channels``; the first axis follows this
        ordering.
    phase_freqs : ndarray, shape (n_phase_freqs,)
        Phase-band center frequencies (Hz); filter band per center ``f`` and
        full width ``w`` is ``[f - w/2, f + w/2]``.
    amp_freqs : ndarray, shape (n_amp_freqs,)
        Amplitude-band center frequencies (Hz).
    method : str
        Method key: ``"pac"`` or ``"ppc_{n}x{m}"`` (e.g. ``"ppc_2x1"``).
    n_epochs : int or None
        Trial count of the subject data the map came from; ``None`` on
        derived (averaged/differenced) maps.
    valid_mask : ndarray of bool, shape (n_channels, n_phase_freqs, n_amp_freqs)
        False marks a cell that carries no measurement: every PPC cell whose
        frequencies miss the ``f_amp ≈ (n/m) * f_phase`` relation within the
        configured tolerance. PAC masks are all True.
    """

    values: NDArray[np.float64]
    ch_names: tuple[str, ...]
    phase_freqs: NDArray[np.float64]
    amp_freqs: NDArray[np.float64]
    method: str
    n_epochs: int | None
    valid_mask: NDArray[np.bool_]


@dataclass
class SubjectCFCResult:
    """Per-method subject evidence: observed map, full null, p-values.

    Attributes
    ----------
    observed : CFCMap
        Observed map, shape ``(n_channels, n_phase_freqs, n_amp_freqs)``.
    null : ndarray, shape (n_surrogates, n_channels, n_phase_freqs, n_amp_freqs)
        Full surrogate distribution (intentionally retained as evidence —
        compressed archives only; keep test grids tiny). NaN on cells where
        ``observed.valid_mask`` is False.
    p_values : ndarray, shape (n_channels, n_phase_freqs, n_amp_freqs)
        One-sided empirical p-values ``(1 + #null >= observed) / (N + 1)`` in
        ``(0, 1]``. NaN on invalid cells. Uncorrected descriptive
        subject-level evidence — never cohort evidence.
    n_surrogates : int
        Surrogate count ``N`` (``null.shape[0]``).
    seed : int or None
        Seed handed to the surrogate generator.
    evidence_level : str
        Always ``"subject"``.
    """

    observed: CFCMap
    null: NDArray[np.float64]
    p_values: NDArray[np.float64]
    n_surrogates: int
    seed: int | None
    evidence_level: str


@dataclass
class GroupCFCResult:
    """Equal-weight descriptive cohort means at any ``n_subjects >= 1``.

    Attributes
    ----------
    averages : dict of str -> CFCMap
        Method key -> arithmetic subject mean, same shape/axes as the subject
        maps with ``n_epochs=None`` and NaN preserved off-ratio.
    n_subjects : int
        Number of contributing subjects.
    evidence_level : str
        Always ``"cohort_descriptive"`` — no p-values are averaged.
    """

    averages: dict[str, CFCMap]
    n_subjects: int
    evidence_level: str


@dataclass
class CFCContrastResult:
    """Subject-as-unit cohort condition contrast with full-map correction.

    Attributes
    ----------
    difference : CFCMap
        Mean of per-subject ``first - second`` differences, shape
        ``(n_channels, n_phase_freqs, n_amp_freqs)``.
    p_values_corrected : ndarray, shape (n_channels, n_phase_freqs, n_amp_freqs)
        ``(1 + #null_max >= |observed|) / (n_resamples + 1)`` per cell; NaN
        on invalid cells. Family = every valid cell of this method map.
    null_max : ndarray, shape (n_resamples,)
        Max-|statistic| over valid cells for each sign-flip resample.
    n_subjects : int
        Paired subject count.
    n_permutations : int
        The requested cap on sign patterns.
    n_resamples : int
        Actual resample count (``null_max.size``): ``2 ** (n_subjects - 1)``
        when that fits under ``n_permutations``, else ``n_permutations``.
    seed : int or None
        Seed for random sign-pattern draws (``None`` for exact enumeration).
    conditions : tuple of str, length 2
        ``(first, second)`` — the difference is ``first - second``.
    correction : str
        ``"full_map_max_abs_mean"``.
    evidence_level : str
        Always ``"cohort_subject_unit"``.
    """

    difference: CFCMap
    p_values_corrected: NDArray[np.float64]
    null_max: NDArray[np.float64]
    n_subjects: int
    n_permutations: int
    n_resamples: int
    seed: int | None
    conditions: tuple[str, str]
    correction: str
    evidence_level: str


# --- validation helpers -----------------------------------------------------


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    """Validate and de-duplicate the requested method names (order kept)."""
    named = tuple(methods)
    if not named:
        raise ValueError("cfc: methods must contain at least one of 'pac' / 'phase_phase'")
    unknown = sorted(set(named) - set(_METHODS))
    if unknown:
        raise ValueError(f"cfc: unknown method(s) {unknown}; expected any of {list(_METHODS)}")
    return tuple(dict.fromkeys(named))


def _validate_grid(
    freqs: Sequence[float] | NDArray[np.float64],
    bandwidth: float,
    sfreq: float,
    label: str,
) -> NDArray[np.float64]:
    """Validate one center grid + full band width against ``freq_grid`` rules."""
    grid = np.asarray(freqs, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"cfc: {label} grid must be a 1-D non-empty array, got {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"cfc: {label} grid carries non-finite centers: {grid}")
    if grid.size < 2:
        raise ValueError(f"cfc: {label} grid needs >= 2 bins (matching freq_grid), got {grid.size}")
    if not np.all(np.diff(grid) > 0):
        raise ValueError(f"cfc: {label} grid must be strictly increasing, got {grid}")
    width = float(bandwidth)
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"cfc: {label} bandwidth must be positive and finite, got {bandwidth}")
    low = grid - width / 2.0
    high = grid + width / 2.0
    if low[0] <= 0:
        raise ValueError(
            f"cfc: {label} band lower edge must be > 0 Hz, got {low[0]:.3f} Hz "
            f"(center {grid[0]} Hz, width {width} Hz)"
        )
    if high[-1] >= sfreq / 2.0:
        raise ValueError(
            f"cfc: {label} band upper edge must be < Nyquist ({sfreq / 2.0} Hz), "
            f"got {high[-1]:.3f} Hz (center {grid[-1]} Hz, width {width} Hz)"
        )
    return grid


def _validate_ratio(ratio: object) -> tuple[int, int]:
    """Validate one ``(n, m)`` integer pair: positive, coprime, non-trivial."""
    pair = tuple(ratio)  # type: ignore[arg-type]
    if len(pair) != 2:
        raise ValueError(f"cfc: ratio {ratio!r} must be a (n, m) pair, got {len(pair)} values")
    for value in pair:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"cfc: ratio {ratio!r} entries must be integers, got {value!r}")
        if value <= 0:
            raise ValueError(f"cfc: ratio {ratio!r} entries must be positive, got {value}")
    n, m = int(pair[0]), int(pair[1])
    if gcd(n, m) != 1:
        raise ValueError(
            f"cfc: ratio ({n}, {m}) is reducible — use ({n // gcd(n, m)}, {m // gcd(n, m)})"
        )
    return n, m


def _padlen() -> int:
    """Edge padding scipy's ``sosfiltfilt`` needs for the PPC band-pass."""
    sos = butter(_PPC_ORDER, [1.0, 2.0], btype="bandpass", fs=100.0, output="sos")
    return 3 * (2 * len(sos) + 1)


def _band_phases(
    data: NDArray[np.float64],
    sfreq: float,
    centers: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Zero-phase band-pass + Hilbert phase per center frequency.

    Each epoch is filtered independently (no epoch concatenation) with a
    stable SOS Butterworth filter and ``scipy.signal.hilbert(..., axis=-1)``.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        Volts. Filtering never crosses the time axis of a single epoch.
    sfreq : float
        Sampling frequency (Hz).
    centers : ndarray, shape (n_bands,)
        Band center frequencies (Hz).
    bandwidth : float
        Full band width (Hz): band = ``[f - w/2, f + w/2]``.

    Returns
    -------
    ndarray, shape (n_bands, n_epochs, n_channels, n_times)
        Wrapped phases (rad) per band. Looping over frequency bins is
        allowed; channels/epochs ride inside the vectorized call.
    """
    phases = np.empty((centers.size, *data.shape), dtype=np.float64)
    for index, center in enumerate(centers):
        sos = butter(
            _PPC_ORDER,
            [center - bandwidth / 2.0, center + bandwidth / 2.0],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        analytic = hilbert(sosfiltfilt(sos, data, axis=-1), axis=-1)
        phases[index] = np.angle(analytic)
    return phases


def _empirical_p(
    observed: NDArray[np.float64],
    null: NDArray[np.float64],
) -> NDArray[np.float64]:
    """One-sided p-value ``(1 + #null >= observed) / (N + 1)`` per cell."""
    null_maps = np.asarray(null, dtype=float)
    exceedances = np.count_nonzero(null_maps >= np.asarray(observed), axis=0)
    p_values = (1.0 + exceedances) / (null_maps.shape[0] + 1.0)
    return np.asarray(np.where(np.isnan(observed), np.nan, p_values), dtype=np.float64)


# --- PAC --------------------------------------------------------------------


def _pac_subject_maps(
    data: NDArray[np.float64],
    sfreq: float,
    phase_freqs: NDArray[np.float64],
    amp_freqs: NDArray[np.float64],
    phase_bandwidth: float,
    amp_bandwidth: float,
    pac_method: str,
    n_surrogates: int,
    seed: int | None,
    n_jobs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observed + full null PAC maps from one tensorpac call per stage.

    Returns
    -------
    observed : ndarray, shape (n_channels, n_phase_freqs, n_amp_freqs)
        Gaussian-copula (or MVL) PAC, mean over epochs.
    null : ndarray, shape (n_surrogates, n_channels, n_phase_freqs, n_amp_freqs)
        tensorpac's native time-lag surrogate maps, reduced identically.
    """
    n_epochs, n_channels, n_times = data.shape
    # Fold channels into tensorpac's series axis: one call covers every channel.
    series = data.reshape(n_epochs * n_channels, n_times)
    pac = Pac(
        idpac=(_PAC_IDS[pac_method], _PAC_SURROGATE_ID, _PAC_NORM_ID),
        f_pha=np.column_stack(
            (
                phase_freqs - phase_bandwidth / 2.0,
                phase_freqs + phase_bandwidth / 2.0,
            )
        ),
        f_amp=np.column_stack((amp_freqs - amp_bandwidth / 2.0, amp_freqs + amp_bandwidth / 2.0)),
        dcomplex="hilbert",
        verbose="ERROR",
    )
    phases = pac.filter(sfreq, series, ftype="phase", n_jobs=n_jobs)
    amplitudes = pac.filter(sfreq, series, ftype="amplitude", n_jobs=n_jobs)
    observed = pac.fit(
        phases,
        amplitudes,
        n_perm=n_surrogates,
        random_state=seed,
        n_jobs=n_jobs,
    )
    n_amp, n_pha = amp_freqs.size, phase_freqs.size
    # tensorpac native order (n_amp, n_pha, n_series) -> SCHEMA (ch, pha, amp).
    observed_map = (
        np.asarray(observed, dtype=float)
        .reshape(n_amp, n_pha, n_epochs, n_channels)
        .mean(axis=2)
        .transpose(2, 1, 0)
    )
    null_map = (
        np.asarray(pac.surrogates, dtype=float)
        .reshape(n_surrogates, n_amp, n_pha, n_epochs, n_channels)
        .mean(axis=3)
        .transpose(0, 3, 2, 1)
    )
    return observed_map, null_map


# --- n:m phase-phase coupling -----------------------------------------------


def _ppc_valid_mask(
    phase_freqs: NDArray[np.float64],
    amp_freqs: NDArray[np.float64],
    n: int,
    m: int,
    ratio_tolerance: float,
) -> NDArray[np.bool_]:
    """Cells whose frequencies satisfy ``|f_amp - (n/m) * f_phase| <= tol``."""
    expected = (n / m) * phase_freqs  # (n_phase,)
    distance = np.abs(amp_freqs[np.newaxis, :] - expected[:, np.newaxis])
    return np.asarray(distance <= ratio_tolerance, dtype=bool)


def _ppc_subject_maps(
    data: NDArray[np.float64],
    sfreq: float,
    phase_freqs: NDArray[np.float64],
    amp_freqs: NDArray[np.float64],
    phase_bandwidth: float,
    amp_bandwidth: float,
    n: int,
    m: int,
    valid_mask: NDArray[np.bool_],
    n_surrogates: int,
    seed: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observed + full null n:m PPC maps (mean resultant length).

    Coupling is ``|mean(exp(1j * (n * phi_low - m * phi_high)))|`` over trials
    and time, one value per channel and valid frequency pair. Surrogates
    circularly shift the high-frequency phase within every epoch/channel,
    which keeps each epoch's filtered phase sequence intact while breaking
    its cross-frequency alignment.

    Returns
    -------
    observed : ndarray, shape (n_channels, n_phase_freqs, n_amp_freqs)
        ``[0, 1]`` coupling; NaN off-ratio.
    null : ndarray, shape (n_surrogates, n_channels, n_phase_freqs, n_amp_freqs)
        Full null maps; NaN off-ratio.
    """
    n_epochs, n_channels, n_times = data.shape
    phase_phases = _band_phases(data, sfreq, phase_freqs, phase_bandwidth)
    low_index, high_index = np.nonzero(valid_mask)
    # E_low/E_high carry the (n, m) multipliers so the phase difference is one
    # elementwise product; the pair sum is a single einsum over trials+time.
    e_low = np.exp(1j * n * phase_phases)[low_index]  # (n_pairs, ep, ch, t)
    needed = np.unique(amp_freqs[high_index])
    amp_phases = _band_phases(data, sfreq, needed, amp_bandwidth)
    e_high = np.exp(1j * m * amp_phases)[np.searchsorted(needed, amp_freqs[high_index])]

    def _couple(high: NDArray[np.complex128]) -> NDArray[np.float64]:
        # Sum over trials and time only — channels stay a separate axis.
        pair_values = np.abs(np.einsum("pict,pict->pc", e_low, np.conj(high)))
        return pair_values / float(n_epochs * n_times)

    def _scatter(pair_values: NDArray[np.float64]) -> NDArray[np.float64]:
        full = np.full((n_channels,) + valid_mask.shape, np.nan, dtype=np.float64)
        full[:, low_index, high_index] = pair_values.T
        return full

    observed = _scatter(_couple(e_high))
    rng = np.random.default_rng(seed)
    stride = np.arange(n_times)
    null = np.empty((n_surrogates,) + observed.shape, dtype=np.float64)
    for surrogate in range(n_surrogates):
        # Per-epoch/per-channel circular shift of the high-band phase.
        shifts = rng.integers(0, n_times, size=(n_epochs, n_channels, 1))
        index = (stride + shifts) % n_times
        null[surrogate] = _scatter(_couple(np.take_along_axis(e_high, index[np.newaxis], axis=-1)))
    return observed, null


# --- public API -------------------------------------------------------------


def estimate_cfc(
    epochs: mne.BaseEpochs,
    *,
    methods: Sequence[str],
    phase_freqs: Sequence[float] | NDArray[np.float64],
    amp_freqs: Sequence[float] | NDArray[np.float64],
    phase_bandwidth: float,
    amp_bandwidth: float,
    pac_method: str = "gaussian_copula",
    ratios: Sequence[tuple[int, int]] = ((2, 1),),
    ratio_tolerance: float = 0.5,
    n_surrogates: int = 200,
    seed: int | None = None,
    condition_query: str | None = None,
    n_jobs: int = 1,
) -> dict[str, SubjectCFCResult]:
    """Per-subject CFC maps with reproducible surrogate nulls and p-values.

    PAC runs through tensorpac (all channels in one call, result transposed
    to the public shape); n:m PPC runs on SciPy Hilbert phases. Every method
    gets its own observed map, full surrogate null and one-sided empirical
    p-values — **subject-level descriptive evidence**, uncorrected, never
    cohort evidence.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs with data shaped ``(n_epochs, n_channels, n_times)`` in Volts.
        At least 2 epochs (after condition selection) and finite data are
        required.
    methods : sequence of str
        Any of ``"pac"`` / ``"phase_phase"`` (order preserved). ``"pac"``
        produces the ``"pac"`` key; ``"phase_phase"`` one ``"ppc_{n}x{m}"``
        key per configured ratio.
    phase_freqs, amp_freqs : sequence of float
        Center-frequency grids (Hz), strictly increasing, >= 2 bins, finite.
        Bands are ``[f - w/2, f + w/2]`` for the matching full width.
    phase_bandwidth, amp_bandwidth : float
        Full band widths (Hz). Must be positive; every band edge must stay
        inside ``(0, sfreq / 2)``.
    pac_method : str
        tensorpac PAC estimator: ``"gaussian_copula"`` (default) or
        ``"mean_vector_length"`` (both probe-verified IDs).
    ratios : sequence of (int, int)
        Integer ``(n, m)`` pairs for PPC, with ``f_amp ≈ (n/m) * f_phase``
        and the phase difference ``n * phi_low - m * phi_high``. Positive,
        coprime; reducible pairs such as ``(4, 2)`` are rejected.
    ratio_tolerance : float
        Absolute Hz tolerance for a PPC cell to count as ratio-valid. Cells
        outside stay NaN and are excluded from every downstream statistic.
    n_surrogates : int
        Surrogate count per method (>= 1). Artifacts retain the full null
        ``(n_surrogates, n_channels, n_phase_freqs, n_amp_freqs)`` — keep test
        grids and counts tiny.
    seed : int or None
        Seed for tensorpac's time-lag draw and the PPC circular shifts. The
        same data/config/seed reproduces observed and null arrays exactly.
    condition_query : str or None
        Pandas ``DataFrame.eval`` query on ``epochs.metadata``; ``None`` uses
        every epoch (``all_trials``). Zero/two-few matches fail loud.
    n_jobs : int
        Parallelism forwarded to tensorpac's filter/surrogate stages.

    Returns
    -------
    dict of str -> SubjectCFCResult
        Method key -> subject evidence. PAC ``valid_mask`` is all True; PPC
        masks encode ratio-compatible cells.

    Raises
    ------
    ValueError
        Invalid grids/bandwidths/Nyquist edges, fewer than 2 selected epochs,
        too few time samples for the chosen bands, non-finite data, unknown
        method/``pac_method``, reducible or non-positive ratios,
        non-positive ``ratio_tolerance``/``n_surrogates``, or a condition
        query that matches fewer than 2 epochs.
    """
    method_keys = _validate_methods(methods)
    if pac_method not in _PAC_IDS:
        raise ValueError(f"cfc: pac_method must be one of {sorted(_PAC_IDS)}, got {pac_method!r}")
    if isinstance(n_surrogates, bool) or not isinstance(n_surrogates, (int, np.integer)):
        raise ValueError(f"cfc: n_surrogates must be an integer, got {n_surrogates!r}")
    if n_surrogates < 1:
        raise ValueError(f"cfc: n_surrogates must be >= 1, got {n_surrogates}")
    tolerance = float(ratio_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"cfc: ratio_tolerance must be positive and finite, got {ratio_tolerance}")
    ratio_pairs = tuple(_validate_ratio(ratio) for ratio in ratios)
    if "phase_phase" in method_keys and not ratio_pairs:
        raise ValueError("cfc: method 'phase_phase' needs at least one (n, m) ratio")

    sfreq = float(epochs.info["sfreq"])
    phase_grid = _validate_grid(phase_freqs, phase_bandwidth, sfreq, "phase_freqs")
    amp_grid = _validate_grid(amp_freqs, amp_bandwidth, sfreq, "amp_freqs")

    selected = epochs if condition_query is None else epochs[_epoch_mask(epochs, condition_query)]
    chosen = len(epochs) if condition_query is None else len(selected)
    if chosen < 2:
        raise ValueError(
            f"cfc: {chosen} epoch(s) selected"
            f"{'' if condition_query is None else f' by query {condition_query!r}'}"
            " — cross-frequency coupling is an across-trials statistic, need >= 2"
        )
    data = np.asarray(selected.get_data(), dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("cfc: epochs carry non-finite samples — interpolate/drop before CFC")
    n_times = data.shape[-1]
    needed_samples = _padlen() + 1
    if n_times < needed_samples:
        raise ValueError(
            f"cfc: epochs have {n_times} samples; the {needed_samples}-sample floor "
            "is what a zero-phase SOS band-pass needs for the configured bands"
        )

    subjects: dict[str, SubjectCFCResult] = {}
    if "pac" in method_keys:
        observed, null = _pac_subject_maps(
            data,
            sfreq,
            phase_grid,
            amp_grid,
            float(phase_bandwidth),
            float(amp_bandwidth),
            pac_method,
            int(n_surrogates),
            seed,
            n_jobs,
        )
        subjects["pac"] = _subject_result(
            observed, null, selected.ch_names, phase_grid, amp_grid, "pac", chosen, seed
        )
    if "phase_phase" in method_keys:
        for n, m in ratio_pairs:
            key = f"ppc_{n}x{m}"
            valid_mask = _ppc_valid_mask(phase_grid, amp_grid, n, m, tolerance)
            if not valid_mask.any():
                raise ValueError(
                    f"cfc: ratio ({n}, {m}) with tolerance {tolerance} Hz selects no "
                    f"(phase_freq, amp_freq) cell — widen ratio_tolerance or the grids"
                )
            observed, null = _ppc_subject_maps(
                data,
                sfreq,
                phase_grid,
                amp_grid,
                float(phase_bandwidth),
                float(amp_bandwidth),
                n,
                m,
                valid_mask,
                int(n_surrogates),
                seed,
            )
            subjects[key] = _subject_result(
                observed,
                null,
                selected.ch_names,
                phase_grid,
                amp_grid,
                key,
                chosen,
                seed,
                valid_mask=valid_mask,
            )
    return subjects


def _subject_result(
    observed: NDArray[np.float64],
    null: NDArray[np.float64],
    ch_names: Sequence[str],
    phase_grid: NDArray[np.float64],
    amp_grid: NDArray[np.float64],
    method: str,
    n_epochs: int,
    seed: int | None,
    valid_mask: NDArray[np.bool_] | None = None,
) -> SubjectCFCResult:
    """Assemble one :class:`SubjectCFCResult` from observed/null arrays."""
    mask = (
        np.broadcast_to(np.asarray(valid_mask, dtype=bool), observed.shape).copy()
        if valid_mask is not None
        else np.ones(observed.shape, dtype=bool)
    )
    return SubjectCFCResult(
        observed=CFCMap(
            values=observed,
            ch_names=tuple(str(name) for name in ch_names),
            phase_freqs=phase_grid.copy(),
            amp_freqs=amp_grid.copy(),
            method=method,
            n_epochs=n_epochs,
            valid_mask=mask,
        ),
        null=null,
        p_values=_empirical_p(observed, null),
        n_surrogates=int(null.shape[0]),
        seed=seed,
        evidence_level="subject",
    )


def _check_comparable(maps: Sequence[CFCMap], label: str) -> None:
    """Raise unless every map shares channels, grids, shape and valid mask."""
    reference = maps[0]
    for candidate in maps[1:]:
        if candidate.ch_names != reference.ch_names:
            raise ValueError(
                f"{label}: channel mismatch — {candidate.ch_names} vs {reference.ch_names}"
            )
        for name, ref_axis, axis in (
            ("phase_freqs", reference.phase_freqs, candidate.phase_freqs),
            ("amp_freqs", reference.amp_freqs, candidate.amp_freqs),
        ):
            if ref_axis.shape != axis.shape or not np.allclose(
                ref_axis, axis, rtol=_GRID_RTOL, atol=0.0
            ):
                raise ValueError(f"{label}: {name} grid mismatch ({axis} vs {ref_axis})")
        if candidate.values.shape != reference.values.shape:
            raise ValueError(
                f"{label}: shape mismatch {candidate.values.shape} vs {reference.values.shape}"
            )
        if not np.array_equal(candidate.valid_mask, reference.valid_mask):
            raise ValueError(
                f"{label}: valid_mask mismatch — recompute every subject with identical "
                "ratio_tolerance/grids so off-ratio cells line up"
            )


def _check_map_set(maps: Sequence[Mapping[str, CFCMap]], label: str) -> tuple[str, ...]:
    """Validate one collection of method-keyed maps; return the method keys."""
    entries = list(maps)
    if not entries:
        raise ValueError(f"{label}: received no subjects — nothing to combine")
    keys = tuple(entries[0])
    for entry in entries:
        for value in entry.values():
            if not isinstance(value, CFCMap):
                raise TypeError(
                    f"{label}: accepts observed CFCMap objects only, got "
                    f"{type(value).__name__} — subject nulls/p-values must not enter "
                    "cohort math"
                )
        if tuple(entry) != keys:
            raise ValueError(f"{label}: method mismatch — {tuple(entry)} vs {keys}")
    return keys


def group_cfc(
    subject_maps: Sequence[Mapping[str, CFCMap]],
) -> GroupCFCResult:
    """Equal-subject-weight descriptive cohort mean at any ``n_subjects >= 1``.

    Subject means are arithmetic and trial-count-blind (mirroring
    :func:`p0ly_utils.analysis.group_tfr` / ``group_connectivity``); no
    p-values are averaged — this is ``"cohort_descriptive"`` evidence, not
    inference.

    Parameters
    ----------
    subject_maps : sequence of mapping of str -> CFCMap
        One mapping per subject, method key -> observed map. Every subject
        must carry the same method keys, channel set, phase/amplitude grids
        and valid masks; valid cells must be finite.

    Returns
    -------
    GroupCFCResult
        Method key -> mean map ``(n_channels, n_phase_freqs, n_amp_freqs)``
        with ``n_epochs=None``; off-ratio cells stay NaN.

    Raises
    ------
    ValueError
        No subjects, method/channel/grid/mask/shape mismatch, or a non-finite
        value in a valid cell.
    """
    entries = list(subject_maps)
    keys = _check_map_set(entries, "group_cfc")
    averages: dict[str, CFCMap] = {}
    for method in keys:
        per_subject = [entry[method] for entry in entries]
        _check_comparable(per_subject, "group_cfc")
        reference = per_subject[0]
        stacked = np.stack([m.values for m in per_subject], axis=0)
        mask = reference.valid_mask
        if not np.all(np.isfinite(stacked[:, mask])):
            raise ValueError(
                f"group_cfc: method {method!r} has non-finite values in ratio-valid "
                "cells — a valid cell must be finite in every subject"
            )
        mean = np.full(reference.values.shape, np.nan, dtype=np.float64)
        mean[mask] = stacked[:, mask].mean(axis=0)
        averages[method] = CFCMap(
            values=mean,
            ch_names=reference.ch_names,
            phase_freqs=reference.phase_freqs.copy(),
            amp_freqs=reference.amp_freqs.copy(),
            method=method,
            n_epochs=None,
            valid_mask=mask.copy(),
        )
    return GroupCFCResult(
        averages=averages,
        n_subjects=len(entries),
        evidence_level="cohort_descriptive",
    )


def _sign_patterns(n_subjects: int, n_permutations: int, seed: int | None) -> NDArray[np.float64]:
    """Sign-flip patterns with the first subject fixed to ``+1``.

    Exact enumeration when ``2 ** (n_subjects - 1) <= n_permutations``; else
    ``n_permutations`` drawn patterns with the observed (all-``+``) row
    included exactly once.
    """
    n_exact = 2 ** (n_subjects - 1)
    if n_exact <= n_permutations:
        rest = np.array(list(product((-1.0, 1.0), repeat=n_subjects - 1)), dtype=float)
        return np.concatenate([np.ones((n_exact, 1)), rest], axis=1)
    draw = np.random.default_rng(seed).choice(
        [-1.0, 1.0], size=(n_permutations - 1, n_subjects - 1)
    )
    return np.vstack(
        [
            np.ones((1, n_subjects)),
            np.hstack([np.ones((n_permutations - 1, 1)), draw]),
        ]
    )


def cfc_condition_contrast(
    first: Sequence[Mapping[str, CFCMap]],
    second: Sequence[Mapping[str, CFCMap]],
    *,
    conditions: tuple[str, str],
    n_permutations: int = 1024,
    seed: int | None = None,
) -> dict[str, CFCContrastResult]:
    """Subject-as-unit condition contrast with full-map max-statistic control.

    Per subject the same ordered cohort position contributes ``first -
    second``; under the null each subject's **whole difference map** flips
    sign (a paired condition-label exchange at the subject unit), the first
    subject stays ``+1`` to remove the global sign symmetry, and each resample
    contributes the max ``|statistic|`` over that method's valid cells.
    ``p_corrected = (1 + #null_max >= |observed|) / (n_resamples + 1)``.

    Subject surrogate p-values never enter this computation: the inputs are
    observed :class:`CFCMap` objects only.

    **Resolution:** exact enumeration yields at most ``2 ** (n_subjects - 1)``
    null maxima, so with the ``+1`` convention n=2 has minimum p = 1/3 and
    n=6 has minimum p = 1/33 — small cohorts cannot necessarily reach
    p < 0.05. Larger cohorts draw ``n_permutations`` seeded patterns.

    Parameters
    ----------
    first, second : sequence of mapping of str -> CFCMap
        Condition A and B observed maps, paired **by position** (the pipeline
        resolves each arm over the same ordered subject list). At least 2
        subjects with identical method keys, channels, grids and masks.
    conditions : tuple of (str, str)
        ``(first, second)`` labels; the difference is ``first - second``.
    n_permutations : int
        Cap on sign-flip resamples (>= 2).
    seed : int or None
        Seed for the random sign-pattern draw (unused for exact enumeration).

    Returns
    -------
    dict of str -> CFCContrastResult
        One full-map family per method key (PAC and every PPC ratio are
        corrected separately — no cross-method family claim).

    Raises
    ------
    ValueError
        Fewer than 2 subjects, unequal arm lengths, a non-positive cap, or
        any method/channel/grid/mask mismatch between arms.
    """
    arm_first, arm_second = list(first), list(second)
    if len(arm_first) != len(arm_second):
        raise ValueError(
            f"cfc_condition_contrast: arm sizes differ ({len(arm_first)} vs "
            f"{len(arm_second)}) — paired subjects must be the same cohort"
        )
    if len(arm_first) < 2:
        raise ValueError(
            f"cfc_condition_contrast: {len(arm_first)} subject(s) — cohort inference "
            "needs >= 2 paired subjects (the DAG omits these targets at n<2)"
        )
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, (int, np.integer)):
        raise ValueError(
            f"cfc_condition_contrast: n_permutations must be an int, got {n_permutations!r}"
        )
    if n_permutations < 2:
        raise ValueError(
            f"cfc_condition_contrast: n_permutations must be >= 2, got {n_permutations}"
        )
    if len(conditions) != 2:
        raise ValueError(
            f"cfc_condition_contrast: conditions must be a (first, second) pair, got {conditions!r}"
        )
    keys = _check_map_set(arm_first, "cfc_condition_contrast")
    second_keys = _check_map_set(arm_second, "cfc_condition_contrast")
    if second_keys != keys:
        raise ValueError(
            f"cfc_condition_contrast: method mismatch between arms — {second_keys} vs {keys}"
        )

    n_subjects = len(arm_first)
    patterns = _sign_patterns(n_subjects, int(n_permutations), seed)
    results: dict[str, CFCContrastResult] = {}
    for method in keys:
        stacked_first = [entry[method] for entry in arm_first]
        stacked_second = [entry[method] for entry in arm_second]
        both_arms = stacked_first + stacked_second
        _check_comparable(both_arms, f"cfc_condition_contrast[{method}]")
        reference = both_arms[0]
        differences = np.stack(
            [a.values - b.values for a, b in zip(stacked_first, stacked_second, strict=True)],
            axis=0,
        )
        valid = reference.valid_mask.copy()
        valid &= np.all(np.isfinite(differences), axis=0)
        if not valid.any():
            raise ValueError(
                f"cfc_condition_contrast: method {method!r} has no ratio-valid cell that "
                "is finite in both conditions — nothing to test"
            )
        difference = np.full(reference.values.shape, np.nan, dtype=np.float64)
        difference[valid] = differences[:, valid].mean(axis=0)

        null_max = np.empty(patterns.shape[0], dtype=np.float64)
        for index, pattern in enumerate(patterns):
            flipped = (pattern[:, np.newaxis, np.newaxis, np.newaxis] * differences).mean(axis=0)
            null_max[index] = np.abs(flipped[valid]).max()
        exceedances = np.count_nonzero(
            null_max[:, np.newaxis] >= np.abs(difference[valid])[np.newaxis, :], axis=0
        )
        p_corrected = np.full(reference.values.shape, np.nan, dtype=np.float64)
        p_corrected[valid] = (1.0 + exceedances) / (null_max.size + 1.0)
        results[method] = CFCContrastResult(
            difference=CFCMap(
                values=difference,
                ch_names=reference.ch_names,
                phase_freqs=reference.phase_freqs.copy(),
                amp_freqs=reference.amp_freqs.copy(),
                method=method,
                n_epochs=None,
                valid_mask=valid,
            ),
            p_values_corrected=p_corrected,
            null_max=null_max,
            n_subjects=n_subjects,
            n_permutations=int(n_permutations),
            n_resamples=int(null_max.size),
            seed=seed,
            conditions=(str(conditions[0]), str(conditions[1])),
            correction="full_map_max_abs_mean",
            evidence_level="cohort_subject_unit",
        )
    return results
