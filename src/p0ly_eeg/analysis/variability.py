"""Across-trial variability maps for voltage and oscillatory power (US-029).

Design (recorded in ``scrum/01_Backlog/US-029_neural-variability.md``):

- Within one subject/condition every measure is a reduction over the trial
  axis: ``sd = sqrt(var(data, ddof=1))``, ``fano = var / mean``,
  ``cv = sd / mean``. Sample estimators need ``n_trials >= 2``; the trial
  count is carried on the result so it never has to be guessed.
- Signed voltage ``(n_epochs, n_channels, n_times)`` supports **SD only**.
  Fano/CV describe relative dispersion of nonnegative quantities, so
  requesting them for voltage is a hard error, not a silent skip.
- Power ``(n_epochs, n_channels, n_freqs, n_times)`` comes from the existing
  complex TFR transform (:func:`~p0ly_eeg.analysis.tfr.tfr_epochs` +
  :func:`~p0ly_eeg.analysis.tfr.tfr_power_epochs`) — one spectral
  implementation, no second convolution. Negative or non-finite power
  rejects the entire input.
- Ratio measures are defined **only where mean power > 0**: undefined cells
  stay NaN (``np.divide(..., where=...)``, never a zero-replacement epsilon)
  and are identified by the persisted ``positive_mean`` mask. SD remains
  defined at zero mean.
- Cohort maps are equal-weight summaries of per-subject maps (never pooled
  trials). Ratio cells average only the subjects whose mean is positive and
  record a per-cell contributor count; ``n_subjects`` is always the full
  cohort size, so a changing denominator stays visible.
- Small-sample bias: two trials are mathematically sufficient but unstable;
  the sample SD is unbiased for the variance, not for the SD itself, and
  Fano/CV are especially noisy near zero mean. No bias correction is
  applied. Also out of scope: variability-of-variability, cross-channel or
  cross-subject ratios, log-power, and any inferential contrast.
- All functions are pure: no filesystem access, no I/O, no config reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mne
import numpy as np
from numpy.typing import NDArray

from p0ly_eeg.analysis.psd import _epoch_mask
from p0ly_eeg.analysis.tfr import freq_grid, tfr_epochs, tfr_power_epochs

__all__ = [
    "GroupVariabilityResult",
    "VariabilityResult",
    "group_variability",
    "variability_epochs",
]

_DOMAINS = ("voltage", "power")
_MEASURES = ("sd", "fano", "cv")
_RATIO_MEASURES = frozenset({"fano", "cv"})
_RANK = {"voltage": 3, "power": 4}


@dataclass
class VariabilityResult:
    """Across-trial variability maps of one subject's condition.

    Attributes
    ----------
    maps : dict of str to ndarray
        Measure name → map. Requested measure order is preserved. Voltage
        maps are ``(n_channels, n_times)``; power maps are
        ``(n_channels, n_freqs, n_times)``. Ratio cells with zero mean power
        are NaN.
    ch_names : tuple of str
        Channel names, length ``n_channels``.
    times : ndarray, shape (n_times,)
        Time axis in seconds (power: the decimated TFR grid).
    freqs : ndarray, shape (n_freqs,) or None
        Frequency axis in Hz; None for the voltage domain.
    domain : str
        ``"voltage"`` or ``"power"``.
    n_trials : int
        Number of trials reduced (>= 2). SD/variances are sample estimators.
    positive_mean : ndarray of bool or None
        Power domain only: ``True`` where mean power > 0, i.e. where Fano/CV
        are defined. Same shape as the power maps and present even when only
        ``sd`` was requested. None for voltage.
    """

    maps: dict[str, NDArray[np.float64]]
    ch_names: tuple[str, ...]
    times: NDArray[np.float64]
    freqs: NDArray[np.float64] | None
    domain: str
    n_trials: int
    positive_mean: NDArray[np.bool_] | None


@dataclass
class GroupVariabilityResult:
    """Equal-subject-weight cohort summary of variability maps.

    Attributes
    ----------
    maps : dict of str to ndarray
        Measure name → cohort map. Voltage ``(n_channels, n_times)``, power
        ``(n_channels, n_freqs, n_times)``. Ratio cells with no contributing
        subject are NaN.
    n_contributing : dict of str to ndarray of int64
        Measure name → per-cell number of contributing subjects, same shape
        as the map. SD is defined for every subject (count = ``n_subjects``);
        Fano/CV count only subjects whose mean was positive.
    ch_names : tuple of str
        Channel names, length ``n_channels``.
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    freqs : ndarray, shape (n_freqs,) or None
        Frequency axis in Hz; None for the voltage domain.
    domain : str
        ``"voltage"`` or ``"power"``.
    n_subjects : int
        Full cohort size, even where fewer subjects contributed a ratio cell.
    """

    maps: dict[str, NDArray[np.float64]]
    n_contributing: dict[str, NDArray[np.int64]]
    ch_names: tuple[str, ...]
    times: NDArray[np.float64]
    freqs: NDArray[np.float64] | None
    domain: str
    n_subjects: int


def _validated_measures(domain: str, measures: Sequence[str]) -> list[str]:
    """Domain + measure guards; returns the requested measures in order."""
    if domain not in _DOMAINS:
        raise ValueError(
            f"variability: unknown domain {domain!r} (voltage | power)"
        )
    names = list(measures)
    if not names:
        raise ValueError(
            f"variability: no measures requested for domain {domain!r} — "
            f"pick from {list(_MEASURES)}"
        )
    unknown = [name for name in names if name not in _MEASURES]
    if unknown:
        raise ValueError(
            f"variability: unknown measure(s) {unknown} for domain "
            f"{domain!r} (sd | fano | cv)"
        )
    ratios = [name for name in names if name in _RATIO_MEASURES]
    if domain == "voltage" and ratios:
        raise ValueError(
            f"variability: measure(s) {ratios} are undefined for signed "
            "voltage — Fano/CV describe relative dispersion of nonnegative "
            "quantities (use domain='power')"
        )
    return names


def _checked(data: NDArray[np.float64], domain: str) -> NDArray[np.float64]:
    """Rank / trial-count / finite / nonnegativity guards (§2)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != _RANK[domain]:
        raise ValueError(
            f"variability: {domain} data must be rank {_RANK[domain]} with "
            f"the trial axis first, got shape {data.shape}"
        )
    n_trials = data.shape[0]
    if n_trials < 2:
        raise ValueError(
            f"variability: {domain} domain needs >= 2 trials for sample "
            f"statistics (ddof=1), got {n_trials}"
        )
    n_nonfinite = int(np.count_nonzero(~np.isfinite(data)))
    if n_nonfinite:
        raise ValueError(
            f"variability: {domain} data has {n_nonfinite} non-finite "
            "cell(s) — rejected, not imputed"
        )
    if domain == "power":
        n_negative = int(np.count_nonzero(data < 0))
        if n_negative:
            raise ValueError(
                f"variability: power data has {n_negative} negative cell(s) "
                "— power is nonnegative"
            )
    return data


def _reduced(
    data: NDArray[np.float64], measures: Sequence[str]
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_]]:
    """Shared validated mean/sample-variance reduction over the trial axis."""
    mean = data.mean(axis=0)
    var = data.var(axis=0, ddof=1)
    sd = np.sqrt(var)
    positive_mean = mean > 0
    nan = np.full(mean.shape, np.nan)
    maps: dict[str, NDArray[np.float64]] = {}
    for measure in measures:
        if measure == "sd":
            maps["sd"] = sd
        elif measure == "fano":
            maps["fano"] = np.divide(var, mean, out=nan.copy(), where=positive_mean)
        else:  # "cv", already guarded
            maps["cv"] = np.divide(sd, mean, out=nan.copy(), where=positive_mean)
    return maps, positive_mean


def variability_epochs(
    epochs: mne.BaseEpochs,
    *,
    domain: str,
    measures: Sequence[str],
    condition_query: str | None = None,
    tfr_settings: Mapping[str, object] | None = None,
) -> VariabilityResult:
    """Across-trial variability maps for one subject's condition.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs of one subject, ``(n_epochs, n_channels, n_times)``, loaded
        (preloaded) and optionally carrying ``epochs.metadata`` for condition
        selection. The input object is not mutated.
    domain : str
        ``"voltage"`` (signed ERP-scale data, SD only) or ``"power"``
        (nonnegative TFR power from ``tfr_settings``).
    measures : sequence of str
        ``"sd"``, and for power also ``"fano"`` / ``"cv"``. Order is
        preserved in ``maps``.
    condition_query : str or None
        Pandas ``DataFrame.eval`` query on ``epochs.metadata`` selecting the
        condition subset (``None`` = all trials, the ``all_trials``
        convention). Zero matches fail loud with the query and counts.
    tfr_settings : Mapping[str, object] or None
        ``config.yaml tfr`` block (``fmin``/``fmax``/``step`` required;
        ``method``/``n_cycles``/``time_bandwidth``/``decim`` optional).
        Required for the power domain, unused otherwise.

    Returns
    -------
    VariabilityResult
        Voltage: SD map ``(n_channels, n_times)``. Power: maps
        ``(n_channels, n_freqs, n_times)`` plus the ``positive_mean`` mask.

    Raises
    ------
    ValueError
        Unknown domain/measure, empty measure selection, Fano/CV requested
        for voltage, fewer than 2 trials, a query without metadata or with
        zero matches, negative/non-finite power, or missing ``tfr_settings``
        for the power domain.
    """
    names = _validated_measures(domain, measures)
    if len(epochs) == 0:
        raise ValueError(
            f"variability_epochs: received 0 epochs for domain {domain!r} — "
            "nothing to reduce"
        )
    mask = _epoch_mask(epochs, condition_query)
    n_trials = int(mask.sum())
    if n_trials == 0:
        raise ValueError(
            f"variability_epochs: condition query {condition_query!r} "
            f"matched 0 of {len(epochs)} epochs"
        )
    selected = epochs[mask]

    freqs: NDArray[np.float64] | None = None
    positive_mean: NDArray[np.bool_] | None = None
    if domain == "voltage":
        maps, _ = _reduced(_checked(selected.get_data(), domain), names)
        times = np.asarray(selected.times, dtype=np.float64)
    else:
        if not tfr_settings:
            raise ValueError(
                "variability_epochs: the power domain needs tfr_settings — "
                "the config.yaml tfr: block"
            )
        settings = dict(tfr_settings)
        grid = freq_grid(
            float(settings["fmin"]), float(settings["fmax"]), float(settings["step"])
        )
        tfr = tfr_epochs(
            selected,
            grid,
            settings.get("n_cycles"),  # type: ignore[arg-type]
            method=str(settings.get("method", "morlet")),
            time_bandwidth=float(settings.get("time_bandwidth", 4.0)),  # type: ignore[arg-type]
            decim=int(settings.get("decim", 1)),  # type: ignore[arg-type]
        )
        maps, positive_mean = _reduced(
            _checked(tfr_power_epochs(tfr), domain), names
        )
        freqs = np.asarray(tfr.freqs, dtype=np.float64)
        times = np.asarray(tfr.times, dtype=np.float64)

    return VariabilityResult(
        maps=maps,
        ch_names=tuple(selected.ch_names),
        times=times,
        freqs=freqs,
        domain=domain,
        n_trials=n_trials,
        positive_mean=positive_mean,
    )


def _check_comparable(results: Sequence[VariabilityResult]) -> None:
    """Loud comparability guards across subjects (§3 cohort handling)."""
    first = results[0]
    needs_mask = any(name in _RATIO_MEASURES for name in first.maps)
    for other in results[1:]:
        if other.domain != first.domain:
            raise ValueError(
                f"group_variability: domain mismatch — {other.domain!r} vs "
                f"{first.domain!r}"
            )
        if other.ch_names != first.ch_names:
            raise ValueError(
                f"group_variability: channel mismatch — {other.ch_names} vs "
                f"{first.ch_names}"
            )
        if tuple(other.maps) != tuple(first.maps):
            raise ValueError(
                f"group_variability: measure mismatch — {tuple(other.maps)} vs "
                f"{tuple(first.maps)}"
            )
        for name, array in first.maps.items():
            if other.maps[name].shape != array.shape:
                raise ValueError(
                    f"group_variability: map shape mismatch for {name!r} — "
                    f"{other.maps[name].shape} vs {array.shape}"
                )
        if (other.freqs is None) != (first.freqs is None):
            raise ValueError(
                "group_variability: frequency-grid mismatch — results mix "
                "the voltage domain (no freqs) and the power domain"
            )
        if first.freqs is not None and (
            other.freqs is None
            or other.freqs.size != first.freqs.size
            or not np.allclose(other.freqs, first.freqs)
        ):
            raise ValueError(
                "group_variability: subjects have different frequency grids — "
                "compute all results with identical fmin/fmax/step settings"
            )
        if other.times.size != first.times.size or not np.allclose(
            other.times, first.times
        ):
            raise ValueError(
                "group_variability: subjects have different time grids — "
                f"{other.times.size} vs {first.times.size} samples"
            )
        if first.domain == "power" and other.positive_mean is None:
            raise ValueError(
                "group_variability: power results must carry a positive_mean "
                "mask — recompute with variability_epochs"
            )
        if first.positive_mean is not None and other.positive_mean is not None:
            if other.positive_mean.shape != first.positive_mean.shape:
                raise ValueError(
                    "group_variability: positive_mean mask shape mismatch — "
                    f"{other.positive_mean.shape} vs {first.positive_mean.shape}"
                )
    if needs_mask:
        for result in results:
            if result.positive_mean is None:
                raise ValueError(
                    "group_variability: ratio measures need the positive_mean "
                    "mask on every result"
                )


def group_variability(
    results: Sequence[VariabilityResult],
) -> GroupVariabilityResult:
    """Equal-subject-weight cohort summary of per-subject variability maps.

    SD averages over **all** subjects. Fano/CV average only the subjects whose
    ``positive_mean`` is true at that cell — equal weight per contributing
    subject, never pooled trials — and record the per-cell contributor count;
    cells with zero contributors stay NaN. Note that this makes the ratio
    denominators vary across cells by design: read ``n_contributing``
    alongside the map.

    Parameters
    ----------
    results : sequence of VariabilityResult
        Per-subject results sharing domain, channel names, frequency/time
        grids, measure set, map shapes and mask shape.

    Returns
    -------
    GroupVariabilityResult
        Cohort maps plus per-cell contributor counts and the full cohort size.

    Raises
    ------
    ValueError
        Empty input, or any domain/channel/grid/measure/shape/mask mismatch.
    """
    result_list = list(results)
    if not result_list:
        raise ValueError("group_variability: received no results — nothing to aggregate")
    first = result_list[0]
    _check_comparable(result_list)

    n_subjects = len(result_list)
    maps: dict[str, NDArray[np.float64]] = {}
    n_contributing: dict[str, NDArray[np.int64]] = {}
    for name in first.maps:
        stack = np.stack([result.maps[name] for result in result_list])
        if name == "sd":
            maps[name] = stack.mean(axis=0)
            n_contributing[name] = np.full(
                stack.shape[1:], n_subjects, dtype=np.int64
            )
            continue
        valid = np.stack(
            [np.asarray(result.positive_mean, dtype=bool) for result in result_list]
        )
        counts = valid.sum(axis=0, dtype=np.int64)
        total = np.where(valid, stack, 0.0).sum(axis=0)
        maps[name] = np.divide(
            total,
            counts,
            out=np.full(stack.shape[1:], np.nan),
            where=counts > 0,
        )
        n_contributing[name] = counts

    return GroupVariabilityResult(
        maps=maps,
        n_contributing=n_contributing,
        ch_names=first.ch_names,
        times=np.asarray(first.times, dtype=np.float64),
        freqs=None if first.freqs is None else np.asarray(first.freqs, dtype=np.float64),
        domain=first.domain,
        n_subjects=n_subjects,
    )
