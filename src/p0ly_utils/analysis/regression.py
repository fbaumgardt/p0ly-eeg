"""Single-trial OLS regression of epochs on declared metadata predictors (US-014).

Design (recorded in ``scrum/01_Backlog/US-014_single-trial-regression.md``):

- One model per (subject, timelock) fit on **all retained trials** of the
  ``mne.Epochs``: an intercept, every declared continuous column as a linear
  term, and every declared categorical column dummy-coded against its
  lexicographically first level (reference coding). Predictors are never
  standardized, so a beta carries response units per predictor unit and a
  categorical beta is a difference from the reference level adjusted for the
  other terms.
- Rows with a null/non-finite declared predictor are excluded explicitly
  (:attr:`RegressionResult.excluded_rows`, zero-based metadata positions) —
  never silently, and never independently of the response rows.
- OLS is solved once: responses are reshaped to ``(n_used, n_response_cells)``
  and one :func:`numpy.linalg.lstsq` call yields every coefficient, reshaped
  back to ``(n_terms, ...)``. Full rank and enough rows are required; a
  rank-deficient design fails loud with column names, row counts and rank.
- ``domain="voltage"`` regresses ``epochs.get_data()`` →
  ``(n_channels, n_times)`` maps per term; ``domain="power"`` regresses
  per-epoch TFR power (``tfr_epochs`` → ``tfr_power_epochs``, no log
  transform) → ``(n_channels, n_freqs, n_times)`` maps per term.
- :func:`group_regression` averages subject beta maps with equal subject
  weight (never trial-count weighted — that would be a pooled-trial estimate).

Explanatory within-subject estimation only: no inferential contrasts, no
standard errors, no decoding (those are US-030 and follow-ups). All functions
are pure: no filesystem access, no config reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from p0ly_utils.analysis.tfr import freq_grid, tfr_epochs, tfr_power_epochs

__all__ = [
    "GroupRegressionResult",
    "RegressionResult",
    "group_regression",
    "regression_epochs",
]


@dataclass
class RegressionResult:
    """OLS beta maps of one (subject, timelock) model.

    Attributes
    ----------
    beta_maps : dict of str -> ndarray
        Term name -> coefficient map. Term order is ``"intercept"``, the
        continuous predictors in declared order, then the categorical dummy
        terms in declared-column / level order (``"col[level]"``). Shapes:
        ``(n_channels, n_times)`` for ``domain="voltage"`` and
        ``(n_channels, n_freqs, n_times)`` for ``domain="power"``.
    ch_names : tuple of str
        Channel labels, length ``n_channels``; the maps' first axis follows
        this ordering.
    times : ndarray, shape (n_times,)
        Response time axis (s); the maps' last axis.
    freqs : ndarray, shape (n_freqs,) or None
        Frequency axis (Hz) for ``domain="power"``; ``None`` for voltage.
    domain : str
        ``"voltage"`` or ``"power"`` — the response the betas are in units of.
    n_input : int
        Number of metadata rows (trials) presented to the model.
    n_used : int
        Number of rows retained by the complete-case exclusion.
    n_excluded : int
        Number of rows dropped; always ``n_input - n_used``.
    excluded_rows : ndarray, shape (n_excluded,)
        Zero-based positions in the original metadata of every dropped row.
    reference_levels : dict of str -> str
        Categorical column -> omitted (reference) level.
    """

    beta_maps: dict[str, NDArray[np.float64]]
    ch_names: tuple[str, ...]
    times: NDArray[np.float64]
    freqs: NDArray[np.float64] | None
    domain: str
    n_input: int
    n_used: int
    n_excluded: int
    excluded_rows: NDArray[np.int64]
    reference_levels: dict[str, str]


@dataclass
class GroupRegressionResult:
    """Equal-subject-weight cohort beta maps.

    Attributes
    ----------
    beta_maps : dict of str -> ndarray
        Term name -> mean coefficient map over subjects (same term order and
        domain-specific shapes as the per-subject results).
    ch_names : tuple of str
        Shared channel labels, length ``n_channels``.
    times : ndarray, shape (n_times,)
        Shared response time axis (s).
    freqs : ndarray, shape (n_freqs,) or None
        Shared frequency axis (Hz) for the power domain; ``None`` otherwise.
    domain : str
        ``"voltage"`` or ``"power"``.
    reference_levels : dict of str -> str
        Shared categorical reference levels (must match across subjects).
    n_subjects : int
        Number of subject results averaged.
    n_trials_used_total : int
        Informational sum of ``n_used`` over subjects — trial counts never
        weight the average.
    """

    beta_maps: dict[str, NDArray[np.float64]]
    ch_names: tuple[str, ...]
    times: NDArray[np.float64]
    freqs: NDArray[np.float64] | None
    domain: str
    reference_levels: dict[str, str]
    n_subjects: int
    n_trials_used_total: int


def _declared_predictors(
    continuous: Sequence[str], categorical: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Validate the declared predictor lists (§3.2 rule 1)."""
    cont, cat = list(continuous), list(categorical)
    if not cont and not cat:
        raise ValueError(
            "regression_epochs: declare at least one predictor in "
            "continuous= or categorical="
        )
    for label, cols in (("continuous", cont), ("categorical", cat)):
        duplicate = sorted({col for col in cols if cols.count(col) > 1})
        if duplicate:
            raise ValueError(
                f"regression_epochs: {label} predictors declared twice: {duplicate}"
            )
    overlap = sorted(set(cont) & set(cat))
    if overlap:
        raise ValueError(
            f"regression_epochs: predictor(s) declared as both continuous and "
            f"categorical: {overlap}"
        )
    return cont, cat


def _metadata_frame(epochs: mne.BaseEpochs) -> pd.DataFrame:
    """The epochs' trial metadata, validated against the trial axis."""
    metadata = epochs.metadata
    if metadata is None or len(metadata) == 0:
        raise ValueError(
            "regression_epochs: epochs carry no trial metadata — align "
            "metadata before regressing (US-016/US-017)"
        )
    if len(metadata) != len(epochs):
        raise ValueError(
            f"regression_epochs: metadata has {len(metadata)} rows but the "
            f"epochs carry {len(epochs)} trials"
        )
    return metadata


def _continuous_matrix(
    metadata: pd.DataFrame, columns: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Numeric predictor values plus a per-row usability mask."""
    values = np.empty((len(metadata), len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        try:
            converted = pd.to_numeric(metadata[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"regression_epochs: column {column!r} is declared continuous "
                f"but holds non-numeric values ({error})"
            ) from error
        values[:, index] = converted.to_numpy(dtype=np.float64)
    return values, np.isfinite(values).all(axis=1)


def _categorical_matrix(
    metadata: pd.DataFrame, columns: Sequence[str]
) -> tuple[dict[str, NDArray[np.str_]], NDArray[np.bool_]]:
    """String predictor values plus a per-row usability mask (nulls unusable)."""
    values: dict[str, NDArray[np.str_]] = {}
    usable = np.ones(len(metadata), dtype=bool)
    for column in columns:
        series = metadata[column]
        usable &= series.notna().to_numpy()
        values[column] = series.astype(str).to_numpy(dtype=np.str_)
    return values, usable


def _design_matrix(
    continuous: Sequence[str],
    categorical: Sequence[str],
    continuous_values: NDArray[np.float64],
    categorical_values: Mapping[str, NDArray[np.str_]],
    rows: NDArray[np.int64],
) -> tuple[NDArray[np.float64], list[str], dict[str, str]]:
    """Intercept + continuous + reference-coded dummy design matrix."""
    terms = ["intercept"]
    blocks = [np.ones((rows.size, 1), dtype=np.float64)]
    for index, column in enumerate(continuous):
        blocks.append(continuous_values[rows, index : index + 1])
        terms.append(column)
    reference_levels: dict[str, str] = {}
    for column in categorical:
        column_rows = categorical_values[column][rows]
        levels = sorted(set(column_rows.tolist()))
        if len(levels) < 2:
            raise ValueError(
                f"regression_epochs: categorical predictor {column!r} has "
                f"{len(levels)} retained level(s) ({levels}) — a contrast "
                "needs at least two"
            )
        reference_levels[column] = levels[0]
        for level in levels[1:]:
            blocks.append((column_rows == level).astype(np.float64)[:, None])
            terms.append(f"{column}[{level}]")
    return np.hstack(blocks), terms, reference_levels


def _response(
    epochs: mne.BaseEpochs,
    domain: str,
    tfr_settings: Mapping[str, object] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Response array, time axis and (power only) frequency axis."""
    if domain == "voltage":
        return (
            np.asarray(epochs.get_data(), dtype=np.float64),
            np.asarray(epochs.times, dtype=np.float64),
            None,
        )
    if domain != "power":
        raise ValueError(
            f"regression_epochs: unknown signal domain {domain!r} "
            "(expected 'voltage' or 'power')"
        )
    if tfr_settings is None:
        raise ValueError(
            "regression_epochs: domain='power' needs tfr_settings "
            "(fmin/fmax/step/method/n_cycles/time_bandwidth/decim)"
        )
    missing = [key for key in ("fmin", "fmax", "step") if key not in tfr_settings]
    if missing:
        raise ValueError(
            f"regression_epochs: tfr_settings is missing {missing} — the power "
            "domain reuses the pipeline's tfr: block"
        )
    freqs = freq_grid(
        float(tfr_settings["fmin"]),  # type: ignore[arg-type]
        float(tfr_settings["fmax"]),  # type: ignore[arg-type]
        float(tfr_settings["step"]),  # type: ignore[arg-type]
    )
    n_cycles = tfr_settings.get("n_cycles")
    tfr = tfr_epochs(
        epochs,
        freqs,
        None if n_cycles is None else float(n_cycles),  # type: ignore[arg-type]
        method=str(tfr_settings.get("method", "morlet")),
        time_bandwidth=float(tfr_settings.get("time_bandwidth", 4.0)),  # type: ignore[arg-type]
        decim=int(tfr_settings.get("decim", 1)),  # type: ignore[arg-type]
    )
    return (
        np.asarray(tfr_power_epochs(tfr), dtype=np.float64),
        np.asarray(tfr.times, dtype=np.float64),
        np.asarray(tfr.freqs, dtype=np.float64),
    )


def regression_epochs(
    epochs: mne.BaseEpochs,
    *,
    continuous: Sequence[str],
    categorical: Sequence[str],
    domain: str = "voltage",
    tfr_settings: Mapping[str, object] | None = None,
) -> RegressionResult:
    """Fit one OLS model of all retained trials on the declared predictors.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs carrying aligned trial metadata (``epochs.metadata``, one row
        per trial); data shape ``(n_epochs, n_channels, n_times)``.
    continuous : sequence of str
        Metadata columns entered as linear terms (never standardized).
    categorical : sequence of str
        Metadata columns entered as reference-coded dummy terms.
    domain : str
        ``"voltage"`` (default; response is ``epochs.get_data()``) or
        ``"power"`` (response is per-epoch TFR power).
    tfr_settings : mapping or None
        Required for ``domain="power"``; the pipeline's ``tfr:`` block —
        ``fmin``, ``fmax``, ``step`` are required, ``method`` (``"morlet"``),
        ``n_cycles`` (``None`` = adaptive), ``time_bandwidth`` (4.0) and
        ``decim`` (1) default to the ``tfr_epochs`` conventions.

    Returns
    -------
    RegressionResult
        Beta maps per term — ``(n_channels, n_times)`` for voltage and
        ``(n_channels, n_freqs, n_times)`` for power — plus axes, exclusion
        bookkeeping and the categorical reference levels.

    Raises
    ------
    ValueError
        No/empty/mismatched metadata, no declared predictor, duplicate or
        unknown predictor columns, non-numeric continuous values, null or
        non-finite predictors (excluded — so this never raises for those),
        a categorical column with fewer than two retained levels,
        non-finite responses, an unknown domain, too few rows or a
        rank-deficient design (US-014 §3.2).
    """
    cont, cat = _declared_predictors(continuous, categorical)
    metadata = _metadata_frame(epochs)
    unknown = [column for column in [*cont, *cat] if column not in metadata.columns]
    if unknown:
        raise ValueError(
            f"regression_epochs: declared predictor(s) {unknown} not in the "
            f"metadata columns {list(metadata.columns)}"
        )

    continuous_values, continuous_usable = _continuous_matrix(metadata, cont)
    categorical_values, categorical_usable = _categorical_matrix(metadata, cat)
    used_mask = continuous_usable & categorical_usable
    rows = np.flatnonzero(used_mask).astype(np.int64)
    n_input, n_used = len(metadata), int(rows.size)

    design, terms, reference_levels = _design_matrix(
        cont, cat, continuous_values, categorical_values, rows
    )
    n_terms = design.shape[1]
    if n_used < n_terms:
        raise ValueError(
            f"regression_epochs: only {n_used} usable row(s) for {n_terms} "
            f"design column(s) {terms} (n_input={n_input}, "
            f"n_excluded={n_input - n_used}) — need n_used >= n_design_columns"
        )
    rank = int(np.linalg.matrix_rank(design))
    if rank != n_terms:
        raise ValueError(
            f"regression_epochs: rank-deficient design — rank {rank} < "
            f"{n_terms} columns {terms} (n_input={n_input}, n_used={n_used}); "
            "a constant or duplicated predictor cannot be estimated"
        )

    response, times, freqs = _response(epochs, domain, tfr_settings)
    if response.shape[0] != n_input:
        raise ValueError(
            f"regression_epochs: response carries {response.shape[0]} trial(s) "
            f"but metadata has {n_input} row(s)"
        )
    used_response = response[rows]
    if not np.isfinite(used_response).all():
        raise ValueError(
            "regression_epochs: response data contains non-finite values in "
            "retained trial(s) — reject or interpolate them upstream"
        )

    coefficients, *_ = np.linalg.lstsq(
        design, used_response.reshape(n_used, -1), rcond=None
    )
    maps = coefficients.reshape((n_terms, *response.shape[1:]))
    return RegressionResult(
        beta_maps={
            term: np.asarray(maps[index], dtype=np.float64)
            for index, term in enumerate(terms)
        },
        ch_names=tuple(epochs.ch_names),
        times=times,
        freqs=freqs,
        domain=domain,
        n_input=n_input,
        n_used=n_used,
        n_excluded=n_input - n_used,
        excluded_rows=np.flatnonzero(~used_mask).astype(np.int64),
        reference_levels=reference_levels,
    )


def group_regression(results: Sequence[RegressionResult]) -> GroupRegressionResult:
    """Equal-subject-weight mean of per-subject beta maps.

    Parameters
    ----------
    results : sequence of RegressionResult
        Per-subject (per-timelock) results sharing domain, term order,
        reference levels, channel set and response grid.

    Returns
    -------
    GroupRegressionResult
        Arithmetic subject mean per term (equal weight — a subject with 10
        trials counts as much as one with 100; weighting by ``n_used`` would
        be a pooled-trial estimate, which this story excludes) plus the
        shared axes and the total retained trial count as information only.

    Raises
    ------
    ValueError
        No input, or any domain / term-order / term-set / reference-level /
        channel / time-grid / frequency-grid / beta-shape mismatch
        (US-024 fail-loud).
    """
    items = list(results)
    if not items:
        raise ValueError("group_regression: received no results — nothing to average")
    first = items[0]
    terms = list(first.beta_maps)
    for other in items[1:]:
        if other.domain != first.domain:
            raise ValueError(
                f"group_regression: signal domain mismatch — {other.domain!r} "
                f"vs {first.domain!r}"
            )
        if list(other.beta_maps) != terms:
            raise ValueError(
                f"group_regression: term order mismatch — {list(other.beta_maps)} "
                f"vs {terms}"
            )
        if other.reference_levels != first.reference_levels:
            raise ValueError(
                f"group_regression: categorical reference-level mismatch — "
                f"{other.reference_levels} vs {first.reference_levels}"
            )
        if other.ch_names != first.ch_names:
            raise ValueError(
                f"group_regression: channel-set mismatch — {other.ch_names} "
                f"vs {first.ch_names}"
            )
        if other.times.shape != first.times.shape or not np.allclose(
            other.times, first.times
        ):
            raise ValueError(
                "group_regression: time-grid mismatch — compute all subject "
                "epochs with identical tmin/tmax/baseline"
            )
        if (other.freqs is None) != (first.freqs is None):
            raise ValueError(
                "group_regression: frequency-axis mismatch — one subject has a "
                "frequency grid and another does not"
            )
        if first.freqs is not None and (
            other.freqs.shape != first.freqs.shape
            or not np.allclose(other.freqs, first.freqs)
        ):
            raise ValueError(
                "group_regression: frequency-grid mismatch — compute all "
                "power regressions with identical fmin/fmax/step"
            )
        for term in terms:
            if other.beta_maps[term].shape != first.beta_maps[term].shape:
                raise ValueError(
                    f"group_regression: beta-shape mismatch for term {term!r} — "
                    f"{other.beta_maps[term].shape} vs "
                    f"{first.beta_maps[term].shape}"
                )

    return GroupRegressionResult(
        beta_maps={
            term: np.asarray(
                np.stack([item.beta_maps[term] for item in items], axis=0).mean(
                    axis=0
                ),
                dtype=np.float64,
            )
            for term in terms
        },
        ch_names=first.ch_names,
        times=first.times.copy(),
        freqs=None if first.freqs is None else first.freqs.copy(),
        domain=first.domain,
        reference_levels=dict(first.reference_levels),
        n_subjects=len(items),
        n_trials_used_total=sum(item.n_used for item in items),
    )
