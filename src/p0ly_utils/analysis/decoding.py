"""Cross-validated temporal decoding of categorical ``Epochs.metadata`` targets
(US-030).

Within-subject only: every subject is fit and scored on its own trials, so
cohort summaries are equal-weight means of subject scores — trials are never
pooled across subjects (see :func:`group_decoding`).

Design (recorded in ``scrum/01_Backlog/US-030_temporal-decoding.md``):

- Every time point is a separate scikit-learn problem. A
  ``make_pipeline(StandardScaler(), LinearModel(LogisticRegression(...)))``
  pipeline is wrapped by :class:`mne.decoding.SlidingEstimator`
  (``mode="time_resolved"``, scores ``(n_times,)``) or
  :class:`mne.decoding.GeneralizingEstimator`
  (``mode="temporal_generalization"``, scores ``(n_train_times,
  n_test_times)``).
- Cross-validation is ``StratifiedKFold(n_splits, shuffle=True,
  random_state=seed)`` and scoring goes through
  :func:`mne.decoding.cross_val_multiscore`, MNE's multi-score wrapper around
  sklearn's ``cross_validate`` (plain ``cross_val_score`` rejects the per-time
  score arrays). Folds are reported un-averaged as ``fold_scores``.
- The scaler lives **inside** the pipeline, so it is cloned and fit on each
  training fold only (never on the full data) — no scaling leakage.
- Estimator families, continuous targets, and group inference are out of
  scope; unknown estimator/metric names fail loud instead of guessing.

Stage 0 probe (MNE 1.12.1, scikit-learn 1.9.0, NumPy 2.4.6) — run on a tiny
in-memory fixture (60 epochs x 3 channels x 20 times, 5 folds)::

    LinearModel(LogisticRegression())            # LinearModel(model=None)
    SlidingEstimator(pipeline, scoring=..., n_jobs=...).fit(X, y).predict(X)
        -> (60, 20)                       # one prediction per epoch/time point
    cross_val_multiscore(SlidingEstimator(...), X, y, cv=StratifiedKFold(5))
        -> (5, 20)                        # (n_splits, n_times)
    cross_val_multiscore(GeneralizingEstimator(...), X, y, cv=...)
        -> (5, 20, 20)                    # (n_splits, n_train_times, n_test_times)

MNE's per-time progress bars are silenced with ``mne.utils.use_log_level``
around the fit/scoring call; the caller's log level is restored afterwards.

Array shapes follow [[SCHEMA]] §1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mne
import numpy as np
from mne.decoding import (
    GeneralizingEstimator,
    LinearModel,
    SlidingEstimator,
    cross_val_multiscore,
)
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "DecodingResult",
    "GroupDecodingResult",
    "decode_epochs",
    "group_decoding",
]

_MODES = ("time_resolved", "temporal_generalization")
_SCORINGS = ("accuracy", "balanced_accuracy")
_ESTIMATORS = ("logistic_regression",)
_ESTIMATOR_DEFAULTS: dict[str, object] = {"C": 1.0, "max_iter": 1000, "class_weight": None}


@dataclass
class DecodingResult:
    """Within-subject cross-validated decoding scores for one target.

    Attributes
    ----------
    scores : ndarray, shape (n_times,) or (n_train_times, n_test_times)
        Fold-averaged scores. ``(n_times,)`` for ``mode="time_resolved"``,
        ``(n_train_times, n_test_times)`` for
        ``mode="temporal_generalization"``.
    fold_scores : ndarray, shape (n_splits, n_times) or (n_splits,
        n_train_times, n_test_times)
        Per-fold scores; ``scores`` is the arithmetic mean over axis 0.
    times : ndarray, shape (n_times,)
        Epoch sample grid in seconds (train and test axes share it in this
        story).
    target : str
        ``Epochs.metadata`` column that was decoded.
    classes : tuple of str
        Class labels in deterministic sorted order (numeric labels are
        stringified for persistence).
    scoring : str
        Metric name (``"accuracy"`` or ``"balanced_accuracy"``).
    chance_level : float
        ``1 / n_classes`` for both supported metrics.
    mode : str
        ``"time_resolved"`` or ``"temporal_generalization"``.
    n_splits : int
        Number of stratified CV folds.
    seed : int | None
        Seed for fold shuffling and classifier resampling.
    n_epochs : int
        Number of decoded trials (no trials are dropped).
    class_counts : ndarray, shape (n_classes,)
        Trial count per class, in ``classes`` order.
    """

    scores: NDArray[np.float64]
    fold_scores: NDArray[np.float64]
    times: NDArray[np.float64]
    target: str
    classes: tuple[str, ...]
    scoring: str
    chance_level: float
    mode: str
    n_splits: int
    seed: int | None
    n_epochs: int
    class_counts: NDArray[np.int64]


@dataclass
class GroupDecodingResult:
    """Equal-subject-weight cohort summary of :class:`DecodingResult` objects.

    Attributes
    ----------
    scores : ndarray, shape (n_times,) or (n_train_times, n_test_times)
        Arithmetic mean of the subject-level ``scores`` (one weight per
        subject, independent of trial counts).
    times : ndarray, shape (n_times,)
        Shared epoch sample grid.
    target, classes, scoring, chance_level, mode
        Configuration shared by every contributing subject.
    n_subjects : int
        Number of averaged subjects (1 is valid; no p-values are produced).
    """

    scores: NDArray[np.float64]
    times: NDArray[np.float64]
    target: str
    classes: tuple[str, ...]
    scoring: str
    chance_level: float
    mode: str
    n_subjects: int


def _check_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"unknown decoding mode {mode!r}; expected one of {_MODES}")


def _check_scoring(scoring: str) -> None:
    if scoring not in _SCORINGS:
        raise ValueError(f"unsupported scoring {scoring!r}; expected one of {_SCORINGS}")


def _labels(epochs: mne.BaseEpochs, target: str) -> NDArray[np.object_]:
    """Validate the target column and return its raw labels (no dropping)."""
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(f"epochs must be an mne.BaseEpochs, got {type(epochs).__name__}")
    if len(epochs) == 0:
        raise ValueError("cannot decode empty epochs (0 trials)")
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is None; decoding needs aligned trial metadata")
    if target not in epochs.metadata.columns:
        raise ValueError(
            f"target {target!r} not in epochs.metadata columns "
            f"{list(epochs.metadata.columns)}"
        )
    column = epochs.metadata[target]
    if column.isna().any():
        raise ValueError(f"target {target!r} has missing labels; decoding never drops trials")
    labels = column.to_numpy()
    if labels.shape[0] != len(epochs):
        raise ValueError(
            f"target {target!r} has {labels.shape[0]} labels for {len(epochs)} epochs"
        )
    if np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all():
        raise ValueError(f"target {target!r} has non-finite labels")
    return labels


def _pipeline(
    estimator: str,
    estimator_params: Mapping[str, object] | None,
    seed: int | None,
) -> object:
    """Build the unfitted scaler+classifier pipeline (folds clone it)."""
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unsupported estimator {estimator!r}; expected one of {_ESTIMATORS}")
    params = dict(_ESTIMATOR_DEFAULTS)
    unknown = sorted(set(estimator_params or {}) - set(params))
    if unknown:
        raise ValueError(
            f"unsupported estimator_params {unknown}; allowed keys are {sorted(params)}"
        )
    params.update(estimator_params or {})
    return make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(random_state=seed, **params)),
    )


def decode_epochs(
    epochs: mne.BaseEpochs,
    *,
    target: str,
    mode: str = "time_resolved",
    estimator: str = "logistic_regression",
    estimator_params: Mapping[str, object] | None = None,
    n_splits: int = 5,
    scoring: str = "accuracy",
    seed: int | None = None,
    n_jobs: int = 1,
) -> DecodingResult:
    """Decode a categorical ``Epochs.metadata`` column with within-subject CV.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data, shape ``(n_epochs, n_channels, n_times)`` (Volts), with
        ``epochs.metadata`` aligned to the trials.
    target : str
        Categorical metadata column to decode. Every distinct value is a
        class; trials with missing labels are rejected, never dropped.
    mode : {"time_resolved", "temporal_generalization"}
        ``"time_resolved"`` scores every time point independently
        (``SlidingEstimator``). ``"temporal_generalization"`` trains at every
        time point and tests at every time point
        (``GeneralizingEstimator``), scoring a square matrix.
    estimator : {"logistic_regression"}
        Classifier family. Unknown names fail loud.
    estimator_params : mapping or None
        Overrides for the allow-listed ``LogisticRegression`` keys ``C``,
        ``max_iter``, ``class_weight`` (defaults ``1.0``, ``1000``, ``None``).
        ``random_state`` is always ``seed``.
    n_splits : int
        Stratified fold count; each class needs at least this many trials.
    scoring : {"accuracy", "balanced_accuracy"}
        Metric passed to scikit-learn; chance is ``1 / n_classes`` for both.
    seed : int | None
        Seed for fold shuffling and classifier resampling (reproducible).
    n_jobs : int
        Parallelism across time points only; scores do not depend on it.

    Returns
    -------
    DecodingResult
        Fold and fold-averaged scores plus the configuration that produced
        them. Score shapes: ``(n_times,)`` / ``(n_times, n_times)`` for
        ``scores`` and ``(n_splits, ...)`` for ``fold_scores``.

    Raises
    ------
    ValueError
        Unknown target/mode/scoring/estimator/parameter, absent metadata,
        missing or non-finite labels, fewer than two classes, a class with
        fewer than ``n_splits`` trials, ``n_splits < 2``, or non-finite data.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")
    _check_mode(mode)
    _check_scoring(scoring)

    labels = _labels(epochs, target)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"target {target!r} has a single class {classes.tolist()}")
    if int(counts.min()) < n_splits:
        raise ValueError(
            f"target {target!r} cannot be split into {n_splits} stratified folds: "
            f"class counts {dict(zip((str(c) for c in classes), counts.tolist(), strict=True))}"
        )

    data = epochs.get_data(copy=False)
    if data.shape[-1] == 0:
        raise ValueError("epochs have an empty time axis")
    if not np.isfinite(data).all():
        raise ValueError("epochs contain non-finite data")

    pipeline = _pipeline(estimator, estimator_params, seed)
    mne_estimator = (GeneralizingEstimator if mode == "temporal_generalization" else SlidingEstimator)(
        pipeline, scoring=scoring, n_jobs=n_jobs
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with mne.utils.use_log_level("ERROR"):
        fold_scores = np.asarray(
            cross_val_multiscore(mne_estimator, data, labels, cv=cv), dtype=np.float64
        )

    return DecodingResult(
        scores=fold_scores.mean(axis=0),
        fold_scores=fold_scores,
        times=np.asarray(epochs.times, dtype=np.float64),
        target=target,
        classes=tuple(str(label) for label in classes),
        scoring=scoring,
        chance_level=1.0 / classes.size,
        mode=mode,
        n_splits=n_splits,
        seed=seed,
        n_epochs=len(epochs),
        class_counts=counts.astype(np.int64),
    )


def group_decoding(results: Sequence[DecodingResult]) -> GroupDecodingResult:
    """Average subject decoding scores with equal subject weights.

    Parameters
    ----------
    results : sequence of DecodingResult
        One result per subject (n >= 1). Every result must share the target,
        mode, scoring, class order, chance level, time grid, and score shape;
        mismatches fail loud instead of silently broadcasting.

    Returns
    -------
    GroupDecodingResult
        Equal-weight mean of the subject-level ``scores`` — never fold means
        weighted by trial counts, and no pooling of trials across subjects.
    """
    subjects = list(results)
    if not subjects:
        raise ValueError("no subject decoding results to aggregate")
    first = subjects[0]
    for index, result in enumerate(subjects[1:], start=2):
        for field in ("target", "mode", "scoring", "classes"):
            if getattr(result, field) != getattr(first, field):
                raise ValueError(
                    f"subject result {index} has {field}={getattr(result, field)!r}, "
                    f"expected {getattr(first, field)!r}"
                )
        if not np.isclose(result.chance_level, first.chance_level):
            raise ValueError(
                f"subject result {index} has chance_level={result.chance_level}, "
                f"expected {first.chance_level}"
            )
        if result.scores.shape != first.scores.shape:
            raise ValueError(
                f"subject result {index} has score shape {result.scores.shape}, "
                f"expected {first.scores.shape}"
            )
        if not np.array_equal(result.times, first.times):
            raise ValueError(f"subject result {index} has a different time grid")

    return GroupDecodingResult(
        scores=np.mean([result.scores for result in subjects], axis=0),
        times=first.times,
        target=first.target,
        classes=first.classes,
        scoring=first.scoring,
        chance_level=first.chance_level,
        mode=first.mode,
        n_subjects=len(subjects),
    )
