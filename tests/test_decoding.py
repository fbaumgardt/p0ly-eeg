"""Tests for p0ly_utils.analysis.decoding (US-030).

All MNE objects are simulated in-memory (``EpochsArray``) — no fixture files,
no ``@pytest.mark.slow``. The decodable fixture puts opposite class means in a
central time window on the first channel (2x the unit noise) and leaves the
rest noise; the null control reuses that data with permuted labels, so a real
decoder must beat it. Every test pins its seed, so score assertions are
deterministic rather than statistical.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info
from sklearn.preprocessing import StandardScaler

from p0ly_utils.analysis import decode_epochs, group_decoding

_SFREQ = 100.0
_TMIN = -0.1
_N_TIMES = 20
_T_START, _T_STOP = 8, 12  # central signal window
_SEED = 20261020
_EFFECT = 2.0
_NOISE = 1.0


def _epochs(
    labels: object,
    *,
    signal: bool = True,
    seed: int = 0,
    n_channels: int = 3,
    n_times: int = _N_TIMES,
) -> EpochsArray:
    """Simulated epochs: class means differ in the central window on channel 0."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=_NOISE, size=(labels.size, n_channels, n_times))
    if signal:
        first = np.unique(labels)[0]
        sign = np.where(labels == first, 1.0, -1.0)
        stop = min(_T_START, n_times)
        end = min(_T_STOP, n_times)
        data[:, 0, stop:end] += _EFFECT * sign[:, None]
    info = create_info([f"EEG{index:03d}" for index in range(n_channels)], _SFREQ, "eeg")
    return EpochsArray(data, info, tmin=_TMIN, metadata=pd.DataFrame({"Condition": labels}))


def _binary(seed: int = 0, n_per_class: int = 30) -> EpochsArray:
    return _epochs(["A", "B"] * n_per_class, seed=seed)


def _null(epochs: EpochsArray, *, seed: int = 99) -> EpochsArray:
    """Same data with permuted labels — the decoding null control."""
    labels = np.random.default_rng(seed).permutation(epochs.metadata["Condition"].to_numpy())
    shuffled = epochs.copy()
    shuffled.metadata = pd.DataFrame({"Condition": labels})
    return shuffled


def _central(scores: np.ndarray) -> float:
    return float(np.mean(scores[..., _T_START:_T_STOP]))


# --- Stage 2: modes, shapes, decodable vs null -------------------------------


def test_time_resolved_shapes_and_signal() -> None:
    epochs = _binary()
    result = decode_epochs(epochs, target="Condition", n_splits=5, seed=_SEED)
    assert result.scores.shape == (_N_TIMES,)
    assert result.fold_scores.shape == (5, _N_TIMES)
    assert result.times.shape == (_N_TIMES,)
    assert result.mode == "time_resolved"
    assert result.n_epochs == len(epochs)
    assert result.class_counts.tolist() == [30, 30]
    assert result.classes == ("A", "B")
    assert result.chance_level == pytest.approx(0.5)
    np.testing.assert_allclose(result.scores, result.fold_scores.mean(axis=0))

    null = decode_epochs(_null(epochs), target="Condition", n_splits=5, seed=_SEED)
    central = _central(result.scores)
    assert central > 0.85
    assert central > _central(null.scores) + 0.25
    assert central > float(np.mean(result.scores[: _T_START])) + 0.2
    assert central > float(np.mean(result.scores[_T_STOP:])) + 0.2


def test_temporal_generalization_shapes_and_signal() -> None:
    epochs = _binary()
    result = decode_epochs(
        epochs,
        target="Condition",
        mode="temporal_generalization",
        n_splits=5,
        seed=_SEED,
    )
    assert result.scores.shape == (_N_TIMES, _N_TIMES)
    assert result.fold_scores.shape == (5, _N_TIMES, _N_TIMES)
    css = float(np.mean(result.scores[_T_START:_T_STOP, _T_START:_T_STOP]))
    assert css > 0.85

    null = decode_epochs(
        _null(epochs),
        target="Condition",
        mode="temporal_generalization",
        n_splits=5,
        seed=_SEED,
    )
    null_css = float(np.mean(null.scores[_T_START:_T_STOP, _T_START:_T_STOP]))
    assert css > null_css + 0.25
    # Generalizes across the whole window, not only the diagonal.
    assert float(np.mean(np.diag(result.scores)[_T_START:_T_STOP])) > 0.8


def test_same_seed_is_reproducible() -> None:
    epochs = _binary()
    first = decode_epochs(epochs, target="Condition", n_splits=5, seed=_SEED)
    second = decode_epochs(epochs, target="Condition", n_splits=5, seed=_SEED)
    np.testing.assert_array_equal(first.fold_scores, second.fold_scores)
    np.testing.assert_array_equal(first.scores, second.scores)


def test_multiclass_decoding_records_chance() -> None:
    labels = ["A", "B", "C"] * 20
    result = decode_epochs(_epochs(labels, seed=3), target="Condition", n_splits=4, seed=_SEED)
    assert result.classes == ("A", "B", "C")
    assert result.class_counts.tolist() == [20, 20, 20]
    assert result.chance_level == pytest.approx(1 / 3)
    assert result.scores.shape == (_N_TIMES,)
    assert np.all(result.scores >= 0.0)


def test_numeric_labels_are_stringified_in_sorted_order() -> None:
    labels = np.array([1, 0] * 20)
    epochs = _epochs(labels)
    epochs.metadata["Condition"] = labels  # numeric categorical column
    result = decode_epochs(epochs, target="Condition", n_splits=4, seed=1)
    assert result.classes == ("0", "1")
    assert result.chance_level == pytest.approx(0.5)


def test_balanced_accuracy_scoring() -> None:
    result = decode_epochs(
        _binary(), target="Condition", scoring="balanced_accuracy", n_splits=5, seed=_SEED
    )
    assert result.scoring == "balanced_accuracy"
    assert result.chance_level == pytest.approx(0.5)
    assert _central(result.scores) > 0.8


# --- Stage 1: guards ---------------------------------------------------------


def test_unknown_target_fails() -> None:
    with pytest.raises(ValueError, match="not in epochs.metadata"):
        decode_epochs(_binary(), target="Nope", n_splits=5)


def test_missing_metadata_fails() -> None:
    epochs = _binary()
    epochs.metadata = None
    with pytest.raises(ValueError, match="metadata"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_missing_labels_fail() -> None:
    epochs = _binary()
    labels = epochs.metadata["Condition"].astype(object)
    labels.iloc[0] = None
    epochs.metadata = pd.DataFrame({"Condition": labels})
    with pytest.raises(ValueError, match="missing labels"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_non_finite_labels_fail() -> None:
    epochs = _binary()
    labels = np.zeros(len(epochs))
    labels[0] = np.inf
    epochs.metadata = pd.DataFrame({"Condition": labels})
    with pytest.raises(ValueError, match="non-finite labels"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_object_dtype_numeric_labels_fail() -> None:
    epochs = _binary()
    labels = pd.Series([0, 1] * 30, dtype=object)
    epochs.metadata = pd.DataFrame({"Condition": labels})
    with pytest.raises(ValueError, match=r"'Condition'.*object-dtype"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_mixed_label_types_fail() -> None:
    epochs = _binary()
    epochs.metadata = pd.DataFrame({"Condition": pd.Series(["A", 1] * 30, dtype=object)})
    with pytest.raises(ValueError, match=r"'Condition'.*mixed"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_single_class_fails() -> None:
    with pytest.raises(ValueError, match="single class"):
        decode_epochs(_epochs(["A"] * 30), target="Condition", n_splits=5)


def test_minority_class_smaller_than_folds_fails() -> None:
    epochs = _epochs(["A"] * 58 + ["B"] * 2)
    with pytest.raises(ValueError, match="class counts"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_invalid_fold_count_fails() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        decode_epochs(_binary(), target="Condition", n_splits=1)


def test_empty_epochs_fail() -> None:
    empty = _binary().copy()[:0]
    with pytest.raises(ValueError, match="empty epochs"):
        decode_epochs(empty, target="Condition", n_splits=2)


def test_invalid_mode_fails() -> None:
    with pytest.raises(ValueError, match="decoding mode"):
        decode_epochs(_binary(), target="Condition", mode="regression", n_splits=5)


def test_invalid_scoring_fails() -> None:
    with pytest.raises(ValueError, match="scoring"):
        decode_epochs(_binary(), target="Condition", scoring="roc_auc", n_splits=5)


def test_invalid_estimator_fails() -> None:
    with pytest.raises(ValueError, match="estimator"):
        decode_epochs(_binary(), target="Condition", estimator="svm", n_splits=5)


def test_invalid_estimator_params_fail() -> None:
    with pytest.raises(ValueError, match="estimator_params"):
        decode_epochs(
            _binary(),
            target="Condition",
            estimator_params={"gamma": 0.1},
            n_splits=5,
        )


def test_non_finite_data_fails() -> None:
    epochs = _binary()
    data = epochs.get_data()
    data[0, 0, 0] = np.nan
    epochs._data = data
    with pytest.raises(ValueError, match="non-finite data"):
        decode_epochs(epochs, target="Condition", n_splits=5)


def test_estimator_params_are_honoured() -> None:
    result = decode_epochs(
        _binary(),
        target="Condition",
        estimator_params={"C": 0.5, "max_iter": 200, "class_weight": "balanced"},
        n_splits=5,
        seed=_SEED,
    )
    assert result.scores.shape == (_N_TIMES,)


# --- Stage 2: fold isolation -------------------------------------------------


def test_scaler_fits_training_folds_only(monkeypatch: pytest.MonkeyPatch) -> None:
    epochs = _binary()
    n_epochs = len(epochs)
    calls: list[int] = []
    original = StandardScaler.fit

    def spy(self: StandardScaler, X: np.ndarray, y: np.ndarray | None = None) -> StandardScaler:
        calls.append(np.asarray(X).shape[0])
        return original(self, X, y)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    result = decode_epochs(epochs, target="Condition", n_splits=5, seed=_SEED)

    # Folds are counted separately from the repeated per-time-point fits.
    assert result.fold_scores.shape[0] == 5
    assert len(calls) == 5 * _N_TIMES
    assert set(calls) == {n_epochs - n_epochs // 5}
    assert n_epochs not in calls


# --- Stage 2: cohort aggregation --------------------------------------------


def test_group_decoding_equal_subject_weights() -> None:
    small = decode_epochs(_epochs(["A", "B"] * 10, seed=1), target="Condition", n_splits=4, seed=7)
    large = decode_epochs(_epochs(["A", "B"] * 30, seed=2), target="Condition", n_splits=4, seed=7)
    assert small.n_epochs != large.n_epochs
    group = group_decoding([small, large])
    assert group.n_subjects == 2
    assert group.target == "Condition"
    assert group.classes == ("A", "B")
    np.testing.assert_allclose(group.scores, (small.scores + large.scores) / 2)


def test_group_decoding_single_subject_is_identity() -> None:
    result = decode_epochs(_binary(), target="Condition", n_splits=5, seed=_SEED)
    group = group_decoding([result])
    assert group.n_subjects == 1
    np.testing.assert_array_equal(group.scores, result.scores)


def test_group_decoding_rejects_mismatches() -> None:
    result = decode_epochs(_binary(), target="Condition", n_splits=5, seed=_SEED)
    with pytest.raises(ValueError, match="target"):
        group_decoding([result, replace(result, target="Other")])
    with pytest.raises(ValueError, match="classes"):
        group_decoding([result, replace(result, classes=("A", "Z"))])
    with pytest.raises(ValueError, match="time grid"):
        group_decoding([result, replace(result, times=result.times + 0.1)])
    with pytest.raises(ValueError, match="score shape"):
        group_decoding([result, replace(result, scores=result.scores[:-1])])
    with pytest.raises(ValueError, match="n_splits"):
        group_decoding([result, replace(result, n_splits=3)])
    with pytest.raises(ValueError, match="seed"):
        group_decoding([result, replace(result, seed=_SEED + 1)])
    with pytest.raises(ValueError, match="no subject"):
        group_decoding([])
