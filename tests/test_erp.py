"""Tests for p0ly_utils.analysis.erp (US-010).

All MNE objects are simulated in-memory (EpochsArray / EvokedArray) per the
US-010 AC "unit tests with simulated evoked data" — no fixture files, no
``@pytest.mark.slow``. Cluster permutation tests use small n_permutations and
a fixed seed; with 6 subjects the sign-flip null has 2 ** 5 = 32 unique
permutations, so p-values are exact multiples of 1/32 and the minimum
achievable p is 0.03125.

Simulated effect layout: a 3-channel (Fz/Cz/Pz = indices 2:5) x 15-sample
(10:25) amplitude blob added to 'cond_a' difference waves. The significant
cluster must intersect those blob channels within a few samples of the blob
window, which pins the ``(channel_indices, time_indices)`` cluster tuple
orientation returned by the library.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, EvokedArray, create_info
from scipy import sparse

from p0ly_utils.analysis import (
    ClusterPermStats,
    GroupERPResult,
    average_epochs,
    cluster_permutation_1samp,
    eeg_adjacency,
    evoked_difference,
    grand_average,
    group_erp,
)

_SFREQ = 100.0
_TMIN = -0.2
_N_TIMES = 40
_N_CHANNELS = 8
_CH_NAMES = ("Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "C3", "C4")
_BLOB_CH_START, _BLOB_CH_STOP = 2, 5  # Fz, Cz, Pz
_BLOB_T_START, _BLOB_T_STOP = 10, 25
_EFFECT = 2e-6  # V
_SIGMA = 5e-7  # V
_N_SUBJECTS = 6
_N_PERMS = 128
_SEED = 20261020


@pytest.fixture(scope="module")
def info() -> mne.Info:
    """8 EEG channels on standard-1020 positions for montage-derived adjacency."""
    built = create_info(list(_CH_NAMES), _SFREQ, "eeg")
    built.set_montage("standard_1020")
    return built


def _eog_info() -> mne.Info:
    built = create_info([*_CH_NAMES, "EOG"], _SFREQ, ["eeg"] * _N_CHANNELS + ["eog"])
    built.set_montage("standard_1020", on_missing="ignore")
    return built


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _make_epochs(info: mne.Info, rng: np.random.Generator) -> mne.Epochs:
    """12 simulated epochs with Correct/RT metadata."""
    n_epochs = 12
    data = rng.normal(0.0, _SIGMA, (n_epochs, _N_CHANNELS, _N_TIMES))
    metadata = pd.DataFrame(
        {
            "Correct": [True, False] * 6,
            "RT": np.linspace(200.0, 800.0, n_epochs),
        }
    )
    events = np.column_stack(
        [
            np.arange(n_epochs) * 100,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    ).astype(int)
    return EpochsArray(data, info, events, tmin=_TMIN, metadata=metadata, verbose="ERROR")


def _make_epochs_without_metadata(info: mne.Info, rng: np.random.Generator) -> mne.Epochs:
    n_epochs = 6
    data = rng.normal(0.0, _SIGMA, (n_epochs, _N_CHANNELS, _N_TIMES))
    events = np.column_stack(
        [
            np.arange(n_epochs) * 100,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    ).astype(int)
    return EpochsArray(data, info, events, tmin=_TMIN, verbose="ERROR")


def _sim_diff_waves(
    rng: np.random.Generator, *, effect: bool, n_subjects: int = _N_SUBJECTS
) -> np.ndarray:
    """Per-subject difference waves, shape (n_subjects, n_channels, n_times)."""
    X = rng.normal(0.0, np.sqrt(2.0) * _SIGMA, (n_subjects, _N_CHANNELS, _N_TIMES))
    if effect:
        X[:, _BLOB_CH_START:_BLOB_CH_STOP, _BLOB_T_START:_BLOB_T_STOP] += _EFFECT
    return X


def _sim_subject_evokeds(
    rng: np.random.Generator,
    info: mne.Info,
    *,
    effect: bool = True,
    with_eog: bool = False,
    n_subjects: int = _N_SUBJECTS,
) -> list[dict[str, mne.Evoked]]:
    """Per-subject ``{condition: evoked}`` mappings; blob effect on 'cond_a'.

    ``info`` is unused when ``with_eog=True`` (an EOG-carrying info is
    built internally); it is always passed so callers stay uniform.
    """
    built_info = _eog_info() if with_eog else info
    n_channels = len(built_info.ch_names)
    subjects: list[dict[str, mne.Evoked]] = []
    for subject in range(n_subjects):
        data_a = rng.normal(0.0, _SIGMA, (n_channels, _N_TIMES))
        data_b = rng.normal(0.0, _SIGMA, (n_channels, _N_TIMES))
        if effect:
            data_a[_BLOB_CH_START:_BLOB_CH_STOP, _BLOB_T_START:_BLOB_T_STOP] += _EFFECT
        base = f"sub-{subject:02d}"
        subjects.append(
            {
                "cond_a": EvokedArray(
                    data_a, built_info, tmin=_TMIN, nave=30, comment=f"{base}_cond_a"
                ),
                "cond_b": EvokedArray(
                    data_b, built_info, tmin=_TMIN, nave=30, comment=f"{base}_cond_b"
                ),
            }
        )
    return subjects


class TestAverageEpochs:
    def test_average_all_epochs(self, info: mne.Info, rng: np.random.Generator) -> None:
        epochs = _make_epochs(info, rng)
        evoked = average_epochs(epochs)
        assert evoked.data.shape == (_N_CHANNELS, _N_TIMES)
        assert evoked.nave == len(epochs)
        assert np.allclose(evoked.data, epochs.get_data().mean(axis=0))

    def test_condition_query_filters(self, info: mne.Info, rng: np.random.Generator) -> None:
        epochs = _make_epochs(info, rng)
        evoked = average_epochs(epochs, "Correct == True")
        selected = epochs.metadata["Correct"].to_numpy(dtype=bool)
        assert evoked.nave == int(selected.sum())
        assert np.allclose(evoked.data, epochs.get_data()[selected].mean(axis=0))

    def test_condition_query_zero_matches_raises(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        epochs = _make_epochs(info, rng)
        with pytest.raises(ValueError, match="matched 0 of 12"):
            average_epochs(epochs, "RT > 1000")

    def test_query_without_metadata_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        epochs = _make_epochs_without_metadata(info, rng)
        with pytest.raises(ValueError, match="requires epochs.metadata"):
            average_epochs(epochs, "Correct == True")

    def test_empty_epochs_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        epochs = _make_epochs(info, rng)
        empty = epochs[np.zeros(len(epochs), dtype=bool)]
        assert len(empty) == 0
        with pytest.raises(ValueError, match="received 0 epochs"):
            average_epochs(empty)


class TestGrandAverage:
    def test_equal_weight_mean(self, info: mne.Info, rng: np.random.Generator) -> None:
        evokeds = [
            EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
            for _ in range(3)
        ]
        result = grand_average(evokeds)
        assert np.allclose(result.data, np.mean([ev.data for ev in evokeds], axis=0))
        assert result.nave == pytest.approx(3.0)
        assert result.ch_names == list(_CH_NAMES)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="no evokeds"):
            grand_average([])

    def test_channel_mismatch_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        a = EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
        odd_info = create_info(["Fz", "Cz", "Pz", "Oz"], _SFREQ, "eeg")
        b = EvokedArray(rng.normal(size=(4, _N_TIMES)), odd_info, tmin=_TMIN)
        with pytest.raises(ValueError, match="channel mismatch"):
            grand_average([a, b])

    def test_time_grid_mismatch_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        a = EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
        b = EvokedArray(rng.normal(size=(_N_CHANNELS, 30)), info, tmin=_TMIN)
        with pytest.raises(ValueError, match="time grid mismatch"):
            grand_average([a, b])


class TestEvokedDifference:
    def test_difference_data(self, info: mne.Info, rng: np.random.Generator) -> None:
        a = EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
        b = EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
        diff = evoked_difference(a, b)
        assert np.allclose(diff.data, a.data - b.data)
        assert diff.data.shape == (_N_CHANNELS, _N_TIMES)

    def test_mismatch_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        a = EvokedArray(rng.normal(size=(_N_CHANNELS, _N_TIMES)), info, tmin=_TMIN)
        b = EvokedArray(rng.normal(size=(_N_CHANNELS, 30)), info, tmin=_TMIN)
        with pytest.raises(ValueError, match="time grid mismatch"):
            evoked_difference(a, b)


class TestEEGAdjacency:
    def test_shape_and_symmetry(self, info: mne.Info) -> None:
        adjacency = eeg_adjacency(info)
        assert sparse.issparse(adjacency)
        assert adjacency.shape == (_N_CHANNELS, _N_CHANNELS)
        assert adjacency.nnz > 0
        assert (adjacency != adjacency.T).nnz == 0

    def test_non_eeg_excluded(self) -> None:
        adjacency = eeg_adjacency(_eog_info())
        assert adjacency.shape == (_N_CHANNELS, _N_CHANNELS)


class TestClusterPermutation1samp:
    def test_detects_embedded_effect(self, info: mne.Info, rng: np.random.Generator) -> None:
        stats = cluster_permutation_1samp(
            _sim_diff_waves(rng, effect=True),
            eeg_adjacency(info),
            n_permutations=_N_PERMS,
            seed=_SEED,
        )
        assert isinstance(stats, ClusterPermStats)
        assert stats.t_obs.shape == (_N_CHANNELS, _N_TIMES)
        assert 0 < stats.h0.size <= _N_PERMS
        assert stats.n_significant() >= 1
        sig_ch, sig_t = stats.clusters[int(np.argmin(stats.cluster_pvals))]
        # cluster tuple is (channel_indices, time_indices): channels hit the
        # blob channels, times stay within a few samples of the blob window
        blob_channels = set(range(_BLOB_CH_START, _BLOB_CH_STOP))
        assert set(sig_ch.tolist()) & blob_channels
        assert sig_t.min() >= _BLOB_T_START - 3
        assert sig_t.max() <= _BLOB_T_STOP + 3
        assert sig_ch.max() < _N_CHANNELS
        assert sig_t.max() < _N_TIMES

    def test_null_data_has_no_significant_clusters(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        stats = cluster_permutation_1samp(
            _sim_diff_waves(rng, effect=False),
            eeg_adjacency(info),
            n_permutations=_N_PERMS,
            seed=_SEED,
        )
        assert stats.n_significant() == 0
        assert len(stats.clusters) == len(stats.cluster_pvals)

    def test_single_subject_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="needs >= 2 subjects"):
            cluster_permutation_1samp(
                _sim_diff_waves(rng, effect=False, n_subjects=1), eeg_adjacency(info)
            )

    def test_2d_input_raises(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            cluster_permutation_1samp(rng.normal(size=(_N_CHANNELS, _N_TIMES)), None)

    def test_adjacency_shape_mismatch_raises(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        four = create_info(["Fz", "Cz", "Pz", "Oz"], _SFREQ, "eeg")
        four.set_montage("standard_1020")
        with pytest.raises(ValueError, match="does not match"):
            cluster_permutation_1samp(_sim_diff_waves(rng, effect=False), eeg_adjacency(four))

    def test_runs_without_adjacency(self, rng: np.random.Generator) -> None:
        stats = cluster_permutation_1samp(
            _sim_diff_waves(rng, effect=False), None, n_permutations=_N_PERMS, seed=_SEED
        )
        assert stats.t_obs.shape == (_N_CHANNELS, _N_TIMES)


class TestGroupERP:
    def test_averages_and_contrast(self, info: mne.Info, rng: np.random.Generator) -> None:
        subjects = _sim_subject_evokeds(rng, info, effect=True)
        result = group_erp(
            subjects,
            [("cond_a", "cond_b")],
            n_permutations=_N_PERMS,
            seed=_SEED,
        )
        assert isinstance(result, GroupERPResult)
        assert result.n_subjects == _N_SUBJECTS
        assert result.ch_names == _CH_NAMES
        assert set(result.averages) == {"cond_a", "cond_b"}
        expected_a = np.mean([s["cond_a"].data for s in subjects], axis=0)
        assert np.allclose(result.averages["cond_a"].data, expected_a)
        stats = result.contrasts[("cond_a", "cond_b")]
        assert stats.t_obs.shape == (_N_CHANNELS, _N_TIMES)
        assert stats.n_significant() >= 1
        # difference direction: effect was added to cond_a, so blob t is positive
        assert stats.t_obs[_BLOB_CH_START:_BLOB_CH_STOP].mean() > 0.0

    def test_no_pairs_averages_only(self, info: mne.Info, rng: np.random.Generator) -> None:
        result_single = group_erp(_sim_subject_evokeds(rng, info, n_subjects=1))
        assert list(result_single.averages) == ["cond_a", "cond_b"]
        assert result_single.contrasts == {}
        assert result_single.n_subjects == 1

    def test_single_subject_with_pairs_raises(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        subjects = _sim_subject_evokeds(rng, info, n_subjects=1)
        with pytest.raises(ValueError, match="need >= 2 subjects"):
            group_erp(subjects, [("cond_a", "cond_b")])

    def test_unknown_contrast_condition_raises(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        subjects = _sim_subject_evokeds(rng, info)
        with pytest.raises(ValueError, match="not among the declared conditions"):
            group_erp(subjects, [("cond_a", "cond_z")])

    def test_self_pair_raises(self, info: mne.Info, rng: np.random.Generator) -> None:
        subjects = _sim_subject_evokeds(rng, info)
        with pytest.raises(ValueError, match="contrasts a condition with itself"):
            group_erp(subjects, [("cond_a", "cond_a")])

    def test_condition_key_set_mismatch_raises(
        self, info: mne.Info, rng: np.random.Generator
    ) -> None:
        subjects = _sim_subject_evokeds(rng, info)
        broken = [{"cond_a": subjects[0]["cond_a"]}] + subjects[1:]
        with pytest.raises(ValueError, match="declares conditions"):
            group_erp(broken)

    def test_non_eeg_channels_excluded(self, info: mne.Info, rng: np.random.Generator) -> None:
        subjects = _sim_subject_evokeds(rng, info, effect=True, with_eog=True)
        result = group_erp(subjects, [("cond_a", "cond_b")], n_permutations=_N_PERMS, seed=_SEED)
        assert result.ch_names == _CH_NAMES  # EOG dropped
        assert set(result.averages["cond_a"].ch_names) == set(_CH_NAMES)
        assert result.contrasts[("cond_a", "cond_b")].n_significant() >= 1

    def test_no_subjects_raises(self) -> None:
        with pytest.raises(ValueError, match="received no subjects"):
            group_erp([])
