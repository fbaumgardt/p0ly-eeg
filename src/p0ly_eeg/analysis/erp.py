"""ERP analysis: subject averages, group grand averages, and cluster-based
permutation statistics (US-010).

Design (recorded in ``scrum/01_Backlog/US-010_erp-analysis.md``, locked
2026-10-20):

- Condition comparison at group level is treated as a **paired design**:
  per-subject difference waves ``d_i = cond_a_i - cond_b_i`` are stacked into
  ``X (n_subjects, n_channels, n_times)`` and tested against zero with a
  one-sample cluster permutation test
  (:func:`mne.stats.spatio_temporal_cluster_1samp_test`).
- MNE's cluster API (1.12) expects the vertex (channel) axis **last**:
  ``X (n_observations, n_times, n_vertices)``. :func:`cluster_permutation_1samp`
  accepts the sensor convention ``(n_subjects, n_channels, n_times)`` and
  transposes in/out, so results keep the SCHEMA ``(n_channels, n_times)``
  orientation and cluster index tuples are ``(channel_indices, time_indices)``.
- Grand averages are equal-weight arithmetic means across subjects
  (``mne.combine_evoked(weights="equal")``), so every subject contributes
  equally regardless of trial counts.
- All functions are pure: no filesystem access, no I/O, no config reads.

Array shapes follow [[SCHEMA]] §1. Sensor-space channel adjacency comes from
:func:`mne.channels.find_ch_adjacency`; it is computed from channel positions
(Delaunay triangulation when the montage has no built-in adjacency table, which
emits an MNE INFO message once per call).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mne
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # scipy arrives transitively via mne; only needed for typing
    from scipy.sparse import sparray, spmatrix

__all__ = [
    "ClusterPermStats",
    "GroupERPResult",
    "average_epochs",
    "cluster_permutation_1samp",
    "eeg_adjacency",
    "evoked_difference",
    "grand_average",
    "group_erp",
]


@dataclass
class ClusterPermStats:
    """Result of one cluster-based permutation test.

    Attributes
    ----------
    t_obs : ndarray, shape (n_channels, n_times)
        Observed t-statistic per channel-time point. The TFR variant
        (:func:`p0ly_eeg.analysis.tfr_cluster_permutation`) widens this to
        ``(n_channels, n_freqs, n_times)``.
    clusters : list of tuple of index arrays
        One tuple of index arrays per suprathreshold cluster
        (``out_type="indices"`` from the MNE API, reordered channel-first).
        ERP: ``(channel_indices, time_indices)``; TFR: ``(channel_indices,
        freq_indices, time_indices)``.
    cluster_pvals : ndarray, shape (n_clusters,)
        Permutation p-value for each cluster (same order as ``clusters``).
    h0 : ndarray, shape (n_max_stats,)
        Maximum cluster statistics under the sign-flip null. Length equals the
        number of unique permutations actually evaluated — for a one-sample
        test with ``n_subjects`` that is ``2 ** (n_subjects - 1)`` when small,
        which can be below ``n_permutations`` and quantizes p-values at their
        minimum achievable resolution ``1 / h0.size``.
    n_permutations : int
        Requested permutation count (MNE may use fewer unique sign-flips).
    tail : int
        Test tail: 0 (two-tailed), 1, or -1.
    threshold : float | None
        Cluster-forming threshold; ``None`` means MNE's automatic t-threshold
        for p < 0.05 with df = n_subjects - 1.
    seed : int | None
        Random seed for permutation draw reproducibility.

    Notes
    -----
    Requires ``n_subjects >= 2``: a one-sample t-test over a single subject's
    difference wave has no within-group variance and no meaningful null.
    """

    t_obs: NDArray[np.float64]
    clusters: list[tuple[NDArray[np.intp], ...]]
    cluster_pvals: NDArray[np.float64]
    h0: NDArray[np.float64]
    n_permutations: int
    tail: int
    threshold: float | None
    seed: int | None

    def n_significant(self, p_threshold: float = 0.05) -> int:
        """Number of clusters with ``cluster_pvals < p_threshold``."""
        return int((self.cluster_pvals < p_threshold).sum())


@dataclass
class GroupERPResult:
    """Group-level ERP output for one timelock (US-010 pipeline contract).

    Attributes
    ----------
    averages : dict[str, mne.Evoked]
        Condition name -> equal-weight grand average across subjects,
        shape ``(n_channels, n_times)`` per Evoked.
    contrasts : dict[tuple[str, str], ClusterPermStats]
        ``(cond_a, cond_b)`` -> cluster permutation results for the paired
        difference wave ``cond_a - cond_b`` (same key order as requested).
    n_subjects : int
        Number of subjects contributing.
    ch_names : tuple[str, ...]
        EEG channel names the analysis ran on (non-EEG channels are dropped).
    """

    averages: dict[str, mne.Evoked]
    contrasts: dict[tuple[str, str], ClusterPermStats]
    n_subjects: int
    ch_names: tuple[str, ...]


def _pick_eeg(evoked: mne.Evoked) -> mne.Evoked:
    """Return an EEG-only copy of ``evoked`` (the input is not modified).

    Raises
    ------
    ValueError
        If the evoked carries no EEG channels.
    """
    picked = evoked.copy().pick("eeg")
    if len(picked.ch_names) == 0:
        raise ValueError(
            f"ERP analysis requires EEG channels, but {evoked.comment!r} has none "
            f"(channels present: {evoked.ch_names})"
        )
    return picked


def _validate_consistent_grid(evokeds: Sequence[mne.Evoked], *, label: str) -> None:
    """Raise :class:`ValueError` unless all evokeds share channels and time grid.

    Parameters
    ----------
    evokeds : sequence of mne.Evoked
        Evokeds to cross-check.
    label : str
        Caller name used in error messages.
    """
    if len(evokeds) == 0:
        raise ValueError(f"{label}: received no evokeds")
    first = evokeds[0]
    for evoked in evokeds[1:]:
        if evoked.ch_names != first.ch_names:
            raise ValueError(
                f"{label}: channel mismatch — {evoked.comment!r} has {evoked.ch_names}, "
                f"expected {first.ch_names}"
            )
        if evoked.data.shape != first.data.shape or not np.allclose(evoked.times, first.times):
            raise ValueError(
                f"{label}: time grid mismatch — {evoked.comment!r} spans "
                f"{len(evoked.times)} samples from {evoked.times[0]:.3f} s, expected "
                f"{len(first.times)} from {first.times[0]:.3f} s"
            )


def average_epochs(
    epochs: mne.BaseEpochs,
    condition_query: str | None = None,
) -> mne.Evoked:
    """Average epochs into a subject-level evoked response.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Input epochs, shape ``(n_epochs, n_channels, n_times)``; optionally
        carrying ``epochs.metadata`` for condition selection.
    condition_query : str | None
        Pandas ``DataFrame.eval`` query string applied to ``epochs.metadata``
        (e.g. ``"Correct == True"`` from ``config.yaml``
        ``group_aggregation.condition_splits``). ``None`` averages all epochs.

    Returns
    -------
    mne.Evoked
        Subject average, shape ``(n_channels, n_times)``; ``nave`` equals the
        number of averaged epochs.

    Raises
    ------
    ValueError
        Fail-loud (US-024 philosophy): empty epochs, a query applied to epochs
        without metadata, or a query matching zero epochs.
    """
    if len(epochs) == 0:
        raise ValueError("average_epochs: received 0 epochs")
    if condition_query is not None:
        if epochs.metadata is None:
            raise ValueError(
                f"average_epochs: condition query {condition_query!r} requires "
                "epochs.metadata, but the epochs carry none"
            )
        mask = epochs.metadata.eval(condition_query)
        if int(mask.sum()) == 0:
            raise ValueError(
                f"average_epochs: condition query {condition_query!r} matched 0 of "
                f"{len(epochs)} epochs"
            )
        epochs = epochs[mask.to_numpy(dtype=bool)]
    return epochs.average()


def grand_average(evokeds: Sequence[mne.Evoked]) -> mne.Evoked:
    """Equal-weight grand average across subjects.

    Parameters
    ----------
    evokeds : sequence of mne.Evoked
        One evoked per subject, each shape ``(n_channels, n_times)``. All must
        share identical channel lists and time grids.

    Returns
    -------
    mne.Evoked
        Arithmetic mean across subjects, ``(n_channels, n_times)``. Weights are
        equal (not nave-weighted), so a subject with 10 trials counts as much
        as one with 100; the result ``nave`` is the total trial count.
    """
    evoked_list = list(evokeds)
    _validate_consistent_grid(evoked_list, label="grand_average")
    return mne.combine_evoked(evoked_list, weights="equal")


def evoked_difference(a: mne.Evoked, b: mne.Evoked) -> mne.Evoked:
    """Point-wise difference wave ``a - b``.

    Parameters
    ----------
    a, b : mne.Evoked
        Same grid: shape ``(n_channels, n_times)`` each.

    Returns
    -------
    mne.Evoked
        Difference wave, ``(n_channels, n_times)``. ``nave`` is not meaningful
        for a mixed-sign combination and should not be interpreted.
    """
    _validate_consistent_grid([a, b], label="evoked_difference")
    return mne.combine_evoked([a, b], weights=[1, -1])


def eeg_adjacency(info: mne.Info) -> spmatrix | sparray:
    """Sensor adjacency among the EEG channels of ``info``.

    Wraps :func:`mne.channels.find_ch_adjacency(info, "eeg")`; adjacency order
    matches :meth:`mne.Evoked.pick("eeg")` channel order, so it pairs directly
    with data stacked from EEG-picked evokeds.

    Parameters
    ----------
    info : mne.Info
        Any info carrying EEG channel positions (e.g. from an Evoked).

    Returns
    -------
    scipy.sparse matrix or array, shape (n_channels, n_channels)
        Symmetric binary sensor adjacency.
    """
    adjacency, _ch_names = mne.channels.find_ch_adjacency(info, "eeg")
    return adjacency


def cluster_permutation_1samp(
    X: NDArray[np.float64],
    adjacency: spmatrix | sparray | None = None,
    *,
    n_permutations: int = 1024,
    tail: int = 0,
    seed: int | None = None,
) -> ClusterPermStats:
    """One-sample cluster permutation test over subject difference waves.

    Wraps :func:`mne.stats.spatio_temporal_cluster_1samp_test` with the
    sensor-convention API boundary: input ``X`` follows the SCHEMA layout
    ``(n_subjects, n_channels, n_times)``; MNE's vertices-last convention
    ``(n_subjects, n_times, n_channels)`` is handled internally.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_channels, n_times)
        Per-subject difference waves (e.g. condition A evoked minus condition
        B evoked), tested against zero.
    adjacency : sparse matrix/array, shape (n_channels, n_channels) | None
        Sensor adjacency from :func:`eeg_adjacency`. ``None`` falls back to
        MNE's regular-lattice adjacency over channel index order, which is not
        position-based — prefer an explicit montage-derived adjacency.
    n_permutations : int
        Maximum sign-flip permutations. With few subjects the unique-permutation
        count ``2 ** (n_subjects - 1)`` bounds the null (see ClusterPermStats).
    tail : int
        0 (two-tailed, default), 1, or -1.
    seed : int | None
        Seed for reproducible permutation draws.

    Returns
    -------
    ClusterPermStats
        Statistics with ``(n_channels, n_times)`` orientation; the cluster
        index tuples are ``(channel_indices, time_indices)``.

    Raises
    ------
    ValueError
        If ``X`` is not 3-D, has fewer than 2 subjects, or the adjacency shape
        does not match the channel count.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(
            f"cluster_permutation_1samp: X must be 3-D (n_subjects, n_channels, "
            f"n_times), got shape {X.shape}"
        )
    n_subjects, n_channels, _n_times = X.shape
    if n_subjects < 2:
        raise ValueError(
            f"cluster_permutation_1samp: needs >= 2 subjects for a group test, "
            f"got {n_subjects}. Grand averages remain valid for a single subject."
        )
    if adjacency is not None and tuple(adjacency.shape) != (n_channels, n_channels):
        raise ValueError(
            f"cluster_permutation_1samp: adjacency shape {adjacency.shape} does not "
            f"match X channel count ({n_channels},)"
        )
    t_obs_st, clusters_st, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        np.transpose(X, (0, 2, 1)),
        threshold=None,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        seed=seed,
        out_type="indices",
    )
    clusters = [(np.asarray(ch_idx), np.asarray(time_idx)) for time_idx, ch_idx in clusters_st]
    return ClusterPermStats(
        t_obs=np.asarray(t_obs_st).T,
        clusters=clusters,
        cluster_pvals=np.asarray(cluster_pv, dtype=np.float64),
        h0=np.asarray(h0, dtype=np.float64),
        n_permutations=n_permutations,
        tail=tail,
        threshold=None,
        seed=seed,
    )


def group_erp(
    subject_evokeds: Sequence[Mapping[str, mne.Evoked]],
    condition_pairs: Sequence[tuple[str, str]] = (),
    *,
    adjacency: spmatrix | sparray | None = None,
    n_permutations: int = 1024,
    tail: int = 0,
    seed: int | None = None,
) -> GroupERPResult:
    """Group-level ERP for one timelock: grand averages + pairwise contrasts.

    This is the entry point the p0ly-flow ``run_erp`` rule calls. It never
    touches the filesystem; the rule script owns reading per-condition evoked
    files into the ``subject_evokeds`` mappings and writing the outputs.

    Parameters
    ----------
    subject_evokeds : sequence of mapping str -> mne.Evoked
        One mapping per subject: condition name -> that subject's condition
        evoked (from :func:`average_epochs`). Every subject must declare the
        same condition key set — differences fail loud. All evokeds must share
        an EEG channel grid; non-EEG channels are dropped before averaging and
        statistics so adjacency and data rows stay aligned.
    condition_pairs : sequence of (cond_a, cond_b), default ()
        Contrasts to test; the per-subject difference wave is ``cond_a -
        cond_b``. Requires >= 2 subjects (single-subject runs may only request
        grand averages).
    adjacency : sparse matrix/array, shape (n_channels, n_channels) | None
        Sensor adjacency; built from the first evoked's info when omitted.
    n_permutations, tail, seed
        Passed through to :func:`cluster_permutation_1samp`.

    Returns
    -------
    GroupERPResult
        Per-condition grand averages (every declared condition, including
        implicit ones such as ``all_trials``) plus one ClusterPermStats per
        requested pair.

    Raises
    ------
    ValueError
        Fail-loud: no subjects, inconsistent condition key sets, fewer than 2
        subjects with pairs requested, unknown or self-comparing conditions in
        a pair, or inconsistent channel grids / time samples across evokeds.
    """
    subjects = list(subject_evokeds)
    if len(subjects) == 0:
        raise ValueError("group_erp: received no subjects")
    keys = list(subjects[0])
    for index, subject in enumerate(subjects[1:], start=1):
        if list(subject) != keys:
            raise ValueError(
                f"group_erp: subject #{index} declares conditions "
                f"{sorted(subject)} but subject #0 declares {sorted(keys)}"
            )
    for cond_a, cond_b in condition_pairs:
        if cond_a == cond_b:
            raise ValueError(
                f"group_erp: condition pair ({cond_a!r}, {cond_b!r}) contrasts a "
                "condition with itself"
            )
        for cond in (cond_a, cond_b):
            if cond not in keys:
                raise ValueError(
                    f"group_erp: contrast condition {cond!r} is not among the "
                    f"declared conditions {keys}"
                )
    if len(subjects) < 2 and len(condition_pairs) > 0:
        raise ValueError(
            f"group_erp: {len(condition_pairs)} condition pair(s) requested but "
            f"only {len(subjects)} subject present; cluster permutation tests "
            "need >= 2 subjects"
        )

    picked = {cond: [_pick_eeg(subject[cond]) for subject in subjects] for cond in keys}
    all_picked = [evoked for evokeds in picked.values() for evoked in evokeds]
    _validate_consistent_grid(all_picked, label="group_erp")

    averages = {cond: grand_average(evokeds) for cond, evokeds in picked.items()}
    contrasts: dict[tuple[str, str], ClusterPermStats] = {}
    if condition_pairs:
        if adjacency is None:
            adjacency = eeg_adjacency(picked[keys[0]][0].info)
        for cond_a, cond_b in condition_pairs:
            diffs = [
                evoked_difference(evoked_a, evoked_b)
                for evoked_a, evoked_b in zip(picked[cond_a], picked[cond_b])
            ]
            X = np.stack([diff.data for diff in diffs])
            contrasts[(cond_a, cond_b)] = cluster_permutation_1samp(
                X,
                adjacency,
                n_permutations=n_permutations,
                tail=tail,
                seed=seed,
            )
    return GroupERPResult(
        averages=averages,
        contrasts=contrasts,
        n_subjects=len(subjects),
        ch_names=tuple(picked[keys[0]][0].ch_names),
    )
