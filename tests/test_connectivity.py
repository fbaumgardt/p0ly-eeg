"""Tests for p0ly_utils.analysis.connectivity (US-013).

All data is simulated in memory per the AC "unit tests with small synthetic
channel sets" — no fixture files, no ``@pytest.mark.slow``.

Synthetic layout: 4 EEG channels at 200 Hz, 200-sample (1 s) epochs of white
noise (sigma 1e-6 V). Channels 0/1 share a common source (strong coherence
~1), channels 0/2 are independent (near-zero). Connectivity is an
across-trials statistic, so every test uses >= 2 epochs by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from mne import EpochsArray, create_info

from p0ly_utils.analysis import (
    ConnectivityMatrices,
    band_average,
    graph_metrics,
    group_connectivity,
    spectral_connectivity,
)

_SFREQ = 200.0
_N_SAMPLES = 200  # 1 s epochs
_N_CHANNELS = 4
_CH_NAMES = ("Fp1", "Fp2", "Fz", "Cz")
_BANDS = {"alpha": [8, 12], "gamma": [30, 45]}


def _epochs(n_epochs: int = 40, coupled: bool = True):
    """Noise epochs with channels 0/1 either sharing a source or not."""
    rng = np.random.default_rng(42)
    src = rng.normal(size=(n_epochs, _N_SAMPLES))
    data = rng.normal(size=(n_epochs, _N_CHANNELS, _N_SAMPLES)) * 1e-6
    if coupled:
        data[:, 0, :] += src
        data[:, 1, :] += src
    info = create_info(list(_CH_NAMES), _SFREQ, "eeg")
    return EpochsArray(data, info)


def _matrices(**kwargs) -> ConnectivityMatrices:
    return spectral_connectivity(_epochs(), **kwargs)


# --- spectral_connectivity --------------------------------------------------


@pytest.mark.parametrize("method", ["wpli", "coh", "plv"])
def test_spectral_connectivity_shape_contract(method: str) -> None:
    """AC: adjacency matrices shaped (n_channels, n_channels, n_freqs)."""
    out = _matrices(methods=[method])
    n_freqs = out.matrices[method].shape[-1]
    assert out.matrices[method].shape == (_N_CHANNELS, _N_CHANNELS, n_freqs)
    assert out.ch_names == _CH_NAMES
    assert n_freqs >= 1


def test_spectral_connectivity_symmetric_zero_diagonal() -> None:
    """Lower-triangle dense output is symmetrized; self-connectivity is 0."""
    for matrix in _matrices().matrices.values():
        assert np.allclose(matrix, matrix.transpose(1, 0, 2))
        assert np.allclose(np.diagonal(matrix, axis1=0, axis2=1), 0.0)


def test_spectral_connectivity_detects_coupled_pair() -> None:
    """Channel 0/1 share a source (coupling ~ max), 0/2 independent."""
    matrices = _matrices()
    mid = matrices.matrices["coh"].shape[-1] // 2
    coupled_vs_independent = (
        matrices.matrices["coh"][0, 1, mid],
        matrices.matrices["coh"][0, 2, mid],
    )
    assert coupled_vs_independent[0] > 0.7
    assert coupled_vs_independent[1] < 0.2


def test_spectral_connectivity_all_methods_in_one_call() -> None:
    """One call computes the default three-method set, keyed in order."""
    out = _matrices()
    assert tuple(out.matrices) == ("wpli", "coh", "plv")
    for method, matrix in out.matrices.items():
        assert method in ("wpli", "coh", "plv")
        assert np.all(np.isfinite(matrix))


def test_spectral_connectivity_needs_two_epochs() -> None:
    """Connectivity is an across-trials statistic — 1 epoch fails loud."""
    with pytest.raises(ValueError, match=">= 2"):
        spectral_connectivity(_epochs(n_epochs=1))


def _epochs_with_metadata(n_used: int | None = None):
    """Noise epochs carrying a boolean ``used`` metadata column.

    ``n_used`` epochs match ``"used"`` (default: all but one).
    """
    import pandas as pd

    epochs = _epochs()
    n = len(epochs)
    flags = [True] * (n_used if n_used is not None else n - 1)
    flags += [False] * (n - len(flags))
    epochs.metadata = pd.DataFrame({"used": flags})
    return epochs


def test_spectral_connectivity_condition_query() -> None:
    """A metadata query selects the condition epochs before computing."""
    epochs = _epochs_with_metadata()
    out = spectral_connectivity(epochs, condition_query="used")
    assert set(out.matrices) == {"wpli", "coh", "plv"}
    assert out.ch_names == _CH_NAMES


def test_spectral_connectivity_query_too_few_matches_fail_loud() -> None:
    """A query matching < 2 epochs raises (US-024 fail-loud philosophy)."""
    with pytest.raises(ValueError, match="matched 1 of"):
        spectral_connectivity(_epochs_with_metadata(n_used=1), condition_query="used")


def test_spectral_connectivity_query_needs_metadata() -> None:
    """A query without epochs metadata fails loud (US-024 philosophy)."""
    with pytest.raises(ValueError, match="no metadata"):
        _matrices(condition_query="trial == 1")


# --- group_connectivity -----------------------------------------------------


def test_group_connectivity_equal_weight_mean() -> None:
    """Group mean is the plain arithmetic mean over subjects."""
    a = _matrices()
    b = _matrices()
    assert np.allclose(
        group_connectivity([a, b]).matrices["coh"], (a.matrices["coh"] + b.matrices["coh"]) / 2
    )


def test_group_connectivity_single_subject_identity() -> None:
    """A one-subject group is that subject (n=1 bootstrap honesty)."""
    a = _matrices()
    assert np.allclose(group_connectivity([a]).matrices["coh"], a.matrices["coh"])


def test_group_connectivity_freq_grid_mismatch_fails_loud() -> None:
    """Subjects computed with different fmin/fmax cannot be averaged."""
    a = _matrices(fmin=4.0, fmax=45.0)
    b = _matrices(fmin=8.0, fmax=45.0)
    # mne-connectivity grids may differ in bin count only; a same-count grid
    # still differs in values — guard on grid content either way.
    with pytest.raises(ValueError, match="frequency grid"):
        group_connectivity(
            [a, b]
            if a.freqs.size == b.freqs.size
            else [a, ConnectivityMatrices(a.ch_names, a.freqs + 0.5, a.matrices)]
        )


def test_group_connectivity_empty_fails_loud() -> None:
    with pytest.raises(ValueError, match="no results"):
        group_connectivity([])


# --- band_average -----------------------------------------------------------


def test_band_average_shapes_and_inclusion() -> None:
    """Bands are inclusive (fmin, fmax) windows; averaging returns 2-D."""
    out = _matrices()
    freqs = out.freqs
    averaged = band_average(out.matrices["coh"], freqs, _BANDS)
    assert set(averaged) == {"alpha", "gamma"}
    for values in averaged.values():
        assert values.shape == (_N_CHANNELS, _N_CHANNELS)
    alpha_bins = (freqs >= 8) & (freqs <= 12)
    assert np.allclose(
        averaged["alpha"],
        out.matrices["coh"][..., alpha_bins].mean(axis=-1),
    )


def test_band_average_empty_band_fails_loud() -> None:
    """A band outside the frequency grid selects 0 bins — fail loud."""
    out = _matrices(fmin=4.0, fmax=45.0)
    with pytest.raises(ValueError, match="selects 0"):
        band_average(out.matrices["coh"], out.freqs, {"void": [46, 80]})


# --- graph_metrics ----------------------------------------------------------


def _triangle(equal_weights: bool = True) -> np.ndarray:
    """Three strongly coupled nodes plus one isolated channel."""
    w = 0.5 if equal_weights else None
    adjacency = np.zeros((4, 4))
    for i, j, wij in ((0, 1, w or 0.9), (1, 2, w or 0.6), (0, 2, w or 0.3)):
        adjacency[i, j] = adjacency[j, i] = wij
    return adjacency


def test_graph_metrics_density_and_path() -> None:
    """AC: graph metrics of a thresholded band graph."""
    metrics = graph_metrics(_triangle(), min_edge_weight=0.1)
    assert metrics.n_edges == 3
    assert metrics.n_components == 2  # triangle {0,1,2} + isolated {3}
    # Equal weights make every effective distance 1/0.5, so all three
    # triangle pairs are direct shortest paths of the same length.
    assert metrics.path_length == pytest.approx(1 / 0.5)
    assert metrics.clustering[1] > 0  # node 1 closes the triangle
    assert metrics.strength[3] == 0.0  # isolated node


def test_graph_metrics_strength_counts_both_ends() -> None:
    """Strength is the weighted degree over the thresholded graph."""
    metrics = graph_metrics(_triangle(equal_weights=False), min_edge_weight=0.1)
    assert metrics.strength[0] == pytest.approx(0.9 + 0.3)
    assert metrics.strength[1] == pytest.approx(0.9 + 0.6)


def test_graph_metrics_weighted_path_length() -> None:
    """Path distances are 1/weight, so weak edges detour via strong ones."""
    metrics = graph_metrics(_triangle(equal_weights=False), min_edge_weight=0.1)
    # pair (0,1): 1/0.9; (1,2): 1/0.6; (0,2): direct 1/0.3 loses to the
    # bundle 1/0.9 + 1/0.6 -> shortest 2.778.
    assert metrics.path_length == pytest.approx((1 / 0.9 + 1 / 0.6 + (1 / 0.9 + 1 / 0.6)) / 3)


def test_graph_metrics_threshold_drops_edges() -> None:
    """Edges below min_edge_weight vanish; an empty graph fails loud."""
    metrics = graph_metrics(_triangle(equal_weights=False), min_edge_weight=0.5)
    assert metrics.n_edges == 2
    # Remaining graph: 0-1 (1/0.9), 1-2 (1/0.6); 0-2 detours through 1.
    assert metrics.path_length == pytest.approx((1 / 0.9 + 1 / 0.6 + (1 / 0.9 + 1 / 0.6)) / 3)
    with pytest.raises(ValueError, match="no edges"):
        graph_metrics(_triangle(equal_weights=False), min_edge_weight=0.95)


def test_graph_metrics_largest_component_path_length() -> None:
    """path_length describes the largest component only."""
    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 1.0  # 2-node component
    adjacency[2, 3] = adjacency[3, 2] = 0.9  # 2-node component
    metrics = graph_metrics(adjacency, min_edge_weight=0.1)
    assert metrics.n_components == 2
    assert metrics.path_length == pytest.approx(1.0)


def test_graph_metrics_input_validation() -> None:
    """Non-square, non-finite, and non-positive thresholds fail loud."""
    with pytest.raises(ValueError, match="square"):
        graph_metrics(np.ones((3, 4)))
    bad = _triangle()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        graph_metrics(bad)
    with pytest.raises(ValueError, match="positive"):
        graph_metrics(_triangle(), min_edge_weight=0.0)
