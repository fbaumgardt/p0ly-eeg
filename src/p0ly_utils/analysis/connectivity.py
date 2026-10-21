"""Spectral connectivity matrices and graph metrics (US-013).

Design (recorded in ``scrum/01_Backlog/US-013_connectivity.md``, locked
2026-10-21):

- One :func:`~mne_connectivity.spectral_connectivity_epochs` call computes
  all configured methods (``wpli`` / ``coh`` / ``plv`` by default) from the
  per-epoch data ``(n_epochs, n_channels, n_samples)`` — arrays are accepted
  directly, no MNE Epochs object needed inside the library call.
- mne-connectivity's ``get_data(output="dense")`` fills **only the lower
  triangle** (diagonal left at 0, verified live in mne-connectivity 0.9.0
  with a 4-channel synthetic). :func:`spectral_connectivity` therefore
  symmetrizes every matrix ``M = dense + dense.T`` to the SCHEMA shape
  ``(n_channels, n_channels, n_freqs)``; the diagonal stays 0 —
  self-connectivity is 1 for coherence by definition but excluded to share
  one convention across methods, so a thresholded *graph* never gets
  self-loops. Downstream consumers must not interpret diagonal values.
- Graph metrics run on a *thresholded* band-averaged adjacency
  (:func:`graph_metrics`): edges with weight below ``min_edge_weight`` are
  dropped, coupling strength converts to effective distance ``1 / weight``
  for shortest paths (high coupling = close), and the characteristic path
  length is computed on the largest connected component (documented
  convention — a thresholded functional graph need not be connected).
- All functions are pure: no filesystem access, no I/O, no config reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import mne
import networkx as nx
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from numpy.typing import NDArray

from p0ly_utils.analysis.psd import _epoch_mask

__all__ = [
    "ConnectivityMatrices",
    "GraphMetrics",
    "band_average",
    "graph_metrics",
    "group_connectivity",
    "spectral_connectivity",
]

_METHOD_ORDER = ("wpli", "coh", "plv")


@dataclass
class ConnectivityMatrices:
    """Per-method adjacency matrices over a shared frequency grid.

    Attributes
    ----------
    ch_names : tuple of str
        Channel labels, length ``n_channels``; the matrices' first two axes
        follow this ordering.
    freqs : ndarray, shape (n_freqs,)
        The frequency axis (Hz) shared by every matrix.
    matrices : dict of str -> ndarray
        Method name (e.g. ``"wpli"``, ``"coh"``, ``"plv"``) -> adjacency of
        shape ``(n_channels, n_channels, n_freqs)``, symmetric with a zero
        diagonal.
    """

    ch_names: tuple[str, ...]
    freqs: NDArray[np.float64]
    matrices: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class GraphMetrics:
    """Graph metrics of one thresholded (unweighted topologized) graph.

    Attributes
    ----------
    clustering : ndarray, shape (n_nodes,)
        Weighted clustering coefficient per node (Onnela geometric-mean
        definition via :func:`networkx.clustering`); 0 for isolated nodes.
    path_length : float
        Characteristic path length — mean weighted shortest path within the
        largest connected component, distances ``1 / weight``.
    strength : ndarray, shape (n_nodes,)
        Weighted degree (strength) per node over the thresholded graph.
    n_edges : int
        Number of edges above threshold (undirected, counted once).
    n_components : int
        Number of connected components; ``path_length`` describes the
        largest one only.
    """

    clustering: NDArray[np.float64]
    path_length: float
    strength: NDArray[np.float64]
    n_edges: int
    n_components: int


def spectral_connectivity(
    epochs: mne.BaseEpochs,
    *,
    methods: Sequence[str] = _METHOD_ORDER,
    fmin: float = 4.0,
    fmax: float = 45.0,
    mode: str = "multitaper",
    n_jobs: int = 1,
    condition_query: str | None = None,
) -> ConnectivityMatrices:
    """Per-method spectral connectivity over channels for one condition.

    Wraps :func:`mne_connectivity.spectral_connectivity_epochs` (one call
    computes all methods) and symmetrizes its lower-triangle dense output
    into full ``(n_channels, n_channels, n_freqs)`` matrices.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs, shape ``(n_epochs, n_channels, n_times)``; at least 2 epochs
        are required (connectivity is an across-trials statistic). May carry
        ``epochs.metadata`` for condition selection.
    methods : sequence of str
        mne-connectivity method names (default ``("wpli", "coh", "plv"``);
        order in the result follows this sequence.
    fmin, fmax : float
        Frequency bounds (Hz) forwarded to mne-connectivity, which resolves
        its own adaptive frequency grid (recorded in the result).
    mode : str
        Spectral estimation mode (default ``"multitaper"``).
    n_jobs : int
        Parallelism forwarded to mne-connectivity.
    condition_query : str or None
        Pandas ``DataFrame.eval`` query on ``epochs.metadata`` selecting the
        condition subset first; ``None`` uses all epochs (``all_trials``).
        Zero matches fail loud.

    Returns
    -------
    ConnectivityMatrices
        ``matrices[method]`` is ``(n_channels, n_channels, n_freqs)``,
        symmetric with a zero diagonal (self-connectivity excluded — the
        convention lets one thresholded graph carry every method).

    Raises
    ------
    ValueError
        Fewer than 2 epochs, an empty result after the condition query, or
        a query without epochs metadata (US-024 fail-loud philosophy).
    """
    if len(epochs) < 2:
        raise ValueError(
            f"spectral_connectivity: received {len(epochs)} epochs — "
            "connectivity is an across-trials statistic, need >= 2"
        )
    if condition_query is not None:
        mask = _epoch_mask(epochs, condition_query)
        n_matches = int(mask.sum())
        if n_matches < 2:
            raise ValueError(
                f"spectral_connectivity: condition query {condition_query!r} "
                f"matched {n_matches} of {len(epochs)} epochs (need >= 2)"
            )
        epochs = epochs[mask]
    data = np.asarray(epochs.get_data(), dtype=float)  # (n_epochs, n_ch, n_smp)
    con_list = spectral_connectivity_epochs(
        data,
        method=list(methods),
        indices=None,
        sfreq=epochs.info["sfreq"],
        mode=mode,
        fmin=float(fmin),
        fmax=float(fmax),
        n_jobs=n_jobs,
        verbose=False,
    )
    if not isinstance(con_list, list):  # single method -> bare object
        con_list = [con_list]
    matrices: dict[str, NDArray[np.float64]] = {}
    freqs = np.asarray(con_list[0].freqs, dtype=float)
    for con in con_list:
        dense = np.asarray(con.get_data(output="dense"))  # (ch, ch, freqs)
        dense = dense + dense.transpose(1, 0, 2)  # symmetrize lower triangle
        method_key = str(con.method) if not isinstance(con.method, str) else con.method
        method_key = method_key[0] if isinstance(method_key, list) else method_key
        matrices[method_key] = dense
    return ConnectivityMatrices(
        ch_names=tuple(epochs.ch_names),
        freqs=freqs,
        matrices={m: matrices[m] for m in methods if m in matrices},
    )


def _check_comparable(a: ConnectivityMatrices, b: ConnectivityMatrices, label: str) -> None:
    """Raise :class:`ValueError` unless two results share channels, methods, freqs."""
    if a.ch_names != b.ch_names:
        raise ValueError(f"{label}: channel mismatch — {b.ch_names} vs {a.ch_names}")
    if set(a.matrices) != set(b.matrices):
        raise ValueError(f"{label}: method mismatch — {sorted(b.matrices)} vs {sorted(a.matrices)}")
    if a.freqs.size != b.freqs.size or not np.allclose(a.freqs, b.freqs):
        raise ValueError(
            f"{label}: frequency grid mismatch ({a.freqs.size} vs "
            f"{b.freqs.size} bins) — compute every subject with identical "
            f"fmin/fmax settings"
        )


def group_connectivity(
    mats: Sequence[ConnectivityMatrices],
) -> ConnectivityMatrices:
    """Equal-weight grand average of per-subject connectivity results.

    Parameters
    ----------
    mats : sequence of ConnectivityMatrices
        Per-subject results sharing an identical channel set, method set and
        frequency grid.

    Returns
    -------
    ConnectivityMatrices
        Arithmetic mean across subjects per method (equal-weight, not
        nave-weighted, mirroring :func:`p0ly_utils.analysis.group_tfr`).

    Raises
    ------
    ValueError
        No input, or any channel/method/frequency mismatch.
    """
    mats_list = list(mats)
    if not mats_list:
        raise ValueError("group_connectivity: received no results — nothing to average")
    for m in mats_list[1:]:
        _check_comparable(mats_list[0], m, "group_connectivity")
    return ConnectivityMatrices(
        ch_names=mats_list[0].ch_names,
        freqs=mats_list[0].freqs.copy(),
        matrices={
            method: np.stack([m.matrices[method] for m in mats_list], axis=0).mean(axis=0)
            for method in mats_list[0].matrices
        },
    )


def band_average(
    matrix: NDArray[np.float64],
    freqs: NDArray[np.float64],
    bands: Mapping[str, Sequence[float]],
) -> dict[str, NDArray[np.float64]]:
    """Average an adjacency over the frequency bins of each named band.

    Parameters
    ----------
    matrix : ndarray, shape (n_channels, n_channels, n_freqs)
        Symmetric adjacency (e.g. from :func:`spectral_connectivity`).
    freqs : ndarray, shape (n_freqs,)
        Frequency axis (Hz) matching ``matrix``'s last dimension.
    bands : mapping of str -> (fmin, fmax)
        Inclusive (fmin, fmax) Hz band windows; bins equal to a bound are
        included.

    Returns
    -------
    dict of str -> ndarray, shape (n_channels, n_channels)
        Band name -> band-averaged adjacency.

    Raises
    ------
    ValueError
        A band that selects zero frequency bins.
    """
    freqs = np.asarray(freqs, dtype=float)
    averaged: dict[str, NDArray[np.float64]] = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= float(lo)) & (freqs <= float(hi))
        if not sel.any():
            raise ValueError(
                f"band_average: band {name!r} [{lo}, {hi}] Hz selects 0 of "
                f"{freqs.size} bins ({freqs[0]:.1f}–{freqs[-1]:.1f} Hz)"
            )
        averaged[name] = matrix[..., sel].mean(axis=-1)
    return averaged


def graph_metrics(
    adjacency: NDArray[np.float64],
    *,
    min_edge_weight: float = 0.1,
) -> GraphMetrics:
    """Graph metrics of one band-averaged ``(n_channels, n_channels)`` matrix.

    The matrix is treated as an undirected weighted graph: edges are the
    upper-triangle entries with weight >= ``min_edge_weight`` (the
    symmetric duplicate and the zero diagonal are ignored), weight is
    coupling strength, and path computations use the effective distance
    ``1 / weight``. The characteristic path length describes the largest
    connected component only (see :class:`GraphMetrics`).

    Parameters
    ----------
    adjacency : ndarray, shape (n_channels, n_channels)
        Symmetric coupling matrix.
    min_edge_weight : float
        Minimum coupling for an edge to exist (noise-floor lever; a
        thresholded functional graph is NOT complete — without this, the
        weighted path length degenerates toward the complete graph).

    Returns
    -------
    GraphMetrics
        Weighted clustering per node, characteristic path length (largest
        component), strength per node, edge count, component count.

    Raises
    ------
    ValueError
        Non-square or non-finite input, or no edges above threshold.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    if adjacency.ndim != 2 or adjacency.shape[1] != n:
        raise ValueError(f"graph_metrics: expected a square adjacency, got shape {adjacency.shape}")
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("graph_metrics: adjacency carries non-finite values")
    if min_edge_weight <= 0:
        raise ValueError(f"graph_metrics: min_edge_weight must be positive, got {min_edge_weight}")
    edges = [
        (i, j, float(adjacency[i, j]))
        for i, j in zip(*np.triu_indices(n, k=1), strict=True)
        if adjacency[i, j] >= min_edge_weight
    ]
    if not edges:
        raise ValueError(
            f"graph_metrics: no edges with weight >= {min_edge_weight} — "
            "lower min_edge_weight or the adjacency is below the noise floor"
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j, w in edges:
        graph.add_edge(i, j, weight=w, distance=1.0 / w)
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest)
    path_length = (
        nx.average_shortest_path_length(subgraph, weight="distance") if len(largest) > 1 else 0.0
    )
    clustering = nx.clustering(graph, weight="weight")
    strength = np.zeros(n)
    for i, j, w in edges:
        strength[i] += w
        strength[j] += w
    return GraphMetrics(
        clustering=np.array([clustering[i] for i in range(n)]),
        path_length=path_length,
        strength=strength,
        n_edges=len(edges),
        n_components=len(components),
    )
