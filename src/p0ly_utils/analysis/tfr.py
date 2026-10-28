"""Time-frequency analysis: Morlet / multitaper TFR power and ITPC (US-012).

Design (recorded in ``scrum/01_Backlog/US-012_tfr-itpc.md``, locked 2026-10-20):

- One complex transform feeds both products: :func:`tfr_epochs` computes a
  single-trial complex TFR via MNE native APIs (``output="complex"``),
  giving an :class:`~mne.time_frequency.EpochsTFR` whose complex data is
  ``(n_epochs, n_channels, n_freqs, n_times)`` for Morlet. Power
  (:func:`tfr_power_epochs`, per-epoch ``|c|²`` → 4-D) and ITPC
  (:func:`itpc`, ``|mean_epochs(c/|c|)|`` → 3-D, bounded [0, 1]) are
  vectorized NumPy reductions over the epochs axis — no second convolution.
- Morlet vs multitaper: ``n_cycles=None`` is adaptive (``freqs / 2``, time
  resolution scales with frequency). Multitaper adds ``time_bandwidth``
  (canonical 4.0). MNE 1.12.1 multitaper ``output="complex"`` carries a
  **taper axis**: data is 5-D ``(n_epochs, n_channels, n_tapers, n_freqs,
  n_times)`` with :attr:`~mne.time_frequency.EpochsTFR.weights`
  ``(n_tapers, n_freqs)``. Collapsing rules (documented contract):
  power follows MNE's own weighting, ``Σ_k w_k|c_k|² / Σ_k w_k²``; ITPC is
  computed per taper over epochs and averaged uniformly across tapers.
- ``EpochsTFR.average()`` raises for multitaper tapers in MNE 1.12.1, so
  :func:`average_power` / :func:`itpc` build
  :class:`~mne.time_frequency.AverageTFR` containers directly with
  :class:`~mne.time_frequency.AverageTFRArray` from our own reductions.
- ITPC has a small-n bias: for pure noise ``E[ITPC] ≈ 1/√n_epochs``.
  No bias correction is applied (deferred, see the story file).
- All functions are pure: no filesystem access, no I/O, no config reads.

Engineering note: the complex single-trial transient is large (~1.4 GB at
fixture scale 100 epochs × 59 ch × 41 freqs × 358 times, complex128);
``decim`` shrinks the persisted grid — the transient stays full-rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mne
import numpy as np
from mne.stats import combine_adjacency
from mne.time_frequency import AverageTFRArray, EpochsTFR
from numpy.typing import NDArray

from p0ly_utils.analysis.erp import ClusterPermStats
from p0ly_utils.analysis.psd import _epoch_mask

if TYPE_CHECKING:  # scipy arrives transitively via mne
    from scipy.sparse import sparray, spmatrix

__all__ = [
    "average_power",
    "freq_grid",
    "group_tfr",
    "itpc",
    "itpc_z",
    "tfr_cluster_permutation",
    "tfr_difference",
    "tfr_epochs",
    "tfr_power_epochs",
]


def freq_grid(
    fmin: float, fmax: float, step: float
) -> NDArray[np.float64]:
    """Inclusive, evenly spaced frequency grid (Hz) with float-safety.

    Parameters
    ----------
    fmin, fmax : float
        Frequency bounds (Hz); ``fmax > fmin`` required.
    step : float
        Bin spacing (Hz), positive. The grid always ends at or below
        ``fmax`` — the last bin is kept only if it lands within float
        tolerance of ``fmax``.

    Returns
    -------
    ndarray, shape (n_freqs,)
        ``fmin + step * np.arange(n)`` with ``fmin`` and ``fmax`` inclusive
        whenever ``step`` divides the range exactly.

    Raises
    ------
    ValueError
        Non-positive ``step`` or an empty/invalid range.
    """
    if step <= 0:
        raise ValueError(f"freq_grid: step must be positive, got {step}")
    if fmax <= fmin:
        raise ValueError(f"freq_grid: fmax ({fmax}) must exceed fmin ({fmin})")
    n_bins = int(np.floor((fmax - fmin) / step + 1e-9)) + 1
    freqs = fmin + step * np.arange(n_bins, dtype=np.float64)
    # float-arange edge: drop the last bin if step truly overshoots fmax
    if freqs[-1] > fmax + 1e-9 * step:
        freqs = freqs[:-1]
    if freqs.size < 2:  # pragma: no cover - guarded by fmax > fmin
        raise ValueError(
            f"freq_grid: range {fmin}–{fmax} Hz with step {step} produced "
            "fewer than 2 bins"
        )
    return freqs


def tfr_epochs(
    epochs: mne.BaseEpochs,
    freqs: Sequence[float] | NDArray[np.float64],
    n_cycles: float | Sequence[float] | NDArray[np.float64] | None = None,
    *,
    method: str = "morlet",
    time_bandwidth: float = 4.0,
    decim: int = 1,
    zero_mean: bool = True,
    use_fft: bool = True,
    condition_query: str | None = None,
) -> EpochsTFR:
    """Single-trial complex time-frequency transform (MNE native).

    One convolution produces the complex grid both power and ITPC reduce
    from. MNE object-level API keeps container metadata (``info``, ``times``,
    ``freqs``) free.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs to transform, shape ``(n_epochs, n_channels, n_times)``;
        must be non-empty. Optionally carrying ``epochs.metadata`` for
        condition selection.
    freqs : sequence of float or ndarray, shape (n_freqs,)
        Frequencies of interest (Hz), strictly the ``freq_grid`` output.
    n_cycles : float | sequence of float | ndarray | None
        Wavelet cycle count; float applies everywhere, an array must be
        length ``len(freqs)``. ``None`` (default) is adaptive: ``freqs / 2``,
        so time resolution improves with frequency.
    method : str
        ``"morlet"`` (default) or ``"multitaper"``. Anything else fails loud.
    time_bandwidth : float
        Multitaper spectral smoothing (canonical 4.0); ignored for Morlet.
    decim : int
        Temporal decimation applied by MNE (after the full-rate transform);
        result times are ``epoch times[::decim]``.
    zero_mean, use_fft
        Forwarded to MNE; both default to MNE's recommended values.
    condition_query : str or None
        Pandas ``DataFrame.eval`` query on ``epochs.metadata`` selecting the
        condition subset before transformation; ``None`` transforms all
        epochs (the ``all_trials`` convention). Zero matches fail loud.

    Returns
    -------
    EpochsTFR
        Complex single-trial transform. Morlet: data
        ``(n_epochs, n_channels, n_freqs, n_times)``. Multitaper: data
        ``(n_epochs, n_channels, n_tapers, n_freqs, n_times)`` with
        ``tfr.weights`` of shape ``(n_tapers, n_freqs)``.

    Raises
    ------
    ValueError
        Empty epochs, unknown method, a query without epochs metadata, or a
        query matching zero epochs (US-024 fail-loud philosophy).
    """
    if len(epochs) == 0:
        raise ValueError("tfr_epochs: received 0 epochs — nothing to transform")
    if method not in ("morlet", "multitaper"):
        raise ValueError(f"tfr_epochs: unknown method {method!r} (morlet | multitaper)")
    if condition_query is not None:
        mask = _epoch_mask(epochs, condition_query)
        n_matches = int(mask.sum())
        if n_matches == 0:
            raise ValueError(
                f"tfr_epochs: condition query {condition_query!r} matched 0 of "
                f"{len(epochs)} epochs"
            )
        epochs = epochs[mask]
    freqs = np.asarray(freqs, dtype=float)
    if n_cycles is None:
        n_cycles = freqs / 2.0
    kwargs: dict[str, object] = {
        "freqs": freqs,
        "n_cycles": n_cycles,
        "output": "complex",
        "average": False,  # single-trial complex grid
        "decim": decim,
        "zero_mean": zero_mean,
        "use_fft": use_fft,
    }
    if method == "multitaper":
        kwargs["time_bandwidth"] = time_bandwidth
    return epochs.compute_tfr(method=method, **kwargs)  # type: ignore[arg-type,return-value]


def _complex_data(tfr: EpochsTFR) -> NDArray[np.complex128]:
    """Validated complex data of ``tfr`` (4-D Morlet or 5-D multitaper)."""
    if not isinstance(tfr, EpochsTFR):
        raise TypeError(
            "expected an EpochsTFR (from tfr_epochs); got "
            f"{type(tfr).__name__}"
        )
    data = np.asarray(tfr.data)
    if data.ndim not in (4, 5):
        raise ValueError(
            f"expected complex TFR data of rank 4 (Morlet) or 5 "
            f"(multitaper, taper axis), got rank {data.ndim}"
        )
    if not np.iscomplexobj(data):
        raise ValueError(
            f"expected complex TFR data, got dtype {tfr.data.dtype} — "
            "was output='complex'?"
        )
    return data


def tfr_power_epochs(tfr: EpochsTFR) -> NDArray[np.float64]:
    """Per-epoch TFR power ``|c|²`` from a complex transform.

    Parameters
    ----------
    tfr : EpochsTFR
        Complex single-trial transform from :func:`tfr_epochs`.

    Returns
    -------
    ndarray, shape (n_epochs, n_channels, n_freqs, n_times)
        Per-epoch power. Multitaper input (5-D, taper axis) is collapsed
        following MNE's weighting convention
        ``Σ_k w_k |c_k|² / Σ_k w_k²`` — Morlet input is plain ``|c|²``.
    """
    data = _complex_data(tfr)
    power = np.abs(data) ** 2
    if data.ndim == 5:  # multitaper: (e, ch, taper, freq, time)
        if tfr.weights is None:
            raise ValueError(
                "multitaper TFR carries no taper weights — cannot collapse "
                "the taper axis (recompute with tfr_epochs / output='complex')"
            )
        weights = np.asarray(tfr.weights, dtype=float)  # (n_tapers, n_freqs)
        w = weights[None, None, :, :, None]  # broadcast over epochs/channels/time
        power = (power * w).sum(axis=2) / (weights**2).sum(axis=0)[
            None, None, :, None
        ]
    return power


def itpc(tfr: EpochsTFR) -> mne.time_frequency.AverageTFR:
    """Inter-Trial Phase Coherence from a complex transform.

    ``ITPC(f, t) = |mean_epochs(c / |c|)|`` — the length of the mean unit
    phasor over trials: 1 when phase is locked, ~0 when phases cancel. This
    is computed here from our single complex transform, deliberately *not*
    MNE's legacy ``output='itc'`` (whose multitaper accumulation is not
    bounded by 1).

    Small-n bias: for pure noise ``E[ITPC] ≈ 1/√n_epochs``. No correction is
    applied (deferred — see the story file).

    Parameters
    ----------
    tfr : EpochsTFR
        Complex single-trial transform from :func:`tfr_epochs`.

    Returns
    -------
    AverageTFR
        ``(n_channels, n_freqs, n_times)`` ITPC, values in [0, 1]; ``nave``
        equals the number of epochs reduced.

    Raises
    ------
    ValueError
        If any complex coefficient is exactly zero (phase undefined for
        silent data).
    """
    data = _complex_data(tfr)
    if np.any(np.abs(data) == 0):
        raise ValueError(
            "itpc: encountered zero-valued coefficients — phase is undefined "
            "(all-zero signal in at least one epoch/channel/frequency/time)"
        )
    phasor = data / np.abs(data)  # per-taper unit phasors, vectorized
    itc = np.abs(phasor.mean(axis=0))
    if data.ndim == 5:  # multitaper: mean ITPC over tapers (bounded [0, 1])
        itc = itc.mean(axis=1)
    return AverageTFRArray(
        tfr.info,
        itc,
        times=tfr.times,
        freqs=tfr.freqs,
        nave=int(data.shape[0]),
        comment="itpc",
    )


def average_power(tfr: EpochsTFR) -> mne.time_frequency.AverageTFR:
    """Mean TFR power over epochs.

    Parameters
    ----------
    tfr : EpochsTFR
        Complex single-trial transform from :func:`tfr_epochs`.

    Returns
    -------
    AverageTFR
        ``(n_channels, n_freqs, n_times)`` epoch-mean power; ``nave`` equals
        the number of epochs reduced.
    """
    power = tfr_power_epochs(tfr)
    return AverageTFRArray(
        tfr.info,
        power.mean(axis=0),
        times=tfr.times,
        freqs=tfr.freqs,
        nave=int(power.shape[0]),
        comment="power",
    )


def _check_comparable(a: mne.time_frequency.AverageTFR, b: mne.time_frequency.AverageTFR, label: str) -> None:
    """Raise :class:`ValueError` unless two AverageTFRs share channels, freqs and times."""
    if a.ch_names != b.ch_names:
        raise ValueError(
            f"{label}: channel mismatch — {b.ch_names} vs {a.ch_names}"
        )
    if a.freqs.size != b.freqs.size or not np.allclose(a.freqs, b.freqs):
        raise ValueError(
            f"{label}: frequency grid mismatch — compute all TFRs with an "
            f"identical freq_grid"
        )
    if a.data.shape != b.data.shape or not np.allclose(a.times, b.times):
        raise ValueError(
            f"{label}: time grid mismatch — {b.times[0]:.3f}–{b.times[-1]:.3f} s "
            f"({b.times.size} samples) vs {a.times[0]:.3f}–{a.times[-1]:.3f} s "
            f"({a.times.size} samples)"
        )


def group_tfr(
    tfrs: Sequence[mne.time_frequency.AverageTFR]
) -> mne.time_frequency.AverageTFR:
    """Equal-weight grand average of per-subject TFR averages.

    Parameters
    ----------
    tfrs : sequence of AverageTFR
        Per-subject ``(n_channels, n_freqs, n_times)`` averages (power and
        ITPC both reduce through this function) sharing an identical
        frequency grid and channel set.

    Returns
    -------
    AverageTFR
        Arithmetic mean across subjects. Equal-weight (not nave-weighted),
        so a subject with 10 trials counts as much as one with 100; the
        result ``nave`` is the total trial count. Fisher-z transformation of
        ITPC before averaging is deferred — see the story file.

    Raises
    ------
    ValueError
        No input, or any frequency-grid / channel-set mismatch.
    """
    tfr_list = list(tfrs)
    if not tfr_list:
        raise ValueError("group_tfr: received no TFRs — nothing to average")
    first = tfr_list[0]
    for tfr in tfr_list[1:]:
        if len(tfr.freqs) != len(first.freqs) or not np.allclose(tfr.freqs, first.freqs):
            raise ValueError(
                "group_tfr: subjects have different frequency grids — "
                "compute all TFRs with identical fmin/fmax/step settings"
            )
        if tfr.ch_names != first.ch_names:
            raise ValueError(
                f"group_tfr: subjects have different channel sets — "
                f"{tfr.ch_names} vs {first.ch_names}"
            )
    return mne.grand_average(tfr_list)


def tfr_difference(
    a: mne.time_frequency.AverageTFR, b: mne.time_frequency.AverageTFR
) -> mne.time_frequency.AverageTFR:
    """Point-wise difference TFR ``a - b`` (e.g. power or ITPC of two conditions).

    Parameters
    ----------
    a, b : AverageTFR
        Averages to subtract; must share channels and frequency/time grids.

    Returns
    -------
    AverageTFR
        ``a - b``, ``(n_channels, n_freqs, n_times)``. ``nave`` is not
        meaningful for a mixed-sign combination and should not be interpreted.
    """
    _check_comparable(a, b, label="tfr_difference")
    return AverageTFRArray(
        a.info,
        a.data - b.data,
        times=a.times,
        freqs=a.freqs,
        comment=f"{a.comment}-{b.comment}",
    )


def itpc_z(itpc: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Fisher z-transform of ITPC values (US-026 statistics transform).

    Raw ITPC is bounded ``[0, 1]`` and carries the small-n bias
    (``E[ITPC | noise] ≈ 1/√n_epochs``); the cluster-permutation statistic
    for ITPC contrasts runs on ``atanh(ITPC)`` differences so values compress
    near the [0, 1] ceiling symmetrically around zero. Values are clipped to
    ``[-1 + 1e-7, 1 - 1e-7]`` first — per-subject ITPC hits exactly 1.0 with
    few epochs, and ``atanh(1)`` diverges.

    Parameters
    ----------
    itpc : ndarray, shape (...)
        ITPC values in ``[0, 1]`` (any dimensionality).

    Returns
    -------
    ndarray, same shape
        ``arctanh(clip(itpc))``; bounded by ``±atanh(1 - 1e-7) ≈ ±8.1``.
    """
    values = np.clip(np.asarray(itpc, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arctanh(values)


def tfr_cluster_permutation(
    X: NDArray[np.float64],
    ch_adjacency: spmatrix | sparray,
    *,
    n_permutations: int = 1024,
    tail: int = 0,
    seed: int | None = None,
) -> ClusterPermStats:
    """One-sample cluster permutation over TFR difference grids (US-026).

    Generalizes :func:`p0ly_utils.analysis.erp.cluster_permutation_1samp` to
    the channel x freq x time space: per-subject difference maps stacked into
    ``X (n_subjects, n_channels, n_freqs, n_times)`` are tested against zero
    with :func:`mne.stats.spatio_temporal_cluster_1samp_test`. The 3-D
    adjacency is built with :func:`mne.stats.combine_adjacency` — the sensor
    adjacency for the channel axis plus regular lattices for the frequency
    and time axes (the freq × time pixel grid flattening of the story spike;
    a pixel's neighbors are its ±1 step in each axis).

    Reuses :class:`~p0ly_utils.analysis.erp.ClusterPermStats` with widened
    dimensionality: ``t_obs`` is ``(n_channels, n_freqs, n_times)`` and the
    cluster tuples are ``(channel_indices, freq_indices, time_indices)``.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_channels, n_freqs, n_times)
        Per-subject difference maps (e.g. ``itpc_z(cond_a) - itpc_z(cond_b)``
        for ITPC, raw power difference for power), tested against zero.
    ch_adjacency : sparse matrix/array, shape (n_channels, n_channels)
        Sensor adjacency from :func:`p0ly_utils.analysis.erp.eeg_adjacency`.
    n_permutations, tail, seed
        As in :func:`cluster_permutation_1samp`; with few subjects the unique
        sign-flip count ``2 ** (n_subjects - 1)`` bounds the null.

    Returns
    -------
    ClusterPermStats
        Statistics with ``(n_channels, n_freqs, n_times)`` orientation.

    Raises
    ------
    ValueError
        If ``X`` is not 4-D, has fewer than 2 subjects, or the sensor
        adjacency shape does not match the channel count.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 4:
        raise ValueError(
            f"tfr_cluster_permutation: X must be 4-D (n_subjects, n_channels, "
            f"n_freqs, n_times), got shape {X.shape}"
        )
    n_subjects, n_channels, n_freqs, n_times = X.shape
    if n_subjects < 2:
        raise ValueError(
            f"tfr_cluster_permutation: needs >= 2 subjects for a group test, "
            f"got {n_subjects}"
        )
    if tuple(ch_adjacency.shape) != (n_channels, n_channels):
        raise ValueError(
            f"tfr_cluster_permutation: adjacency shape {ch_adjacency.shape} "
            f"does not match X channel count ({n_channels},)"
        )
    adjacency = combine_adjacency(ch_adjacency, n_freqs, n_times)
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        threshold=None,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        seed=seed,
        out_type="indices",
    )
    index_tuples = [(np.asarray(ch), np.asarray(fr), np.asarray(t)) for ch, fr, t in clusters]
    return ClusterPermStats(
        t_obs=np.asarray(t_obs),
        clusters=index_tuples,
        cluster_pvals=np.asarray(cluster_pv, dtype=np.float64),
        h0=np.asarray(h0, dtype=np.float64),
        n_permutations=n_permutations,
        tail=tail,
        threshold=None,
        seed=seed,
    )
