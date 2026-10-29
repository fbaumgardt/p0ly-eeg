"""Multi-segment raw recording merge (US-009).

Pure, stateless library function that concatenates a list of
:class:`mne.io.BaseRaw` segments — produced when acquisition was interrupted
or split across files — into a single continuous raw object so downstream
preprocessing and analysis treat the session as one uninterrupted recording.

The function is **path- and layout-agnostic** (ADR-001 / ADR-005): it operates
on already-loaded ``Raw`` objects. The pipeline (`p0ly-flow`) owns the
detection of how many segments exist for a subject/session (manifest-driven,
see ADR-005 §3 amendment) and is responsible for loading each segment and
calling :func:`merge_recordings`.

Discontinuities at segment boundaries are flagged with a ``BAD_break``
annotation (configurable label) so the downstream epoching rule can reject
epochs that straddle a boundary. Channel set, channel order, and sampling
rate are validated across all segments; a mismatch raises ``ValueError``.
"""

from __future__ import annotations

import mne

__all__ = ["merge_recordings"]


def merge_recordings(
    raws: list[mne.io.BaseRaw],
    gap_label: str = "BAD_break",
    gap_duration: float | None = None,
) -> mne.io.BaseRaw:
    """Concatenate raw recording segments into one continuous raw.

    Segments are abutted in the given order. At each boundary ``i -> i+1`` a
    ``gap_label`` annotation is inserted covering ``gap_duration`` seconds
    (default: one sample) so epochs crossing a boundary can be rejected.

    A single-element list returns a copy of that segment unchanged (no
    ``gap_label``) — lets the pipeline call merge unconditionally even for
    single-segment subjects.

    Parameters
    ----------
    raws
        Non-empty list of :class:`mne.io.BaseRaw` segments in acquisition
        order. Shape of each ``raw.get_data()`` is ``(n_channels, n_times)``.
    gap_label
        Annotation description inserted at each segment boundary. Default
        ``"BAD_break"`` (recognised by MNE as a BAD annotation).
    gap_duration
        Duration of the boundary annotation in seconds. ``None`` (default)
        uses one sample (``1 / sfreq``); a float uses that many seconds.

    Returns
    -------
    merged : mne.io.BaseRaw
        Concatenated raw of shape ``(n_channels, sum(n_times))`` carrying the
        union of all segment annotations (onsets shifted by cumulative
        segment durations) plus ``len(raws) - 1`` boundary annotations.

    Raises
    ------
    ValueError
        If ``raws`` is empty, or if any two segments disagree on sampling
        rate, channel names, or channel order.
    """
    if not raws:
        raise ValueError("merge_recordings requires at least one Raw segment.")

    _validate_consistency(raws)

    # Single segment: passthrough copy, no boundary annotation.
    if len(raws) == 1:
        return raws[0].copy()

    sfreq = float(raws[0].info["sfreq"])
    gap_dur = (1.0 / sfreq) if gap_duration is None else float(gap_duration)

    # mne.concatenate_raws shifts each segment's annotations by the cumulative
    # duration of preceding segments and preserves channel info / sfreq.
    segments = [r.copy() for r in raws]
    merged = mne.concatenate_raws(segments)

    # Insert a BAD_break annotation at each boundary onset. Boundary i (between
    # segment i and i+1) lands at the cumulative sample count of the first i
    # segments, expressed in seconds from the merged recording's start.
    cumulative = 0.0
    boundary_onsets: list[float] = []
    for seg in raws[:-1]:
        cumulative += seg.n_times / sfreq
        boundary_onsets.append(cumulative)

    merged.annotations.append(
        onset=boundary_onsets,
        duration=[gap_dur] * len(boundary_onsets),
        description=[gap_label] * len(boundary_onsets),
    )
    return merged


def _validate_consistency(raws: list[mne.io.BaseRaw]) -> None:
    """Raise ValueError if segments disagree on sfreq or channel layout."""
    ref = raws[0]
    ref_sfreq = float(ref.info["sfreq"])
    ref_ch = list(ref.ch_names)
    for idx, seg in enumerate(raws[1:], start=1):
        if float(seg.info["sfreq"]) != ref_sfreq:
            raise ValueError(
                f"Sampling rate mismatch at segment {idx}: "
                f"segment 0 has {ref_sfreq} Hz, segment {idx} has "
                f"{float(seg.info['sfreq'])} Hz."
            )
        if list(seg.ch_names) != ref_ch:
            raise ValueError(
                f"Channel mismatch at segment {idx}: segment 0 has "
                f"{ref_ch}, segment {idx} has {list(seg.ch_names)}. "
                "Channel names and order must be identical across segments."
            )
