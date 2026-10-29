from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

    from p0ly_eeg.metadata.core import ExperimentSpec


def _match_onsets(
    epoch_onsets_ms: np.ndarray,
    meta_onsets_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align epoch and metadata trial lists by reciprocal nearest onset match."""
    n_epochs = len(epoch_onsets_ms)
    n_meta = len(meta_onsets_ms)
    if n_epochs == 0 or n_meta == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    arr = epoch_onsets_ms[:, np.newaxis] - meta_onsets_ms[np.newaxis, :]
    df_match_to_evt = np.sum(arr >= 0, axis=1) - 1
    evt_match_to_df = np.sum(arr < 0, axis=0)

    epoch_idx = np.arange(n_epochs)
    meta_idx = df_match_to_evt
    in_bounds = (meta_idx >= 0) & (meta_idx < n_meta)
    mutual = in_bounds & (evt_match_to_df[meta_idx] == epoch_idx)
    matched_epoch_idx = epoch_idx[mutual]
    matched_meta_idx = meta_idx[mutual]

    if len(matched_epoch_idx) == 0:
        return matched_epoch_idx, matched_meta_idx

    if n_meta >= 2:
        tolerance_ms = float(np.mean(np.diff(meta_onsets_ms)))
    else:
        tolerance_ms = float("inf")

    close_enough = np.abs(arr[matched_epoch_idx, matched_meta_idx]) < tolerance_ms
    return matched_epoch_idx[close_enough], matched_meta_idx[close_enough]


def align_epochs_metadata(epochs: mne.BaseEpochs, metadata: pd.DataFrame) -> mne.BaseEpochs:
    """Align events and metadata by their onset times (seconds).

    Thin Path-A onset matcher: subsets ``epochs`` to those whose event onset
    reciprocal-nearest-matches a metadata ``Onset`` and attaches the matched
    rows as ``epochs.metadata``. Mismatches are silently dropped here — use
    :func:`epoch_with_metadata` for the explicit mismatch bookkeeping
    (``excluded_trials.csv``) required by the pipeline epoch rule.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs whose event onsets are matched to metadata. Event onsets are
        read from ``epochs.events[:, 0]`` (samples) and converted to seconds
        via ``epochs.info["sfreq"]``.
    metadata : pd.DataFrame
        DataFrame with an ``Onset`` column in **seconds** (SCHEMA §2), e.g. the
        US-016 ``events_from_raw`` → ``parse_metadata`` output.

    Returns
    -------
    mne.BaseEpochs
        Epochs subset aligned to matching metadata rows, with
        ``.metadata`` attached.
    """
    epoch_onsets_s = epochs.events[:, 0] / epochs.info["sfreq"]
    meta_onsets_s = metadata["Onset"].to_numpy()
    matched_epoch_idx, matched_meta_idx = _match_onsets(epoch_onsets_s, meta_onsets_s)

    epochs = epochs[matched_epoch_idx]
    epochs.metadata = metadata.iloc[matched_meta_idx].reset_index(drop=True)
    return epochs


# Columns written to ``excluded_trials.csv`` (SCHEMA §6). ``subject`` /
# ``timelock`` scope a row to one (subject, timelock) epoch run; ``Block`` /
# ``Trial`` / ``Onset`` identify the trial; ``reason`` classifies the mismatch.
_EXCLUDED_COLUMNS = ["subject", "timelock", "Block", "Trial", "Onset", "reason"]


def _empty_excluded(subject: str | None, timelock: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in _EXCLUDED_COLUMNS})


def _excluded_rows(
    block,
    trial,
    onset,
    reason,
    subject: str | None,
    timelock: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": subject,
            "timelock": timelock,
            "Block": block,
            "Trial": trial,
            "Onset": onset,
            "reason": reason,
        }
    )


def _bad_epoch_mask(
    epochs: mne.BaseEpochs,
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Boolean per epoch: ``True`` where ``[onset+tmin, onset+tmax]`` overlaps a ``bad*`` annotation.

    Epochs are cut with ``reject_by_annotation=False`` so every candidate
    survives segmentation; this mask flags the ones whose window intersects
    any ``bad``-prefixed annotation on ``raw.annotations`` (e.g. US-008's
    ``bad_minmax_zscore``). Two half-open intervals coincide when
    ``ep_start < a_end`` and ``a_start < ep_end``.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs cut with ``reject_by_annotation=False`` (all candidates present).
    raw : mne.io.BaseRaw
        Raw whose ``annotations`` carry the bad-interval markers.
    tmin, tmax : float
        Epoch window relative to the timelock onset (seconds).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_epochs,)``.
    """
    sfreq = float(epochs.info["sfreq"])
    onsets = epochs.events[:, 0] / sfreq
    ep_start = onsets + tmin
    ep_end = onsets + tmax
    mask = np.zeros(len(onsets), dtype=bool)
    for ann in raw.annotations:
        if not str(ann["description"]).lower().startswith("bad"):
            continue
        a_start = float(ann["onset"])
        a_end = a_start + float(ann["duration"])
        mask |= (ep_start < a_end) & (a_start < ep_end)
    return mask


def validate_intervals(
    spec_timelocks: Mapping[str, object],
    config_intervals: Mapping[str, object],
    *,
    timelock: str | None = None,
) -> None:
    """Cross-check ``config`` epoch intervals against the spec's timelock keys.

    Epoch windows live in the pipeline config (``epoching.intervals``), the
    timelock event-code map lives in the experiment spec (``ExperimentSpec.
    timelocks``). They share keys but are sourced independently, so drift is
    possible. The guards are **asymmetric** (ADR-006):

    * a spec timelock with no config interval → ``warnings.warn`` and proceed
      (the spec may declare timelocks the current analysis isn't running);
    * a config interval key that is not a spec timelock → ``ValueError``
      (almost certainly a typo — no valid epoch can be cut for it).

    Parameters
    ----------
    spec_timelocks
        ``ExperimentSpec.timelocks`` (or any mapping keyed by timelock name).
    config_intervals
        ``config["epoching"]["intervals"]`` keyed by timelock name.
    timelock
        Optional single timelock being processed — narrows the spec-extra
        warning to that timelock only; ``None`` checks the full key sets.

    Raises
    ------
    ValueError
        If ``config_intervals`` has a key absent from ``spec_timelocks``.
    """
    spec_keys = set(spec_timelocks)
    cfg_keys = set(config_intervals)
    unknown = cfg_keys - spec_keys
    if unknown:
        raise ValueError(
            f"config epoching.intervals has key(s) {sorted(unknown)!r} "
            f"not present in the spec's timelocks {sorted(spec_keys)!r}; "
            f"this is likely a typo."
        )
    missing = spec_keys - cfg_keys
    if timelock is not None:
        if timelock in spec_keys and timelock not in cfg_keys:
            warnings.warn(
                f"timelock {timelock!r} is in the spec but has no entry in "
                f"config epoching.intervals; skipping it for this analysis run.",
                stacklevel=2,
            )
    elif missing:
        warnings.warn(
            f"spec timelock(s) {sorted(missing)!r} have no entry in config "
            f"epoching.intervals; skipping them for this analysis run.",
            stacklevel=2,
        )


def _trial_lookup(trial_onsets_s: np.ndarray):
    """Return (sorted_onsets, order) so searchsorted maps onsets to trial rows."""
    order = np.argsort(trial_onsets_s, kind="stable")
    return trial_onsets_s[order], order


def epoch_with_metadata(
    raw: mne.io.BaseRaw,
    spec: ExperimentSpec,
    timelock: str,
    metadata: pd.DataFrame,
    baseline: list[float] | tuple[float, float] | None,
    *,
    interval: list[float] | tuple[float, float],
    subject: str | None = None,
) -> tuple[mne.BaseEpochs, pd.DataFrame]:
    """Segment ``raw`` into epochs for one timelock and align trial metadata.

    Cuts ``mne.Epochs`` from the timelock's annotation events using the
    analysis-run ``interval`` → ``(tmin, tmax)`` (from ``config.yaml``
    ``epoching.intervals``, ADR-006) and the shared ``baseline`` window, then
    reconciles the by-trial ``metadata`` (US-016 output) with the
    cut epochs **explicitly** — mismatches are retained/logged, never silently
    dropped:

    * epochs with no matching metadata row → **retained with NaN metadata**
      and logged to the excluded frame as ``extra_epoch``;
    * metadata rows with no matching epoch → excluded frame, reason
      ``no_timelock_event`` (a pure alignment miss — a trial whose epoch was
      bad is *matched* during alignment, then logged as ``bad_interval``);
    * epochs overlapping a ``bad*`` annotation → flagged ``BAD=True`` in
      ``epochs.metadata`` after segmentation (so they stay present during
      alignment, keeping Path B ``Num_Sel`` contiguous), then dropped at the
      end and logged as ``bad_interval``.

    Alignment is key-based on ``(Block, Trial)`` for one-epoch-per-trial
    timelocks, and on ``(Block, Trial, Num_Sel)`` for the experiment's
    ``ExpandOnEvent`` timelock (e.g. IGT ``select``). Trial identity is
    assigned to each epoch by onset proximity to the metadata ``Onset`` column
    (seconds, SCHEMA §2).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Cleaned continuous raw (US-008 output) carrying event-marker
        ``Annotations`` and bad-interval ``Annotations``. Underlying data
        shape ``(n_channels, n_times)``.
    spec : ExperimentSpec
        Experiment spec providing ``timelocks`` / ``intervals`` / optional
        ``trial_expander``.
    timelock : str
        Key into ``spec.timelocks`` (the event-code map lives in the spec;
        the ``(tmin, tmax)`` window is passed via ``interval``).
    metadata : pd.DataFrame
        By-trial metadata (US-016 ``parse_metadata`` output) with ``Block``,
        ``Trial``, ``Onset`` (seconds) and, when ``expand_trials`` was set,
        ``Num_Sel``. One row per trial (or per selection for expanded rows).
    baseline : list[float] | tuple[float, float] | None
        Baseline window ``[start, end]`` in seconds (shared across timelocks,
        from ``config.yaml`` ``epoching.baseline``); ``None`` disables.
    interval : list[float] | tuple[float, float]
        Epoch window ``[tmin, tmax]`` in seconds for this timelock —
        analysis-run config from ``config.yaml`` ``epoching.intervals[timelock]``
        (ADR-006; the spec no longer carries windows).
    subject : str | None
        Subject id stamped onto excluded-frame rows for ``excluded_trials.csv``
        (SCHEMA §6). Pure label — no filesystem or signal use.

    Returns
    -------
    epochs : mne.BaseEpochs
        Epochs for this timelock (shape ``(n_epochs, n_channels, n_times)``)
        with ``.metadata`` attached; extra epochs carry NaN metadata rows.
        A boolean ``BAD`` column in ``.metadata`` flags bad-interval epochs
        before they are dropped (survivors all carry ``BAD=False``).
    excluded : pd.DataFrame
        ``excluded_trials.csv`` content (columns ``subject, timelock, Block,
        Trial, Onset, reason``). One row per mismatched metadata row or
        dropped/extra epoch.
    """
    # ---- events from annotations ----
    events, event_id = mne.events_from_annotations(raw)
    # Recordings may label markers "Stimulus/..."; specs use "Stim/...".
    event_id = {k.replace("Stimulus", "Stim"): v for k, v in event_id.items()}

    tl_map: dict[str, str] = spec.timelocks[timelock]
    evt_id: dict[str, int] = {
        label: event_id[code] for label, code in tl_map.items() if code in event_id
    }
    tmin, tmax = interval

    target_ids = np.fromiter(evt_id.values(), dtype=int)
    sel = (
        np.isin(events[:, 2], target_ids) if len(target_ids) else np.zeros(len(events), dtype=bool)
    )
    tl_events = events[sel]

    expander_code = spec.trial_expander.event_code if spec.trial_expander is not None else None
    is_expander = expander_code is not None and expander_code in set(tl_map.values())

    meta_cols = list(metadata.columns)

    # ---- empty-events guard: nothing to segment ----
    if len(tl_events) == 0:
        epochs = _empty_epochs(raw, tmin, tmax, baseline)
        excluded = _excluded_rows(
            metadata["Block"].to_numpy(),
            metadata["Trial"].to_numpy(),
            metadata["Onset"].to_numpy(),
            np.full(len(metadata), "no_timelock_event", dtype=object),
            subject,
            timelock,
        )
        return epochs, excluded

    # ---- segment (keep all candidates; flag bad intervals after) ----
    epochs = mne.Epochs(
        raw,
        tl_events,
        evt_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject_by_annotation=False,
        preload=True,
        verbose=False,
    )
    sfreq = float(epochs.info["sfreq"])

    # Bad-interval flag: ``True`` where the epoch window overlaps a ``bad*``
    # annotation. Bad epochs stay present through alignment (so Path B
    # ``Num_Sel`` stays contiguous) and drop at the very end.
    bad_flag = _bad_epoch_mask(epochs, raw, tmin, tmax)
    n_epochs = len(epochs)
    all_onsets_s = epochs.events[:, 0] / sfreq

    # ---- trial-level metadata view + trial onsets (seconds) ----
    has_num_sel = "Num_Sel" in metadata.columns
    if is_expander and has_num_sel:
        meta_exp = metadata.reset_index(drop=True)
        trial_meta = meta_exp.groupby(["Block", "Trial"], as_index=False).first()
    else:
        # Collapse to one row per (Block, Trial). For expanded metadata used
        # against a non-expander timelock (e.g. IGT submit/fdb), the first row
        # per trial carries the trial-level columns shared across selections.
        if has_num_sel:
            trial_meta = metadata.groupby(["Block", "Trial"], as_index=False).first()
        else:
            trial_meta = metadata.reset_index(drop=True)
        meta_exp = None

    trial_onsets_s = trial_meta["Onset"].to_numpy(dtype=float)
    sorted_onsets, order = _trial_lookup(trial_onsets_s)
    bt = trial_meta[["Block", "Trial"]].to_numpy()

    # Map epoch onsets to nearest preceding trial (searchsorted, seconds).
    # -1 ⇒ onset precedes the first trial (NaN Block/Trial downstream).
    def _assign_trials(onsets_s: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_onsets, onsets_s, side="right") - 1
        trial_row = np.where(idx >= 0, order[np.clip(idx, 0, None)], -1)
        return trial_row

    def _block_trial(trial_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Block, Trial) for each row index, NaN where ``trial_row < 0``."""
        safe = trial_row.clip(0, None)
        block = np.where(trial_row >= 0, bt[safe, 0], np.nan)
        trial = np.where(trial_row >= 0, bt[safe, 1], np.nan)
        return block, trial

    excluded_frames: list[pd.DataFrame] = []

    def _log_positions(pos: np.ndarray, reason: str) -> None:
        """Log epoch positions to the excluded frame, trial id by onset proximity."""
        if len(pos) == 0:
            return
        onsets = all_onsets_s[pos]
        block, trial = _block_trial(_assign_trials(onsets))
        excluded_frames.append(
            _excluded_rows(
                block, trial, onsets,
                np.full(len(pos), reason, dtype=object),
                subject, timelock,
            )
        )

    def _log_meta_rows(frame: pd.DataFrame) -> None:
        """Log metadata rows with no matching epoch as ``no_timelock_event``."""
        if len(frame) == 0:
            return
        excluded_frames.append(
            _excluded_rows(
                frame["Block"].to_numpy(),
                frame["Trial"].to_numpy(),
                frame["Onset"].to_numpy(),
                np.full(len(frame), "no_timelock_event", dtype=object),
                subject, timelock,
            )
        )

    if is_expander and meta_exp is not None:
        # ---- Path B: N epochs per trial, key-join on (Block, Trial, Num_Sel) ----
        ep = pd.DataFrame({"Onset_s": all_onsets_s})
        ep["Block"], ep["Trial"] = _block_trial(_assign_trials(all_onsets_s))

        # Rank epochs within (Block, Trial) by onset → Num_Sel (vectorized
        # groupby cumcount on the sorted-by-onset order).
        ep_sorted = ep.loc[ep["Block"].notna()].sort_values(["Block", "Trial", "Onset_s"])
        ep_sorted["Num_Sel"] = ep_sorted.groupby(["Block", "Trial"]).cumcount()
        ep["Num_Sel"] = np.nan
        ep.loc[ep_sorted.index, "Num_Sel"] = ep_sorted["Num_Sel"].to_numpy()

        merged = ep.merge(meta_exp, on=["Block", "Trial", "Num_Sel"], how="left", sort=False)
        aligned = merged[meta_cols].reset_index(drop=True)

        # Excluded metadata rows: expanded rows with no matching epoch.
        matched_keys = merged.loc[
            merged["Num_Sel"].notna(), ["Block", "Trial", "Num_Sel"]
        ].drop_duplicates()
        unmatched = meta_exp.merge(
            matched_keys, on=["Block", "Trial", "Num_Sel"], how="left", indicator=True
        )
        _log_meta_rows(unmatched[unmatched["_merge"] == "left_only"])
    else:
        # ---- Path A: 1 epoch per trial, reciprocal onset match on seconds ----
        matched_ep, matched_meta = _match_onsets(all_onsets_s, trial_onsets_s)
        aligned = pd.DataFrame(np.nan, index=np.arange(n_epochs), columns=meta_cols)
        if len(matched_ep):
            aligned.iloc[matched_ep] = trial_meta.iloc[matched_meta][meta_cols].to_numpy()

        # Excluded metadata rows (trials with no surviving epoch).
        unmatched_rows = np.setdiff1d(np.arange(len(trial_meta)), matched_meta)
        _log_meta_rows(trial_meta.iloc[unmatched_rows])

    # Attach metadata (BAD flags bad-interval epochs; survivors carry False).
    aligned["BAD"] = bad_flag
    epochs.metadata = aligned.reset_index(drop=True)

    # Extra epochs: retained NaN-metadata epochs that are not bad. Bad no-trial
    # epochs are excluded below as ``bad_interval`` (precedence over extra_epoch).
    extra_pos = np.flatnonzero(aligned["Block"].isna().to_numpy() & ~bad_flag)
    _log_positions(extra_pos, "extra_epoch")

    # Bad-interval epochs drop at the very end: they survived segmentation and
    # participated in alignment (keeping Path B ``Num_Sel`` contiguous), so each
    # is logged as ``bad_interval`` with ``(Block, Trial)`` when it mapped to a
    # trial, else NaN trial.
    _log_positions(np.flatnonzero(bad_flag), "bad_interval")

    # Drop bad epochs now (the ``BAD`` column stays in ``epochs.metadata`` for
    # audit); survivors all carry ``BAD=False``.
    epochs = epochs[~bad_flag]

    excluded = (
        pd.concat(excluded_frames, ignore_index=True)
        if excluded_frames
        else _empty_excluded(subject, timelock)
    )
    return epochs, excluded


def _empty_epochs(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    baseline: list[float] | tuple[float, float] | None,
) -> mne.BaseEpochs:
    """Construct a zero-epoch ``mne.Epochs`` matching ``raw``'s info/timing."""
    sfreq = float(raw.info["sfreq"])
    n_times = int(round((tmax - tmin) * sfreq))
    data = np.zeros((0, len(raw.info["ch_names"]), max(n_times, 1)), dtype=np.float64)
    # Explicit ``drop_log=()`` avoids MNE's default drop-log computation, which
    # calls ``max(self.selection)`` and raises on an empty (zero-epoch) selection.
    return mne.EpochsArray(data, raw.info, tmin=tmin, baseline=baseline, drop_log=(), verbose=False)
