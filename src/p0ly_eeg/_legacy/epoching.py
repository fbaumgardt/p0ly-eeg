from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import p0ly_eeg.metadata as _meta_pkg
from p0ly_eeg.epoching import align_epochs_metadata
from p0ly_eeg.metadata import ExperimentSpec

logger = logging.getLogger(__name__)


def compute_epochs(
    raw: mne.io.BaseRaw,
    experiment: ExperimentSpec,
    intervals: dict[str, tuple[float, float]],
    csv_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, mne.Epochs]:
    """Create epochs for each timelock defined in an experiment spec.

    Metadata is parsed from annotations using the experiment's own
    ``get_metadata`` function, and epochs are aligned to metadata rows by
    reciprocal nearest-onset matching via :func:`align_epochs_metadata`.

    .. deprecated:: US-021
        This legacy module is unimported. ``ExperimentSpec.intervals`` was
        removed (windows now live in ``config.yaml epoching.intervals``);
        callers must pass an explicit ``intervals`` dict here.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data with annotations.
    experiment : ExperimentSpec
        Experiment specification defining timelocks and metadata extraction
        rules (no longer carries windows).
    intervals : dict[str, tuple[float, float]]
        Per-timelock ``(tmin, tmax)`` windows in seconds — sourced from
        ``config.yaml epoching.intervals`` (ADR-006).
    csv_path : str | None
        Path to an external trial-level CSV (e.g. PTB log for ``intwm``)
        whose columns are merged into the metadata by the experiment spec.
    output_dir : str | None
        Directory for ``-epo.fif.gz`` files.  Defaults to the raw file's
        parent directory.

    Returns
    -------
    dict[str, mne.Epochs]
        Mapping from each timelock name (e.g. ``"stim"``, ``"resp"``) to the
        corresponding Epochs object.
    """
    raw_file: str = str(raw.filenames[0])
    stem: str = Path(raw_file).stem
    logger.info("Loading %s", raw_file)

    # Work on a copy so we can clear the bads list without mutating the
    # caller's raw object.
    r: mne.io.BaseRaw = raw.copy()
    r.info["bads"] = []

    # ---- event extraction ----
    evts: np.ndarray
    ids: dict[str, int]
    evts, ids = mne.events_from_annotations(r)

    # Normalise "Stimulus/…" to "Stim/…" (some recording setups emit the
    # long prefix).
    ids = {k.replace("Stimulus", "Stim"): v for k, v in ids.items()}

    # Build a DataFrame for the metadata parser.  Annotations.onset is in
    # seconds; ``align_epochs_metadata`` matches on the ``Onset`` column in
    # seconds (SCHEMA §2).
    annot_df: pd.DataFrame = pd.DataFrame({
        "onset": r.annotations.onset,
        "description": r.annotations.description,
    })

    # ---- metadata parsing ----
    # Each experiment module (dmss, dotprobe, …) exposes a ``get_metadata``
    # function.  Dispatch by ExperimentSpec.name.
    exp_mod = getattr(_meta_pkg, experiment.name)
    md: pd.DataFrame = exp_mod.get_metadata(annot_df, f=csv_path)

    # Export metadata for inspection / debugging.
    md_csv: Path = (
        Path(output_dir or Path(raw_file).parent) / f"{stem}_metadata.csv"
    )
    md.to_csv(md_csv, index=False)
    logger.info("Metadata written to %s", md_csv)

    # ---- epoch each timelock ----
    epochs_dict: dict[str, mne.Epochs] = {}
    for lck in experiment.timelocks:
        try:
            tmin: float
            tmax: float
            tmin, tmax = intervals[lck]

            # Map spec code labels → MNE integer event IDs.
            evt_id: dict[str, int] = {
                k: ids[v] for k, v in experiment.timelocks[lck].items()
            }

            # Filter events to only those relevant to this timelock
            # (vectorized boolean indexing).
            target_ids: list[int] = list(evt_id.values())
            evt_lck: np.ndarray = evts[np.isin(evts[:, 2], target_ids)]

            # Epoch with ±1 s padding (headroom for downstream filtering /
            # time-frequency) and a 200 ms pre-event baseline.
            epoch: mne.Epochs = mne.Epochs(
                r,
                evt_lck,
                evt_id,
                tmin=tmin - 1.0,
                tmax=tmax + 1.0,
                baseline=(tmin, tmin + 0.2),
                preload=True,
                reject_by_annotation=False,
            )

            # Align epochs to metadata by reciprocal nearest onset match.
            # This subsets epochs to only those with a matching metadata row
            # and attaches the correctly-matched metadata.
            epoch = align_epochs_metadata(epoch, md)

            # Flag epochs whose annotations contain "bad".
            bad_mask: np.ndarray = np.array([
                any("bad" in str(annot[2]).lower() for annot in annots)
                for annots in epoch.get_annotations_per_epoch()
            ])
            epoch.metadata["BAD"] = bad_mask

            # Persist.
            out_dir: Path = Path(output_dir or Path(raw_file).parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path: Path = out_dir / f"{stem}_{lck}-epo.fif.gz"
            epoch.save(out_path, overwrite=True)
            logger.info("Saved %s", out_path)

            epochs_dict[lck] = epoch
        except Exception:
            logger.exception(
                "Failed to epoch timelock %r for %s", lck, raw_file
            )
            continue

    return epochs_dict
