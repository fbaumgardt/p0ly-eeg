"""Raw EEG ingestion dispatcher.

Single entry point :func:`load_raw` selects a per-format reader by dispatching
on the **type** of ``source``:

* ``str | Path | os.PathLike`` → BrainVision reader (a concrete ``.vhdr`` path).
* :class:`mne_bids.BIDSPath`   → mne-bids reader (``read_raw_bids``).

The library is **path- and layout-agnostic** (ADR-005): it composes no
filesystem paths and builds no :class:`BIDSPath`. The pipeline owns all layout
(``sub-{subject}_task-{task}.vhdr`` naming, BIDS entity construction) and
hands the library a fully resolved ``source``.

This module is stateless and decoupled from Snakemake: it reads from the
filesystem and returns an :class:`mne.io.BaseRaw` but never writes derivative
files. Pipeline scripts are responsible for saving.

Forward compatibility (US-009): the ``str | Path`` branch will widen to also
accept ``list[str | Path]`` for multi-segment recording merge, with no
dispatcher rewrite.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids

__all__ = ["load_raw", "load_raw_brainvision", "load_raw_bids"]


def load_raw_brainvision(
    vhdr: str | Path | os.PathLike,
    montage: str | None = "easycap-M1",
) -> mne.io.BaseRaw:
    """Read a BrainVision recording from a concrete ``.vhdr`` file path.

    The ``.eeg`` / ``.vmrk`` siblings are resolved by
    :func:`mne.io.read_raw_brainvision`. EOG channels are marked and ``montage``
    is applied only when the recording has no montage set.

    Parameters
    ----------
    vhdr
        Concrete path to the BrainVision ``.vhdr`` header file (any path-like
        object). Pipeline-owned; the library composes no paths.
    montage
        MNE builtin montage name applied when the recording has none. Default
        ``"easycap-M1"``; ``None`` leaves the recording without a montage.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw object with montage set. Shape of ``raw.get_data()`` is
        ``(n_channels, n_times)`` (see SCHEMA §1).
    """
    raw = mne.io.read_raw_brainvision(Path(vhdr), preload=False, verbose="ERROR")
    raw.set_channel_types({k: "eog" for k in raw.ch_names if "EOG" in k.upper()})
    if montage is not None and raw.get_montage() is None:
        raw.set_montage(montage, on_missing="ignore")
    return raw


def load_raw_bids(
    bids_path: BIDSPath,
    montage: str | None = "easycap-M1",
) -> mne.io.BaseRaw:
    """Read a BIDS EEG recording via :func:`mne_bids.read_raw_bids`.

    The :class:`BIDSPath` is constructed by the **pipeline** (ADR-005); the
    library builds no BIDS entities. ``montage`` is applied only when the
    sidecar did not provide one.

    Parameters
    ----------
    bids_path
        Pipeline-constructed :class:`mne_bids.BIDSPath` identifying the
        recording. ``read_raw_bids`` resolves the concrete file at runtime.
    montage
        MNE builtin montage name applied when the sidecar lacks one. Default
        ``"easycap-M1"``; ``None`` leaves the recording without a montage.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw object with montage set. Shape of ``raw.get_data()`` is
        ``(n_channels, n_times)`` (see SCHEMA §1).
    """
    raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
    raw.set_channel_types({k: "eog" for k in raw.ch_names if "EOG" in k.upper()})
    if montage is not None and raw.get_montage() is None:
        raw.set_montage(montage, on_missing="ignore")
    return raw


def load_raw(
    source: str | Path | os.PathLike | BIDSPath,
    montage: str | None = "easycap-M1",
) -> mne.io.BaseRaw:
    """Dispatch to a per-format raw reader based on the type of ``source``.

    The library composes no paths and builds no :class:`BIDSPath` (ADR-005);
    the pipeline owns all filesystem layout and BIDS entity construction.

    Parameters
    ----------
    source
        Pipeline-resolved source identifying the recording:

        * ``str | Path | os.PathLike`` → a concrete BrainVision ``.vhdr`` path.
        * :class:`mne_bids.BIDSPath`    → a BIDS recording.

        US-009 will widen the ``str | Path`` branch to also accept
        ``list[str | Path]`` for multi-segment recording merge.
    montage
        MNE builtin montage name. Applied to BrainVision always (when missing)
        and to BIDS only when the sidecar lacks one. Default ``"easycap-M1"``;
        ``None`` keeps the recording's own (possibly absent) montage.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw object with montage set. Shape of ``raw.get_data()`` is
        ``(n_channels, n_times)`` (see SCHEMA §1).

    Raises
    ------
    TypeError
        If ``source`` is not a supported type (a path-like object for
        BrainVision or a :class:`BIDSPath` for BIDS).
    """
    # BIDSPath first: it is not a Path subclass, but checking it before the
    # path-like arm keeps the dispatch unambiguous.
    if isinstance(source, BIDSPath):
        return load_raw_bids(source, montage)
    if isinstance(source, (str, Path, os.PathLike)):
        return load_raw_brainvision(source, montage)
    raise TypeError(
        f"Unsupported source type {type(source).__name__!r}. "
        "Expected a concrete path (str | Path) for BrainVision or "
        "an mne_bids.BIDSPath for BIDS."
    )
