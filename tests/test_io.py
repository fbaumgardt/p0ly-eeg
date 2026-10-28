from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
import pybv
import pytest
from mne_bids import BIDSPath, write_raw_bids

from p0ly_utils.io import load_raw, load_raw_bids, load_raw_brainvision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CH_NAMES = ["Cz", "Fz", "Pz", "Oz"]
_SFREQ = 256.0


def _synthetic_raw(n_channels: int = 4, duration: float = 5.0) -> mne.io.RawArray:
    """Gaussian-noise RawArray on a few easycap-M1 channels (volts)."""
    ch_names = _CH_NAMES[:n_channels]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, int(_SFREQ * duration))) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=UTC))
    raw.info["line_freq"] = 50.0
    return raw


def _write_brainvision(raw: mne.io.RawArray, data_dir: Path, subject: str, task: str) -> Path:
    """Write a BrainVision triplet under {data_dir}/sub-{subject}/ and return the .vhdr."""
    subj_dir = data_dir / f"sub-{subject}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    pybv.write_brainvision(
        data=raw.get_data() * 1e6,  # pybv expects µV
        sfreq=_SFREQ,
        ch_names=raw.ch_names,
        fname_base=f"sub-{subject}_task-{task}",
        folder_out=str(subj_dir),
        overwrite=True,
        unit="µV",
    )
    return subj_dir / f"sub-{subject}_task-{task}.vhdr"


def _write_bids(raw: mne.io.RawArray, bids_root: Path, subject: str, task: str) -> BIDSPath:
    """Write a synthetic BIDS EEG dataset and return the recording BIDSPath."""
    bp = BIDSPath(subject=subject, task=task, datatype="eeg", root=str(bids_root))
    write_raw_bids(
        raw,
        bp,
        overwrite=True,
        verbose="ERROR",
        allow_preload=True,
        format="BrainVision",
    )
    return bp


# ---------------------------------------------------------------------------
# BrainVision
# ---------------------------------------------------------------------------


class TestLoadRawBrainVision:
    def test_reads_vhdr_and_sets_montage(self, tmp_path: Path) -> None:
        raw = _synthetic_raw()
        vhdr = _write_brainvision(raw, tmp_path, "001", "rest")

        out = load_raw_brainvision(vhdr)

        assert vhdr.exists()
        assert out.ch_names == raw.ch_names
        assert out.get_montage() is not None
        # easycap-M1 maps our channels; all four should have positions.
        pos = out.get_montage().get_positions()["ch_pos"]
        assert {ch: pos[ch] for ch in raw.ch_names} and not any(
            np.isnan(pos[ch]).any() for ch in raw.ch_names
        )

    def test_montage_is_configurable(self, tmp_path: Path) -> None:
        vhdr = _write_brainvision(_synthetic_raw(), tmp_path, "002", "rest")
        out = load_raw_brainvision(vhdr, montage="easycap-M10")
        assert out.get_montage() is not None

    def test_montage_none_leaves_recording_without_one(self, tmp_path: Path) -> None:
        vhdr = _write_brainvision(_synthetic_raw(), tmp_path, "002b", "rest")
        out = load_raw_brainvision(vhdr, montage=None)
        assert out.get_montage() is None

    def test_accepts_str_and_path(self, tmp_path: Path) -> None:
        vhdr = _write_brainvision(_synthetic_raw(), tmp_path, "003", "rest")
        from_str = load_raw_brainvision(str(vhdr))
        from_path = load_raw_brainvision(vhdr)
        assert from_str.ch_names == from_path.ch_names == _CH_NAMES

    def test_load_raw_dispatches_path_to_brainvision(self, tmp_path: Path) -> None:
        vhdr = _write_brainvision(_synthetic_raw(), tmp_path, "004", "rest")
        via_str = load_raw(str(vhdr))
        via_path = load_raw(vhdr)
        assert via_str.ch_names == via_path.ch_names == _CH_NAMES
        assert via_str.get_montage() is not None


# ---------------------------------------------------------------------------
# BIDS
# ---------------------------------------------------------------------------


class TestLoadRawBids:
    def test_reads_bids_recording(self, tmp_path: Path) -> None:
        raw = _synthetic_raw()
        bp = _write_bids(raw, tmp_path, "001", "rest")

        out = load_raw_bids(bp)

        assert bp.fpath.exists()
        assert out.ch_names == raw.ch_names

    def test_applies_montage_when_sidecar_lacks_one(self, tmp_path: Path) -> None:
        # Synthetic RawArray has no montage -> BIDS sidecar carries none -> fallback applies.
        bp = _write_bids(_synthetic_raw(), tmp_path, "001", "rest")
        out = load_raw_bids(bp)
        assert out.get_montage() is not None

    def test_load_raw_dispatches_bidspath_to_bids(self, tmp_path: Path) -> None:
        raw = _synthetic_raw()
        bp = _write_bids(raw, tmp_path, "001", "rest")

        out = load_raw(bp)

        assert out.ch_names == raw.ch_names


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestLoadRawDispatch:
    @pytest.mark.parametrize("bad_source", [123, None, 3.14, ["a", "b"], {"k": 1}])
    def test_unsupported_source_type_raises_type_error(
        self, bad_source: object, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="Unsupported source type"):
            load_raw(bad_source)  # type: ignore[arg-type]

    def test_default_montage_is_easycap_m1(self, tmp_path: Path) -> None:
        vhdr = _write_brainvision(_synthetic_raw(), tmp_path, "001", "rest")
        out = load_raw(vhdr)
        # easycap-M1 is the documented default; confirm a position is present.
        assert out.get_montage() is not None
