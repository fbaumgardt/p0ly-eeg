"""Tests for p0ly_eeg.merge.merge_recordings (US-009)."""

from __future__ import annotations

from datetime import UTC, datetime

import mne
import numpy as np
import pytest

from p0ly_eeg.merge import merge_recordings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CH_NAMES = ["Cz", "Fz", "Pz", "Oz"]
_SFREQ = 256.0


def _synthetic_raw(
    n_channels: int = 4,
    duration: float = 5.0,
    seed: int = 42,
    first_samp: int = 0,
) -> mne.io.RawArray:
    """Gaussian-noise RawArray on a few channels (volts)."""
    ch_names = _CH_NAMES[:n_channels]
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_channels, int(_SFREQ * duration))) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose="ERROR")
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=UTC))
    raw.info["line_freq"] = 50.0
    return raw


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestMergeRecordings:
    def test_merge_two_segments_concatenates_length(self) -> None:
        a = _synthetic_raw(duration=5.0, seed=1)
        b = _synthetic_raw(duration=3.0, seed=2)
        merged = merge_recordings([a, b])

        assert merged.n_times == a.n_times + b.n_times
        assert merged.ch_names == a.ch_names
        assert float(merged.info["sfreq"]) == _SFREQ

    def test_merge_inserts_bad_break_at_each_boundary(self) -> None:
        a = _synthetic_raw(seed=1)
        b = _synthetic_raw(seed=2)
        c = _synthetic_raw(seed=3)
        merged = merge_recordings([a, b, c])

        breaks = [ann for ann in merged.annotations if ann["description"] == "BAD_break"]
        assert len(breaks) == 2  # one per boundary (n-1)
        # First boundary at end of segment a, second at end of a+b.
        np.testing.assert_allclose(
            breaks[0]["onset"], a.n_times / _SFREQ, atol=1e-9
        )
        np.testing.assert_allclose(
            breaks[1]["onset"], (a.n_times + b.n_times) / _SFREQ, atol=1e-9
        )

    def test_merge_single_segment_passthrough_no_break(self) -> None:
        a = _synthetic_raw(seed=1)
        merged = merge_recordings([a])

        assert merged.n_times == a.n_times
        assert not any(ann["description"] == "BAD_break" for ann in merged.annotations)
        # Passthrough is a copy, not the same object.
        assert merged is not a

    def test_merge_preserves_annotations_with_shifted_onsets(self) -> None:
        a = _synthetic_raw(duration=5.0, seed=1)
        b = _synthetic_raw(duration=3.0, seed=2)
        # An annotation 2s into segment a; one 1s into segment b.
        a.annotations.append(onset=[2.0], duration=[0.1], description=["stim"])
        b.annotations.append(onset=[1.0], duration=[0.1], description=["stim"])
        merged = merge_recordings([a, b])

        stims = [ann for ann in merged.annotations if ann["description"] == "stim"]
        assert len(stims) == 2
        onsets = sorted(float(ann["onset"]) for ann in stims)
        # Segment-a annotation keeps its onset; segment-b's shifts by a's duration.
        np.testing.assert_allclose(onsets[0], 2.0, atol=1e-9)
        np.testing.assert_allclose(onsets[1], 1.0 + a.n_times / _SFREQ, atol=1e-9)

    def test_merge_gap_label_configurable(self) -> None:
        a = _synthetic_raw(seed=1)
        b = _synthetic_raw(seed=2)
        merged = merge_recordings([a, b], gap_label="BAD_gap")

        assert any(ann["description"] == "BAD_gap" for ann in merged.annotations)
        assert not any(ann["description"] == "BAD_break" for ann in merged.annotations)

    def test_merge_gap_duration_configurable(self) -> None:
        a = _synthetic_raw(seed=1)
        b = _synthetic_raw(seed=2)
        merged = merge_recordings([a, b], gap_duration=0.5)

        breaks = [ann for ann in merged.annotations if ann["description"] == "BAD_break"]
        assert len(breaks) == 1
        np.testing.assert_allclose(float(breaks[0]["duration"]), 0.5, atol=1e-9)

    def test_merge_data_is_segment_abutment(self) -> None:
        a = _synthetic_raw(duration=2.0, seed=1)
        b = _synthetic_raw(duration=2.0, seed=2)
        merged = merge_recordings([a, b])

        got = merged.get_data()
        np.testing.assert_allclose(got[:, : a.n_times], a.get_data())
        np.testing.assert_allclose(got[:, a.n_times :], b.get_data())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMergeRecordingsValidation:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            merge_recordings([])

    def test_mismatched_sfreq_raises(self) -> None:
        a = _synthetic_raw(seed=1)
        b_data = np.random.default_rng(2).standard_normal((4, int(128 * 5.0))) * 1e-6
        info = mne.create_info(ch_names=_CH_NAMES, sfreq=128.0, ch_types="eeg")
        b = mne.io.RawArray(b_data, info, verbose="ERROR")
        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            merge_recordings([a, b])

    def test_mismatched_channels_raises(self) -> None:
        a = _synthetic_raw(seed=1)
        b = _synthetic_raw(seed=2)
        b.rename_channels({"Oz": "P4"})
        with pytest.raises(ValueError, match="Channel mismatch"):
            merge_recordings([a, b])

    def test_mismatched_channel_order_raises(self) -> None:
        a = _synthetic_raw(seed=1)
        b = _synthetic_raw(seed=2)
        b.reorder_channels(["Oz", "Pz", "Fz", "Cz"])
        with pytest.raises(ValueError, match="Channel mismatch"):
            merge_recordings([a, b])
