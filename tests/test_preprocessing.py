from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest
from mne.preprocessing import ICA
from mne_icalabel import label_components

from p0ly_eeg.preprocessing import (
    _minmax_zscore,
    _resolve_fit_window,
    artefact_rejection,
    fix_channels,
    ica_clean_dnn,
    ica_clean_regression,
    interpolate_bads,
    preprocess_raw,
)

_CH_NAMES = json.loads((Path(__file__).parent / "data" / "ch_names.json").read_text())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(n_channels: int = 10, duration: float = 10.0, sfreq: float = 256.0) -> mne.io.RawArray:
    """Gaussian noise on all channels."""
    ch_names = _CH_NAMES["eeg"][:n_channels]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False).set_montage("easycap-M1", on_missing="ignore")


def _make_raw_with_spike(
    channel: int,
    sample: int,
    value: float = 10.0,
    *,
    n_channels: int = 10,
    duration: float = 10.0,
    sfreq: float = 256.0,
    all_channels: bool = False,
) -> mne.io.RawArray:
    """Build raw data with an injected amplitude outlier at construction time."""
    ch_names = _CH_NAMES["eeg"][:n_channels]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    if all_channels:
        data[:, sample] = value
    else:
        data[channel, sample] = value
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False).set_montage("easycap-M1", on_missing="ignore")


def _make_raw_with_eog(
    n_eeg: int = 10, duration: float = 60.0, sfreq: float = 256.0
) -> mne.io.RawArray:
    """EEG + EOG channels; EOG carries a 1 Hz blink-like sinusoid mixed into EEG."""
    ch_names = _CH_NAMES["eeg"][:n_eeg]
    eog_names = _CH_NAMES["eog"]
    n_eog = len(eog_names)
    rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq

    # One sinusoid per EOG channel, each with independent noise
    blink = 50e-6 * np.sin(2 * np.pi * 1.0 * t)
    eog_data = blink[np.newaxis, :] + rng.standard_normal((n_eog, n_samples)) * 2e-6

    eeg_data = rng.standard_normal((n_eeg, n_samples)) * 1e-6
    # Mix mean EOG signal into EEG; amplitude diminishes away from frontal channels
    eeg_data += eog_data.mean(axis=0) * 0.5

    data = np.vstack([eeg_data, eog_data])
    info = mne.create_info(
        ch_names=ch_names + eog_names,
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["eog"] * n_eog,
    )
    return mne.io.RawArray(data, info, verbose=False)


# Threshold slightly below 3.0: with one dominant outlier among many tiny
# channels, |z| converges to 3.0 and float rounding can land just below it.
DETECTION_THRESHOLD = 2.9


class TestMinmaxZscore:
    def test_detects_outlier_channel(self):
        raw = _make_raw_with_spike(channel=5, sample=100)
        mask = _minmax_zscore(raw, axis="channels", threshold=DETECTION_THRESHOLD)
        assert mask[5]
        assert mask.sum() == 1

    def test_no_outlier_all_false(self):
        raw = _make_raw(n_channels=10)
        mask = _minmax_zscore(raw, axis="channels", threshold=3.0)
        assert not mask.any()

    def test_iterative_refinement_catches_more(self):
        raw = _make_raw(n_channels=20)
        data = raw.get_data()
        data[3, 50] = 0.5
        data[7, 50] = 10.0
        raw._data[:] = data
        mask_1 = _minmax_zscore(raw, threshold=3.0, max_iter=1)
        mask_3 = _minmax_zscore(raw, threshold=3.0, max_iter=3)
        assert mask_3.sum() >= mask_1.sum()

    def test_time_axis_flags_bad_epochs(self):
        raw = _make_raw_with_spike(
            channel=0, sample=768, duration=10.0, all_channels=True
        )
        epo = mne.make_fixed_length_epochs(raw, duration=1.0, verbose=False)
        mask = _minmax_zscore(epo, axis="time", threshold=DETECTION_THRESHOLD)
        assert mask[3]

    def test_mask_excludes_from_baseline(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 2560)) * 1e-6
        data[0, 50] = 0.01
        data[1, 50] = 10.0
        info = mne.create_info([f"EEG{i:03d}" for i in range(10)], 256.0, "eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        without_mask = _minmax_zscore(raw, threshold=3.0)
        pre_mask = np.zeros(10, dtype=bool)
        pre_mask[1] = True
        with_mask = _minmax_zscore(raw, threshold=3.0, mask=pre_mask)
        assert not without_mask.any()
        assert with_mask[1]


# ---------------------------------------------------------------------------
# fix_channels
# ---------------------------------------------------------------------------


class TestFixChannels:
    def test_outlier_added_to_bads(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert "Pz" in result.info["bads"]

    def test_does_not_mutate_input(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        _ = fix_channels(raw)
        assert raw.info["bads"] == []

    def test_preserves_existing_bads(self):
        raw = _make_raw_with_spike(channel=5, sample=100)
        raw.info["bads"] = ["Cz"]
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert "Cz" in result.info["bads"]
        assert "C4" in result.info["bads"]

    def test_no_bad_channels_returns_empty_bads(self):
        raw = _make_raw(n_channels=10)
        result = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert result.info["bads"] == []


# ---------------------------------------------------------------------------
# interpolate_bads
# ---------------------------------------------------------------------------


class TestInterpolateBads:
    def test_interpolates_flagged_channels(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        flagged = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        assert flagged.info["bads"] != []
        interp = interpolate_bads(flagged)
        assert interp.info["bads"] == []

    def test_does_not_mutate_input(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        flagged = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        bads_before = list(flagged.info["bads"])
        _ = interpolate_bads(flagged)
        assert flagged.info["bads"] == bads_before

    def test_reset_bads_false_keeps_bads(self):
        raw = _make_raw_with_spike(channel=3, sample=100)
        flagged = fix_channels(raw, threshold=DETECTION_THRESHOLD)
        interp = interpolate_bads(flagged, reset_bads=False)
        assert interp.info["bads"] != []

    def test_no_bads_is_noop(self):
        raw = _make_raw(n_channels=10)
        interp = interpolate_bads(raw)
        assert interp.info["bads"] == []


# ---------------------------------------------------------------------------
# artefact_rejection
# ---------------------------------------------------------------------------


class TestArtefactRejection:
    def test_flags_segment_with_spike(self):
        raw = _make_raw_with_spike(
            channel=0, sample=768, duration=10.0, sfreq=256.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=DETECTION_THRESHOLD, duration=1.0)
        assert len(annot) >= 1
        assert any(abs(onset - 3.0) < 1.0 for onset in annot.onset)

    def test_clean_recording_no_annotations(self):
        raw = _make_raw(duration=5.0)
        annot = artefact_rejection(raw, threshold=3.0, duration=0.5)
        assert len(annot) == 0

    def test_annotation_label(self):
        raw = _make_raw_with_spike(
            channel=0, sample=500, duration=5.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=DETECTION_THRESHOLD, duration=0.5)
        for desc in annot.description:
            assert desc == "bad_minmax_zscore"

    def test_stimulation_mask_excludes_window(self):
        raw = _make_raw_with_spike(
            channel=0, sample=512, duration=10.0, sfreq=256.0, all_channels=True
        )
        data = raw.get_data()
        data[:, 1792] = 0.01
        raw._data[:] = data
        annot_masked = artefact_rejection(
            raw, threshold=DETECTION_THRESHOLD, duration=1.0, stimulation=(0.0, 5.0)
        )
        assert len(annot_masked) >= 1

    def test_stimulation_none_endpoints(self):
        raw = _make_raw_with_spike(
            channel=0, sample=256, duration=10.0, all_channels=True
        )
        annot = artefact_rejection(raw, threshold=3.0, duration=1.0, stimulation=(None, None))
        assert len(annot) == 0

    def test_partial_rejection(self):
        """Only some epochs contain artefacts → only those annotated.

        10s / 1s epochs → 10 epochs. One spike in epoch 4 (sample 1152).
        With 1 of 10 epochs as an outlier, the z-score clears the threshold;
        the remaining 9 clean epochs are not flagged.
        """
        raw = _make_raw(duration=10.0, sfreq=256.0)
        data = raw.get_data()
        data[:, 1152] = 10.0  # epoch 4 (samples 1024–1279)
        raw._data[:] = data
        annot = artefact_rejection(raw, threshold=DETECTION_THRESHOLD, duration=1.0)
        assert len(annot) == 1
        assert any(abs(onset - 4.0) < 1.0 for onset in annot.onset)


# ---------------------------------------------------------------------------
# ica_clean_dnn
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIcaCleanDnn:
    @pytest.fixture
    def long_raw(self):
        """ICA needs enough data; 60s at 256 Hz with 20 channels. Reference is set to average of all channels."""
        return _make_raw(n_channels=20, duration=60.0, sfreq=256.0).set_eeg_reference()

    def test_returns_tuple(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_does_not_mutate_input(self, long_raw):
        orig_data = long_raw.get_data().copy()
        _ = ica_clean_dnn(long_raw)
        np.testing.assert_array_equal(long_raw.get_data(), orig_data)

    def test_fit_window_float_centered_crop(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, fit_window=30.0)
        assert ica.n_iter_ is not None

    def test_fit_window_tuple_crop(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, fit_window=(5.0, 55.0))
        assert ica.n_iter_ is not None

    def test_fit_window_list_crop(self, long_raw):
        # The list form is what YAML config round-trips [tmin, tmax] to.
        cleaned, ica = ica_clean_dnn(long_raw, fit_window=[5.0, 55.0])
        assert ica.n_iter_ is not None

    def test_fit_window_none_uses_full(self, long_raw):
        cleaned, ica = ica_clean_dnn(long_raw, fit_window=None)
        assert ica.n_iter_ is not None

    def test_exclude_components_empty(self, long_raw):
        _, ica = ica_clean_dnn(long_raw, exclude_components=[])
        assert ica.exclude == []

    def test_threshold_excludes_subset_of_label_matches(self, long_raw):
        # First establish the label-only exclusions and their probabilities.
        r_ic = long_raw.copy().filter(1.0, 100.0)
        ica_ref = ICA(
            n_components=None,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=97,
            max_iter="auto",
        )
        ica_ref.fit(r_ic)
        labels = label_components(r_ic, ica_ref, method="iclabel")
        non_brain = ["eye blink", "muscle artifact"]
        matching = [
            (i, float(labels["y_pred_proba"][i]))
            for i, lab in enumerate(labels["labels"])
            if lab in non_brain
        ]
        if not matching:
            pytest.skip("No non-brain components in fixture; threshold path untestable.")
        # A threshold just above the max matching probability excludes nothing.
        max_p = max(p for _, p in matching)
        _, ica_high = ica_clean_dnn(
            long_raw, threshold=max_p + 1e-6
        )
        assert ica_high.exclude == []
        # A threshold of 0.0 excludes every label match (== label-only behaviour).
        _, ica_zero = ica_clean_dnn(long_raw, threshold=0.0)
        assert ica_zero.exclude == [i for i, _ in matching]

    def test_threshold_none_matches_label_only(self, long_raw):
        _, ica_none = ica_clean_dnn(long_raw, threshold=None)
        _, ica_label = ica_clean_dnn(long_raw)
        assert ica_none.exclude == ica_label.exclude


# ---------------------------------------------------------------------------
# ica_clean_regression
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIcaCleanRegression:
    @pytest.fixture
    def raw_with_eog(self):
        return _make_raw_with_eog(n_eeg=20, duration=60.0, sfreq=256.0)

    def test_returns_tuple(self, raw_with_eog):
        cleaned, ica = ica_clean_regression(raw_with_eog, components=10)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_does_not_mutate_input(self, raw_with_eog):
        orig_data = raw_with_eog.get_data().copy()
        _ = ica_clean_regression(raw_with_eog, components=10)
        np.testing.assert_array_equal(raw_with_eog.get_data(), orig_data)

    def test_auto_detects_eog_channels(self, raw_with_eog):
        _, ica = ica_clean_regression(raw_with_eog, components=10)
        assert isinstance(ica.exclude, list)

    def test_no_eog_channels_no_exclusions(self):
        raw = _make_raw(n_channels=20, duration=60.0)
        _, ica = ica_clean_regression(raw, components=10)
        assert ica.exclude == []

    def test_fit_window_tuple_restricts_fit(self, raw_with_eog):
        _, ica = ica_clean_regression(raw_with_eog, components=10, fit_window=(5.0, 50.0))
        assert ica.n_iter_ is not None

    def test_fit_window_float_centered(self, raw_with_eog):
        # Regression previously could not take a centred window; now it can.
        _, ica = ica_clean_regression(raw_with_eog, components=10, fit_window=30.0)
        assert ica.n_iter_ is not None


# ---------------------------------------------------------------------------
# preprocess_raw (integration of the full chain)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPreprocessRaw:
    """End-to-end chain: filter -> bad-channels -> ICA -> sliding-window reject.

    Exercises the same path the pipeline ``preprocess`` rule calls, so the
    chain is unit-tested here instead of only via a slow ``snakemake`` run.
    """

    @pytest.fixture
    def raw(self):
        # 20 channels / 20 s referenced to a common average so ICLabel does not
        # warn about CAR; montage applied (mirrors US-007 ingestion contract).
        return (
            _make_raw(n_channels=20, duration=20.0, sfreq=256.0)
            .set_eeg_reference(verbose=False)
        )

    def _cfg(self, **overrides):
        cfg = dict(
            l_freq=1.0,
            h_freq=45.0,
            bad_channel_z_thresh=2.5,
            ica_strategy="mne-icalabel",
            icalabel_threshold=0.75,
            interval_window_ms=500,
            interval_reject_z_thresh=3.0,
        )
        cfg.update(overrides)
        return cfg

    def test_returns_cleaned_raw_bad_channels_and_ica(self, raw):
        cleaned, bad_channels, ica = preprocess_raw(raw, **self._cfg())
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(bad_channels, list)
        assert isinstance(ica, mne.preprocessing.ICA)
        # interpolation clears bads on the cleaned raw
        assert cleaned.info["bads"] == []

    def test_ica_object_carries_exclude_list(self, raw):
        _, _, ica = preprocess_raw(raw, **self._cfg())
        assert isinstance(ica.exclude, list)
        # exclude indices are within the fitted component count
        assert all(0 <= i < ica.n_components_ for i in ica.exclude)

    def test_does_not_mutate_input(self, raw):
        orig = raw.get_data().copy()
        _ = preprocess_raw(raw, **self._cfg())
        np.testing.assert_array_equal(raw.get_data(), orig)

    def test_requires_montage(self):
        raw = _make_raw(n_channels=20, duration=20.0, sfreq=256.0)
        raw.set_eeg_reference(verbose=False)
        # strip montage to assert the precondition guard
        raw = raw.copy().set_montage(None)
        with pytest.raises(RuntimeError, match="Montage missing"):
            preprocess_raw(raw, **self._cfg())

    def test_unknown_ica_strategy_raises(self, raw):
        with pytest.raises(ValueError, match="Unknown ica_strategy"):
            preprocess_raw(raw, **self._cfg(ica_strategy="bogus"))

    def test_find_bads_eog_strategy_runs(self):
        # The regression strategy needs EOG channels; reuse the EOG helper.
        raw = _make_raw_with_eog(n_eeg=20, duration=20.0, sfreq=256.0)
        raw = raw.set_eeg_reference(verbose=False).set_montage(
            "easycap-M1", on_missing="ignore"
        )
        cleaned, bad_channels, ica = preprocess_raw(
            raw, **self._cfg(ica_strategy="find_bads_eog")
        )
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert isinstance(bad_channels, list)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_persists_bad_interval_annotations(self, raw):
        # Inject a moderate-amplitude transient to trigger sliding-window
        # rejection without destabilising the ICA fit (a huge spike raises
        # infomax ``log(n_features**2)`` to a ZeroDivisionError).
        data = raw.get_data()
        data[:, 4000:4050] += 1e-3  # ~1 mV step across all channels
        raw._data[:] = data
        cleaned, _, _ = preprocess_raw(raw, **self._cfg(interval_reject_z_thresh=2.5))
        descs = list(cleaned.annotations.description)
        assert "bad_minmax_zscore" in descs

    def test_clean_recording_has_no_bad_annotations(self, raw):
        cleaned, _, _ = preprocess_raw(raw, **self._cfg())
        assert all(d != "bad_minmax_zscore" for d in cleaned.annotations.description)


# ---------------------------------------------------------------------------
# preprocess_raw — optional steps (each parameter None skips its step)
# ---------------------------------------------------------------------------


class TestPreprocessRawOptional:
    """Every step is optional; a None parameter skips its step."""

    @pytest.fixture
    def raw(self):
        return (
            _make_raw(n_channels=20, duration=20.0, sfreq=256.0)
            .set_eeg_reference(verbose=False)
        )

    def test_all_none_returns_raw_with_no_steps_applied(self, raw):
        cleaned, bad_channels, ica = preprocess_raw(raw)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert bad_channels == []
        assert ica is None
        # only the preliminary average reference ran — a numerical no-op on an
        # already-referenced fixture (float noise only), no bad-interval annotations
        np.testing.assert_allclose(cleaned.get_data(), raw.get_data())
        assert all(d != "bad_minmax_zscore" for d in cleaned.annotations.description)

    def test_all_none_does_not_mutate_input(self, raw):
        orig = raw.get_data().copy()
        _ = preprocess_raw(raw)
        np.testing.assert_array_equal(raw.get_data(), orig)

    def test_all_none_still_requires_montage(self):
        raw = _make_raw(n_channels=20, duration=20.0, sfreq=256.0)
        raw.set_eeg_reference(verbose=False)
        raw = raw.copy().set_montage(None)
        with pytest.raises(RuntimeError, match="Montage missing"):
            preprocess_raw(raw)  # no kwargs -> all steps skipped, guard still fires

    def test_filter_only_runs_without_other_steps(self, raw):
        cleaned, bad_channels, ica = preprocess_raw(raw, l_freq=1.0, h_freq=45.0)
        assert bad_channels == []
        assert ica is None
        # filter ran: data differs from input
        assert not np.array_equal(cleaned.get_data(), raw.get_data())
        assert all(d != "bad_minmax_zscore" for d in cleaned.annotations.description)

    def test_filter_one_sided_lowpass_only(self, raw):
        # l_freq=None, h_freq set -> lowpass-only; must run without error (S1).
        cleaned, _, _ = preprocess_raw(raw, l_freq=None, h_freq=20.0)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert not np.array_equal(cleaned.get_data(), raw.get_data())

    def test_filter_both_none_skips_filter(self, raw):
        cleaned, _, _ = preprocess_raw(raw, l_freq=None, h_freq=None)
        np.testing.assert_allclose(cleaned.get_data(), raw.get_data())

    def test_bad_channels_only_flags_and_interpolates(self):
        # Outlier channel -> flagged + interpolated; no filter/ICA/annotations.
        raw = _make_raw_with_spike(channel=3, sample=100, n_channels=20, duration=20.0)
        raw = raw.set_eeg_reference(verbose=False)
        cleaned, bad_channels, ica = preprocess_raw(raw, bad_channel_z_thresh=2.5)
        assert len(bad_channels) >= 1
        assert ica is None
        assert cleaned.info["bads"] == []  # interpolated -> cleared
        assert all(d != "bad_minmax_zscore" for d in cleaned.annotations.description)

    def test_bad_channel_z_thresh_none_skips_bad_channels(self, raw):
        cleaned, bad_channels, ica = preprocess_raw(
            raw, l_freq=None, h_freq=None, bad_channel_z_thresh=None
        )
        assert bad_channels == []
        assert ica is None
        np.testing.assert_allclose(cleaned.get_data(), raw.get_data())

    def test_ica_strategy_none_skips_ica_returns_none(self, raw, caplog):
        # With ICA skipped the run should be fast, emit no 'Fitting ICA' log,
        # and return None for the ICA object.
        import logging

        with caplog.at_level(logging.INFO):
            cleaned, _, ica = preprocess_raw(
                raw, l_freq=None, h_freq=None, ica_strategy=None
            )
        assert not any("Fitting ICA" in rec.message for rec in caplog.records)
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert ica is None

    @pytest.mark.slow
    def test_ica_strategy_set_returns_ica_object(self, raw):
        # ICA enabled -> a fitted ICA object is returned (reportable).
        _, _, ica = preprocess_raw(
            raw, l_freq=None, h_freq=None, ica_strategy="mne-icalabel", icalabel_threshold=0.75
        )
        assert isinstance(ica, mne.preprocessing.ICA)
        assert ica.n_components_ is not None

    def test_icalabel_threshold_ignored_when_ica_skipped(self, raw):
        # threshold is inert when ica_strategy is None; no error raised.
        cleaned, _, ica = preprocess_raw(
            raw, l_freq=None, h_freq=None, ica_strategy=None, icalabel_threshold=0.9
        )
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert ica is None

    def test_unknown_ica_strategy_still_raises(self, raw):
        # None skips ICA; an unrecognised string is still an error (S2).
        with pytest.raises(ValueError, match="Unknown ica_strategy"):
            preprocess_raw(raw, l_freq=None, h_freq=None, ica_strategy="bogus")

    def test_sliding_window_partial_skips_step(self, raw):
        # interval_window_ms set but interval_reject_z_thresh=None -> step skipped,
        # no bad_minmax_zscore annotations added (S4 + both-required rule).
        cleaned, _, _ = preprocess_raw(
            raw, l_freq=None, h_freq=None, interval_window_ms=500, interval_reject_z_thresh=None
        )
        assert all(d != "bad_minmax_zscore" for d in cleaned.annotations.description)

    def test_sliding_window_both_set_runs_step(self, raw):
        # Inject a transient so the step has something to flag.
        data = raw.get_data()
        data[:, 4000:4050] += 1e-3
        raw._data[:] = data
        cleaned, _, _ = preprocess_raw(
            raw, l_freq=None, h_freq=None, interval_window_ms=500, interval_reject_z_thresh=2.5
        )
        descs = list(cleaned.annotations.description)
        assert "bad_minmax_zscore" in descs


# ---------------------------------------------------------------------------
# _resolve_fit_window / preprocess_raw ica_fit_window (US-022)
# ---------------------------------------------------------------------------


class TestResolveFitWindow:
    @pytest.fixture
    def raw(self):
        # 10 s at 256 Hz; times run [0, 10). Midpoint = 5.0.
        return _make_raw(n_channels=4, duration=10.0, sfreq=256.0)

    def test_none_returns_none(self, raw):
        assert _resolve_fit_window(raw, None) is None

    def test_float_centers_on_midpoint(self, raw):
        # Centred on the time-axis midpoint (raw.times[-1] is the last SAMPLE
        # time, not the duration): mid = (times[0] + times[-1]) / 2.
        mid = (raw.times[0] + raw.times[-1]) / 2.0
        tmin, tmax = _resolve_fit_window(raw, 4.0)
        np.testing.assert_allclose([tmin, tmax], [mid - 2.0, mid + 2.0], atol=1e-9)

    def test_tuple_passthrough(self, raw):
        tmin, tmax = _resolve_fit_window(raw, (1.0, 9.0))
        assert (tmin, tmax) == (1.0, 9.0)

    def test_list_passthrough(self, raw):
        # YAML config round-trips [tmin, tmax] to a list.
        tmin, tmax = _resolve_fit_window(raw, [1.0, 9.0])
        assert (tmin, tmax) == (1.0, 9.0)

    def test_negative_float_raises(self, raw):
        with pytest.raises(ValueError, match="non-negative"):
            _resolve_fit_window(raw, -2.0)

    def test_inverted_tuple_raises(self, raw):
        with pytest.raises(ValueError, match="tmin < tmax"):
            _resolve_fit_window(raw, (7.0, 3.0))

    def test_equal_bounds_raise(self, raw):
        with pytest.raises(ValueError, match="tmin < tmax"):
            _resolve_fit_window(raw, (5.0, 5.0))

    def test_wrong_arity_raises(self, raw):
        with pytest.raises(ValueError, match="arity"):
            _resolve_fit_window(raw, (1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# preprocess_raw (orchestrator: ica_fit_window is a pure pass-through)
# ---------------------------------------------------------------------------


class TestPreprocessRawIcaFitWindow:
    @pytest.fixture
    def raw(self):
        # Montage is applied (preprocess_raw asserts it). Enough data for ICA.
        return _make_raw(n_channels=20, duration=60.0, sfreq=256.0).set_eeg_reference()

    def test_ica_skipped_ignores_fit_window(self, raw):
        # ica_strategy=None -> no ICA-fit copy is built, so fit_window is moot.
        cleaned, bads, ica = preprocess_raw(
            raw, ica_strategy=None, ica_fit_window=30.0
        )
        assert ica is None
        # Output spans the full recording (no crop anywhere).
        n_full = raw.get_data().shape[1]
        assert cleaned.get_data().shape[1] == n_full

    def test_dnn_float_fit_window_runs(self, raw):
        cleaned, _bads, ica = preprocess_raw(
            raw, ica_strategy="mne-icalabel", ica_fit_window=30.0
        )
        assert ica is not None
        assert ica.n_iter_ is not None
        # Cleaned output spans the FULL recording; crop is fit-only.
        n_full = raw.get_data().shape[1]
        assert cleaned.get_data().shape[1] == n_full

    def test_dnn_tuple_fit_window_runs(self, raw):
        cleaned, _bads, ica = preprocess_raw(
            raw, ica_strategy="mne-icalabel", ica_fit_window=(5.0, 55.0)
        )
        assert ica is not None
        assert ica.n_iter_ is not None
        assert cleaned.get_data().shape[1] == raw.get_data().shape[1]

    def test_regression_fit_window_runs(self, raw):
        cleaned, _bads, ica = preprocess_raw(
            raw, ica_strategy="find_bads_eog", ica_fit_window=30.0
        )
        assert ica is not None
        assert ica.n_iter_ is not None
        assert cleaned.get_data().shape[1] == raw.get_data().shape[1]

    def test_validation_negative_float_propagates(self, raw):
        with pytest.raises(ValueError, match="non-negative"):
            preprocess_raw(raw, ica_strategy="mne-icalabel", ica_fit_window=-5.0)

    def test_validation_inverted_tuple_propagates(self, raw):
        with pytest.raises(ValueError, match="tmin < tmax"):
            preprocess_raw(raw, ica_strategy="mne-icalabel", ica_fit_window=(55.0, 5.0))
