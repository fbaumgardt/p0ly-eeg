from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components


def _minmax_zscore(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    axis: str = "channels",
    threshold: float = 3.0,
    max_iter: int = 1,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean mask flagging outlier channels or epochs.

    Peak-to-peak amplitude is computed per channel (axis='channels') or per
    epoch (axis='time'), then outliers (|z| > threshold) are identified
    iteratively. At each step the mean and std are re-estimated on the unmasked
    subset, so earlier outliers don't inflate the baseline.

    Parameters
    ----------
    inst
        MNE object to score. When ``axis='channels'``, ``inst.get_data()``
        has shape ``(n_channels, n_times)`` (a ``Raw``). When ``axis='time'``,
        ``inst.get_data()`` has shape ``(n_epochs, n_channels, n_times)``
        (an ``Epochs`` object) — the data is averaged over channels first so
        that the score reflects per-epoch amplitude.
    axis
        ``'channels'`` to flag outlier channels (per-channel peak-to-peak),
        ``'time'`` to flag outlier epochs (per-epoch peak-to-peak).
    threshold
        Z-score cutoff; samples with ``|z| >= threshold`` are flagged.
    max_iter
        Number of iterative refinement passes (re-estimate mean/std on the
        unmasked subset each pass).
    mask
        Optional pre-existing boolean mask of shape ``(n_channels,)`` or
        ``(n_epochs,)``; masked entries are excluded from the baseline.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_channels,)`` when ``axis='channels'`` or
        ``(n_epochs,)`` when ``axis='time'``. ``True`` marks an outlier.
    """
    data = inst.get_data()
    if axis == "time":
        # Average over channels first so that X reflects per-epoch amplitude.
        data = np.mean(data, axis=-2)  # axis=-2 works for both Epochs and Raw
    X: np.ndarray = np.max(data, axis=-1) - np.min(data, axis=-1)
    current_mask: np.ndarray = mask if mask is not None else np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        Y = np.ma.masked_array(X, current_mask)
        mn = float(np.mean(Y))
        sd = float(np.std(Y))
        current_mask = np.abs((X - mn) / sd) >= threshold
    return current_mask


def fix_channels(
    raw: mne.io.BaseRaw,
    threshold: float = 3.0,
    max_iter: int = 1,
) -> mne.io.BaseRaw:
    """Return a copy of raw with statistically extreme channels added to bads.

    Flags outlier channels via peak-to-peak Z-scoring (see ``_minmax_zscore``)
    and appends them to ``info["bads"]``. Interpolation is **not** performed
    here — call :func:`interpolate_bads` afterwards to reconstruct the dropped
    channels with MNE's spherical-spline interpolator.

    Parameters
    ----------
    raw
        Continuous raw recording. ``raw.get_data()`` has shape
        ``(n_channels, n_times)`` (see SCHEMA §1).
    threshold
        Z-score cutoff for peak-to-peak distance.
    max_iter
        Number of iterative refinement passes in :func:`_minmax_zscore`.

    Returns
    -------
    mne.io.BaseRaw
        Copy of ``raw`` with ``info["bads"]`` updated. Data shape is
        unchanged: ``(n_channels, n_times)``.
    """
    r = raw.copy()
    bad_mask = _minmax_zscore(r, threshold=threshold, max_iter=max_iter)
    new_bads = np.array(r.ch_names)[bad_mask].tolist()
    r.info["bads"] = np.unique(
        np.concatenate((np.array(r.info["bads"]), np.array(new_bads)))
    ).tolist()
    return r


def interpolate_bads(
    raw: mne.io.BaseRaw,
    reset_bads: bool = True,
) -> mne.io.BaseRaw:
    """Return a copy of raw with ``info["bads"]`` channels interpolated.

    Spherical-spline interpolation (MNE default). With ``reset_bads=True`` the
    interpolated channels are cleared from ``info["bads"]`` so downstream
    steps (ICA, sliding-window rejection) see a full, clean sensor array.
    The input raw is never mutated.

    Parameters
    ----------
    raw
        Continuous raw recording with ``info["bads"]`` populated.
        ``raw.get_data()`` has shape ``(n_channels, n_times)`` (see SCHEMA §1).
    reset_bads
        If ``True``, clear interpolated channels from ``info["bads"]``.

    Returns
    -------
    mne.io.BaseRaw
        Copy of ``raw`` with bad channels interpolated. Data shape is
        unchanged: ``(n_channels, n_times)``.
    """
    r = raw.copy()
    r.interpolate_bads(reset_bads=reset_bads)
    return r


def artefact_rejection(
    raw: mne.io.BaseRaw,
    threshold: float = 3.0,
    max_iter: int = 3,
    duration: float = 0.5,
    stimulation: tuple[float | None, float | None] | None = None,
) -> mne.Annotations:
    """Identify bad time segments via peak-to-peak z-score on fixed-length epochs.

    Returns Annotations marking each bad segment onset with label
    'bad_minmax_zscore'.

    stimulation: optional (start, end) window in seconds to exclude from the
    z-score baseline (e.g. a stimulation artefact window). None endpoints mean
    the beginning / end of the recording. Pass None to skip masking entirely.

    Parameters
    ----------
    raw
        Continuous raw recording. ``raw.get_data()`` has shape
        ``(n_channels, n_times)`` (see SCHEMA §1).
    threshold
        Z-score cutoff for peak-to-peak distance per epoch.
    max_iter
        Number of iterative refinement passes in :func:`_minmax_zscore`.
    duration
        Fixed-length epoch size in seconds used to segment the raw.
    stimulation
        Optional ``(start, end)`` window in seconds to exclude from the
        z-score baseline. ``None`` endpoints mean beginning / end of recording.
        Pass ``None`` to skip masking entirely.

    Returns
    -------
    mne.Annotations
        Annotations for each bad segment. ``onset``, ``duration``, and
        ``description`` arrays each have shape ``(n_annotations,)``. The label
        is ``'bad_minmax_zscore'`` for every annotation.
    """
    epo = mne.make_fixed_length_epochs(raw, duration=duration, reject_by_annotation=False)

    stim_mask: np.ndarray | None = None
    if stimulation is not None:
        # Mark epochs whose onset falls within the stimulation window so they
        # don't inflate the baseline mean/std used for z-score thresholding.
        epoch_onsets = epo.events[:, 0] / epo.info["sfreq"]
        stim_start = stimulation[0] if stimulation[0] is not None else raw.times[0]
        stim_end = stimulation[1] if stimulation[1] is not None else raw.times[-1]
        stim_mask = (epoch_onsets >= stim_start) & (epoch_onsets <= stim_end)

    bad_mask = _minmax_zscore(
        epo, axis="time", threshold=threshold, max_iter=max_iter, mask=stim_mask
    )
    bad_starts = epo.events[bad_mask, 0] / epo.info["sfreq"]
    return mne.Annotations(
        bad_starts, duration, "bad_minmax_zscore", orig_time=raw.annotations.orig_time
    )


def _resolve_fit_window(
    raw: mne.io.BaseRaw,
    fit_window: float | list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Translate an ICA fit-window spec into ``(tmin, tmax)`` bounds, or ``None``.

    Single source of truth for the fit-window interpretation shared by
    :func:`ica_clean_dnn` and :func:`ica_clean_regression`. No centred-window
    math or validation is duplicated in the callers.

    Parameters
    ----------
    raw
        Continuous raw recording whose ``times`` define the midpoint used for
        the centred-window form.
    fit_window
        ``None`` → no crop (full recording). ``float`` → a centred window of
        that duration (seconds) around the recording midpoint
        ``mid = (raw.times[0] + raw.times[-1]) / 2``. ``tuple`` or ``list`` →
        explicit ``(tmin, tmax)`` bounds (seconds, ``tmin < tmax``). The
        ``list`` form is accepted because YAML config round-trips ``[tmin, tmax]``
        to a list (mirrors the ``intervals`` convention).

    Returns
    -------
    tuple[float, float] | None
        ``(tmin, tmax)`` bounds, or ``None`` when no crop is requested.

    Raises
    ------
    ValueError
        If ``fit_window`` is a negative ``float``, or a tuple/list with
        ``tmin >= tmax`` (or wrong arity). Out-of-range vs the recording is left
        to MNE's ``crop`` (already raises).
    """
    if fit_window is None:
        return None
    if isinstance(fit_window, (list, tuple)):
        if len(fit_window) != 2:
            raise ValueError(
                f"fit_window sequence must be (tmin, tmax), got arity {len(fit_window)}"
            )
        tmin, tmax = float(fit_window[0]), float(fit_window[1])
        if tmin >= tmax:
            raise ValueError(
                f"fit_window bounds must satisfy tmin < tmax, got ({tmin}, {tmax})"
            )
        return tmin, tmax
    if fit_window < 0:
        raise ValueError(f"fit_window float must be non-negative, got {fit_window}")
    midpoint = (raw.times[0] + raw.times[-1]) / 2.0
    return midpoint - fit_window / 2.0, midpoint + fit_window / 2.0


def ica_clean_dnn(
    raw: mne.io.BaseRaw,
    exclude_components: list[str] = ["eye blink", "muscle artifact"],
    threshold: float | None = None,
    bandpass: tuple[float, float] = (1.0, 100.0),
    fit_window: float | list[float] | tuple[float, float] | None = None,
    n_jobs: int = 1,
) -> tuple[mne.io.BaseRaw, ICA]:
    """Remove artefact ICA components identified by ICLabel (DNN-based).

    A component ``i`` is excluded when its ICLabel ``labels[i]`` is in
    ``exclude_components``. When ``threshold`` is not None, the component is
    additionally required to carry a predicted-class probability
    ``y_pred_proba[i] >= threshold`` (range 0–1); this guards against removing
    low-confidence classifications. Pass ``threshold=None`` to keep the
    label-only behaviour.

    fit_window controls how much of the recording is used to fit the ICA:
      - None: use the full recording.
      - (tmin, tmax) / [tmin, tmax]: crop to that explicit window.
      - float: crop a centred window of that duration (seconds) around the
        recording midpoint.

    The input raw is never mutated; a cleaned copy is returned together with the
    fitted ICA object for inspection.

    Parameters
    ----------
    raw
        Continuous raw recording. ``raw.get_data()`` has shape
        ``(n_channels, n_times)`` (see SCHEMA §1).
    exclude_components
        ICLabel label names to exclude (e.g. ``'eye blink'``, ``'muscle artifact'``).
    threshold
        ICLabel predicted-class probability cutoff. ``None`` keeps label-only
        behaviour.
    bandpass
        ``(l_freq, h_freq)`` applied to the copy used for ICA fitting.
    fit_window
        Controls the fitting window: ``None`` for full recording,
        ``(tmin, tmax)`` (or list) for explicit crop, ``float`` for a centred
        window of that duration (seconds). Interpretation lives in
        :func:`_resolve_fit_window`; the crop is applied to the ICA-fit copy
        ``r_ic`` only, never to the cleaned output.
    n_jobs
        Number of parallel jobs for filtering.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        Copy of ``raw`` with artefact components removed. Data shape is
        unchanged: ``(n_channels, n_times)``.
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``.exclude`` populated.
    """

    r_ic = raw.copy().filter(*bandpass, n_jobs=n_jobs)

    bounds = _resolve_fit_window(raw, fit_window)
    if bounds is not None:
        tmin, tmax = bounds
        r_ic = r_ic.crop(tmin=tmin, tmax=tmax)

    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        max_iter="auto",
    )
    ica.fit(r_ic)
    labels = label_components(r_ic, ica, method="iclabel")
    proba = labels["y_pred_proba"]
    label_names = labels["labels"]
    if threshold is not None:
        ica.exclude = [
            i
            for i, (label, p) in enumerate(zip(label_names, proba))
            if label in exclude_components and float(p) >= threshold
        ]
    else:
        ica.exclude = [i for i, label in enumerate(label_names) if label in exclude_components]

    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica


def ica_clean_regression(
    raw: mne.io.BaseRaw,
    bandpass: tuple[float, float] = (1.0, 40.0),
    decim: int = 1,
    components: int = 20,
    fit_window: float | list[float] | tuple[float, float] | None = None,
) -> tuple[mne.io.BaseRaw, ICA]:
    """Remove EOG artefact components via regression-based ICA.

    EOG channels are detected automatically from raw.info. The search window
    for find_bads_eog is capped at 1000 s from the fit-window start to keep it
    tractable on long recordings. The input raw is never mutated.

    Parameters
    ----------
    raw
        Continuous raw recording with EEG + EOG channels.
        ``raw.get_data()`` has shape ``(n_channels, n_times)`` where
        ``n_channels`` includes both EEG and EOG channels (see SCHEMA §1).
    bandpass
        ``(l_freq, h_freq)`` applied to the copy used for ICA fitting.
    decim
        Decimation factor for ICA fitting.
    components
        Number of ICA components to compute.
    fit_window
        Controls the fitting window: ``None`` for full recording,
        ``(tmin, tmax)`` (or list) for explicit crop, ``float`` for a centred
        window of that duration (seconds). Interpretation lives in
        :func:`_resolve_fit_window`; the crop is applied to the ICA-fit copy
        ``r_ic`` only, never to the cleaned output.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        Copy of ``raw`` with EOG artefact components removed. Data shape is
        unchanged: ``(n_channels, n_times)``.
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``.exclude`` populated.
    """
    r_ic = raw.copy().filter(*bandpass)
    bounds = _resolve_fit_window(raw, fit_window)
    if bounds is not None:
        tmin, tmax = bounds
        r_ic = r_ic.crop(tmin=tmin, tmax=tmax)
    else:
        tmin = raw.times[0]
        tmax = raw.times[-1]
    r_ic = r_ic.set_eeg_reference()
    ica = ICA(n_components=components).fit(r_ic, decim=decim)

    eog_indices = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
    eog_channels = [raw.ch_names[i] for i in eog_indices]

    # Limit the EOG correlation search to a 1000 s window from the fit-window
    # start for efficiency on long recordings.
    search_start = max(tmin, raw.times[0])
    search_stop = min(tmin + 1000.0, tmax)

    bad_components: list[int] = []
    for ch_name in eog_channels:
        found, _ = ica.find_bads_eog(raw, ch_name=ch_name, start=search_start, stop=search_stop)
        bad_components.extend(found)
    ica.exclude = list(set(bad_components))

    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica


def preprocess_raw(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float | None = None,
    h_freq: float | None = None,
    bad_channel_z_thresh: float | None = None,
    ica_strategy: str | None = None,
    icalabel_threshold: float | None = None,
    ica_fit_window: float | list[float] | tuple[float, float] | None = None,
    interval_window_ms: int | None = None,
    interval_reject_z_thresh: float | None = None,
) -> tuple[mne.io.BaseRaw, list[str], ICA | None]:
    """Chain the continuous-data preprocessing steps on one raw recording.

    Single-rule pipeline (no per-step FIF intermediates): bandpass filter ->
    bad-channel flag + interpolate -> ICA -> sliding-window artefact rejection.
    Operates on **continuous (un-segmented)** data.

    Every step is **optional**: a parameter left ``None`` skips its step. This
    lets the pipeline run partial chains (e.g. filter-only, or ICA without
    bad-channel detection.

    A montage must already be applied; this function asserts it is present. The
    input raw is never mutated (a private copy is made unconditionally).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous raw recording with montage already applied.
        ``raw.get_data()`` has shape ``(n_channels, n_times)`` (see SCHEMA §1).
    l_freq, h_freq
        Bandpass cutoffs in Hz. If both are ``None`` the filter step is
        skipped; one-sided filters are valid (``l_freq=None`` for lowpass-only,
        ``h_freq=None`` for highpass-only) and passed through to MNE.
    bad_channel_z_thresh
        Z-score cutoff for :func:`fix_channels`. If ``None``, the
        bad-channel flag + interpolate step is skipped and ``bad_channels``
        is empty.
    ica_strategy
        ``"mne-icalabel"`` (ICLabel DNN), ``"find_bads_eog"`` (regression),
        or ``None`` to skip ICA. An unrecognised string raises ``ValueError``
        (``None`` is distinct from an invalid strategy).
    icalabel_threshold
        ICLabel predicted-class probability cutoff; ignored unless
        ``ica_strategy == "mne-icalabel"`` (and ignored when ICA is skipped).
    ica_fit_window
        ICA fit-window control, forwarded as ``fit_window`` to both
        :func:`ica_clean_dnn` and :func:`ica_clean_regression` (pure pass-through,
        no translation): ``None`` for the full recording, ``float`` for a
        centred window of that duration (seconds) around the recording midpoint,
        ``(tmin, tmax)`` (or list) for an explicit window. The crop applies to
        the ICA-fit copy ``r_ic`` only; the cleaned output spans the full
        recording. Silently ignored when ``ica_strategy is None``.
    interval_window_ms
        Sliding-window length in milliseconds (converted to seconds). The
        sliding-window step is skipped unless **both** ``interval_window_ms`` and
        ``interval_reject_z_thresh`` are set.
    interval_reject_z_thresh
        Peak-to-peak Z-score cutoff for :func:`artefact_rejection`.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        Cleaned continuous raw; when the sliding-window step runs it carries
        bad-interval ``Annotations`` (label ``bad_minmax_zscore``) ready for
        downstream epoching. ``get_data()`` shape: ``(n_channels, n_times)``.
    bad_channels : list[str]
        Channels flagged before interpolation (for the pipeline sidecar log).
        Empty when the bad-channel step is skipped.
    ica : mne.preprocessing.ICA | None
        The fitted ICA object (with ``.exclude`` / ``.labels_`` populated) when
        the ICA step ran, else ``None``. Callers can persist it (``ica.save``)
        or read ``ica.exclude`` for component-exclusion reporting.
    """
    if raw.get_montage() is None:
        raise RuntimeError(
            "Montage missing on input raw; ingestion must apply it before preprocessing."
        )

    # Work on a private copy so the caller's raw is never mutated. The copy is
    # unconditional — non-mutation is a library invariant independent of which
    # steps run.
    raw = raw.copy().load_data()
    bad_channels: list[str] = []
    ica_obj: ICA | None = None

    # 1. bandpass filter on the full continuous recording. Skipped only when
    #    both cutoffs are None; one-sided filters are delegated to MNE.
    if l_freq is not None or h_freq is not None:
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq)

    # 2. bad channels -> flag + interpolate (spherical spline). One guard gates
    #    the flag+interpolate pair; the pre-interpolation set is captured for
    #    the caller's sidecar log.
    if bad_channel_z_thresh is not None:
        raw = fix_channels(raw, threshold=bad_channel_z_thresh)
        bad_channels = list(raw.info["bads"])
        raw = interpolate_bads(raw)

    # 3. Set preliminary average EEG reference
    raw.set_eeg_reference()

    # 4. ICA on the cleaned continuous raw. ica_strategy=None skips ICA;
    #    an unrecognised string is still an error (None != invalid strategy).
    if ica_strategy is not None:
        if ica_strategy == "mne-icalabel":
            raw, ica_obj = ica_clean_dnn(
                raw, threshold=icalabel_threshold, fit_window=ica_fit_window
            )
        elif ica_strategy == "find_bads_eog":
            raw, ica_obj = ica_clean_regression(raw, fit_window=ica_fit_window)
        else:
            raise ValueError(
                f"Unknown ica_strategy {ica_strategy!r} "
                "(expected 'mne-icalabel', 'find_bads_eog', or None to skip ICA)."
            )

    # 4. sliding-window reject on continuous data -> Annotations persisted.
    #    Both the window size and the threshold are required to run the step.
    if interval_window_ms is not None and interval_reject_z_thresh is not None:
        annots = artefact_rejection(
            raw,
            threshold=interval_reject_z_thresh,
            duration=interval_window_ms / 1000.0,
        )
        raw = raw.set_annotations(raw.annotations + annots)

    return raw, bad_channels, ica_obj
