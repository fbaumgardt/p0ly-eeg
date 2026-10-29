"""Quality report generation for EEG preprocessing pipelines.

Builds an MNE Report styled with a neo-brutalist, ultra-minimalist design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mne.preprocessing import ICA

# ---------------------------------------------------------------------------
# CSS loading
# ---------------------------------------------------------------------------

_CSS_CACHE: str | None = None


def _load_css() -> str:
    """Read the neo-brutalist stylesheet shipped alongside this module."""
    css_path = Path(__file__).parent / "report.css"
    return css_path.read_text()


def get_report_css() -> str:
    """Return the neo-brutalist report CSS, loading from disk on first call."""
    global _CSS_CACHE
    if _CSS_CACHE is None:
        _CSS_CACHE = _load_css()
    return _CSS_CACHE


# ---------------------------------------------------------------------------
# Inline JavaScript helpers (injected once via add_custom_js)
# ---------------------------------------------------------------------------

_TAB_SWITCH_JS = """
function switchMetaTab(tabId) {
  var panels = document.querySelectorAll('.meta-panel');
  for (var i = 0; i < panels.length; i++) {
    panels[i].style.display = 'none';
  }
  var target = document.getElementById(tabId);
  if (target) target.style.display = 'block';

  var tabs = document.querySelectorAll('.meta-tab');
  for (var j = 0; j < tabs.length; j++) {
    tabs[j].classList.remove('active');
  }
  event.currentTarget.classList.add('active');
}
"""

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ChannelQuality:
    """Per-channel quality metrics vector.

    All fields are None when the metric could not be computed (e.g. no
    impedance data available, or channel was excluded from analysis).
    """

    ch_name: str
    impedance_kohm: float | None = None
    imp_in_range: bool | None = None
    ptp_uv: float | None = None
    ptp_zscore: float | None = None
    line_noise_db: float | None = None
    low_freq_db: float | None = None
    muscle_noise_pct: float | None = None
    neighbor_correlation: float | None = None
    is_bridged: bool = False
    bridged_with: list[str] = field(default_factory=list)
    is_flat: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Aggregate quality report for one recording + preprocessing run."""

    subject_id: str
    recording_date: datetime | None = None
    duration_s: float | None = None
    sfreq: float | None = None
    n_channels: int = 0
    n_bads_initial: int = 0
    n_bridged_pairs: int = 0
    bridged_pairs: list[tuple[str, str]] = field(default_factory=list)
    line_freq: float | None = None
    highpass: float | None = None
    lowpass: float | None = None
    channels: list[ChannelQuality] = field(default_factory=list)
    ica_components_fitted: int | None = None
    ica_components_excluded: list[int] = field(default_factory=list)
    ica_labels_excluded: list[str] = field(default_factory=list)
    bad_segment_total_s: float = 0.0
    bad_segment_pct: float = 0.0
    trial_retention: pd.DataFrame | None = None
    global_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_ptp_zscore(epochs: mne.BaseEpochs, max_iter: int = 3) -> np.ndarray:
    """Iterative z-score of per-channel peak-to-peak amplitudes."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ptp = np.ptp(data, axis=-1)  # (n_epochs, n_channels)
    mean_ptp = np.mean(ptp, axis=0)  # (n_channels,)
    mask = np.zeros(mean_ptp.shape, dtype=bool)
    for _ in range(max_iter):
        masked = np.ma.masked_array(mean_ptp, mask)
        mn = float(np.mean(masked))
        sd = float(np.std(masked))
        mask = np.abs((mean_ptp - mn) / sd) >= 3.0
    return (mean_ptp - np.mean(mean_ptp[~mask])) / np.std(mean_ptp[~mask])


def _compute_band_power(
    raw: mne.io.BaseRaw, fmin: float, fmax: float
) -> np.ndarray:
    """Mean power (dB) per channel in [fmin, fmax] Hz."""
    spectrum = raw.compute_psd(
        fmin=fmin, fmax=fmax, method="welch", verbose=False
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    # psd shape: (n_channels, n_freqs); average across the band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    return 10 * np.log10(np.mean(psd[:, band_mask], axis=1))


def _compute_line_noise_power(raw: mne.io.BaseRaw) -> np.ndarray | None:
    """Mean power at line_freq ± 1 Hz, per channel (dB)."""
    lf = raw.info.get("line_freq")
    if lf is None:
        return None
    return _compute_band_power(raw, lf - 1, lf + 1)


def _compute_muscle_noise_pct(raw: mne.io.BaseRaw) -> np.ndarray | None:
    """Percentage of recording annotated as BAD_muscle, per channel.

    Returns None if annotate_muscle_zscore fails or produces no annotations.
    """
    try:
        raw_copy = raw.copy()
        annot, scores = mne.preprocessing.annotate_muscle_zscore(
            raw_copy, ch_type="eeg", min_length_good=0.1
        )
    except Exception:
        return None
    total_dur = raw.times[-1] - raw.times[0]
    if total_dur <= 0:
        return None
    muscle_annot = [
        a for a in raw_copy.annotations if a["description"] == "BAD_muscle"
    ]
    if not muscle_annot:
        return np.zeros(raw.info["nchan"])
    # For each channel, check how much of the recording is covered
    pcts = np.zeros(raw.info["nchan"])
    for a in muscle_annot:
        pcts += (a["duration"] / total_dur) * 100.0
    return pcts


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------


def _hero_html(report_data: QualityReport) -> str:
    """Build the executive summary hero section."""
    subj = report_data.subject_id
    fs = f"{report_data.sfreq:.0f}" if report_data.sfreq else "—"
    nch = report_data.n_channels
    dur = f"{report_data.duration_s:.1f}" if report_data.duration_s else "—"

    status = "Preprocessing Required"
    status_class = "status-warn"
    if report_data.ica_components_excluded and report_data.n_bridged_pairs == 0:
        status = "Cleaned &amp; Epoch-Ready"
        status_class = "status-ready"

    date_str = ""
    if report_data.recording_date is not None:
        date_str = report_data.recording_date.strftime("%Y-%m-%d %H:%M:%S UTC")

    hp = f"{report_data.highpass:.1f}" if report_data.highpass else "—"
    lp = f"{report_data.lowpass:.1f}" if report_data.lowpass else "—"

    return f"""\
<div style="margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1.5px solid #000;">
<p class="hero-status">Recording: <span class="{status_class}">{status}</span></p>
<p class="hero-meta">
  {subj} &nbsp;&nbsp;|&nbsp;&nbsp;
  Fs: {fs} Hz &nbsp;|&nbsp; Channels: {nch} &nbsp;|&nbsp; Duration: {dur} s
</p>
<p class="hero-desc">
  EEG recording with {nch}-channel system.
  Hardware bandpass: {hp} – {lp} Hz.
  Line frequency: {report_data.line_freq or "—"} Hz.
  {report_data.n_bads_initial} channel(s) marked bad initially,
  {report_data.n_bridged_pairs} bridged pair(s) detected.
</p>

<div class="meta-tabs">
  <button class="meta-tab active" onclick="switchMetaTab('meta-raw')">Raw Info</button>
  <button class="meta-tab" onclick="switchMetaTab('meta-events')">Events/Triggers</button>
  <button class="meta-tab" onclick="switchMetaTab('meta-bads')">Bad Channels</button>
  <button class="meta-tab" onclick="switchMetaTab('meta-ica')">ICA Summary</button>
</div>

<div id="meta-raw" class="meta-panel" style="display:block">
  <div class="meta-row"><span class="meta-key">Recording Date</span><span>{date_str}</span></div>
  <div class="meta-row"><span class="meta-key">Sampling Rate</span><span>{fs} Hz</span></div>
  <div class="meta-row"><span class="meta-key">Channels</span><span>{nch} EEG</span></div>
  <div class="meta-row"><span class="meta-key">Line Frequency</span><span>{report_data.line_freq or "—"} Hz</span></div>
  <div class="meta-row"><span class="meta-key">Hardware Filter</span><span>{hp} – {lp} Hz</span></div>
  <div class="meta-row"><span class="meta-key">Duration</span><span>{dur} s</span></div>
  <div class="meta-row"><span class="meta-key">Bad Segments</span><span>{report_data.bad_segment_total_s:.1f} s ({report_data.bad_segment_pct:.1f}%)</span></div>
</div>
<div id="meta-events" class="meta-panel" style="display:none">
  <div class="meta-row"><span class="meta-key">Event Count</span><span>— (post-epoching)</span></div>
</div>
<div id="meta-bads" class="meta-panel" style="display:none">
  <div class="meta-row"><span class="meta-key">Initial Bads</span><span>{report_data.n_bads_initial}</span></div>
  <div class="meta-row"><span class="meta-key">Bridged Pairs</span><span>{report_data.n_bridged_pairs}</span></div>
</div>
<div id="meta-ica" class="meta-panel" style="display:none">
  <div class="meta-row"><span class="meta-key">Components Fitted</span><span>{report_data.ica_components_fitted or "—"}</span></div>
  <div class="meta-row"><span class="meta-key">Components Excluded</span><span>{len(report_data.ica_components_excluded)} ({", ".join(f"ICA{i:03d}" for i in report_data.ica_components_excluded) or "none"})</span></div>
  <div class="meta-row"><span class="meta-key">Excluded Labels</span><span>{", ".join(report_data.ica_labels_excluded) or "—"}</span></div>
</div>
</div>"""


def _impedance_html(impedances: dict[str, dict]) -> str:
    """Build the impedance table and bar-chart pipeline-stage HTML."""
    rows = []
    for ch_name, imp_data in impedances.items():
        imp_val = imp_data.get("imp", float("nan"))
        unit = imp_data.get("imp_unit", "kOhm")
        upper = imp_data.get("imp_upper_bound")
        in_range = "WARN" if (upper is not None and imp_val > upper) else "OK"
        rows.append(
            {
                "Channel": ch_name,
                "Impedance": f"{imp_val:.1f}",
                "Unit": unit,
                "Upper Bound": f"{upper:.1f}" if upper is not None else "—",
                "Status": in_range,
            }
        )

    df = pd.DataFrame(rows)
    n_warn = (df["Status"] == "WARN").sum()
    summary = (
        f"Impedance values measured at recording start. "
        f"{len(rows)} electrodes checked, {n_warn} exceed calibration range."
    )
    table = df.to_html(border=0, classes="table", index=False)

    return f"""\
<div class="pipeline-stage">
<div class="pipeline-left">
  <p class="pipeline-title">Preprocessing — Electrode Impedances</p>
  <p class="pipeline-summary">{summary}</p>
  <a class="pipeline-log-link" href="#">View raw impedance log &rarr;</a>
</div>
<div class="pipeline-right">
  {table}
  <p class="figure-caption" style="margin-top:0.25rem;">Fig. 01 | BRAINVISION_IMPEDANCE_CHECK</p>
</div>
</div>"""


def _channel_quality_html(channels: list[ChannelQuality]) -> str:
    """Build the per-channel quality metrics pipeline-stage."""
    rows = []
    warn_count = 0
    for ch in channels:
        w = len(ch.warnings)
        warn_count += w
        rows.append(
            {
                "Channel": ch.ch_name,
                "PTP (µV)": f"{ch.ptp_uv:.1f}" if ch.ptp_uv is not None else "—",
                "PTP z": f"{ch.ptp_zscore:.1f}" if ch.ptp_zscore is not None else "—",
                "Line Noise (dB)": f"{ch.line_noise_db:.1f}" if ch.line_noise_db is not None else "—",
                "Drift (dB)": f"{ch.low_freq_db:.1f}" if ch.low_freq_db is not None else "—",
                "Muscle %": f"{ch.muscle_noise_pct:.1f}" if ch.muscle_noise_pct is not None else "—",
                "Bridged": "YES" if ch.is_bridged else "—",
                "Flat": "YES" if ch.is_flat else "—",
                "Flags": w,
            }
        )

    df = pd.DataFrame(rows)
    summary = (
        f"Per-channel peak-to-peak amplitude, line noise, drift, "
        f"and muscle artifact metrics. {warn_count} flag(s) raised."
    )
    table = df.to_html(border=0, classes="table", index=False)

    return f"""\
<div class="pipeline-stage">
<div class="pipeline-left">
  <p class="pipeline-title">Preprocessing — Channel Quality Metrics</p>
  <p class="pipeline-summary">{summary}</p>
  <a class="pipeline-log-link" href="#">View raw channel stats &rarr;</a>
</div>
<div class="pipeline-right">
  {table}
  <p class="figure-caption" style="margin-top:0.25rem;">Fig. 02 | CHANNEL_QUALITY_MATRIX</p>
</div>
</div>"""


def _warnings_html(warnings: list[str]) -> str:
    """Build quality-flag alert lines."""
    if not warnings:
        return '<div class="quality-ok">&#10003; No quality flags raised.</div>'
    return "\n".join(
        f'<div class="quality-warning">&#9888; {w}</div>' for w in warnings
    )


def _trial_retention_html(df: pd.DataFrame) -> str:
    """Build trial retention pipeline-stage from a summary DataFrame."""
    total = df["N Total"].sum()
    kept = total - df["N Rejected"].sum()
    retention_pct = kept / total * 100 if total else 0
    summary = (
        f"Trial-level rejection summary after artifact detection. "
        f"Overall retention: {retention_pct:.1f}% ({int(kept)}/{int(total)} trials)."
    )
    table = df.to_html(border=0, classes="table", index=False)

    return f"""\
<div class="pipeline-stage">
<div class="pipeline-left">
  <p class="pipeline-title">Epoching — Trial Retention</p>
  <p class="pipeline-summary">{summary}</p>
  <a class="pipeline-log-link" href="#">View drop log &rarr;</a>
</div>
<div class="pipeline-right">
  {table}
  <p class="figure-caption" style="margin-top:0.25rem;">Fig. 03 | TRIAL_RETENTION_BY_CONDITION</p>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_quality_report(
    raw: mne.io.BaseRaw,
    *,
    subject_id: str = "Unknown",
    impedances: dict[str, dict] | None = None,
    ica: ICA | None = None,
    epochs: mne.BaseEpochs | None = None,
    metadata: pd.DataFrame | None = None,
) -> QualityReport:
    """Compute all quality metrics for a recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The (potentially preprocessed) raw recording.
    subject_id : str
        Human-readable identifier for the recording.
    impedances : dict | None
        Impedance dict from ``raw.impedances`` (BrainVision only).
        Provide ``None`` to skip impedance reporting.
    ica : mne.preprocessing.ICA | None
        Fitted ICA object. Its ``exclude`` list and ICLabel labels
        are extracted if available.
    epochs : mne.BaseEpochs | None
        Epoched data. Used for trial retention summary.
    metadata : pd.DataFrame | None
        Trial-level metadata (required if *epochs* is provided for
        per-condition breakdown).

    Returns
    -------
    QualityReport
        Dataclass holding all computed metrics.
    """
    info = raw.info
    report = QualityReport(
        subject_id=subject_id,
        recording_date=info.get("meas_date"),
        duration_s=raw.times[-1] - raw.times[0] if len(raw.times) else None,
        sfreq=info["sfreq"],
        n_channels=info["nchan"],
        n_bads_initial=len(info["bads"]),
        line_freq=info.get("line_freq"),
        highpass=info.get("highpass"),
        lowpass=info.get("lowpass"),
    )

    # --- Impedance data ---
    if impedances:
        for ch_name, imp_data in impedances.items():
            imp_val = imp_data.get("imp")
            upper = imp_data.get("imp_upper_bound")
            in_range = not (upper is not None and imp_val is not None and imp_val > upper)
            cq = ChannelQuality(
                ch_name=ch_name,
                impedance_kohm=float(imp_val) if imp_val is not None else None,
                imp_in_range=in_range,
            )
            if not in_range:
                cq.warnings.append(
                    f"{ch_name}: impedance {imp_val} {imp_data.get('imp_unit','kOhm')} exceeds range"
                )
            report.channels.append(cq)

    # --- PSD-based metrics ---
    try:
        raw_band = raw.copy().filter(0.5, 80, verbose=False)
    except Exception:
        raw_band = raw

    line_noise = _compute_line_noise_power(raw_band)
    low_freq = _compute_band_power(raw_band, 0.1, 2.0)
    muscle_pct = _compute_muscle_noise_pct(raw_band)

    # --- Epoch-level PTP z-scores ---
    ptp_z: np.ndarray | None = None
    ptp_mean: np.ndarray | None = None
    try:
        epo = mne.make_fixed_length_epochs(
            raw_band, duration=1.0, reject_by_annotation=False, verbose=False
        )
        ptp_z = _compute_ptp_zscore(epo)
        ptp_mean = np.mean(np.ptp(epo.get_data(), axis=-1), axis=0)
    except Exception:
        pass

    # --- Bridged electrodes ---
    try:
        bridged_idx = mne.preprocessing.compute_bridged_electrodes(
            raw_band, verbose=False
        )
        bridged_pairs = [
            (raw_band.ch_names[i], raw_band.ch_names[j]) for i, j in bridged_idx
        ]
    except Exception:
        bridged_idx = []
        bridged_pairs = []

    report.n_bridged_pairs = len(bridged_pairs)
    report.bridged_pairs = bridged_pairs

    # --- Per-channel quality assembly ---
    ch_names = raw_band.ch_names
    for i, ch_name in enumerate(ch_names):
        # Find existing or create new
        cq = next((c for c in report.channels if c.ch_name == ch_name), None)
        if cq is None:
            cq = ChannelQuality(ch_name=ch_name)
            report.channels.append(cq)

        if ptp_mean is not None:
            cq.ptp_uv = float(ptp_mean[i] * 1e6)
        if ptp_z is not None:
            cq.ptp_zscore = float(ptp_z[i])
            if abs(cq.ptp_zscore) > 2.5:
                cq.warnings.append(
                    f"{ch_name}: PTP z-score {cq.ptp_zscore:.1f} — extreme amplitude"
                )
        if line_noise is not None:
            cq.line_noise_db = float(line_noise[i])
            if cq.line_noise_db > -40:
                cq.warnings.append(
                    f"{ch_name}: elevated line noise ({cq.line_noise_db:.1f} dB)"
                )
        if low_freq is not None:
            cq.low_freq_db = float(low_freq[i])
            if cq.low_freq_db > -25:
                cq.warnings.append(
                    f"{ch_name}: elevated low-frequency drift ({cq.low_freq_db:.1f} dB)"
                )
        if muscle_pct is not None and i < len(muscle_pct):
            cq.muscle_noise_pct = float(muscle_pct[i])
            if cq.muscle_noise_pct > 5.0:
                cq.warnings.append(
                    f"{ch_name}: {cq.muscle_noise_pct:.1f}% muscle artifact"
                )

        # Bridged status
        for a, b in bridged_pairs:
            if ch_name == a:
                cq.is_bridged = True
                cq.bridged_with.append(b)
            elif ch_name == b:
                cq.is_bridged = True
                cq.bridged_with.append(a)

        # Flat detection
        if cq.ptp_uv is not None and cq.ptp_uv < 1.0:
            cq.is_flat = True
            cq.warnings.append(f"{ch_name}: flat channel (PTP < 1 µV)")

    # --- Bad segment summary ---
    total_dur = raw.times[-1] - raw.times[0] if len(raw.times) else 0.0
    bad_dur = sum(
        a["duration"]
        for a in raw.annotations
        if a["description"].startswith("BAD_")
    )
    report.bad_segment_total_s = bad_dur
    report.bad_segment_pct = (bad_dur / total_dur * 100) if total_dur > 0 else 0.0

    # --- ICA ---
    if ica is not None:
        report.ica_components_fitted = ica.n_components_
        report.ica_components_excluded = list(ica.exclude)
        if hasattr(ica, "labels_"):
            report.ica_labels_excluded = [
                ica.labels_[i] for i in ica.exclude if i < len(ica.labels_)
            ]

    # --- Trial retention ---
    if epochs is not None and metadata is not None:
        drop_mask = np.array(
            [len(reason) > 0 for reason in epochs.drop_log], dtype=bool
        )
        if len(drop_mask) == len(metadata):
            metadata = metadata.copy()
            metadata["_rejected"] = drop_mask
            grouped = (
                metadata.groupby(
                    [c for c in metadata.columns if c not in ("Onset", "_rejected")]
                )
                .agg(N_Total=("_rejected", "count"), N_Rejected=("_rejected", "sum"))
                .reset_index()
            )
            grouped["Retention %"] = (
                (grouped["N_Total"] - grouped["N_Rejected"])
                / grouped["N_Total"]
                * 100
            ).round(1)
            # Pick a reasonable subset of columns for display
            cond_cols = [
                c
                for c in grouped.columns
                if c not in ("N_Total", "N_Rejected", "Retention %")
            ]
            display_cols = cond_cols + ["N_Total", "N_Rejected", "Retention %"]
            report.trial_retention = grouped[display_cols]

    # --- Global warnings ---
    report.global_warnings = [
        f"{ch.ch_name}: {w}"
        for ch in report.channels
        for w in ch.warnings
    ]

    return report


def build_quality_report(
    raw: mne.io.BaseRaw,
    *,
    subject_id: str = "Unknown",
    impedances: dict[str, dict] | None = None,
    ica: ICA | None = None,
    epochs: mne.BaseEpochs | None = None,
    metadata: pd.DataFrame | None = None,
    title: str | None = None,
) -> mne.Report:
    """Build a neo-brutalist styled quality report for an EEG recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The (preprocessed) raw recording.
    subject_id : str
        Identifier shown in the hero section.
    impedances : dict | None
        Impedance dict from ``raw.impedances`` after ``read_raw_brainvision``.
    ica : mne.preprocessing.ICA | None
        Fitted ICA with ``exclude`` set.
    epochs : mne.BaseEpochs | None
        Epoched data for trial retention summary.
    metadata : pd.DataFrame | None
        Trial metadata (needed if *epochs* is provided).
    title : str | None
        Report title. Defaults to *subject_id*.

    Returns
    -------
    mne.Report
        Fully populated report. Call ``.save("report.html")`` to write.
    """
    report_title = title or subject_id
    report = mne.Report(title=report_title)

    # --- CSS + JS ---
    report.add_custom_css(get_report_css())
    report.add_custom_js(_TAB_SWITCH_JS)

    # --- Compute all metrics ---
    qr = compute_quality_report(
        raw,
        subject_id=subject_id,
        impedances=impedances,
        ica=ica,
        epochs=epochs,
        metadata=metadata,
    )

    # --- Hero section ---
    report.add_html(
        _hero_html(qr),
        title="Executive Summary",
        tags=("summary",),
        section="Overview",
    )

    # --- Impedances ---
    if impedances:
        report.add_html(
            _impedance_html(impedances),
            title="Electrode Impedances",
            tags=("quality",),
            section="Data Quality",
        )

    # --- Channel quality ---
    if qr.channels:
        report.add_html(
            _channel_quality_html(qr.channels),
            title="Channel Quality Metrics",
            tags=("quality",),
            section="Data Quality",
        )

    # --- Warnings ---
    report.add_html(
        _warnings_html(qr.global_warnings),
        title="Quality Flags",
        tags=("quality",),
        section="Data Quality",
    )

    # --- Trial retention ---
    if qr.trial_retention is not None:
        report.add_html(
            _trial_retention_html(qr.trial_retention),
            title="Trial Retention",
            tags=("epochs",),
            section="Trial Quality",
        )

    # --- Built-in MNE plots ---
    report.add_raw(
        raw,
        title="Raw Data — Butterfly &amp; PSD",
        psd=True,
        tags=("raw",),
    )

    if ica is not None:
        report.add_ica(
            ica,
            title="ICA — Component Topographies",
            inst=raw,
            tags=("ica",),
        )

    if epochs is not None:
        report.add_epochs(
            epochs,
            title="Epochs — Trials &amp; PSD",
            tags=("epochs",),
        )

    return report
