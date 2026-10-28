# p0ly-utils

EEG utilities library (`p0ly_utils`) for preprocessing, epoching, metadata parsing, and time/frequency/connectivity/coupling analyses of event-related EEG.

![CI](https://github.com/fbaumgardt/p0ly-utils/actions/workflows/ci.yml/badge.svg?branch=main)

The library powers [p0ly-flow](../p0ly-flow), a Snakemake pipeline used for my PhD research at Boston University's Reinhart Lab for Computational Cognitive Neuroscience. All functions are pure (stateless, no filesystem paths — pipeline-owned) and documented with array shapes.

## Analyses

The `p0ly_utils.analysis` package (one submodule per domain, all importable from `p0ly_utils.analysis`):

| Domain | Highlights |
| :--- | :--- |
| **ERP** (`erp`) | Per-condition evokeds, grand averages, condition differences, one-sample cluster-permutation statistics with EEG adjacency |
| **PSD** (`psd`) | Welch spectra for raw/epoched data, per-condition spectra, band powers, aperiodic (1/f) fits via `specparam` |
| **TFR** (`tfr`) | Time–frequency power and ITPC (+ z-score), group averages, condition differences, TFR cluster-permutation |
| **Connectivity** (`connectivity`) | Spectral connectivity (wPLI, coherence, PLV) per condition, group averaging (plain or Fisher-z), band-averaged graph metrics (clustering, strength, path length), condition-difference adjacencies |
| **Regression** (`regression`) | Single-trial OLS beta maps per term (intercept, continuous + reference-coded categorical `Epochs.metadata` predictors) for voltage or per-epoch TFR power, explicit exclusion counts, equal-subject-weight cohort means |
| **CFC** (`cfc`) | Within-channel PAC (tensorpac, Gaussian-copula default) and n:m phase-phase coupling (Hilbert phases, mean resultant length) with full surrogate nulls, equal-subject descriptive cohort means, and subject-unit condition contrasts with full-map max-statistic correction |
| **Variability** (`variability`) | Across-trial sample SD for signed voltage, SD/Fano/CV for nonnegative TFR power, equal-subject-weight cohort maps with per-cell contributor counts |
| **Decoding** (`decoding`) | Within-subject cross-validated MVPA of categorical metadata columns: time-resolved scores `(n_times,)` and temporal-generalization maps `(n_train_times, n_test_times)`, equal-weight cohort means |

```python
from p0ly_utils.analysis import psd_by_condition, band_powers
from p0ly_utils.analysis import spectral_connectivity, group_connectivity
from p0ly_utils.analysis import decode_epochs, group_decoding

spectra = psd_by_condition(
    epochs,  # carries by-trial metadata from the metadata module
    queries={"correct": "Correct == True", "incorrect": "Correct == False"},
    fmin=2.0, fmax=40.0, n_fft=2048,
)
powers = band_powers(
    spectra["correct"].data, spectra["correct"].freqs,
    bands={"theta": (4, 8), "alpha": (8, 13)}, norm_range=(2.0, 40.0),
)

conn = spectral_connectivity(epochs, methods=["wpli", "coh"], fmin=4.0, fmax=45.0)
average = group_connectivity([conn_subject_a, conn_subject_b], fisher_z=True)

from p0ly_utils.analysis import variability_epochs, group_variability

variability = variability_epochs(epochs, domain="voltage", measures=["sd"])  # (ch, time)
cohort = group_variability([variability_subject_a, variability_subject_b])

# Within-subject decoding of a categorical metadata column (scaler + classifier
# are cloned and fit on training folds only; scaler never sees held-out trials).
subject = decode_epochs(epochs, target="Correct", n_splits=5, seed=20261020)
cohort = group_decoding([subject_a, subject_b])  # equal subject weights
```

Single-trial regression (`regression_epochs` → `group_regression`) fits one OLS
model per subject and timelock on all retained trials: intercept, every
continuous column as a linear term, every categorical column reference-coded
against its lexicographically first level. Beta maps are
`(n_channels, n_times)` for `domain="voltage"` and
`(n_channels, n_freqs, n_times)` for `domain="power"` (per-epoch TFR power —
pass the pipeline's `tfr:` block as `tfr_settings`). Rows with null/non-finite
predictors are excluded and counted (`excluded_rows`), never silently dropped;
empty or rank-deficient designs fail loud. Cohort results are equal-subject
means, not pooled-trial estimates.

```python
from p0ly_utils.analysis import regression_epochs, group_regression

result = regression_epochs(
    epochs, continuous=["RT"], categorical=["Correct"], domain="voltage",
)
result.beta_maps["Correct[True]"]  # (n_channels, n_times) vs the "False" reference
cohort = group_regression([result_subject_a, result_subject_b])
```

### Cross-frequency coupling

`cfc` returns one map per method with axes `(n_channels, n_phase_freqs, n_amp_freqs)`: `"pac"` (tensorpac, Gaussian-copula default) and `"ppc_<n>x<m>"` (Hilbert phases of integer-ratio pairs; off-ratio cells are NaN and flagged by `valid_mask`). Three evidence levels are kept strictly apart:

- `estimate_cfc` → per-subject maps with the **full** surrogate null and one-sided empirical p-values (`evidence_level="subject"`, uncorrected, descriptive);
- `group_cfc` → equal-subject-weight means at any `n >= 1` (`"cohort_descriptive"`, no averaged p-values);
- `cfc_condition_contrast` → per-subject condition differences, sign flips at the subject unit, one full-map max-|statistic| family per method (`"cohort_subject_unit"`). It accepts observed `CFCMap` objects only, so subject p-values cannot be pooled into a cohort claim.

```python
from p0ly_utils.analysis import estimate_cfc, group_cfc

subject = estimate_cfc(
    epochs,  # (n_epochs, n_channels, n_times)
    methods=["pac", "phase_phase"],
    phase_freqs=[4.0, 6.0, 8.0], amp_freqs=[20.0, 25.0, 30.0],
    phase_bandwidth=2.0, amp_bandwidth=4.0,
    ratios=[(2, 1)], ratio_tolerance=0.5, n_surrogates=200, seed=20261020,
)
cohort = group_cfc([subject_a, subject_b])  # observed maps only
```

Cross-channel/region pairs and between-cohort inference are out of scope.

## Metadata parsing

The `metadata` module parses EEG event markers into structured per-trial metadata for trial characterization. An abstract `core`/`parser` layer is instantiated with concrete experiment specs (`dotprobe`, `mgsearch`, `dmss`, `igt`, `intwm`, `simonfb`), each exposing `spec`, `timelocks`, `intervals`, and `get_metadata`. The experiments showcase a broad range of block/trial structures and event-code verbosity.

```python
from p0ly_utils.metadata import dotprobe, parse_metadata

df = dotprobe.get_metadata(evt, ids)  # evt: MNE events array; ids: code -> id map
# or: parse_metadata(dotprobe.spec, evt, ids)
```

## Preprocessing

`preprocessing` (channel Z-scoring bad-channel detection, automated ICA with EOG artefact removal, sliding-window trial clipping) overlaps with published packages (pyprep, autoreject, mne-faster), perhaps more robustly there. We maintain in-house versions for flexibility on edge cases of data corruption (e.g. from stimulation) and specific signal-to-noise compromises (very small or large n). Without such requirements, you probably should prefer the published packages.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # runtime
uv sync --group dev  # + development tools
```

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run mypy
uv run pre-commit install   # optional local hooks before commit
```

CI (GitHub Actions, `main`) runs ruff, scoped mypy (migrated metadata modules + tests), and pytest with coverage. Enable Actions in the repo settings on first push.

## See also

- [p0ly-flow](../p0ly-flow) — Snakemake pipeline orchestrating this library
- [docs/DEFERRED_CI_AUTOMATION.md](docs/DEFERRED_CI_AUTOMATION.md) — automation to enable after legacy metadata and `dump_*` code are removed
