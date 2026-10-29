# Agent Directives: p0ly-eeg (Core Math & Signal Processing Library)

You are an expert neural signal processing engineer. Your sole focus in this directory is maintaining, testing, and expanding the mathematical, statistical, and algorithmic foundations of `p0ly-eeg`.

Import name: `p0ly_eeg` (package name: `p0ly-eeg`).

Shared Python conventions: see root [AGENTS.md](../AGENTS.md).

---

## 1. Module Scope (PRD Alignment)

Every utility must align with [scrum/04_Specs/PRD_Core_App.md](../scrum/04_Specs/PRD_Core_App.md). Implement pure, tested functions accepting MNE objects or NumPy arrays:

1. **Custom Preprocessing**
   - Peak-to-peak Z-score channel detection on continuous data
   - 500ms sliding-window peak-to-peak epoch rejection
   - Automated ICA (find_bads_eog / mne-icalabel)
2. **Sensor & Spectral**
   - Group grand-averaging and cluster-based permutation tests
   - Absolute/relative PSD and 1/f aperiodic parameterization
   - Morlet/multitaper TFR and ITPC
3. **Advanced Dynamics & Networks**
   - Connectivity matrices (Coherence, PLV, wPLI) and graph metrics
   - Phase-amplitude coupling comodulograms
4. **Statistical ML & Decoding**
   - Single-trial OLS regression against `Epochs.metadata`
   - Temporal generalization decoding with scikit-learn estimators

---

## 2. Package Structure

```
src/p0ly_eeg/
├── __init__.py          # Re-exports metadata, preprocessing
├── preprocessing.py     # Channel fix, artefact rejection, ICA
├── epoching.py          # align_epochs_metadata()
└── metadata/
    ├── core.py          # ExperimentSpec, ColumnExtractor ABCs
    ├── parser.py        # parse_metadata() — experiment-agnostic
    └── {experiment}.py  # One spec module per Psychtoolbox experiment
```

- New analysis modules go alongside `preprocessing.py` and `epoching.py`
- New experiments get a spec file in `metadata/` only — never modify `parser.py` for experiment-specific logic
- Legacy code stays in `_legacy/` and is not exported

---

## 3. Architectural Constraints

- **Pure functions:** Stateless, deterministic, decoupled from filesystem paths and Snakemake rules
- **Data shape rigor:** Every function signature includes NumPy-style docstrings with explicit dimensionalities per [SCHEMA](../scrum/04_Specs/SCHEMA.md)
- **Core environment:** MNE-Python, NumPy, SciPy, Pandas, scikit-learn, mne-icalabel, mne-connectivity

---

## 4. Abstraction Hierarchy

Exhaust built-in library methods before dropping to lower abstraction levels:

1. **MNE-Python Native** — `raw.filter()`, `epochs.average()`, `mne.compute_covariance()` before extracting arrays
2. **Pandas Vectorized** — `.isin()`, `.cumsum()`, `.groupby().transform()` for tabular metadata logic
3. **NumPy Vectorized** — `get_data()` / `.to_numpy()` only when passing to vectorized NumPy or scikit-learn
4. **Standard Python** — lists, dicts, for-loops only for I/O, config, or when libraries require it

---

## 5. Negative Constraints (Banned Patterns)

- **No `for` loops over channels or epochs**
- **No premature data extraction** — keep data in MNE objects until vectorized NumPy is required
- **No row-wise DataFrame iteration** — no `iterrows()`, no `apply()` with row-wise lambdas
- **Prefer inline composition** for linear transforms; named intermediates when reused or branching

---

## 6. Translation Anchors

| Library | Banned | Required |
| :--- | :--- | :--- |
| **MNE** | `[e.mean() for e in epochs]` | `epochs.average()` |
| **MNE** | Loop over channels to rename | `raw.rename_channels(mapping_dict)` |
| **NumPy** | Nested loops for matrix math | `np.einsum()` or broadcasting |
| **Pandas** | `for i, row in df.iterrows()` | `df["description"].isin(codes).cumsum()` |

---

## 7. Metadata Architecture

When working in `src/p0ly_eeg/metadata/`:

- Each experiment defines an `ExperimentSpec` instance in its own module
- Column extraction uses `ColumnExtractor` subclasses: `CodeLookup`, `IntSum`, `BoolPresence`, `ListCollect`, `DerivedColumn`
- Generic parser drives all experiments via `parse_metadata(spec, df)`
- New extraction patterns → new `ColumnExtractor` subclass in `core.py`, not parser changes

See [ADR-002](../scrum/04_Specs/ADR-002_declarative-metadata-specs.md).

---

## 8. Testing Conventions

- **Fast loop:** `uv run pytest` before considering any change complete. Slow ICA-fit tests are marked `@pytest.mark.slow` and skipped by default (`addopts = "-ra -m 'not slow'"` in `pyproject.toml`); the loop runs in seconds.
- **Full gate (DoD / pre-merge):** `uv run pytest -m ""` runs the slow ICA suite too (~10–15 min). `uv run pytest -m slow` runs only the slow tests in isolation.
- **Mark slow tests:** any test that fits an ICA decomposition (calls `ica_clean_*`, `preprocess_raw` with `ica_strategy` set, or `label_components`) must be decorated `@pytest.mark.slow` so the default loop stays fast. Prefer class-level markers when a whole class is ICA-bound.
- `tests/data/` contains real-world fixtures — use them for valid tests
- New analysis functions require unit tests with simulated MNE data or synthetic sinusoids

Verification gates: [DoD](../scrum/04_Specs/DoD.md).
