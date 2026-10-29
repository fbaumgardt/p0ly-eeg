from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import pandas as pd

from p0ly_eeg.metadata.core import DerivedColumn, ExperimentSpec

if TYPE_CHECKING:
    import mne


# Canonical event-frame columns that ``parse_metadata`` consumes. ``raw.annotations``
# provides both; this helper only narrows to the two the parser needs.
_EVENT_COLUMNS = ["description", "onset"]


def events_from_raw(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Extract Psychtoolbox event-marker annotations from a raw object.

    Returns a DataFrame with the ``description`` / ``onset`` columns that
    ``parse_metadata`` expects (onset in seconds, per ``raw.annotations``).
    No epoching, no filtering, no file writes — the parser selects and
    assigns blocks/trials downstream.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object carrying event-marker ``Annotations`` (shape ``(n_channels,
        n_times)`` for the underlying data; ``n_annotations`` markers).

    Returns
    -------
    pd.DataFrame
        Columns ``description`` (str), ``onset`` (float, seconds). One row per
        annotation, in marker order. Empty frame with the correct columns when
        the raw carries no annotations.
    """
    ann = raw.annotations
    if len(ann) == 0:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    return pd.DataFrame({"description": ann.description, "onset": ann.onset})

# Canonical "Nothing" value: carries the exact schema that the downstream
# groupby(["_Block", "_Trial"]) in parse_metadata depends on, so an empty
# pipeline still produces a well-typed (zero-row) frame instead of a KeyError.
_EMPTY_EVENTS = pd.DataFrame(columns=["description", "onset", "_Block", "_Trial"])


def _bind(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Maybe-monad bind: an empty frame is ``Nothing`` and short-circuits the chain."""
    return _EMPTY_EVENTS.copy() if df.empty else fn(df)


def _infer_blocks(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Block"] = (df[column] != df[column].shift()).cumsum()
    return df


def _select_events(spec: ExperimentSpec, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Recordings label markers "Stimulus/..." but specs use the abbreviated
    # "Stim/..." form; normalise here so spec codes match before filtering.
    df["description"] = df["description"].str.replace("Stimulus", "Stim", regex=False)
    return df[["description", "onset"]].loc[
        lambda e: e["description"].isin(spec.event_codes())
    ]


def _assign_blocks_trials(spec: ExperimentSpec, events: pd.DataFrame) -> pd.DataFrame:
    _col_codes = spec.column_codes()
    events = events.copy()
    # Each block marker opens a new block; cumsum turns the marker positions
    # into a running 1-based-ish block id shared by every row until the next one.
    events["_Block"] = events["description"].isin(spec.block_codes).cumsum()
    # Drop blocks that never carry a data-bearing code (e.g. warm-up / leading
    # markers before the first real block), so they don't become phantom trials.
    events = events[
        events.groupby("_Block")["description"].transform(
            lambda g: g.isin(_col_codes).any()
        )
    ]
    # Re-base surviving block ids to start at 1 after the leading drop above.
    events["_Block"] = events["_Block"] - (events["_Block"].min() - 1)

    # trial_end is a bool used as a shift amount: when trials are delimited by
    # their *end* marker (shift=1) the boundary is moved forward one row so the
    # marker is counted with the trial it closes rather than the next one.
    events['_TrialBegin'] = events["description"].isin(spec.trial_codes).shift(spec.trial_end,fill_value=False)
    events['_Trial'] = events.groupby("_Block")['_TrialBegin'].cumsum()
    # Same phantom-trial guard as for blocks, now per (block, trial) group.
    events = events[
        events.groupby(["_Block", "_Trial"])["description"].transform(
            lambda g: g.isin(_col_codes).any()
        )
    ]
    # Re-base trial ids to start at 1 within each block (cummin == per-block min).
    events['_Trial'] -= events.groupby("_Block")['_Trial'].cummin()-1

    return events.drop(columns=['_TrialBegin'])


def _prepare_event_frame(spec: ExperimentSpec, df: pd.DataFrame) -> pd.DataFrame:
    # `.pipe(_bind, f)` == `_bind(frame, f)`, so each step is a monadic bind:
    # the first empty frame (no input, or nothing survived filtering) short-
    # circuits to _EMPTY_EVENTS and the later stages are skipped.
    return (
        df
        .pipe(_bind, lambda d: _select_events(spec, d))
        .pipe(_bind, lambda e: _assign_blocks_trials(spec, e))
    )


def _extract_row(spec: ExperimentSpec, group: pd.DataFrame, block: int, trial: int) -> dict:
    row: dict = {"Block": block, "Trial": trial, "Onset": float(group["onset"].iloc[0])}
    # Two passes over `columns`: plain extractors read from the event group,
    # while DerivedColumns read from the row built so far. The second loop must
    # run last so every value a DerivedColumn depends on is already populated.
    for col_name, extractor in spec.columns.items():
        if not isinstance(extractor, DerivedColumn):
            row[col_name] = extractor.extract(group)
    for rt_def in spec.rt_defs:
        row[rt_def.name] = rt_def.extract(group)
    for col_name, extractor in spec.columns.items():
        if isinstance(extractor, DerivedColumn):
            row[col_name] = extractor.derive(row)
    return row


def _trial_start_onset(group: pd.DataFrame, spec: ExperimentSpec) -> float:
    # t=0 for per-selection reaction times: prefer the first RT measure's start
    # marker (the experiment's defined trial onset) and fall back to the generic
    # trial code when no RT is defined.
    if spec.rt_defs:
        codes = spec.rt_defs[0].start
    else:
        codes = spec.trial_codes
    return float(group.loc[group["description"].isin(codes), "onset"].sum())


def _expand_trial_rows(
    spec: ExperimentSpec,
    group: pd.DataFrame,
    base_row: dict,
    expand_trials: bool,
) -> list[dict]:
    assert spec.trial_expander is not None
    expander = spec.trial_expander
    sel_rows = group.loc[group["description"] == expander.event_code]
    total_sels = len(sel_rows)
    if total_sels == 0:
        return [base_row]

    # When not expanding, collapse a multi-selection trial to a single row
    # describing only the final selection (Total_Sel still reflects the full count).
    if not expand_trials:
        sel_rows = sel_rows.iloc[-1:]

    begin = _trial_start_onset(group, spec)
    trial_lists = {
        col_name: extractor.extract(group)
        for col_name, extractor in expander.per_event_columns.items()
    }

    expanded: list[dict] = []
    for k, (_, sel) in enumerate(sel_rows.iterrows()):
        sub_row = base_row.copy()
        sub_row["RT_Select"] = float(sel["onset"] - begin)
        sub_row["Num_Sel"] = k
        sub_row["Total_Sel"] = total_sels
        for col_name in expander.per_event_columns:
            val = trial_lists[col_name]
            # Per-event extractors yield a list aligned to selection order; pick
            # the k-th entry (blank if the lists are ragged). Scalars apply to all.
            if isinstance(val, list):
                sub_row[col_name] = val[k] if k < len(val) else ""
            else:
                sub_row[col_name] = val
        expanded.append(sub_row)
    return expanded


def _merge_csv(
    df: pd.DataFrame,
    spec: ExperimentSpec,
    csv_path: str,
) -> pd.DataFrame:
    if spec.csv_columns is None:
        return df
    csv_df = pd.read_csv(csv_path)
    df = df.copy()
    # External CSV is aligned to trials *positionally* (row i <-> trial i), not by
    # any key: truncate when the CSV is longer, pad with NaN (reindex) when shorter.
    # `.values` strips the CSV index so assignment aligns by position, not label.
    for output_col, (csv_col, transform_fn) in spec.csv_columns.items():
        if len(csv_df) >= len(df):
            values = csv_df[csv_col].iloc[: len(df)].map(transform_fn)
        else:
            values = csv_df[csv_col].map(transform_fn)
            values = values.reindex(range(len(df)))
        df[output_col] = values.values
    return df


def _empty_result(spec: ExperimentSpec) -> pd.DataFrame:
    """Zero-row frame carrying the full column set :func:`_extract_row` produces.

    An event-less recording is legitimate (no markers → no trials), but the
    result must stay *typed*: a bare ``pd.DataFrame([])`` has zero columns, so
    ``to_csv`` writes no header and the consumer dies in ``read_csv`` with
    ``EmptyDataError: No columns to parse from file``. Building the columns in
    ``_extract_row`` order keeps an empty run column-identical to a populated
    one (SCHEMA §2).

    Parameters
    ----------
    spec : ExperimentSpec
        Experiment spec whose ``columns`` / ``rt_defs`` define the schema.

    Returns
    -------
    pd.DataFrame
        Frame with ``Block`` / ``Trial`` / ``Onset``, the non-derived spec
        columns, the reaction-time measures, then the derived columns — and
        zero rows.
    """
    columns = ["Block", "Trial", "Onset"]
    columns += [
        name for name, extractor in spec.columns.items() if not isinstance(extractor, DerivedColumn)
    ]
    columns += [rt_def.name for rt_def in spec.rt_defs]
    columns += [
        name for name, extractor in spec.columns.items() if isinstance(extractor, DerivedColumn)
    ]
    # Expander specs (:class:`ExpandOnEvent`) extend each row with the selection
    # bookkeeping columns and one column per per-event extractor.
    if spec.trial_expander is not None:
        columns += ["RT_Select", "Num_Sel", "Total_Sel"]
        columns += list(spec.trial_expander.per_event_columns)
    # A spec column may shadow a base key (simonfb declares its own "Block");
    # the populated path writes into the same dict key, so dedupe to match it.
    seen: set[str] = set()
    unique: list[str] = []
    for name in columns:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return pd.DataFrame(columns=unique)


def parse_metadata(
    spec: ExperimentSpec,
    df: pd.DataFrame,
    csv_path: str | None = None,
    expand_trials: bool = False,
) -> pd.DataFrame:
    events = _prepare_event_frame(spec, df)
    rows: list[dict] = []
    for (block, trial), group in events.groupby(["_Block", "_Trial"], sort=True):
        row = _extract_row(spec, group, cast(int, block), cast(int, trial))
        if spec.trial_expander is not None:
            rows.extend(_expand_trial_rows(spec, group, row, expand_trials))
        else:
            rows.append(row)

    result = _empty_result(spec) if not rows else pd.DataFrame(rows)
    if spec.infer_block_from is not None:
        result = _infer_blocks(result, spec.infer_block_from)
    if csv_path is not None and spec.csv_columns is not None:
        result = _merge_csv(result, spec, csv_path)
    return result
