from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class ColumnExtractor(ABC):
    @abstractmethod
    def extract(self, group: pd.DataFrame) -> Any:
        """Compute one column's value from a single trial's event rows."""

    @abstractmethod
    def codes(self) -> set[str]:
        """Event codes this extractor consumes; used to keep relevant rows."""


@dataclass
class CodeLookup(ColumnExtractor):
    """Map present event codes back to human labels.

    `agg="first"` returns the label of the first matching code (use when exactly
    one of the codes is expected); the default concatenates all matches in event
    order (non-matching rows contribute "").
    """

    code_map: dict[str, str]
    agg: str = "join"

    def extract(self, group: pd.DataFrame) -> str:
        # code_map is label->code; invert because the group gives us codes.
        reverse = {code: label for label, code in self.code_map.items()}
        if self.agg == "first":
            for desc in group["description"]:
                if desc in reverse:
                    return reverse[desc]
            return ""
        return "".join(reverse.get(desc, "") for desc in group["description"])

    def codes(self) -> set[str]:
        return set(self.code_map.values())


@dataclass
class BoolPresence(ColumnExtractor):
    code: str

    def extract(self, group: pd.DataFrame) -> bool:
        return self.code in group["description"].values

    def codes(self) -> set[str]:
        return {self.code}


@dataclass
class IntSum(ColumnExtractor):
    """Sum the integer weights of any matching codes (codes absent => 0).

    Useful for score-like columns where each marker contributes points.
    """

    code_map: dict[str, int]

    def extract(self, group: pd.DataFrame) -> int:
        return int(group["description"].map(self.code_map).fillna(0).sum())

    def codes(self) -> set[str]:
        return set(self.code_map.keys())


@dataclass
class ListCollect(ColumnExtractor):
    """Collect every matching code's value, preserving event (chronological) order.

    Unlike CodeLookup this keeps each occurrence, so it feeds the per-selection
    lists consumed by ExpandOnEvent.
    """

    code_map: dict[str, Any]

    def extract(self, group: pd.DataFrame) -> list[Any]:
        reverse = {code: value for value, code in self.code_map.items()}
        return group["description"].map(reverse).dropna().tolist()

    def codes(self) -> set[str]:
        return set(self.code_map.values())


@dataclass
class DerivedColumn(ColumnExtractor):
    """Column computed from other already-extracted columns, not from raw events.

    The parser detects this type and evaluates it in a final pass (see
    _extract_row), passing the partially-built row instead of the event group.
    """

    depends_on: list[str]
    fn: Callable[[dict[str, Any]], Any]

    def extract(self, group: pd.DataFrame) -> Any:
        # Never called: extract() operates on events, but a derived value needs
        # the assembled row. The parser routes DerivedColumns through derive().
        del group
        raise NotImplementedError("DerivedColumn uses derive()")

    def derive(self, row: dict[str, Any]) -> Any:
        return self.fn({key: row[key] for key in self.depends_on})

    def codes(self) -> set[str]:
        # Consumes columns, not raw event codes, so it contributes none.
        return set()


@dataclass
class ExpandOnEvent:
    """Turn a trial with repeated `event_code` markers into one row per occurrence.

    per_event_columns are extractors (typically ListCollect) whose results are
    indexed positionally against the selections — see parser._expand_trial_rows.
    """

    event_code: str
    per_event_columns: dict[str, ColumnExtractor]


@dataclass
class RTMeasure:
    """Reaction time as onset(end marker) - onset(start marker) within a trial."""

    name: str
    start: list[str]
    end: list[str]
    nan_if_negative: bool = True

    def extract(self, group: pd.DataFrame) -> float:
        # sum() collapses the (expected single) matching onset to a scalar and
        # yields 0 when a marker is missing.
        begin = group.loc[group["description"].isin(self.start), "onset"].sum()
        end = group.loc[group["description"].isin(self.end), "onset"].sum()
        rt = float(end - begin)
        # A negative RT means the markers are missing or out of order; report NaN
        # rather than a nonsensical negative latency.
        if self.nan_if_negative and rt < 0:
            return float("nan")
        return rt


@dataclass
class ExperimentSpec:
    """Declarative description of one experiment's metadata extraction.

    Fields:
      timelocks: event-code map consumed by the parser (``column_codes``)
        and by ``epoch_with_metadata`` (which timelocks exist).
      trial_codes / block_codes: markers that delimit trials and blocks.
      columns: output column name -> extractor.
      trial_end: see parser._assign_blocks_trials (shift amount, 0 or 1).
      rt_defs: reaction-time columns.
      trial_expander: optional one-row-per-selection expansion.
      csv_columns: output col -> (csv col, transform); positionally merged.
      infer_block_from: derive a Block column from an output column post-hoc.
    """

    name: str
    timelocks: dict[str, dict[str, str]]
    trial_codes: list[str]
    columns: dict[str, ColumnExtractor]
    block_codes: list[str] = field(default_factory=list)
    trial_end: bool = False
    rt_defs: list[RTMeasure] = field(default_factory=list)
    trial_expander: ExpandOnEvent | None = None
    csv_columns: dict[str, tuple[str, Callable[[Any], Any]]] | None = None
    infer_block_from: str | None = None

    def column_codes(self) -> set[str]:
        # Union of every code any extractor/RT/expander needs: this is what the
        # parser uses to decide which blocks/trials carry real data.
        codes: set[str] = set()
        for tl in self.timelocks.values():
            codes.update(tl.values())
        for extractor in self.columns.values():
            codes |= extractor.codes()
        for rt_def in self.rt_defs:
            codes.update(rt_def.start)
            codes.update(rt_def.end)
        if self.trial_expander is not None:
            codes.add(self.trial_expander.event_code)
            for extractor in self.trial_expander.per_event_columns.values():
                codes |= extractor.codes()
        return codes

    def event_codes(self) -> set[str]:
        # Full marker allowlist for the initial _select_events filter: column
        # codes plus the structural trial/block delimiters.
        return set(self.trial_codes) | set(self.block_codes) | self.column_codes()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        """Load an ExperimentSpec from a YAML config file.

        See PLAN_US005.md for the YAML format specification.
        """
        # Import here to avoid circular dependency at module level.
        from p0ly_eeg.metadata.transforms import TRANSFORM_REGISTRY

        data = yaml.safe_load(Path(path).read_text())

        # --- Resolve extractors from column configs ---
        columns: dict[str, ColumnExtractor] = {}
        for col_name, cfg in data.get("columns", {}).items():
            col_name = str(col_name)
            cfg = dict(cfg)  # shallow copy to avoid mutating loaded data
            etype = cfg.pop("type")
            if etype == "code_lookup":
                columns[col_name] = CodeLookup(**cfg)
            elif etype == "int_sum":
                columns[col_name] = IntSum(**cfg)
            elif etype == "bool_presence":
                columns[col_name] = BoolPresence(**cfg)
            elif etype == "list_collect":
                columns[col_name] = ListCollect(**cfg)
            elif etype == "derived":
                fn_name: str = cfg.pop("fn")
                fn = TRANSFORM_REGISTRY[fn_name]
                with_kwargs = cfg.pop("with", None)
                if with_kwargs:
                    fn = partial(fn, **with_kwargs)
                columns[col_name] = DerivedColumn(fn=fn, **cfg)
            else:
                raise ValueError(
                    f"Unknown extractor type '{etype}' for column '{col_name}'"
                )

        # --- Resolve RTMeasure list ---
        rt_defs: list[RTMeasure] = [
            RTMeasure(**cfg) for cfg in data.get("rt_defs", [])
        ]

        # --- Resolve trial_expander if present ---
        trial_expander: ExpandOnEvent | None = None
        if data.get("trial_expander") is not None:
            exp_cfg = dict(data["trial_expander"])
            event_code: str = exp_cfg.pop("event_code")
            per_col_configs: dict = exp_cfg.pop("per_event_columns", {})
            per_event_columns: dict[str, ColumnExtractor] = {}
            for col_name, ecfg in per_col_configs.items():
                ecfg = dict(ecfg)
                etype = ecfg.pop("type")
                if etype == "list_collect":
                    per_event_columns[str(col_name)] = ListCollect(**ecfg)
                else:
                    raise ValueError(
                        f"Unsupported per_event_column type '{etype}' "
                        f"for column '{col_name}'"
                    )
            trial_expander = ExpandOnEvent(
                event_code=event_code,
                per_event_columns=per_event_columns,
                **exp_cfg,
            )

        # --- Resolve csv_columns if present ---
        csv_columns: dict[str, tuple[str, Callable[[Any], Any]]] | None = None
        if data.get("csv_columns") is not None:
            csv_columns = {}
            for out_col, (csv_col, xform_name) in data["csv_columns"].items():
                xform = TRANSFORM_REGISTRY[xform_name]
                csv_columns[str(out_col)] = (str(csv_col), xform)

        # --- Handle trial_codes_range shorthand ---
        trial_codes: list[str]
        if "trial_codes_range" in data:
            start, end = data["trial_codes_range"]
            trial_codes = [f"Stim/S {n}" for n in range(start, end + 1)]
        else:
            trial_codes = list(data["trial_codes"])

        return cls(
            name=data["name"],
            timelocks=data["timelocks"],
            trial_codes=trial_codes,
            columns=columns,
            block_codes=data.get("block_codes", []),
            trial_end=data.get("trial_end", False),
            rt_defs=rt_defs,
            trial_expander=trial_expander,
            csv_columns=csv_columns,
            infer_block_from=data.get("infer_block_from"),
        )
