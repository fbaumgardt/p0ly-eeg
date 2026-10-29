"""Named-function registry for ExperimentSpec YAML serialization.

DerivedColumn and csv_columns transforms use lambdas that can't be
represented in YAML. This module maps string names to callables so
YAML specs can reference transform logic by name.

Add new entries to TRANSFORM_REGISTRY when an experiment requires
custom derived columns or csv-column transforms.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


def identity(x: Any) -> Any:
    """Pass-through transform (used by csv_columns that need no conversion)."""
    return x


# ---------------------------------------------------------------------------
# Derived-column functions (row: dict -> Any)
# ---------------------------------------------------------------------------


def _congruent_from_side_response_correct(row: dict[str, Any]) -> bool:
    """simonfb.Congruent: (Side == Response) == Correct."""
    return bool((row["Side"] == row["Response"]) == row["Correct"])


def _trial_to_block_index(row: dict[str, Any], *, blocksize: int = 60) -> int:
    """simonfb.Block: 0-based block index from Trial number.

    blocksize is injected from the YAML ``with:`` key at load time.
    """
    return int(row["Trial"] - 1) // blocksize


# ---------------------------------------------------------------------------
# csv-column transform functions (value -> Any)
# ---------------------------------------------------------------------------


def _lookup_condition(x: Any) -> str | None:
    """intwm.Condition: integer condition code -> label."""
    return {1: "static", 2: "ignore", 3: "constant", 4: "changing"}.get(x)


def _lookup_target(x: Any) -> str | None:
    """intwm.Target: integer target code -> label."""
    return {1: "left", 2: "right"}.get(x)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[str, Callable[..., Any]] = {
    "identity": identity,
    "side_response_correct_congruent": _congruent_from_side_response_correct,
    "trial_block_index": _trial_to_block_index,
    "lookup_condition": _lookup_condition,
    "lookup_target": _lookup_target,
}
