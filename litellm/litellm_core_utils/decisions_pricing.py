"""The Decisions call types and the ``<field>_decisions`` cost map fields they read.

A model that serves both chat and Decisions keeps one cost map row. Rates that
differ for ``/v1/decisions`` and ``/v1/systemone`` live on that row under the
same field name with a ``_decisions`` suffix, e.g. ``output_cost_per_token_decisions``,
and the Rust pricing core lays them over the plain field for those two routes only.
"""

from collections.abc import Mapping
from typing import Final

DECISIONS_PRICING_SUFFIX: Final = "_decisions"
DECISIONS_CALL_TYPES: Final = frozenset({"decisions", "adecisions", "systemone", "asystemone"})


def is_decisions_call_type(call_type: str) -> bool:
    return call_type in DECISIONS_CALL_TYPES


def decisions_fields(entry: Mapping[str, object]) -> Mapping[str, object]:
    """The ``_decisions`` fields of a raw cost map entry, which a model info row leaves out."""
    return {key: value for key, value in entry.items() if key.endswith(DECISIONS_PRICING_SUFFIX)}


def decisions_pricing_fields(row: Mapping[str, object]) -> Mapping[str, object]:
    """``{plain_field: value}`` for every ``<plain_field>_decisions`` field on ``row``."""
    return {key.removesuffix(DECISIONS_PRICING_SUFFIX): value for key, value in decisions_fields(row).items()}


def apply_decisions_pricing(row: Mapping[str, object]) -> Mapping[str, object]:
    """``row`` as a Decisions call reads it: each ``_decisions`` value in place of its plain field."""
    plain: Final = {key: value for key, value in row.items() if not key.endswith(DECISIONS_PRICING_SUFFIX)}
    return {**plain, **decisions_pricing_fields(row)}
