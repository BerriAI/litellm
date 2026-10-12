"""Decisions-specific pricing read from ``<field>_decisions`` cost map fields.

A model that serves both chat and Decisions keeps one cost map row. Rates that
differ for ``/v1/decisions`` and ``/v1/systemone`` live on that row under the
same field name with a ``_decisions`` suffix, e.g. ``output_cost_per_token_decisions``,
and override the plain field for those two routes only. A row without any such
field bills Decisions calls from its plain fields.
"""

from collections.abc import Mapping
from typing import Final

DECISIONS_PRICING_SUFFIX: Final = "_decisions"
DECISIONS_CALL_TYPES: Final = frozenset({"decisions", "adecisions", "systemone", "asystemone"})


def is_decisions_call_type(call_type: str) -> bool:
    return call_type in DECISIONS_CALL_TYPES


def decisions_pricing_fields(row: Mapping[str, object]) -> dict[str, object]:
    """Return ``{plain_field: value}`` for every ``<plain_field>_decisions`` field on ``row``."""
    return {
        key.removesuffix(DECISIONS_PRICING_SUFFIX): value
        for key, value in row.items()
        if key.endswith(DECISIONS_PRICING_SUFFIX)
    }


def apply_decisions_pricing(row: Mapping[str, object]) -> dict[str, object]:
    """Return ``row`` priced for a Decisions call: ``_decisions`` values replace their plain fields."""
    plain: Final = {key: value for key, value in row.items() if not key.endswith(DECISIONS_PRICING_SUFFIX)}
    return {**plain, **decisions_pricing_fields(row)}
