from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from litellm.accounting.types import AggregateQuery, AggregateRow


class CostAggregator(Protocol):
    async def apply(self, rows: tuple[Mapping[str, object], ...]) -> None:
        """Apply stored request contributions atomically with their delivery receipts."""
        ...

    async def query(self, query: AggregateQuery) -> tuple[AggregateRow, ...]:
        """Return disjoint [start, end) rows per group using existing daily projections.

        Filters combine with AND, values within a filter with IN. DAY emits daily
        buckets; None emits whole-interval totals. Reject unsupported metric/scope
        combinations. Different groups may overlap, including non-additive tags.
        """
        ...
