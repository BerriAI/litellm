from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from litellm.accounting.types import AggregateQuery, AggregateRow

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload


class CostAggregator(Protocol):
    async def apply(self, rows: tuple[SpendLogsPayload, ...]) -> None:
        """Apply stored request contributions atomically with their delivery receipts."""
        ...

    async def query(self, query: AggregateQuery) -> tuple[AggregateRow, ...]:
        """Return disjoint [start, end) rows per group using existing daily projections.

        Filters combine with AND, values within a filter with IN. DAY emits daily
        buckets; None emits whole-interval totals. Reject unsupported metric/scope
        combinations. Different groups may overlap, including non-additive tags.
        """
        ...
