"""Contracts only. Implementations own persistence, pricing, projections and live budgets."""

from typing import Protocol

from litellm.accounting.types import (
    AdmissionDecision,
    AggregateQuery,
    AggregateRow,
    CostEvent,
    Dimension,
    InboundUsageEvent,
    Measurement,
    Metric,
    OutboundUsageEvent,
    Unpriced,
)


class UsageMeter(Protocol):
    async def record(self, event: InboundUsageEvent | OutboundUsageEvent) -> None:
        """Acknowledge commit, no-op identical IDs and reject conflicting content.

        HTTP acknowledgement is a separate policy. Unknown usage remains None.
        """
        ...


class CostCompiler(Protocol):
    async def compile(self, event: OutboundUsageEvent) -> CostEvent | Unpriced:
        """Reuse the committed charge or finalize one charge and its delivery intent.

        Resolve captured pricing context by event identity. Replay must not consume
        free allowance again. Missing pricing is Unpriced, never an inferred zero.
        """
        ...


class CostAggregator(Protocol):
    async def apply(self, measurements: tuple[Measurement, ...]) -> None:
        """Apply one source event's complete contribution set per target/version.

        Each target commits all contributions and its receipt atomically. A receipt
        for the first metric must not suppress the rest. Usage need not wait for cost.
        """
        ...

    async def query(self, query: AggregateQuery) -> tuple[AggregateRow, ...]:
        """Return disjoint [start, end) rows per group using existing daily projections.

        Filters combine with AND, values within a filter with IN. DAY emits daily
        buckets; None emits whole-interval totals. Reject unsupported metric/scope
        combinations. Different groups may overlap, including non-additive tags.
        """
        ...


class BudgetEnforcer(Protocol):
    async def reserve(
        self, reservation_id: str, dimensions: tuple[Dimension, ...], estimated: tuple[Metric, ...]
    ) -> AdmissionDecision:
        """Apply existing admission policies and estimates; duplicate IDs add no hold."""
        ...

    async def settle(self, reservation_id: str | None, cost_event: CostEvent) -> None:
        """Settle the committed charge with atomic deduplication and hold mutation.

        Preserve scope/window attribution and retry unfinished settlement after
        restart. SQL projection receipts do not provide live-counter atomicity.
        """
        ...

    async def release(self, reservation_id: str) -> None:
        """Idempotently release unused exposure at the request boundary, preserving charges."""
        ...
