"""Who did what through the control plane. Every event becomes a chained OCSF record, kept in a bounded ring for the
audit API and handed to the configured sinks for delivery."""

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, TypeAlias

from opentelemetry.trace import format_span_id, format_trace_id, get_current_span

from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID, AuditEvent, Decision, EventKind, redact_path
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import NO_PIPELINE, AuditPipeline, Offer, Reservation

DEFAULT_CAPACITY: Final = 10_000
UNKNOWN_PRODUCT: Final = Product(version="unknown")
Trace: TypeAlias = tuple[str | None, str | None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def current_trace() -> Trace:
    context: Final = get_current_span().get_span_context()
    if not context.is_valid:
        return None, None
    return format_trace_id(context.trace_id), format_span_id(context.span_id)


class AuditLog:
    def __init__(
        self,
        records: deque[AuditRecord],  # mutable-ok: a bounded ring buffer is the whole point
        product: Product,
        pipeline: AuditPipeline = NO_PIPELINE,
        now: Callable[[], datetime] = utc_now,
        trace: Callable[[], Trace] = current_trace,
    ) -> None:
        self._records = records
        self._product = product
        self._pipeline = pipeline
        self._now = now
        self._trace = trace
        self._state = ChainState()

    @property
    def pipeline(self) -> AuditPipeline:
        return self._pipeline

    def accepting(self) -> bool:
        return self._pipeline.accepting()

    def reserve(self) -> Reservation | None:
        """Hold one queue slot per strict sink for a write that is about to go out, None when no slot is free."""
        return self._pipeline.reserve()

    def record(
        self,
        actor: str,
        actor_email: str | None,
        dataplane_id: str,
        method: str,
        path: str,
        decision: Decision,
        status: int | None = None,
        credential: str | None = None,
        detail: str | None = None,
        kind: EventKind = "api_activity",
        reservation: Reservation | None = None,
    ) -> AuditEvent:
        trace_id, span_id = self._trace()
        event: Final = AuditEvent(
            at=self._now(),
            actor=actor,
            actor_email=actor_email,
            dataplane_id=dataplane_id,
            method=method,
            path=redact_path(path),
            decision=decision,
            status=status,
            credential=credential,
            detail=detail,
            kind=kind,
            trace_id=trace_id,
            span_id=span_id,
        )
        self._state, record = chain(event, self._product, self._state)
        self._records.append(record)
        if reservation is not None:
            reservation.release()
        self._pipeline.publish(record)
        return event

    def record_login(self, username: str, decision: Decision, detail: str | None = None) -> AuditEvent:
        return self.record(
            username, None, CONTROL_PLANE_ID, "POST", "/auth/login", decision, detail=detail, kind="authentication"
        )

    def recent_records(
        self, limit: int, dataplane_ids: frozenset[str] | None = None, actor: str | None = None
    ) -> tuple[AuditRecord, ...]:
        """``dataplane_ids`` of ``None`` means every dataplane, an empty set means nothing is visible."""
        matching: Final = (
            record
            for record in reversed(self._records)
            if (dataplane_ids is None or record.event.dataplane_id in dataplane_ids)
            and (actor is None or record.event.actor == actor)
        )
        return tuple(record for _, record in zip(range(limit), matching, strict=False))

    def recent(
        self, limit: int, dataplane_ids: frozenset[str] | None = None, actor: str | None = None
    ) -> tuple[AuditEvent, ...]:
        return tuple(record.event for record in self.recent_records(limit, dataplane_ids, actor))


def build_audit_log(
    capacity: int = DEFAULT_CAPACITY,
    now: Callable[[], datetime] = utc_now,
    pipeline: AuditPipeline = NO_PIPELINE,
    product: Product = UNKNOWN_PRODUCT,
    trace: Callable[[], Trace] = current_trace,
) -> AuditLog:
    return AuditLog(deque(maxlen=capacity), product, pipeline, now, trace)


__all__ = ["AuditEvent", "AuditLog", "Decision", "Offer", "Reservation", "build_audit_log", "redact_path", "utc_now"]
