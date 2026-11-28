"""Who did what through the control plane. Every event becomes a chained OCSF record, kept in a bounded ring for the
audit API and handed to the configured sinks for delivery."""

import json
import logging
import sys
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, TextIO, TypeAlias

from opentelemetry.trace import format_span_id, format_trace_id, get_current_span

from litellm._logging import redact_secrets
from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID, AuditEvent, Decision, EventKind, redact_path
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import NO_PIPELINE, Accepted, AuditPipeline, Offer, Reservation

DEFAULT_CAPACITY: Final = 10_000
MAX_AUDIT_FIELD_CHARS: Final = 2048
UNKNOWN_PRODUCT: Final = Product(version="unknown")
Trace: TypeAlias = tuple[str | None, str | None]


def build_audit_logger(stream: TextIO | None = None) -> logging.Logger:
    logger: Final = logging.Logger("litellm.control_plane.audit", logging.INFO)
    logger.propagate = False
    handler: Final = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


AUDIT_LOGGER: Final = build_audit_logger()


def _safe_text(value: str) -> str:
    return value.encode("utf-8", "backslashreplace").decode("utf-8")


def _bounded_redacted(value: str) -> str:
    return redact_secrets(value[:MAX_AUDIT_FIELD_CHARS])[:MAX_AUDIT_FIELD_CHARS]


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
        unauthenticated_records: deque[AuditRecord] | None = None,  # mutable-ok: bounded failure ring
        logger: logging.Logger = AUDIT_LOGGER,
        known_dataplane_ids: frozenset[str] | Callable[[], frozenset[str]] = frozenset(),
    ) -> None:
        self._records = records
        self._unauthenticated_records: deque[AuditRecord] = (  # mutable-ok: bounded failure ring
            unauthenticated_records if unauthenticated_records is not None else deque[AuditRecord]()
        )
        self._product = product
        self._pipeline = pipeline
        self._now = now
        self._trace = trace
        self._logger = logger
        self._known_dataplane_ids = known_dataplane_ids
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
        safe_actor: Final = _bounded_redacted(_safe_text(actor))
        safe_email: Final = None if actor_email is None else _bounded_redacted(_safe_text(actor_email))
        safe_method: Final = _bounded_redacted(_safe_text(method))
        safe_path: Final = _bounded_redacted(redact_path(_safe_text(path)[:MAX_AUDIT_FIELD_CHARS]))
        safe_detail: Final = None if detail is None else _bounded_redacted(_safe_text(detail))
        safe_dataplane_id: Final = _safe_text(dataplane_id)
        known_ids: Final = (
            self._known_dataplane_ids() if callable(self._known_dataplane_ids) else self._known_dataplane_ids
        )
        recorded_dataplane_id: Final = (
            safe_dataplane_id[:MAX_AUDIT_FIELD_CHARS]
            if dataplane_id in known_ids
            else _bounded_redacted(safe_dataplane_id)
        )
        event: Final = AuditEvent(
            at=self._now(),
            actor=safe_actor,
            actor_email=safe_email,
            dataplane_id=recorded_dataplane_id,
            method=safe_method,
            path=safe_path,
            decision=decision,
            status=status,
            credential=credential,
            detail=safe_detail,
            kind=kind,
            trace_id=trace_id,
            span_id=span_id,
        )
        self._state, record = chain(event, self._product, self._state)
        unauthenticated: Final = kind == "authentication" and decision == "denied"
        (self._unauthenticated_records if unauthenticated else self._records).append(record)
        if unauthenticated:
            offers: Final = self._pipeline.publish_unauthenticated(record)
            log_sink_accepted: Final = any(
                worker.sink.name == "log" and isinstance(offer, Accepted)
                for worker, offer in zip(self._pipeline.workers, offers, strict=True)
            )
            if not log_sink_accepted:
                self._logger.info("control_plane.audit %s", json.dumps(record.body, default=str, separators=(",", ":")))
        else:
            self._pipeline.publish(record, reservation)
        if reservation is not None:
            reservation.release()
        return event

    def record_login(self, username: str, decision: Decision, detail: str | None = None) -> AuditEvent:
        return self.record(
            username, None, CONTROL_PLANE_ID, "POST", "/auth/login", decision, detail=detail, kind="authentication"
        )

    def recent_records(
        self, limit: int, dataplane_ids: frozenset[str] | None = None, actor: str | None = None
    ) -> tuple[AuditRecord, ...]:
        """``dataplane_ids`` of ``None`` means every dataplane, an empty set means nothing is visible."""
        ordered: Final = tuple(
            sorted((*self._records, *self._unauthenticated_records), key=lambda record: record.sequence, reverse=True)
        )
        matching: Final = (
            record
            for record in ordered
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
    stream: TextIO | None = None,
    known_dataplane_ids: frozenset[str] | Callable[[], frozenset[str]] = frozenset(),
) -> AuditLog:
    logger: Final = AUDIT_LOGGER if stream is None else build_audit_logger(stream)
    return AuditLog(
        deque[AuditRecord](maxlen=capacity),
        product,
        pipeline,
        now,
        trace,
        deque[AuditRecord](maxlen=capacity),
        logger,
        known_dataplane_ids,
    )


__all__ = ["AuditEvent", "AuditLog", "Decision", "Offer", "Reservation", "build_audit_log", "redact_path", "utc_now"]
