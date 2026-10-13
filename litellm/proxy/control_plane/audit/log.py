"""Who did what on which dataplane through the control plane. Bounded in memory plus a structured log line."""

import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger

Decision: TypeAlias = Literal["allowed", "denied", "unreachable"]
DEFAULT_CAPACITY: Final = 10_000
KEY_PREFIX: Final = "sk-"
REDACTED_KEY: Final = "sk-[redacted]"


def redact_path(path: str) -> str:
    """A virtual key in a path (``/key/sk-.../reset_spend``) must never reach the log or the audit API."""
    return "/".join(REDACTED_KEY if segment.startswith(KEY_PREFIX) else segment for segment in path.split("/"))


@dataclass(frozen=True, slots=True)
class AuditEvent:
    at: datetime
    actor: str
    actor_email: str | None
    dataplane_id: str
    method: str
    path: str
    decision: Decision
    status: int | None
    credential: str | None
    detail: str | None


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class AuditLog:
    events: deque[AuditEvent]  # mutable-ok: a bounded ring buffer of events is the whole point
    now: Callable[[], datetime] = utc_now

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
    ) -> AuditEvent:
        event: Final = AuditEvent(
            at=self.now(),
            actor=actor,
            actor_email=actor_email,
            dataplane_id=dataplane_id,
            method=method,
            path=redact_path(path),
            decision=decision,
            status=status,
            credential=credential,
            detail=detail,
        )
        self.events.append(event)
        verbose_proxy_logger.info("control_plane.audit %s", json.dumps(asdict(event), default=str))
        return event

    def recent(
        self, limit: int, dataplane_ids: frozenset[str] | None = None, actor: str | None = None
    ) -> tuple[AuditEvent, ...]:
        """``dataplane_ids`` of ``None`` means every dataplane, an empty set means nothing is visible."""
        newest_first: Final = reversed(self.events)
        matching: Final = (
            event
            for event in newest_first
            if (dataplane_ids is None or event.dataplane_id in dataplane_ids)
            and (actor is None or event.actor == actor)
        )
        return tuple(event for _, event in zip(range(limit), matching, strict=False))


def build_audit_log(capacity: int = DEFAULT_CAPACITY, now: Callable[[], datetime] = utc_now) -> AuditLog:
    return AuditLog(events=deque(maxlen=capacity), now=now)
