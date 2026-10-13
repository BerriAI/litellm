"""Who did what on which dataplane through the control plane. Bounded in memory plus a structured log line."""

import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import reduce
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

from litellm._logging import verbose_proxy_logger

Decision: TypeAlias = Literal["allowed", "denied", "unreachable"]
DEFAULT_CAPACITY: Final = 10_000
KEY_PREFIX: Final = "sk-"
REDACTED_KEY: Final = "sk-[redacted]"
REDACTION_DECODE_ROUNDS: Final = 8
FINAL_KEY_ACTIONS: Final = frozenset({"reset_spend", "regenerate"})


def _decode_segment(segment: str, remaining: int = REDACTION_DECODE_ROUNDS) -> str:
    return reduce(lambda decoded, _: unquote(decoded), range(remaining), segment)


def redact_path(path: str) -> str:
    """A virtual key in a path (``/key/sk-.../reset_spend``) must never reach the log or the audit API."""
    segments: Final = path.split("/")
    key_index: Final = next(
        (index for index, segment in enumerate(segments) if _decode_segment(segment).lower().startswith(KEY_PREFIX)),
        None,
    )
    if key_index is None:
        return path
    final_action: Final = segments[-1] if segments[-1] in FINAL_KEY_ACTIONS else None
    return "/".join((*segments[:key_index], REDACTED_KEY, *((final_action,) if final_action is not None else ())))


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
