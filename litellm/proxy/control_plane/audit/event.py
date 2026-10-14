"""The audit event value: what happened, by whom, on which dataplane, and which trace it belongs to."""

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, TypeAlias

Decision: TypeAlias = Literal["allowed", "denied", "unreachable"]
EventKind: TypeAlias = Literal["api_activity", "authentication"]
KEY_PREFIX: Final = "sk-"
REDACTED_KEY: Final = "sk-[redacted]"
CONTROL_PLANE_ID: Final = "control-plane"


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
    kind: EventKind = "api_activity"
    trace_id: str | None = None
    span_id: str | None = None
