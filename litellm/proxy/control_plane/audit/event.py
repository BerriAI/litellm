from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

Decision: TypeAlias = Literal["allowed", "denied", "unreachable"]
EventKind: TypeAlias = Literal["api_activity", "authentication"]
KEY_PREFIX: Final = "sk-"
REDACTED_KEY: Final = "sk-[redacted]"
REDACTION_DECODE_ROUNDS: Final = 8
CONTROL_PLANE_ID: Final = "control-plane"
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
    kind: EventKind = "api_activity"
    trace_id: str | None = None
    span_id: str | None = None
