from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional


AUTOQ_REQUEST_KEY_PREFIX = "autoq:req:"
AUTOQ_QUEUE_KEY_PREFIX = "autoq:queue:"
AUTOQ_ACTIVE_KEY_PREFIX = "autoq:active:"
AUTOQ_LIMIT_KEY_PREFIX = "autoq:limit:"
AUTOQ_SUCCESS_KEY_PREFIX = "autoq:success:"
AUTOQ_CEILING_KEY_PREFIX = "autoq:ceiling:"
AUTOQ_CLAIM_KEY_PREFIX = "autoq:claim:"
AUTOQ_ACTIVE_LEASE_KEY_PREFIX = "autoq:active_lease:"


@dataclass(slots=True)
class AutoQueueRequestState:
    request_id: str
    model: str
    priority: int
    state: str
    enqueued_at_ms: int
    deadline_at_ms: int
    worker_id: str
    claim_token: Optional[str] = None
    claimed_at_ms: Optional[int] = None
    started_at_ms: Optional[int] = None
    finished_at_ms: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    def to_hash(self) -> dict[str, str]:
        """Return a Redis-hash-friendly mapping with string values."""
        payload = asdict(self)
        return {key: "" if value is None else str(value) for key, value in payload.items()}

    @classmethod
    def from_json(cls, raw: str | bytes | bytearray) -> "AutoQueueRequestState":
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        payload = json.loads(raw)
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    @classmethod
    def from_hash(cls, raw: dict[str, Any]) -> "AutoQueueRequestState":
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(key, bytes):
                key = key.decode()
            normalized[str(key)] = value

        def _maybe_int(value: Any) -> Optional[int]:
            if value in (None, "", b""):
                return None
            if isinstance(value, bytes):
                value = value.decode()
            return int(value)

        def _maybe_str(value: Any) -> Optional[str]:
            if value in (None, "", b""):
                return None
            if isinstance(value, bytes):
                return value.decode()
            return str(value)

        return cls(
            request_id=_maybe_str(normalized.get("request_id")) or "",
            model=_maybe_str(normalized.get("model")) or "",
            priority=int(normalized.get("priority") or 0),
            state=_maybe_str(normalized.get("state")) or "",
            enqueued_at_ms=int(normalized.get("enqueued_at_ms") or 0),
            deadline_at_ms=int(normalized.get("deadline_at_ms") or 0),
            worker_id=_maybe_str(normalized.get("worker_id")) or "",
            claim_token=_maybe_str(normalized.get("claim_token")),
            claimed_at_ms=_maybe_int(normalized.get("claimed_at_ms")),
            started_at_ms=_maybe_int(normalized.get("started_at_ms")),
            finished_at_ms=_maybe_int(normalized.get("finished_at_ms")),
        )


def request_key(request_id: str) -> str:
    return f"{AUTOQ_REQUEST_KEY_PREFIX}{request_id}"


def queue_key(model: str) -> str:
    return f"{AUTOQ_QUEUE_KEY_PREFIX}{model}"


def active_key(model: str) -> str:
    return f"{AUTOQ_ACTIVE_KEY_PREFIX}{model}"


def limit_key(model: str) -> str:
    return f"{AUTOQ_LIMIT_KEY_PREFIX}{model}"


def success_key(model: str) -> str:
    return f"{AUTOQ_SUCCESS_KEY_PREFIX}{model}"


def ceiling_key(model: str) -> str:
    return f"{AUTOQ_CEILING_KEY_PREFIX}{model}"


def claim_key(request_id: str) -> str:
    return f"{AUTOQ_CLAIM_KEY_PREFIX}{request_id}"


def active_lease_key(request_id: str) -> str:
    return f"{AUTOQ_ACTIVE_LEASE_KEY_PREFIX}{request_id}"


def queue_score(priority: int, now_ms: int) -> int:
    """Build a sortable score where lower values are served first.

    Contract: lower priority values are served first, then earlier enqueue time.
    Redis breaks ties lexicographically by member name, which keeps ordering
    deterministic across workers when score values match exactly.
    """
    return (priority * 10_000_000_000_000) + now_ms


def active_lease_payload(worker_id: str, claim_token: str) -> str:
    return json.dumps({"worker_id": worker_id, "claim_token": claim_token}, separators=(",", ":"))


def active_lease_hash(worker_id: str, claim_token: str) -> dict[str, str]:
    return {"worker_id": worker_id, "claim_token": claim_token}


def request_state_payload(state: AutoQueueRequestState) -> str:
    return state.to_json()


def request_state_from_payload(payload: str | bytes | bytearray) -> AutoQueueRequestState:
    return AutoQueueRequestState.from_json(payload)


def request_state_hash(state: AutoQueueRequestState) -> dict[str, str]:
    return state.to_hash()


def request_state_from_hash(payload: dict[str, Any]) -> AutoQueueRequestState:
    return AutoQueueRequestState.from_hash(payload)
