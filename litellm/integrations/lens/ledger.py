import json
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Final

import litellm
from litellm.integrations.clickhouse.spend_log_types import SpendLogRecord

MAX_HELD_ROWS: Final = 10_000
MAX_HELD_BYTES: Final = 64 * 1024 * 1024
HANDOFF_SECONDS: Final = 300.0
_IDENTITY_FIELDS: Final = ("api_key", "team_id", "user", "organization_id", "end_user")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _milliseconds(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    aware: Final = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return int(aware.timestamp() * 1000)


def _accounting(committed: Mapping[str, object]) -> Mapping[str, object] | None:
    """Cost, usage and identity exactly as the accepted delivery's accounting recorded them.

    Every field is always set: missing or null token counts are zero and missing identity is empty,
    so nothing from the callback's own billing or ownership survives. A delivery without a recorded
    spend has no cost to report and yields None.
    """
    spend: Final = _number(committed.get("spend"))
    if spend is None:
        return None
    prompt: Final = _count(committed.get("prompt_tokens")) or 0
    completion: Final = _count(committed.get("completion_tokens")) or 0
    identity: Final = {field: _text(committed.get(field)) or "" for field in _IDENTITY_FIELDS}
    return {
        "spend": spend,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        **identity,
        **({"end_user": ""} if litellm.disable_end_user_cost_tracking else {}),
    }


def merged_row(companion: SpendLogRecord, committed: Mapping[str, object]) -> Mapping[str, object] | None:
    """The callback's record carrying the committed accounting in place of its own billing values."""
    accounting: Final = _accounting(committed)
    return None if accounting is None else {**companion, **accounting}


def _end_time(committed: Mapping[str, object], start: int) -> int | None:
    end: Final = _milliseconds(committed.get("endTime"))
    if end is not None:
        return end
    duration: Final = _count(committed.get("request_duration_ms"))
    return None if duration is None else start + duration


def minimal_row(committed: Mapping[str, object]) -> Mapping[str, object] | None:
    """A cost record built from the committed accounting alone, when no callback record reached this pod."""
    request_id: Final = _text(committed.get("request_id"))
    start: Final = _milliseconds(committed.get("startTime"))
    accounting: Final = _accounting(committed)
    if not request_id or start is None or accounting is None:
        return None
    end: Final = _end_time(committed, start)
    if end is None:
        return None
    return {
        "request_id": request_id,
        "response_id": request_id,
        "provider_request_id": "",
        "litellm_call_id": _text(committed.get("litellm_call_id")) or "",
        "call_type": _text(committed.get("call_type")) or "",
        "api_key": "",
        "key_alias": "",
        "team_id": "",
        "team_alias": "",
        "organization_id": "",
        "user": "",
        "end_user": "",
        "model": _text(committed.get("model")) or "",
        "model_group": _text(committed.get("model_group")) or "",
        "model_id": _text(committed.get("model_id")) or "",
        "custom_llm_provider": _text(committed.get("custom_llm_provider")) or "",
        "api_base": _text(committed.get("api_base")) or "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "start_time": start,
        "end_time": end,
        "completion_start_time": None,
        "status": _text(committed.get("status")) or "",
        "error_str": "",
        "cache_hit": committed.get("cache_hit") in (True, "True", "true"),
        "session_id": _text(committed.get("session_id")) or "",
        "trace_id": "",
        "span_id": "",
        "request_tags": [],
        "metadata": json.dumps({}),
        "messages": "",
        "response": "",
        **accounting,
    }


@dataclass(frozen=True, slots=True)
class _Held:
    row: SpendLogRecord
    size: int
    at: float


@dataclass(frozen=True, slots=True)
class _Committed:
    request_id: str
    row: Mapping[str, object]
    at: float


@dataclass(frozen=True, slots=True)
class Release:
    ready: tuple[Mapping[str, object], ...] = ()
    dropped: int = 0


def _joined(*releases: Release) -> Release:
    return Release(
        ready=tuple(row for release in releases for row in release.ready),  # comprehension-ok: flattens a few releases
        dropped=sum(release.dropped for release in releases),
    )


def _ready(rows: Sequence[Mapping[str, object] | None]) -> Release:
    ready: Final = tuple(row for row in rows if row is not None)
    return Release(ready=ready, dropped=len(rows) - len(ready))


class LedgerHandoff:
    """Pairs ledger-backed callback records with their confirmed deliveries, whichever arrives first.

    A record is exported only once a delivery for its request is confirmed, never before. Each
    confirmed delivery is kept by its own id: the first one for a request takes the callback
    record, and any further delivery for that request is exported from its accounting alone.
    Both maps stay in arrival order, so expiry pops only the stale prefix.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        max_rows: int = MAX_HELD_ROWS,
        max_bytes: int = MAX_HELD_BYTES,
        wait_seconds: float = HANDOFF_SECONDS,
    ) -> None:
        self.clock: Final = clock
        self.max_rows: Final = max_rows
        self.max_bytes: Final = max_bytes
        self.wait_seconds: Final = wait_seconds
        self.held: Final[OrderedDict[str, _Held]] = OrderedDict()  # mutable-ok: bounded records awaiting commit
        self.committed: Final[OrderedDict[str, _Committed]] = OrderedDict()  # mutable-ok: bounded early deliveries
        self.waiting: Final[dict[str, list[str]]] = {}  # mutable-ok: early delivery ids per request
        self.seen: Final[OrderedDict[str, None]] = OrderedDict()  # mutable-ok: recent delivery ids
        self.held_bytes = 0

    def _take_held(self, key: str) -> _Held | None:
        held: Final = self.held.pop(key, None)
        if held is not None:
            self.held_bytes -= held.size
        return held

    def _pop_committed(self, delivery_id: str) -> _Committed:
        committed: Final = self.committed.pop(delivery_id)
        deliveries: Final = self.waiting.get(committed.request_id, [])
        if delivery_id in deliveries:
            deliveries.remove(delivery_id)
        if not deliveries:
            self.waiting.pop(committed.request_id, None)
        return committed

    def _stale_held(self, now: float) -> Iterator[str]:
        while self.held:
            key, held = next(iter(self.held.items()))
            if now - held.at < self.wait_seconds:
                return
            self._take_held(key)
            yield key

    def _stale_commits(self, now: float) -> Iterator[_Committed]:
        while self.committed:
            delivery_id, committed = next(iter(self.committed.items()))
            if now - committed.at < self.wait_seconds:
                return
            yield self._pop_committed(delivery_id)

    def _expire(self) -> Release:
        now: Final = self.clock()
        dropped: Final = sum(1 for _ in self._stale_held(now))
        orphans: Final = _ready(tuple(minimal_row(committed.row) for committed in self._stale_commits(now)))
        return _joined(Release(dropped=dropped), orphans)

    def hold(self, key: str, row: SpendLogRecord, size: int) -> Release:
        expired: Final = self._expire()
        early: Final = self.waiting.get(key)
        if early:
            committed: Final = self._pop_committed(early[0])
            return _joined(expired, _ready((merged_row(row, committed.row),)))
        if size > self.max_bytes:
            return _joined(expired, Release(dropped=1))
        replaced: Final = self._take_held(key)
        evicted: Final = self._evict(size)
        self.held[key] = _Held(row=row, size=size, at=self.clock())
        self.held_bytes += size
        return _joined(expired, Release(dropped=evicted + (replaced is not None)))

    def _evict(self, incoming: int) -> int:
        evicted = 0  # rebind-ok: counts the oldest records removed to fit a new one
        while self.held and (len(self.held) >= self.max_rows or self.held_bytes + incoming > self.max_bytes):
            self._take_held(next(iter(self.held)))
            evicted += 1
        return evicted

    def _first_report(self, delivery_id: str) -> bool:
        if delivery_id in self.seen:
            return False
        self.seen[delivery_id] = None
        if len(self.seen) > self.max_rows:
            self.seen.popitem(last=False)
        return True

    def _accept(self, row: Mapping[str, object], now: float) -> Release:
        delivery_id: Final = _text(row.get("delivery_id"))
        request_id: Final = _text(row.get("request_id"))
        if delivery_id is None or not self._first_report(delivery_id):
            return Release()
        if row.get("unreportable") is True or not request_id:
            if request_id:
                self._take_held(request_id)
            return Release(dropped=1)
        held: Final = self._take_held(request_id)
        if held is not None:
            return _ready((merged_row(held.row, row),))
        self.committed[delivery_id] = _Committed(request_id=request_id, row=row, at=now)
        self.waiting.setdefault(request_id, []).append(delivery_id)
        return Release()

    def commit(self, rows: Sequence[Mapping[str, object]]) -> Release:
        expired: Final = self._expire()
        now: Final = self.clock()
        accepted: Final = _joined(*(self._accept(row, now) for row in rows))
        excess: Final = tuple(self.committed)[: max(len(self.committed) - self.max_rows, 0)]
        overflow: Final = _ready(tuple(minimal_row(self._pop_committed(delivery).row) for delivery in excess))
        return _joined(expired, accepted, overflow)

    def drain(self) -> Release:
        dropped: Final = len(self.held)
        self.held.clear()
        self.held_bytes = 0
        orphans: Final = tuple(minimal_row(self._pop_committed(delivery).row) for delivery in tuple(self.committed))
        return _joined(Release(dropped=dropped), _ready(orphans))
