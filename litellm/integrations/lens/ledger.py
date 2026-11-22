import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Final, TypeAlias

import litellm
from litellm.integrations.clickhouse.clickhouse_spend_logger import strip_cache_hit_suffix
from litellm.integrations.clickhouse.spend_log_types import SpendLogRecord

MAX_HELD_ROWS: Final = 10_000
MAX_HELD_BYTES: Final = 64 * 1024 * 1024
HANDOFF_SECONDS: Final = 300.0
_IDENTITY_FIELDS: Final = ("api_key", "team_id", "user", "organization_id", "end_user")
_OWNER_FIELDS: Final = ("api_key", "team_id", "user")
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND: Final = timedelta(microseconds=1)
CorrelationKey: TypeAlias = tuple[str, str, str, str, int]
"""The spend log id, the authenticated key hash, team and user, and the start in UTC microseconds."""


def start_micros(moment: datetime) -> int:
    """The instant the spend log row records, in UTC microseconds. A naive time is local, as the row builder reads it."""
    aware: Final = moment if moment.tzinfo is not None else moment.astimezone()
    return (aware - _EPOCH) // _MICROSECOND


def _owner(row: Mapping[str, object]) -> tuple[str, str, str]:
    api_key, team_id, user = (_text(row.get(field)) or "" for field in _OWNER_FIELDS)
    return api_key, team_id, user


def _committed_key(committed: Mapping[str, object]) -> CorrelationKey | None:
    request_id: Final = _text(committed.get("request_id"))
    started: Final = _text(committed.get("startTime"))
    if not request_id or not started:
        return None
    try:
        moment: Final = datetime.fromisoformat(started)
    except ValueError:
        return None
    return (strip_cache_hit_suffix(request_id), *_owner(committed), start_micros(moment))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


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


@dataclass(frozen=True, slots=True)
class _Held:
    row: SpendLogRecord
    size: int
    at: float


@dataclass(frozen=True, slots=True)
class _Committed:
    key: CorrelationKey
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

    A record is exported only once a delivery for its request is confirmed and its callback
    record is here, never before. The two meet on their correlation key, so a delivery only takes
    a record of the same request, owner and start. Two records, or two deliveries, waiting under the
    same key at the same time cannot be told apart and are both dropped. Nothing remembers the key
    afterwards. A record or delivery still unmatched after the wait,
    past the bounds, or at shutdown is dropped and counted. Both maps stay in arrival order, so
    expiry pops only the stale prefix.
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
        self.held: Final[OrderedDict[CorrelationKey, _Held]] = OrderedDict()  # mutable-ok: records awaiting commit
        self.committed: Final[OrderedDict[str, _Committed]] = OrderedDict()  # mutable-ok: bounded early deliveries
        self.waiting: Final[dict[CorrelationKey, str]] = {}  # mutable-ok: the early delivery id per key
        self.seen: Final[OrderedDict[str, None]] = OrderedDict()  # mutable-ok: recent delivery ids
        self.held_bytes = 0

    def _take_held(self, key: CorrelationKey) -> _Held | None:
        held: Final = self.held.pop(key, None)
        if held is not None:
            self.held_bytes -= held.size
        return held

    def _pop_committed(self, delivery_id: str) -> _Committed:
        committed: Final = self.committed.pop(delivery_id)
        if self.waiting.get(committed.key) == delivery_id:
            del self.waiting[committed.key]
        return committed

    def _stale_held(self, now: float) -> Iterator[CorrelationKey]:
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
        dropped: Final = sum(1 for _ in self._stale_held(now)) + sum(1 for _ in self._stale_commits(now))
        return Release(dropped=dropped)

    def hold(self, request_id: str, started: int, row: SpendLogRecord, size: int) -> Release:
        """Holds the callback's record for the request that started at ``started`` UTC microseconds."""
        expired: Final = self._expire()
        key: Final = (strip_cache_hit_suffix(request_id), *_owner(row), started)
        early: Final = self.waiting.get(key)
        if early is not None:
            committed: Final = self._pop_committed(early)
            return _joined(expired, _ready((merged_row(row, committed.row),)))
        if size > self.max_bytes:
            return _joined(expired, Release(dropped=1))
        if self._take_held(key) is not None:
            return _joined(expired, Release(dropped=2))
        evicted: Final = self._evict(size)
        self.held[key] = _Held(row=row, size=size, at=self.clock())
        self.held_bytes += size
        return _joined(expired, Release(dropped=evicted))

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
        if delivery_id is None or not self._first_report(delivery_id):
            return Release()
        key: Final = None if row.get("unreportable") is True else _committed_key(row)
        if key is None:
            return Release(dropped=1)
        held: Final = self._take_held(key)
        if held is not None:
            return _ready((merged_row(held.row, row),))
        twin: Final = self.waiting.get(key)
        if twin is not None:
            _ = self._pop_committed(twin)
            return Release(dropped=2)
        self.committed[delivery_id] = _Committed(key=key, row=row, at=now)
        self.waiting[key] = delivery_id
        return Release()

    def commit(self, rows: Sequence[Mapping[str, object]]) -> Release:
        expired: Final = self._expire()
        now: Final = self.clock()
        accepted: Final = _joined(*(self._accept(row, now) for row in rows))
        excess: Final = tuple(self.committed)[: max(len(self.committed) - self.max_rows, 0)]
        for delivery in excess:
            self._pop_committed(delivery)
        return _joined(expired, accepted, Release(dropped=len(excess)))

    def drain(self) -> Release:
        dropped: Final = len(self.held) + len(self.committed)
        self.held.clear()
        self.held_bytes = 0
        self.committed.clear()
        self.waiting.clear()
        return Release(dropped=dropped)
