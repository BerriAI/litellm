"""In-memory aggregation and durable flush for auto-router session rollups.

Follows ``AdaptiveRouterUpdateQueue``: the logging path only folds into memory,
and a background task drains the aggregate into Postgres with atomic increment
upserts, so two pods writing the same session compose rather than overwrite.

The one departure is that this queue also caches the session state the fold reads
from. A pod that has never seen a session loads its row once and classifies from
memory thereafter, which is what keeps a session correct across a restart or a
move between pods without paying a read per turn.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    SessionState,
    TurnDelta,
    TurnFacts,
    counters_of,
    fold_turn,
    merge_deltas,
    state_column,
    state_from_row,
)
from litellm.repositories.table_repositories import AutoRouterSessionRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

SessionKey = tuple[str, str]

DEFAULT_MAX_TRACKED_SESSIONS = 10_000


@dataclass(frozen=True, slots=True)
class _Pending:
    router_kind: str
    baseline_model: str | None
    first_turn_at: float
    last_turn_at: float
    delta: TurnDelta


@lru_cache(maxsize=1)
def _warn_pending_full(cap: int) -> None:
    verbose_proxy_logger.warning(
        "auto_router_sessions: %d sessions staged for the next flush; new sessions are not being recorded "
        "until it drains. Benchmarks will undercount until then",
        cap,
    )


def _merge_pending(earlier: _Pending, later: _Pending) -> _Pending:
    """Fold two staged batches for one session, oldest first."""
    return _Pending(
        router_kind=later.router_kind,
        baseline_model=later.baseline_model or earlier.baseline_model,
        first_turn_at=min(earlier.first_turn_at, later.first_turn_at),
        last_turn_at=max(earlier.last_turn_at, later.last_turn_at),
        delta=merge_deltas(earlier.delta, later.delta),
    )


def _epoch_to_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


class AutoRouterSessionQueue:
    """Folds auto-routed turns in memory and flushes them to the session rollup."""

    def __init__(self, max_tracked_sessions: int = DEFAULT_MAX_TRACKED_SESSIONS) -> None:
        self._pending: dict[SessionKey, _Pending] = {}  # mutable-ok: drained and replaced wholesale on flush
        self._state: OrderedDict[SessionKey, SessionState] = OrderedDict()  # mutable-ok: bounded LRU cache
        self._lock = asyncio.Lock()
        self._max_tracked_sessions = max_tracked_sessions

    async def record_turn(
        self,
        key: SessionKey,
        router_kind: str,
        baseline_model: str | None,
        turn: TurnFacts,
        prisma_client: "PrismaClient",
    ) -> None:
        """Classify one turn against its session and stage the increments.

        The session id is caller-controlled, so the staged aggregate is capped:
        past the cap a session that is already staged keeps accumulating, but a
        new one is dropped rather than admitted. Without that bound a caller
        sending a fresh id per request grows the aggregate without limit between
        flushes, and benchmark rows are not worth an out-of-memory kill.
        """
        loaded = await self._session_state(key, prisma_client)
        async with self._lock:
            current = self._pending.get(key)
            if current is None and len(self._pending) >= self._max_tracked_sessions:
                _warn_pending_full(self._max_tracked_sessions)
                return
            cached = self._state.get(key)
            delta = fold_turn(cached if cached is not None else loaded, turn)
            self._remember(key, delta.state)
            self._pending[key] = (
                _Pending(
                    router_kind=router_kind,
                    baseline_model=baseline_model,
                    first_turn_at=turn.started_at,
                    last_turn_at=turn.started_at,
                    delta=delta,
                )
                if current is None
                else replace(
                    current,
                    baseline_model=baseline_model or current.baseline_model,
                    first_turn_at=min(current.first_turn_at, turn.started_at),
                    last_turn_at=max(current.last_turn_at, turn.started_at),
                    delta=merge_deltas(current.delta, delta),
                )
            )

    async def _session_state(self, key: SessionKey, prisma_client: "PrismaClient") -> SessionState:
        """The session's state, from memory when this pod has seen it before.

        Loading outside the lock keeps a slow read from stalling every other
        session's fold; a concurrent loader for the same key at worst repeats the
        read, since both resolve to the same stored row.
        """
        async with self._lock:
            cached = self._state.get(key)
            if cached is not None:
                self._state.move_to_end(key)
                return cached
        return await self._load_state(key, prisma_client)

    async def _load_state(self, key: SessionKey, prisma_client: "PrismaClient") -> SessionState:
        session_id, model_group = key
        try:
            row = await AutoRouterSessionRepository(prisma_client).table.find_unique(
                where={  # mutable-ok: prisma's write API takes dict payloads
                    "session_id_model_group": {  # mutable-ok: a JSON object is a dict by definition
                        "session_id": session_id,
                        "model_group": model_group,
                    }
                }
            )
        except Exception as e:  # noqa: BLE001  # a read fault must not fail the spend write
            verbose_proxy_logger.warning(
                "auto_router_sessions: could not load session state for %s (%s); treating as a new session", key, e
            )
            return EMPTY_SESSION_STATE
        if row is None:
            return EMPTY_SESSION_STATE
        return state_from_row(row.last_model, row.last_turn_at, row.model_state)

    def _remember(self, key: SessionKey, state: SessionState) -> None:
        """Cache the session's next state, evicting the least recently used.

        An evicted session is not lost; its next turn reloads the row it was
        already flushed to, which costs one read and classifies identically.
        """
        self._state[key] = state
        self._state.move_to_end(key)
        while len(self._state) > self._max_tracked_sessions:
            self._state.popitem(last=False)

    async def flush(self, prisma_client: "PrismaClient") -> int:
        """Drain the aggregate into the session rollup. Returns rows written.

        A session whose write fails is staged again rather than dropped. Draining
        first and swallowing the error would lose that interval's turns, tokens
        and spend permanently on any transient database fault, and because the
        upsert is atomic a failure means nothing landed, so replaying it cannot
        double-count.
        """
        async with self._lock:
            batch = self._pending
            self._pending = {}  # mutable-ok: fresh aggregate for the next interval

        failed = {  # mutable-ok: built once from the writes that did not land
            key: batch[key] for key in sorted(batch.keys()) if not await self._write(key, batch[key], prisma_client)
        }
        if failed:
            verbose_proxy_logger.warning(
                "auto_router_sessions: %d of %d session writes failed; re-staging them for the next flush",
                len(failed),
                len(batch),
            )
            async with self._lock:
                for key, pending in failed.items():
                    current = self._pending.get(key)
                    # The retried batch is older than anything staged since, so it
                    # merges underneath it and the state of the newer one wins.
                    self._pending[key] = pending if current is None else _merge_pending(pending, current)
        return len(batch) - len(failed)

    async def _write(self, key: SessionKey, pending: _Pending, prisma_client: "PrismaClient") -> bool:
        session_id, model_group = key
        counters = counters_of(pending.delta)
        shared = {  # mutable-ok: prisma's write API takes dict payloads
            "last_turn_at": _epoch_to_datetime(pending.last_turn_at),
            "last_model": pending.delta.state.last_model,
            "model_state": state_column(pending.delta.state),
            "baseline_model": pending.baseline_model,
        }
        try:
            await AutoRouterSessionRepository(
                prisma_client
            ).table.upsert(
                where={  # mutable-ok: prisma's write API takes dict payloads
                    "session_id_model_group": {  # mutable-ok: a JSON object is a dict by definition
                        "session_id": session_id,
                        "model_group": model_group,
                    }
                },
                data={  # mutable-ok: prisma's write API takes dict payloads
                    "create": {  # mutable-ok: prisma's write API takes dict payloads
                        "session_id": session_id,
                        "model_group": model_group,
                        "router_kind": pending.router_kind,
                        "first_turn_at": _epoch_to_datetime(pending.first_turn_at),
                        **shared,
                        **counters,
                    },
                    "update": {  # mutable-ok: prisma's write API takes dict payloads
                        **{  # mutable-ok: a JSON object is a dict by definition
                            field: {"increment": value}  # mutable-ok: a JSON object is a dict by definition
                            for field, value in counters.items()  # mutable-ok: spread into the prisma payload immediately below
                        },  # mutable-ok: spread into the prisma payload immediately below
                        **shared,
                    },
                },
            )
        except Exception as e:  # noqa: BLE001  # one session's write must not drop the rest of the batch
            verbose_proxy_logger.exception("auto_router_sessions: failed to flush session %s (%s)", key, e)
            return False
        return True
