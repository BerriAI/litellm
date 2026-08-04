"""In-memory staging and durable flush for auto-router session rollups.

Follows ``AdaptiveRouterUpdateQueue``: the logging path only stages into memory
and never touches the database, and a background task drains what it staged into
Postgres with atomic increment upserts, so two pods writing the same session
compose rather than overwrite.

Classifying a turn depends on the session's prior state, so the fold happens in
the flusher rather than at arrival. That is the one place the stored state can be
read, folded onto and written back as a single unit of work; a read that fails
there has no half-finished write to corrupt, and the turns simply stage again.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    SessionState,
    StateLookup,
    StateUnavailable,
    TurnDelta,
    TurnFacts,
    counters_of,
    fold_session,
    state_column,
    state_from_row,
)
from litellm.repositories.table_repositories import AutoRouterSessionRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

SessionKey = tuple[str, str]

DEFAULT_MAX_STAGED_TURNS = 50_000
MAX_CACHED_SESSIONS = 10_000


@dataclass(frozen=True, slots=True)
class _Pending:
    router_kind: str
    baseline_model: str | None
    turns: tuple[TurnFacts, ...]


@lru_cache(maxsize=1)
def _warn_staging_full(cap: int) -> None:
    verbose_proxy_logger.warning(
        "auto_router_sessions: %d turns staged for the next flush; further turns are not being recorded "
        "until it drains. Benchmarks will undercount until then",
        cap,
    )


def _epoch_to_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


class AutoRouterSessionQueue:
    """Stages auto-routed turns in memory and folds them into the rollup on flush."""

    def __init__(self, max_staged_turns: int = DEFAULT_MAX_STAGED_TURNS) -> None:
        self._pending: dict[SessionKey, _Pending] = {}  # mutable-ok: drained and replaced wholesale on flush
        self._state: OrderedDict[SessionKey, SessionState] = OrderedDict()  # mutable-ok: bounded LRU cache
        self._staged_turns = 0
        self._lock = asyncio.Lock()
        self._max_staged_turns = max_staged_turns

    async def record_turn(
        self,
        key: SessionKey,
        router_kind: str,
        baseline_model: str | None,
        turn: TurnFacts,
    ) -> None:
        """Stage one turn against its session. Does no I/O; the fold happens on flush.

        ``session_id`` is caller-controlled, so the staging is capped on turns
        held rather than on sessions seen, which is the quantity that actually
        bounds the memory. Past the cap a turn is dropped and logged, because
        benchmark rows are not worth an out-of-memory kill.
        """
        async with self._lock:
            if self._staged_turns >= self._max_staged_turns:
                _warn_staging_full(self._max_staged_turns)
                return
            self._stage(key, router_kind, baseline_model, (turn,))

    def _stage(
        self, key: SessionKey, router_kind: str, baseline_model: str | None, turns: tuple[TurnFacts, ...]
    ) -> None:
        """Add turns to whatever this session already has staged; callers hold the lock.

        Arriving turns and a replayed batch stage identically, because the fold
        sorts by start time and so does not care which of the two came first.
        """
        current = self._pending.get(key)
        self._pending[key] = _Pending(
            router_kind=router_kind,
            baseline_model=baseline_model or (current.baseline_model if current is not None else None),
            turns=(current.turns if current is not None else ()) + turns,
        )
        self._staged_turns += len(turns)

    async def flush(self, prisma_client: "PrismaClient") -> int:
        """Fold and write every staged session. Returns rows written.

        A session that could not be read or written is staged again rather than
        dropped. Reading, folding and writing are one unit here, so a failure
        means nothing landed and replaying it cannot double-count; draining first
        and swallowing the error would lose that interval's turns, tokens and
        spend permanently on any transient database fault.
        """
        async with self._lock:
            batch = self._pending
            self._pending = {}  # mutable-ok: fresh staging for the next interval
            self._staged_turns = 0

        failed = {  # mutable-ok: built once from the sessions that did not land
            key: batch[key] for key in sorted(batch.keys()) if not await self._commit(key, batch[key], prisma_client)
        }
        if failed:
            verbose_proxy_logger.warning(
                "auto_router_sessions: %d of %d sessions could not be folded; re-staging them for the next flush",
                len(failed),
                len(batch),
            )
            async with self._lock:
                for key, pending in failed.items():
                    self._stage(key, pending.router_kind, pending.baseline_model, pending.turns)
        return len(batch) - len(failed)

    async def _commit(self, key: SessionKey, pending: _Pending, prisma_client: "PrismaClient") -> bool:
        """Read the session's state, fold its staged turns onto it, write both back.

        The cache advances only once the write has landed, so a replay folds from
        the state the failed attempt did rather than one that was never persisted.
        An evicted session is not lost either; its next flush reloads the row it
        was already written to, which costs one read and folds identically.
        """
        state = await self._session_state(key, prisma_client)
        if isinstance(state, StateUnavailable):
            return False
        delta = fold_session(state, pending.turns)
        if not await self._write(key, pending, delta, prisma_client):
            return False
        self._state[key] = delta.state
        self._state.move_to_end(key)
        while len(self._state) > MAX_CACHED_SESSIONS:
            self._state.popitem(last=False)
        return True

    async def _session_state(self, key: SessionKey, prisma_client: "PrismaClient") -> StateLookup:
        """The session's state: from memory if this pod has folded it before, else its row.

        Only the flusher touches this cache, so it needs no lock. A session this
        pod has not seen costs one read, which is what keeps it correct across a
        restart or a move between pods.
        """
        cached = self._state.get(key)
        if cached is not None:
            self._state.move_to_end(key)
            return cached
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
        except Exception as e:  # noqa: BLE001  # a read fault re-stages the turns rather than failing the flush
            verbose_proxy_logger.warning("auto_router_sessions: could not load session state for %s (%s)", key, e)
            return StateUnavailable()
        if row is None:
            return EMPTY_SESSION_STATE
        return state_from_row(row.last_model, row.last_turn_at, row.model_state)

    async def _write(self, key: SessionKey, pending: _Pending, delta: TurnDelta, prisma_client: "PrismaClient") -> bool:
        session_id, model_group = key
        counters = counters_of(delta)
        shared = {  # mutable-ok: prisma's write API takes dict payloads
            "last_turn_at": _epoch_to_datetime(max(turn.started_at for turn in pending.turns)),
            "last_model": delta.state.last_model,
            "model_state": state_column(delta.state),
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
                        "first_turn_at": _epoch_to_datetime(min(turn.started_at for turn in pending.turns)),
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
