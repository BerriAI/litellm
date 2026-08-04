"""In-memory staging and durable flush for auto-router session rollups.

Follows ``AdaptiveRouterUpdateQueue``: the logging path only stages into memory
and never touches the database, and a background task drains what it staged into
Postgres with atomic increment upserts, so two pods writing the same session
compose rather than overwrite.

Classifying a turn depends on the session's prior state, so the fold happens in
the flusher rather than at arrival. That is the one place the stored state can be
read, folded onto and written back as a single unit of work; a read that fails
there has no half-finished write to corrupt, and the turns simply stage again.

That unit is a chunk of sessions rather than one session, so an interval costs a
handful of round trips instead of two per session: one ``find_many`` for the
states this pod has not cached, then one transaction carrying every upsert.
"""

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    SessionState,
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
SESSIONS_PER_STATEMENT = 1_000


@dataclass(frozen=True, slots=True)
class _Pending:
    router_kind: str
    baseline_model: str | None
    turns: tuple[TurnFacts, ...]


_Chunk = Mapping[SessionKey, _Pending]
_Folded = tuple[SessionKey, _Pending, TurnDelta]
_ChunkStates = Mapping[SessionKey, SessionState] | StateUnavailable


@lru_cache(maxsize=1)
def _warn_staging_full(cap: int) -> None:
    verbose_proxy_logger.warning(
        "auto_router_sessions: %d turns staged for the next flush; further turns are not being recorded "
        "until it drains. Benchmarks will undercount until then",
        cap,
    )


def _epoch_to_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _chunked(batch: _Chunk, size: int) -> tuple[_Chunk, ...]:
    """``batch`` split into runs of at most ``size``, each in key order.

    Sorting once here is what gives every pod the same lock ordering, and the
    size cap is what stops one statement from carrying the whole staging area.
    """
    keys = sorted(batch)
    return tuple(
        MappingProxyType({key: batch[key] for key in keys[at : at + size]}) for at in range(0, len(keys), size)
    )


def _key_fields(key: SessionKey) -> Mapping[str, str]:
    """One composite primary key as the two columns that make it up."""
    session_id, model_group = key
    return {"session_id": session_id, "model_group": model_group}  # mutable-ok: prisma's query API takes dict payloads


def _unique_where(key: SessionKey) -> Mapping[str, Mapping[str, str]]:
    """The composite primary key, under the name prisma gives the ``@@id`` selector."""
    return {"session_id_model_group": _key_fields(key)}  # mutable-ok: prisma's query API takes dict payloads


def _increment(value: float) -> Mapping[str, float]:
    """One counter's atomic add, the way prisma spells it in an update payload."""
    return {"increment": value}  # mutable-ok: prisma's write API takes dict payloads


def _upsert_data(key: SessionKey, pending: _Pending, delta: TurnDelta) -> Mapping[str, Mapping[str, object]]:
    """One session's write: create its row, or add this interval onto the row already there."""
    counters = counters_of(delta)
    increments = MappingProxyType({name: _increment(value) for name, value in counters.items()})
    shared = {  # mutable-ok: prisma's write API takes dict payloads
        "last_turn_at": _epoch_to_datetime(max(turn.started_at for turn in pending.turns)),
        "last_model": delta.state.last_model,
        "model_state": state_column(delta.state),
        "baseline_model": pending.baseline_model,
    }
    return {  # mutable-ok: prisma's write API takes dict payloads
        "create": {  # mutable-ok: prisma's write API takes dict payloads
            **_key_fields(key),
            "router_kind": pending.router_kind,
            "first_turn_at": _epoch_to_datetime(min(turn.started_at for turn in pending.turns)),
            **shared,
            **counters,
        },
        "update": {**increments, **shared},  # mutable-ok: prisma's write API takes dict payloads
    }


class AutoRouterSessionQueue:
    """Stages auto-routed turns in memory and folds them into the rollup on flush."""

    def __init__(self, max_staged_turns: int = DEFAULT_MAX_STAGED_TURNS) -> None:
        self._pending: dict[SessionKey, _Pending] = {}  # mutable-ok: drained and replaced wholesale on flush
        self._state: OrderedDict[SessionKey, SessionState] = OrderedDict()  # mutable-ok: bounded LRU cache
        self._staged_turns = 0
        self._lock = asyncio.Lock()
        self._max_staged_turns = max_staged_turns

    async def record_turn(self, key: SessionKey, router_kind: str, baseline_model: str | None, turn: TurnFacts) -> None:
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

        A chunk that could not be read or written is staged again rather than
        dropped. Reading, folding and writing are one unit per chunk, so a
        failure means nothing in it landed and replaying it cannot double-count;
        draining first and swallowing the error would lose that interval's turns,
        tokens and spend permanently on any transient database fault.
        """
        async with self._lock:
            batch = self._pending
            self._pending = {}  # mutable-ok: fresh staging for the next interval
            self._staged_turns = 0

        chunks = _chunked(batch, SESSIONS_PER_STATEMENT)
        failed = tuple([key for chunk in chunks if not await self._commit(chunk, prisma_client) for key in chunk])
        if failed:
            verbose_proxy_logger.warning(
                "auto_router_sessions: %d of %d sessions could not be folded; re-staging them for the next flush",
                len(failed),
                len(batch),
            )
            async with self._lock:
                for key in failed:
                    self._stage(key, batch[key].router_kind, batch[key].baseline_model, batch[key].turns)
        return len(batch) - len(failed)

    async def _commit(self, chunk: _Chunk, prisma_client: "PrismaClient") -> bool:
        """Read a chunk's states, fold its staged turns onto them, write them all back.

        The chunk lands whole or not at all: one read covers it and one
        transaction writes it, so a failure at either end leaves nothing written
        for any session in it and the caller re-stages them all. The cache
        advances only once the write has landed, so a replay folds from the state
        the failed attempt did rather than one that was never persisted. An
        evicted session is not lost either; its next flush reloads the row it was
        already written to and folds identically.
        """
        states = await self._session_states(tuple(chunk), prisma_client)
        if isinstance(states, StateUnavailable):
            return False
        folded = tuple((key, pending, fold_session(states[key], pending.turns)) for key, pending in chunk.items())
        if not await self._write(folded, prisma_client):
            return False
        for key, _, delta in folded:
            self._state[key] = delta.state
            self._state.move_to_end(key)
        while len(self._state) > MAX_CACHED_SESSIONS:
            self._state.popitem(last=False)
        return True

    async def _session_states(self, keys: tuple[SessionKey, ...], prisma_client: "PrismaClient") -> _ChunkStates:
        """Each key's state: from memory where this pod has folded it before, else from one read.

        The filter is an OR over whole composite keys rather than ``session_id IN
        (...) AND model_group IN (...)``, because the latter is a cross product:
        it would drag back rows for pairs nobody asked about, growing with the
        number of auto-routers in the flush and leaving Python to discard them.

        Only the flusher touches this cache, so it needs no lock, and a chunk it
        has entirely cached costs no read at all. A fault leaves the whole chunk
        unavailable, which re-stages it; folding onto an empty state instead
        would replace a history that really happened with one derived from a
        single interval.
        """
        missing = tuple(key for key in keys if key not in self._state)
        if not missing:
            return MappingProxyType({key: self._state[key] for key in keys})
        try:
            rows = await AutoRouterSessionRepository(prisma_client).table.find_many(
                where={"OR": [_key_fields(key) for key in missing]}  # mutable-ok: prisma's query API takes dicts
            )
        except Exception as e:  # noqa: BLE001  # a read fault re-stages the chunk rather than failing the flush
            verbose_proxy_logger.warning("auto_router_sessions: could not load %d session states (%s)", len(missing), e)
            return StateUnavailable()
        stored = {  # mutable-ok: built once from the rows this read returned
            (row.session_id, row.model_group): state_from_row(row.last_model, row.last_turn_at, row.model_state)
            for row in rows
        }
        return MappingProxyType(
            {key: self._state[key] if key in self._state else stored.get(key, EMPTY_SESSION_STATE) for key in keys}
        )

    async def _write(self, folded: tuple[_Folded, ...], prisma_client: "PrismaClient") -> bool:
        """Write a chunk's rows as one transaction of atomic increment upserts.

        ``batch_()`` issues its statements sequentially inside the transaction, so
        iteration order is lock acquisition order; the chunk arrives sorted, which
        is what keeps two pods flushing the same sessions from deadlocking.
        """
        try:
            async with prisma_client.db.tx(timeout=timedelta(seconds=60)) as transaction:
                async with transaction.batch_() as batcher:
                    table = batcher.litellm_autoroutersession
                    for key, pending, delta in folded:
                        table.upsert(where=_unique_where(key), data=_upsert_data(key, pending, delta))
        except Exception as e:  # noqa: BLE001  # one chunk's write must not drop the rest of the flush
            verbose_proxy_logger.exception("auto_router_sessions: failed to flush %d sessions (%s)", len(folded), e)
            return False
        return True
