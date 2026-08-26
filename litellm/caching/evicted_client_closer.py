"""
Deferred close of HTTP/SDK clients that the LLM client cache has evicted.

Eviction only drops the cache's reference to a client. Every OpenAI/Azure SDK
client is a reference cycle (each resource namespace holds the client back), so
an evicted client and its pooled TCP connections survive until a generational
collection runs, which under load is thousands of requests later.

Closing at eviction time is not an option: a request that was handed the client
just before it was evicted is still using it, and closing it underneath that
request raises ``RuntimeError: Cannot send a request, as the client has been
closed.``

So an evicted client is closed once two conditions hold. A grace window must
have passed since its eviction, which covers a request that holds the client
but is momentarily not on the wire, and the client must report no connection in
flight. The second condition is what keeps the first honest: a request may run
for ``litellm.request_timeout`` seconds, 6000 by default, and a streaming
response is bounded only by how long the upstream keeps sending, so no deadline
on its own can promise that a request has finished.

Only clients litellm itself created are closed; a client the caller supplied is
left alone because litellm does not own its lifecycle.

A client that closes synchronously is closed from wherever the cache is next
used. One whose close is a coroutine needs the event loop it was evicted on, so
it waits for a call from that loop rather than having work scheduled onto a loop
it does not belong to. Queued clients are therefore bucketed by what it takes to
close them, and each bucket is ordered by deadline, so a reap walks the entries
that are due rather than the whole queue.

The queue holds its clients weakly, so waiting out a grace window never keeps
alive anything the collector would have reclaimed first.
"""

import asyncio
import contextlib
import inspect
import threading
import time
import weakref
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, replace
from typing import Final

from litellm.constants import (
    EVICTED_LLM_CLIENT_CLOSE_GRACE_SECONDS,
    EVICTED_LLM_CLIENT_CLOSE_MAX_PENDING,
)

_CLOSABLE_ANYWHERE: Final = "closable-anywhere"
_CLOSABLE_ON_ANY_LOOP: Final = "closable-on-any-loop"

_BucketKey = str | int


@dataclass(frozen=True, slots=True)
class _PendingClose:
    """A queued close.

    The client is held weakly, so queueing one never keeps alive anything the
    collector would otherwise have reclaimed first.

    ``needs_loop`` is set for a client whose close is a coroutine; those can only
    be closed from the event loop they were evicted on, recorded in ``loop_id``.
    A client that closes synchronously carries neither constraint.
    """

    client_ref: "weakref.ref[object]"
    loop_id: int | None
    needs_loop: bool
    close_after: float


def _bucket_key(pending: _PendingClose) -> _BucketKey:
    """Which reaps can close this entry: any at all, any running a loop, or one loop's."""
    if not pending.needs_loop:
        return _CLOSABLE_ANYWHERE
    if pending.loop_id is None:
        return _CLOSABLE_ON_ANY_LOOP
    return pending.loop_id


def _running_loop_id() -> int | None:
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return None


def _close_function(client: object) -> Callable[[], object] | None:
    close_fn: Final[Callable[[], object] | None] = getattr(client, "aclose", None) or getattr(client, "close", None)
    return close_fn


def _transport_of(client: object) -> object:
    """The httpx transport behind an SDK wrapper, a litellm handler, or a bare client."""
    for holder in (getattr(client, "_client", None), getattr(client, "client", None), client):
        transport: object = getattr(holder, "_transport", None)
        if transport is not None:
            return transport
    return None


def _connection_is_idle(connection: object) -> bool:
    """A pooled connection is idle unless it is servicing a request."""
    is_idle: Final[object] = getattr(connection, "is_idle", None)
    return bool(is_idle()) if callable(is_idle) else True


def _pool_has_busy_connection(transport: object) -> bool | None:
    """Whether the httpcore pool behind the transport is servicing a request.

    ``None`` when there is no such pool, so the caller can ask the other backend.
    """
    pooled: Final[object] = getattr(getattr(transport, "_pool", None), "connections", None)
    if not isinstance(pooled, (list, tuple)):
        return None
    return any(
        not _connection_is_idle(connection)  # pyright: ignore[reportUnknownArgumentType]  # untyped pool list
        for connection in pooled  # pyright: ignore[reportUnknownVariableType]  # untyped pool list
    )


def _has_connection_in_flight(client: object) -> bool:
    """Whether the client is servicing a request right now.

    Both connection backends litellm uses already account for the connections
    they have handed out, so this reads the client's own lease accounting rather
    than inferring it from elapsed time: httpcore reports a non-idle connection
    for the whole of a response including a stream, and aiohttp holds the
    connection in ``_acquired`` over the same span.

    A client that cannot answer is reported as idle, which leaves the grace
    window as the only guard, exactly as it was before this check existed.
    """
    try:
        transport: Final = _transport_of(client)
        pooled_busy: Final = _pool_has_busy_connection(transport)
        if pooled_busy is not None:
            return pooled_busy
        session: Final[object] = getattr(transport, "client", None)
        return bool(getattr(getattr(session, "connector", None), "_acquired", None))
    except Exception:  # noqa: BLE001 - a client that cannot report its state is treated as idle
        return False


async def _close_quietly(closing: Awaitable[object]) -> None:
    with contextlib.suppress(Exception):
        await closing


class EvictedClientCloser:
    """Closes evicted, litellm-owned clients once they are idle and out of grace."""

    def __init__(
        self,
        grace_seconds: float = EVICTED_LLM_CLIENT_CLOSE_GRACE_SECONDS,
        max_pending: int = EVICTED_LLM_CLIENT_CLOSE_MAX_PENDING,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grace_seconds = grace_seconds
        self._max_pending = max_pending
        self._clock = clock
        self._owned: weakref.WeakSet[object] = weakref.WeakSet()
        self._buckets: dict[_BucketKey, deque[_PendingClose]] = {}  # mutable-ok: deadline-ordered queues
        self._pending_count = 0
        self._queue_lock = threading.Lock()  # the cache is reachable from every worker thread's loop
        self._close_tasks: set[asyncio.Task[None]] = set()  # mutable-ok: strong refs to running closes

    def mark_owned(self, client: object) -> None:
        """Record that litellm created this client, so it may be closed on eviction."""
        try:
            self._owned.add(client)
        except TypeError:
            pass  # values that cannot be weak-referenced are never litellm clients

    def _is_owned(self, client: object) -> bool:
        try:
            return client in self._owned
        except TypeError:
            return False  # unhashable values are never litellm clients

    def schedule(self, client: object) -> None:
        """Queue an evicted client for closing once it is idle and out of grace.

        Past ``max_pending`` the client is left to the collector instead, so a
        workload that churns the cache cannot grow this queue without bound.
        Every queued entry comes due within one grace window, so the capacity it
        occupies is returned within that window rather than held.
        """
        if client is None or not self._is_owned(client):
            return
        close_fn: Final = _close_function(client)
        if close_fn is None:
            return
        if self._pending_count >= self._max_pending:
            return
        self._enqueue(
            _PendingClose(
                client_ref=weakref.ref(client),
                loop_id=_running_loop_id(),
                needs_loop=inspect.iscoroutinefunction(close_fn),
                close_after=self._clock() + self._grace_seconds,
            )
        )

    def reap(self) -> None:
        """Close every queued client that is due, idle, and closable from here.

        Called from the cache's read path, so the empty-queue exit comes first and
        the work done past it is proportional to what is due, not to the queue.
        """
        if not self._pending_count:
            return
        now: Final = self._clock()
        for pending in self._take_due(_running_loop_id(), now):
            client = pending.client_ref()
            if client is None:
                continue
            if _has_connection_in_flight(client):
                self._enqueue(replace(pending, close_after=now + self._grace_seconds))
                continue
            self._close(client)

    @property
    def pending_count(self) -> int:
        return self._pending_count

    def _enqueue(self, pending: _PendingClose) -> None:
        """Append to the entry's bucket, dropping any dead entries it queues behind.

        Deadlines only ever move forward, so appending keeps each bucket ordered
        by deadline, and entries whose client the collector already took sit at
        the front rather than having to be searched for.
        """
        with self._queue_lock:
            bucket: Final = self._buckets.setdefault(_bucket_key(pending), deque())  # mutable-ok: FIFO by design
            while bucket and bucket[0].client_ref() is None:
                bucket.popleft()
                self._pending_count -= 1
            bucket.append(pending)
            self._pending_count += 1

    def _take_due(self, loop_id: int | None, now: float) -> tuple[_PendingClose, ...]:
        buckets = (_CLOSABLE_ANYWHERE,) if loop_id is None else (_CLOSABLE_ANYWHERE, _CLOSABLE_ON_ANY_LOOP, loop_id)
        with self._queue_lock:
            return tuple(pending for key in buckets for pending in self._drain_locked(key, now))

    def _drain_locked(self, key: _BucketKey, now: float) -> Iterator[_PendingClose]:
        bucket: Final = self._buckets.get(key)
        if bucket is None:
            return
        while bucket and bucket[0].close_after <= now:
            self._pending_count -= 1
            yield bucket.popleft()
        if not bucket:
            del self._buckets[key]

    def _close(self, client: object) -> None:
        close_fn: Final = _close_function(client)
        if close_fn is None:
            return
        try:
            closing: Final = close_fn()
        except Exception:  # noqa: BLE001 - a discarded client's close must never surface to callers
            return
        if not inspect.isawaitable(closing):
            return
        task: Final = asyncio.get_running_loop().create_task(_close_quietly(closing))
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)


default_evicted_client_closer: Final = EvictedClientCloser()
