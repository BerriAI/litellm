"""Cross-replica ``RefreshCoordinator``: one refresh per ``(user, server)`` across all workers.

Plugs into the foundation's ``RefreshingTokenStore`` via the ``RefreshCoordinator`` seam. A ``SET NX
PX`` lock elects one worker to run the refresh while the rest wait for it and re-read the token it
persisted - so a rotating refresh_token is used once across the fleet, not once per worker. The holder
renews the ``PX`` lease while refresh runs (up to a refresh budget, so a hung endpoint can't hold the
lock forever), and a loser waits longer than that budget - so a loser only re-reads once the holder has
finished or its bounded lease has lapsed, never mid-refresh, and the surrounding store re-checks expiry
on the next fetch, so a crash self-heals rather than serving stale forever. Reading needs no lock, so
losers don't serialize behind each other. The lock is injected (a thin Redis wrapper in production, a
fake in tests).

The lock is a single-flight optimization, not a correctness mutex, so it fails open: when the lock
backend is unreachable, ``acquire`` reports ``ERROR`` (distinct from ``HELD``) and this coordinator
refreshes anyway rather than wait on a holder that may not exist and then serve a still-expired token.
That degrades a Redis outage to the no-coordinator behavior (each worker may refresh), never a stale
bearer the upstream would 401.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import Final, Generic, Protocol, TypeVar

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    InProcessRefreshCoordinator,
    RefreshCoordinator,
)


class LockAcquisition(Enum):
    """Outcome of a best-effort ``acquire``. ``ERROR`` is kept distinct from ``HELD`` so a caller can
    tell "someone else is refreshing" (wait and re-read) from "the lock backend is down" (no election
    happened, so refresh anyway) instead of conflating both into a single ``False``."""

    ACQUIRED = "acquired"  # won the election; this worker refreshes
    HELD = "held"  # another worker holds it; wait then re-read
    ERROR = "error"  # lock backend unreachable; holder unknown, so refresh anyway


class DistributedLock(Protocol):
    """A best-effort cross-replica lock. ``acquire`` is ``SET key token NX PX ttl`` reported as a
    ``LockAcquisition`` (won / held by another / backend error); ``release`` deletes the key only if
    it still holds this caller's ``token`` (so it cannot delete a lock another worker re-acquired
    after PX-expiry); ``extend`` refreshes the ``PX`` lease only for the owner; ``is_held`` is
    ``EXISTS`` (so a waiter can poll without taking the lock)."""

    async def acquire(self, key: str, token: str, ttl_seconds: float) -> LockAcquisition: ...

    async def extend(self, key: str, token: str, ttl_seconds: float) -> bool: ...

    async def release(self, key: str, token: str) -> None: ...

    async def is_held(self, key: str) -> bool: ...

    async def held_token(self, key: str) -> str | None: ...

    async def publish_result(self, key: str, token: str, outcome: str, ttl_seconds: float) -> None: ...

    async def read_result(self, key: str, token: str) -> str | None: ...


OutcomeT = TypeVar("OutcomeT")


def _no_encoded_error(_outcome: object) -> str | None:
    return None


def _no_decoded_error(_outcome: str) -> None:
    return None


@dataclass(frozen=True, slots=True)
class RedisRefreshCoordinator(Generic[OutcomeT]):
    lock: DistributedLock
    _: KW_ONLY
    key_prefix: str = "mcp:refresh_lock:"
    lock_ttl_seconds: float = 10.0
    # The holder renews its lease while a slow token endpoint runs, but only up to this budget; past it
    # it stops renewing and the lock lapses, so a hung refresh degrades to "maybe an extra refresh"
    # rather than holding every loser behind it indefinitely.
    refresh_budget_seconds: float = 20.0
    # How long a loser waits for the holder before giving up and re-reading. It MUST outlast the
    # holder's max lock-hold (refresh_budget_seconds + one lock_ttl_seconds tail); otherwise a loser
    # bails while the holder is still legitimately refreshing, re-reads the still-expired token, and
    # challenges the user mid-refresh.
    wait_timeout_seconds: float = 35.0
    poll_interval_seconds: float = 0.05
    fallback: RefreshCoordinator[OutcomeT] = field(  # pyright: ignore[reportUnknownVariableType]  # dataclass default factories cannot capture the class type parameter
        default_factory=InProcessRefreshCoordinator
    )
    encode_error: Callable[[OutcomeT], str | None] = _no_encoded_error
    decode_error: Callable[[str], OutcomeT | None] = _no_decoded_error
    result_ttl_seconds: float = 60.0
    background_tasks: set[asyncio.Task[None]] = field(  # mutable-ok: live cleanup tasks
        default_factory=set, compare=False, repr=False
    )
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic
    new_token: Callable[[], str] = lambda: uuid.uuid4().hex

    def _key(self, user_id: str, server_id: str) -> str:
        return f"{self.key_prefix}{user_id}:{server_id}"

    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[OutcomeT]],
        reread: Callable[[], Awaitable[OutcomeT]],
    ) -> OutcomeT:
        key: Final = self._key(user_id, server_id)
        token: Final = self.new_token()
        match await self.lock.acquire(key, token, self.lock_ttl_seconds):
            case LockAcquisition.ACQUIRED:
                return await self._refresh_with_lease_renewal(key, token, refresh)
            case LockAcquisition.ERROR:
                # No cross-replica election happened (lock backend down). Keep the same-worker
                # single-flight so concurrent callers here still share one refresh.
                return await self.fallback.run(user_id, server_id, refresh, reread)
            case LockAcquisition.HELD:
                # Another worker holds the lock; wait for that exact generation to finish, then use
                # the typed error it published before falling back to a store re-read. The result key
                # is token-scoped, so a later refresh generation cannot poison this waiter's outcome.
                deadline: Final = self.clock() + self.wait_timeout_seconds
                holder_token = await self.lock.held_token(key)  # rebind-ok: tracks active holder across lease handoffs
                if holder_token is None:
                    return await reread()
                while self.clock() < deadline:
                    current_holder = await self.lock.held_token(key)
                    if current_holder is None:
                        break
                    if current_holder != holder_token:
                        earlier_outcome = await self._read_published_outcome(key, holder_token)
                        if earlier_outcome is not None:
                            return earlier_outcome
                        holder_token = current_holder
                    await self.sleep(self.poll_interval_seconds)
                outcome: Final = await self._read_published_outcome(key, holder_token)
                return outcome if outcome is not None else await reread()

    async def _read_published_outcome(self, key: str, holder_token: str) -> OutcomeT | None:
        published: Final = await self.lock.read_result(key, holder_token)
        return self.decode_error(published) if published is not None else None

    async def _refresh_with_lease_renewal(
        self,
        key: str,
        token: str,
        refresh: Callable[[], Awaitable[OutcomeT]],
    ) -> OutcomeT:
        refresh_task: Final = asyncio.ensure_future(refresh())
        renewal_task: Final = asyncio.create_task(self._renew_lease_until_done(key, token, refresh_task))

        async def release_after_refresh() -> None:
            await asyncio.wait({refresh_task})
            try:
                failed: Final = refresh_task.cancelled() or refresh_task.exception() is not None
                encoded: Final = None if failed else self.encode_error(refresh_task.result())
                if encoded is not None:
                    await self.lock.publish_result(key, token, encoded, self.result_ttl_seconds)
            finally:
                renewal_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renewal_task
                await self.lock.release(key, token)

        release_task: Final = asyncio.create_task(release_after_refresh())
        self.background_tasks.add(release_task)
        release_task.add_done_callback(self.background_tasks.discard)
        try:
            return await asyncio.shield(refresh_task)
        finally:
            if refresh_task.done():
                await release_task

    async def _renew_lease_until_done(
        self,
        key: str,
        token: str,
        refresh_task: asyncio.Future[OutcomeT],
    ) -> None:
        budget_deadline: Final = self.clock() + self.refresh_budget_seconds
        while not refresh_task.done() and self.clock() < budget_deadline:
            await self.sleep(self.lock_ttl_seconds / 2)
            if not refresh_task.done() and not await self.lock.extend(key, token, self.lock_ttl_seconds):
                return
