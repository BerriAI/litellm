import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken

ADVISORY_LOCK_SQL: Final = "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext($1))"


@dataclass(frozen=True, slots=True)
class AdvisoryLockUnavailable:
    pass


class AdvisoryLock(Protocol):
    async def while_held(
        self, key: str, action: Callable[[], Awaitable[OAuthToken | None]]
    ) -> OAuthToken | None | AdvisoryLockUnavailable: ...


class _LockTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class _TransactionalDb(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_LockTransaction]: ...


@dataclass(frozen=True, slots=True)
class PrismaAdvisoryLock:
    db: _TransactionalDb
    timeout: timedelta

    async def while_held(
        self, key: str, action: Callable[[], Awaitable[OAuthToken | None]]
    ) -> OAuthToken | None | AdvisoryLockUnavailable:
        try:
            async with self.db.tx(timeout=self.timeout) as transaction:
                await transaction.query_raw(ADVISORY_LOCK_SQL, key)
                return await action()
        except Exception as exc:  # noqa: BLE001  # without the lock a refresh token could be redeemed twice
            verbose_proxy_logger.warning(
                "Could not hold the Postgres token refresh lock (%s), so the stored token is read instead",
                type(exc).__name__,
            )
            return AdvisoryLockUnavailable()


@dataclass(frozen=True, slots=True)
class AdvisoryLockRefreshCoordinator:
    lock: AdvisoryLock

    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        match await self.lock.while_held(json.dumps([user_id, server_id]), refresh):
            case AdvisoryLockUnavailable():
                return await reread()
            case token:
                return token
