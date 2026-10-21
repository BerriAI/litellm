import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol, TypeAlias, cast  # noqa: TID251  # RedisCache is narrowed to its async Protocol

from litellm.caching.dual_cache import DualCache
from litellm.constants import (
    GITHUB_COPILOT_GITHUB_TOKEN_REFRESH_MARGIN_SECONDS,
    GITHUB_COPILOT_TOKEN_REFRESH_LOCK_PREFIX,
    GITHUB_COPILOT_TOKEN_REFRESH_REREAD_INTERVAL_SECONDS,
    GITHUB_COPILOT_TOKEN_REFRESH_REREAD_SECONDS,
)
from litellm.exceptions import BadRequestError
from litellm.llms.github_copilot.github_host import GithubHostEndpoints
from litellm.llms.github_copilot.per_user_auth import (
    GithubCopilotTokenRejectedError,
    GithubTokenRefreshed,
    GithubTokenRefreshRejected,
    GithubTokenRefreshUnavailable,
    aexchange_github_token,
    arefresh_github_token,
    github_copilot_host_endpoints,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisCommands,
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    RedisRefreshCoordinator,
)
from litellm.proxy.credential_endpoints.user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    compare_and_set_user_provider_credential,
    drop_user_provider_credential_cache,
    get_user_provider_credential_row,
    invalidate_user_provider_credential_cache,
    set_user_provider_credential_cache,
)
from litellm.repositories.prisma_protocols import DatabaseClient

ConnectionPayload: TypeAlias = GithubCopilotUserConnectionPayload
_InFlightTasks: TypeAlias = dict[
    tuple[str, str],
    asyncio.Task[ConnectionPayload | None],
]


class _RedisCache(Protocol):
    def init_async_client(self) -> RedisCommands: ...

    def check_and_fix_namespace(self, key: str) -> str: ...


@dataclass(slots=True)
class GithubCopilotTokenRefreshSingleFlight:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _in_flight: _InFlightTasks = field(
        default_factory=dict,
        repr=False,
    )

    async def run(
        self,
        key: tuple[str, str],
        refresh: Callable[[], Awaitable[ConnectionPayload | None]],
    ) -> ConnectionPayload | None:
        task: Final = await self._get_or_create_task(key, refresh)
        return await asyncio.shield(task)

    async def _get_or_create_task(
        self,
        key: tuple[str, str],
        refresh: Callable[[], Awaitable[ConnectionPayload | None]],
    ) -> asyncio.Task[ConnectionPayload | None]:
        async with self._lock:
            task: Final = self._in_flight.get(key)
            if task is not None and not task.done():
                return task

            async def execute_refresh() -> ConnectionPayload | None:
                return await refresh()

            created_task: Final = asyncio.create_task(execute_refresh())
            self._in_flight[key] = created_task

            def remove_completed_task(completed_task: asyncio.Task[ConnectionPayload | None]) -> None:
                if self._in_flight.get(key) is completed_task:
                    self._in_flight.pop(key, None)

            created_task.add_done_callback(remove_completed_task)
            return created_task


_SINGLE_FLIGHT: Final = GithubCopilotTokenRefreshSingleFlight()


async def arefresh_expiring_connections(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    connections: Mapping[str, ConnectionPayload],
    *,
    single_flight: GithubCopilotTokenRefreshSingleFlight | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Mapping[str, ConnectionPayload]:
    refresher: Final = single_flight if single_flight is not None else _SINGLE_FLIGHT
    items: Final = tuple(connections.items())
    refreshed: Final = await asyncio.gather(
        *(
            _refresh_connection(
                prisma_client=prisma_client,
                cache=cache,
                user_id=user_id,
                credential_name=name,
                stale=payload,
                single_flight=refresher,
                sleep=sleep,
                clock=clock,
            )
            for name, payload in items
        )
    )
    return {name: payload for (name, _), payload in zip(items, refreshed) if payload is not None}


async def _refresh_connection(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    credential_name: str,
    stale: ConnectionPayload,
    single_flight: GithubCopilotTokenRefreshSingleFlight,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> ConnectionPayload | None:
    now: Final = time.time()
    if stale.refresh_token is None:
        return stale
    if stale.refresh_token_expires_at is not None and stale.refresh_token_expires_at <= now:
        return stale
    try:
        endpoints: Final = github_copilot_host_endpoints(credential_name)
    except BadRequestError:
        return stale
    if stale.github_host != endpoints.host:
        return stale
    if stale.access_token_expires_at is None or (
        stale.access_token_expires_at - GITHUB_COPILOT_GITHUB_TOKEN_REFRESH_MARGIN_SECONDS > now
    ):
        try:
            await aexchange_github_token(
                user_id=user_id,
                github_token=stale.access_token,
                credential_name=credential_name,
                endpoints=endpoints,
            )
        except GithubCopilotTokenRejectedError:
            pass
        except Exception:  # noqa: BLE001  # other Copilot exchange errors stay on the existing request path
            return stale
        else:
            return stale
    key: Final = (user_id, credential_name)
    return await single_flight.run(
        key,
        lambda: _coordinate_refresh(
            prisma_client=prisma_client,
            cache=cache,
            user_id=user_id,
            credential_name=credential_name,
            stale_access_token=stale.access_token,
            endpoints=endpoints,
            sleep=sleep,
            clock=clock,
        ),
    )


async def _coordinate_refresh(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    credential_name: str,
    stale_access_token: str,
    endpoints: GithubHostEndpoints,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> ConnectionPayload | None:
    async def refresh() -> ConnectionPayload | None:
        return await _refresh_elected(
            prisma_client=prisma_client,
            cache=cache,
            user_id=user_id,
            credential_name=credential_name,
            stale_access_token=stale_access_token,
            endpoints=endpoints,
            sleep=sleep,
            clock=clock,
        )

    redis_cache_raw: Final = cache.redis_cache
    if redis_cache_raw is None:
        return await refresh()
    try:
        redis_cache: Final = cast(  # cast-ok: runtime RedisCache exposes async methods omitted from its protocol
            _RedisCache, redis_cache_raw
        )
        lock: Final = RedisDistributedLock(
            redis_cache.init_async_client(),
            namespace_key=redis_cache.check_and_fix_namespace,
        )
        coordinator: Final = RedisRefreshCoordinator(
            lock,
            key_prefix=GITHUB_COPILOT_TOKEN_REFRESH_LOCK_PREFIX,
        )
    except Exception:  # noqa: BLE001  # Redis setup failures fail open to an uncoordinated refresh
        return await refresh()
    return await coordinator.run(
        user_id=user_id,
        server_id=credential_name,
        refresh=refresh,
        reread=lambda: _read_connection(prisma_client, user_id, credential_name),
    )


async def _refresh_elected(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    credential_name: str,
    stale_access_token: str,
    endpoints: GithubHostEndpoints,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> ConnectionPayload | None:
    row: Final = await get_user_provider_credential_row(prisma_client, user_id, credential_name)
    if row is None:
        return None
    ciphertext, payload = row
    if payload.access_token != stale_access_token or payload.refresh_token is None:
        return payload
    result: Final = await arefresh_github_token(payload.refresh_token, endpoints)
    match result:
        case GithubTokenRefreshRejected():
            return await _reread_after_rejection(
                prisma_client=prisma_client,
                cache=cache,
                user_id=user_id,
                credential_name=credential_name,
                ciphertext=ciphertext,
                stale_access_token=stale_access_token,
                stale=payload,
                sleep=sleep,
                clock=clock,
            )
        case GithubTokenRefreshUnavailable():
            return payload
        case GithubTokenRefreshed():
            now: Final = time.time()
            refreshed: Final = payload.model_copy(
                update={
                    "access_token": result.access_token,
                    "refresh_token": result.refresh_token,
                    "access_token_expires_at": (now + result.expires_in if result.expires_in is not None else None),
                    "refresh_token_expires_at": (
                        now + result.refresh_token_expires_in if result.refresh_token_expires_in is not None else None
                    ),
                }
            )
    return await _persist_connection(
        prisma_client=prisma_client,
        cache=cache,
        user_id=user_id,
        credential_name=credential_name,
        expected_ciphertext=ciphertext,
        payload=refreshed,
    )


async def _persist_connection(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    credential_name: str,
    expected_ciphertext: str,
    payload: ConnectionPayload,
) -> ConnectionPayload | None:
    if not await compare_and_set_user_provider_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        credential_name=credential_name,
        expected_ciphertext=expected_ciphertext,
        payload=payload,
    ):
        return await _read_connection(prisma_client, user_id, credential_name)
    await set_user_provider_credential_cache(cache, user_id, credential_name, payload)
    current: Final = await _read_connection(prisma_client, user_id, credential_name)
    if current is None:
        await invalidate_user_provider_credential_cache(cache, user_id, credential_name)
        return None
    if current != payload:
        await drop_user_provider_credential_cache(cache, user_id, credential_name)
        return current
    return payload


async def _reread_after_rejection(
    prisma_client: DatabaseClient,
    cache: DualCache,
    user_id: str,
    credential_name: str,
    ciphertext: str,
    stale_access_token: str,
    stale: ConnectionPayload,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> ConnectionPayload | None:
    deadline: Final = clock() + GITHUB_COPILOT_TOKEN_REFRESH_REREAD_SECONDS

    async def read_changed_connection() -> tuple[bool, ConnectionPayload | None]:
        current: Final = await _read_connection(prisma_client, user_id, credential_name)
        if current is None:
            return True, None
        if current.access_token != stale_access_token:
            return True, current
        return False, None

    while True:
        is_changed, current = await read_changed_connection()
        if is_changed:
            return current
        if deadline <= clock():
            return await _persist_connection(
                prisma_client=prisma_client,
                cache=cache,
                user_id=user_id,
                credential_name=credential_name,
                expected_ciphertext=ciphertext,
                payload=stale.model_copy(update={"refresh_token": None, "refresh_token_expires_at": None}),
            )
        await sleep(
            min(
                GITHUB_COPILOT_TOKEN_REFRESH_REREAD_INTERVAL_SECONDS,
                max(0.0, deadline - clock()),
            )
        )


async def _read_connection(
    prisma_client: DatabaseClient,
    user_id: str,
    credential_name: str,
) -> ConnectionPayload | None:
    row: Final = await get_user_provider_credential_row(prisma_client, user_id, credential_name)
    return row[1] if row is not None else None
