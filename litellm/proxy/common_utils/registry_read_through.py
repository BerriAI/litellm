"""Read-through recovery for in-memory registries in multi-replica deployments.

A management write (POST /model/new, /guardrails, /v1/agents) lands on one
replica and reaches Postgres, but sibling replicas only refresh their in-memory
registries on the periodic config reload or the Redis config-sync resync, both
of which lag by seconds. A request that uses the new object immediately can
land on a sibling that has never heard of it and fail with a 400/404.

On a registry miss, callers here fetch the missing object from the DB and load
it into the local registry before giving up. A short negative-result TTL keeps
repeated lookups of genuinely unknown names from hammering the DB.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.agents import AgentResponse

READ_THROUGH_MISS_TTL_SECONDS: Final = 2.0


class RegistryReadThrough:
    __slots__ = ("_lock", "_miss_ttl_seconds", "_recent_misses", "_resync")

    def __init__(
        self,
        resync: Callable[[str], Awaitable[bool]],
        miss_ttl_seconds: float = READ_THROUGH_MISS_TTL_SECONDS,
    ) -> None:
        self._resync = resync
        self._miss_ttl_seconds = miss_ttl_seconds
        self._lock = asyncio.Lock()
        self._recent_misses = InMemoryCache(max_size_in_memory=1000)

    async def attempt(self, key: str) -> bool:
        if self._recent_misses.get_cache(key) is not None:
            return False
        async with self._lock:
            if self._recent_misses.get_cache(key) is not None:
                return False
            try:
                found: Final = await self._resync(key)
            except Exception as e:  # noqa: BLE001  # a failed read-through must surface the original miss error, not a 500
                verbose_proxy_logger.warning("registry read-through for %r failed: %s", key, e)
                return False
            if not found:
                self._recent_misses.set_cache(key, True, ttl=self._miss_ttl_seconds)
            return found


def _db_backed_registries_enabled() -> bool:
    from litellm.proxy import proxy_server

    return proxy_server.prisma_client is not None and proxy_server.store_model_in_db is True


async def _resync_model_deployments(model_name: str) -> bool:
    from litellm.proxy import proxy_server
    from litellm.repositories.model_repository import ModelRepository

    if not _db_backed_registries_enabled():
        return False
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    rows: Final = await ModelRepository(prisma_client).table.find_many(
        where={"OR": [{"model_name": model_name}, {"model_id": model_name}]}
    )
    if not rows:
        return False
    if proxy_server.llm_router is None:
        await proxy_server.proxy_config.add_deployment(
            prisma_client=prisma_client, proxy_logging_obj=proxy_server.proxy_logging_obj
        )
        return proxy_server.llm_router is not None
    proxy_server.proxy_config._add_deployment(db_models=rows)
    proxy_server.llm_model_list = proxy_server.llm_router.get_model_list()
    return True


async def _resync_guardrails(guardrail_name: str) -> bool:
    from litellm.proxy import proxy_server

    if not _db_backed_registries_enabled():
        return False
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    await proxy_server.proxy_config._init_guardrails_in_db(prisma_client=prisma_client)
    return _initialized_guardrail(guardrail_name) is not None


async def _resync_agents(agent_id_or_name: str) -> bool:
    from litellm.proxy import proxy_server

    if not _db_backed_registries_enabled():
        return False
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    await proxy_server.proxy_config._init_agents_in_db(prisma_client=prisma_client)
    return _agent_from_registry(agent_id_or_name) is not None


model_registry_read_through: Final = RegistryReadThrough(resync=_resync_model_deployments)
guardrail_registry_read_through: Final = RegistryReadThrough(resync=_resync_guardrails)
agent_registry_read_through: Final = RegistryReadThrough(resync=_resync_agents)


def _agent_from_registry(agent_id_or_name: str) -> "AgentResponse | None":
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    by_id: Final = global_agent_registry.get_agent_by_id(agent_id=agent_id_or_name)
    if by_id is not None:
        return by_id
    return global_agent_registry.get_agent_by_name(agent_name=agent_id_or_name)


async def get_agent_with_read_through(agent_id_or_name: str) -> "AgentResponse | None":
    agent: Final = _agent_from_registry(agent_id_or_name)
    if agent is not None:
        return agent
    if not await agent_registry_read_through.attempt(agent_id_or_name):
        return None
    return _agent_from_registry(agent_id_or_name)


def _initialized_guardrail(guardrail_name: str) -> "CustomGuardrail | None":
    from litellm.proxy.guardrails import guardrail_endpoints

    return guardrail_endpoints.GUARDRAIL_REGISTRY.get_initialized_guardrail_callback(guardrail_name=guardrail_name)


async def get_initialized_guardrail_with_read_through(guardrail_name: str) -> "CustomGuardrail | None":
    active: Final = _initialized_guardrail(guardrail_name)
    if active is not None:
        return active
    if not await guardrail_registry_read_through.attempt(guardrail_name):
        return None
    return _initialized_guardrail(guardrail_name)
