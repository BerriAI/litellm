"""Read-through recovery for in-memory registries in multi-replica deployments.

A management write (POST /model/new, /guardrails, /v1/agents) lands on one
replica and reaches Postgres, but sibling replicas only refresh their in-memory
registries on the periodic config reload, so a request using the new object
immediately can land on a sibling that has never heard of it and fail 400/404.
On a registry miss, callers here fetch the missing row from the DB and load it
into the local registry before giving up. A short negative-result TTL per key
plus a global resync budget per window bound the DB load from lookups of
genuinely unknown names.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache

if TYPE_CHECKING:
    from prisma.types import (
        LiteLLM_AgentsTableInclude,
        LiteLLM_AgentsTableWhereUniqueInput,
        LiteLLM_GuardrailsTableWhereInput,
        LiteLLM_ProxyModelTableWhereInput,
    )

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.agents import AgentResponse

READ_THROUGH_MISS_TTL_SECONDS: Final = 2.0
READ_THROUGH_RESYNC_WINDOW_SECONDS: Final = 5.0
READ_THROUGH_MAX_RESYNCS_PER_WINDOW: Final = 20


class RegistryReadThrough:
    __slots__ = (
        "_lock",
        "_max_resyncs_per_window",
        "_miss_ttl_seconds",
        "_recent_misses",
        "_resync",
        "_resync_window_seconds",
        "_window_resyncs",
        "_window_started_at",
    )

    def __init__(
        self,
        resync: Callable[[str], Awaitable[bool]],
        miss_ttl_seconds: float = READ_THROUGH_MISS_TTL_SECONDS,
        max_resyncs_per_window: int = READ_THROUGH_MAX_RESYNCS_PER_WINDOW,
        resync_window_seconds: float = READ_THROUGH_RESYNC_WINDOW_SECONDS,
    ) -> None:
        self._resync = resync
        self._miss_ttl_seconds = miss_ttl_seconds
        self._max_resyncs_per_window = max_resyncs_per_window
        self._resync_window_seconds = resync_window_seconds
        self._lock = asyncio.Lock()
        self._recent_misses = InMemoryCache(max_size_in_memory=1000)
        self._window_started_at = float("-inf")
        self._window_resyncs = 0

    def _consume_resync_budget(self) -> bool:
        now: Final = time.monotonic()
        if now - self._window_started_at >= self._resync_window_seconds:
            self._window_started_at = now
            self._window_resyncs = 0
        if self._window_resyncs >= self._max_resyncs_per_window:
            return False
        self._window_resyncs += 1
        return True

    async def attempt(self, key: str) -> bool:
        if self._recent_misses.get_cache(key) is not None:
            return False
        async with self._lock:
            if self._recent_misses.get_cache(key) is not None:
                return False
            if not self._consume_resync_budget():
                verbose_proxy_logger.warning(
                    "registry read-through for %r skipped: resync budget of %s per %ss exhausted",
                    key,
                    self._max_resyncs_per_window,
                    self._resync_window_seconds,
                )
                return False
            try:
                found: Final = await self._resync(key)
            except Exception as e:  # noqa: BLE001  # a failed read-through must surface the original miss error, not a 500
                verbose_proxy_logger.warning("registry read-through for %r failed: %s", key, e)
                return False
            if not found:
                self._recent_misses.set_cache(key, True, ttl=self._miss_ttl_seconds)
            return found


def _db_backed_registries_enabled(object_type: str) -> bool:
    from litellm.proxy import proxy_server

    if proxy_server.prisma_client is None or proxy_server.store_model_in_db is not True:
        return False
    return proxy_server.should_load_db_object(object_type=object_type)


async def _resync_model_deployments(model_name: str) -> bool:
    from litellm.proxy import proxy_server
    from litellm.repositories.model_repository import ModelRepository

    if not _db_backed_registries_enabled("models"):
        return False
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    table: Final = ModelRepository(prisma_client).table
    name_filter: Final[LiteLLM_ProxyModelTableWhereInput] = {"model_name": model_name}
    id_filter: Final[LiteLLM_ProxyModelTableWhereInput] = {"model_id": model_name}
    rows: Final = await table.find_many(where=name_filter) or await table.find_many(where=id_filter)
    if not rows:
        return False
    router: Final = proxy_server.llm_router
    if router is None:
        await proxy_server.proxy_config.add_deployment(
            prisma_client=prisma_client, proxy_logging_obj=proxy_server.proxy_logging_obj
        )
        return proxy_server.llm_router is not None
    async with proxy_server.MODEL_RECONCILE_LOCK:
        proxy_server.proxy_config._add_deployment(db_models=rows)
        proxy_server.llm_model_list = router.get_model_list()
    return True


async def _resync_guardrails(guardrail_name: str) -> bool:
    from litellm.proxy import proxy_server
    from litellm.proxy.guardrails.guardrail_registry import (
        GUARDRAIL_RECONCILE_LOCK,
        IN_MEMORY_GUARDRAIL_HANDLER,
    )
    from litellm.repositories.table_repositories import GuardrailsRepository
    from litellm.types.guardrails import Guardrail

    if not _db_backed_registries_enabled("guardrails"):
        return False
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    active_row_filter: Final[LiteLLM_GuardrailsTableWhereInput] = {
        "guardrail_name": guardrail_name,
        "status": "active",
    }
    row: Final = await GuardrailsRepository(prisma_client).table.find_first(where=active_row_filter)
    if row is None:
        return False
    async with GUARDRAIL_RECONCILE_LOCK:
        IN_MEMORY_GUARDRAIL_HANDLER.sync_guardrail_from_db(guardrail=Guardrail(**dict(row)))
    return _initialized_guardrail(guardrail_name) is not None


async def _resync_agents(agent_id_or_name: str) -> bool:
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints.agent_registry import (
        AGENT_RECONCILE_LOCK,
        agents_table,
        global_agent_registry,
    )
    from litellm.types.agents import AgentResponse

    if not _db_backed_registries_enabled("agents"):
        return False
    if _agent_from_registry(agent_id_or_name) is not None:
        return True
    prisma_client: Final = proxy_server.prisma_client
    assert prisma_client is not None
    table: Final = agents_table(prisma_client)
    id_filter: Final[LiteLLM_AgentsTableWhereUniqueInput] = {"agent_id": agent_id_or_name}
    name_filter: Final[LiteLLM_AgentsTableWhereUniqueInput] = {"agent_name": agent_id_or_name}
    include_permission: Final[LiteLLM_AgentsTableInclude] = {"object_permission": True}
    async with AGENT_RECONCILE_LOCK:
        if _agent_from_registry(agent_id_or_name) is not None:
            return True
        row: Final = await table.find_unique(where=id_filter, include=include_permission) or await table.find_unique(
            where=name_filter, include=include_permission
        )
        if row is None:
            return False
        global_agent_registry.register_agent(agent_config=AgentResponse.model_validate(row.model_dump()))
        return True


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
