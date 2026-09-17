import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import LiteLLM_AccessGroupTable
from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl


class _AgentAccessGroupsRecord(Protocol):
    @property
    def access_group_ids(self) -> Sequence[str] | None: ...


class _AgentIdWhere(TypedDict):
    agent_id: ReadOnly[str]


AccessGroupIds: TypeAlias = tuple[str, ...]
AccessGroupIdsLoader: TypeAlias = Callable[[str], Awaitable[AccessGroupIds]]  # mutable-ok: Callable params
AgentRecordFinder: TypeAlias = Callable[[str], Awaitable[_AgentAccessGroupsRecord | None]]  # mutable-ok: Callable
LoadedAccessGroup: TypeAlias = LiteLLM_AccessGroupTable | None
AccessGroupLoader: TypeAlias = Callable[[str], Awaitable[LoadedAccessGroup]]  # mutable-ok: Callable parameter syntax

_CACHED_IDS: Final = TypeAdapter(list[str])


@dataclass(frozen=True, slots=True)
class AgentAccessGroupCeiling:
    """Everything the agent's attached access groups allow. An empty set denies that resource kind."""

    access_group_ids: AccessGroupIds
    models: frozenset[str]
    mcp_server_ids: frozenset[str]
    agent_ids: frozenset[str]


CeilingResolver: TypeAlias = Callable[[str], Awaitable[AgentAccessGroupCeiling | None]]  # mutable-ok: Callable params


def agent_access_group_ids_cache_key(agent_id: str) -> str:
    return f"agent_access_group_ids:{agent_id}"


def _cached_access_group_ids(cached: object) -> AccessGroupIds | None:
    if cached is None:
        return None
    try:
        return tuple(_CACHED_IDS.validate_python(cached))
    except ValidationError:
        return None


async def _registry_access_group_ids(agent_id: str) -> AccessGroupIds:
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent: Final = await get_agent_with_read_through(agent_id)
    return tuple(agent.access_group_ids or ()) if agent is not None else ()


async def load_agent_access_group_ids(
    agent_id: str,
    cache: DualCache,
    find_agent: AgentRecordFinder,
    fallback: AccessGroupIdsLoader,
) -> AccessGroupIds:
    """The agent row's groups, cached for the management-object TTL and evicted on every agent write."""
    cache_key: Final = agent_access_group_ids_cache_key(agent_id)
    cached: Final = _cached_access_group_ids(await cache.async_get_cache(key=cache_key))
    if cached is not None:
        return cached
    try:
        record: Final = await find_agent(agent_id)
    except Exception as e:  # noqa: BLE001  # prisma raises many error types; the registry snapshot answers instead
        verbose_proxy_logger.warning("Failed to read access groups for agent %r, using registry: %s", agent_id, e)
        return await fallback(agent_id)
    access_group_ids: Final = tuple(record.access_group_ids or ()) if record is not None else ()
    await cache.async_set_cache(key=cache_key, value=access_group_ids, ttl=get_management_object_ttl(cache))
    return access_group_ids


async def _load_agent_access_group_ids(agent_id: str) -> AccessGroupIds:
    from litellm.proxy.agent_endpoints.agent_registry import agents_table
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return await _registry_access_group_ids(agent_id)
    db: Final = prisma_client

    async def find_agent(row_agent_id: str) -> _AgentAccessGroupsRecord | None:
        return await agents_table(db).find_unique(where=_AgentIdWhere(agent_id=row_agent_id))

    return await load_agent_access_group_ids(agent_id, user_api_key_cache, find_agent, _registry_access_group_ids)


async def evict_agent_access_group_ids(agent_ids: Sequence[str]) -> None:
    from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
    from litellm.proxy.proxy_server import user_api_key_cache

    await evict_and_broadcast(
        cache_keys=tuple(agent_access_group_ids_cache_key(agent_id) for agent_id in agent_ids),
        user_api_key_cache=user_api_key_cache,
    )


async def _load_access_group(access_group_id: str) -> LoadedAccessGroup:
    from litellm.proxy.auth.auth_checks import get_access_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        verbose_proxy_logger.warning("Agent access group %s cannot be loaded without a DB", access_group_id)
        return None
    try:
        return await get_access_object(
            access_group_id=access_group_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except HTTPException as e:
        verbose_proxy_logger.warning(
            "Agent access group %s could not be loaded, treating it as empty: %s", access_group_id, e.detail
        )
        return None


async def resolve_agent_access_group_ceiling(
    agent_id: str,
    load_access_group_ids: AccessGroupIdsLoader = _load_agent_access_group_ids,
    load_access_group: AccessGroupLoader = _load_access_group,
) -> AgentAccessGroupCeiling | None:
    """``None`` when the agent has no access groups attached, so nothing is capped."""
    access_group_ids: Final = await load_access_group_ids(agent_id)
    if not access_group_ids:
        return None

    loaded: Final = await asyncio.gather(*(load_access_group(group_id) for group_id in access_group_ids))
    groups: Final = tuple(group for group in loaded if group is not None)
    return AgentAccessGroupCeiling(
        access_group_ids=access_group_ids,
        models=frozenset(model for group in groups for model in group.access_model_names),
        mcp_server_ids=frozenset(server_id for group in groups for server_id in group.access_mcp_server_ids),
        agent_ids=frozenset(target_id for group in groups for target_id in group.access_agent_ids),
    )
