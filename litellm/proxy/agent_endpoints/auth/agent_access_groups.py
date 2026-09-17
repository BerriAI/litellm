import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_AccessGroupTable

AccessGroupIds: TypeAlias = tuple[str, ...]
AccessGroupIdsLoader: TypeAlias = Callable[[str], Awaitable[AccessGroupIds]]  # mutable-ok: Callable params
LoadedAccessGroup: TypeAlias = LiteLLM_AccessGroupTable | None
AccessGroupLoader: TypeAlias = Callable[[str], Awaitable[LoadedAccessGroup]]  # mutable-ok: Callable parameter syntax


@dataclass(frozen=True, slots=True)
class AgentAccessGroupCeiling:
    """Everything the agent's attached access groups allow. An empty set denies that resource kind."""

    access_group_ids: AccessGroupIds
    models: frozenset[str]
    mcp_server_ids: frozenset[str]
    agent_ids: frozenset[str]


CeilingResolver: TypeAlias = Callable[[str], Awaitable[AgentAccessGroupCeiling | None]]  # mutable-ok: Callable params


async def _registry_access_group_ids(agent_id: str) -> AccessGroupIds:
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent: Final = await get_agent_with_read_through(agent_id)
    return tuple(agent.access_group_ids or ()) if agent is not None else ()


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
    load_access_group_ids: AccessGroupIdsLoader = _registry_access_group_ids,
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
