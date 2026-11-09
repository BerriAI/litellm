from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm.proxy._types import LiteLLM_AccessGroupTable

if TYPE_CHECKING:
    from litellm.types.agents import AgentResponse

AccessGroupIds: TypeAlias = tuple[str, ...]
AccessGroupIdsLoader: TypeAlias = Callable[[str], Awaitable[AccessGroupIds]]
LoadedAccessGroup: TypeAlias = LiteLLM_AccessGroupTable | None
AccessGroupLoader: TypeAlias = Callable[[str], Awaitable[LoadedAccessGroup]]


@dataclass(frozen=True, slots=True)
class AgentAccessGroupCeiling:
    """Everything the agent's attached access groups allow. An empty set denies that resource kind."""

    access_group_ids: AccessGroupIds
    models: frozenset[str]
    mcp_server_ids: frozenset[str]
    agent_ids: frozenset[str]
    access_groups: tuple[LiteLLM_AccessGroupTable, ...] = ()


CeilingResolver: TypeAlias = Callable[[str], Awaitable[AgentAccessGroupCeiling | None]]


async def _registry_access_group_ids(agent_id: str) -> AccessGroupIds:
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent: Final = await get_agent_with_read_through(agent_id)
    return tuple(agent.access_group_ids or ()) if agent is not None else ()


async def _load_access_group(access_group_id: str, *, check_db_only: bool = False) -> LoadedAccessGroup:
    from litellm.proxy.auth.agent_ceiling_adapter import load_agent_access_group

    return await load_agent_access_group(access_group_id, check_db_only=check_db_only)


async def resolve_agent_access_group_ceiling(
    agent_id: str,
    load_access_group_ids: AccessGroupIdsLoader = _registry_access_group_ids,
    load_access_group: AccessGroupLoader = _load_access_group,
    *,
    check_db_only: bool = False,
) -> AgentAccessGroupCeiling | None:
    """``None`` when the agent has no access groups attached, so nothing is capped. A named group that
    provably does not exist is dropped from the ceiling, it grants and bounds nothing, while a group
    that exists but could not be read stays named and loads as ``None``. The ceiling itself stays once
    any group is named, so an agent whose every group is missing is still capped to nothing."""
    from litellm.proxy.auth.agent_ceiling_adapter import project_agent_ceiling, read_agent_ceiling_facts

    return project_agent_ceiling(
        await read_agent_ceiling_facts(agent_id, load_access_group_ids, load_access_group, check_db_only=check_db_only)
    )


async def resolve_managed_agent_ceilings(agent: "AgentResponse") -> tuple[AgentAccessGroupCeiling, ...]:
    async def authoritative_group(group_id: str) -> LoadedAccessGroup:
        return await _load_access_group(group_id, check_db_only=True)

    async def manual_ids(_agent_id: str) -> AccessGroupIds:
        return tuple(agent.access_group_ids or ())

    manual: Final = await resolve_agent_access_group_ceiling(
        agent.agent_id, load_access_group_ids=manual_ids, load_access_group=authoritative_group
    )
    return (manual,) if manual is not None else ()
