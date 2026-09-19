"""Display names for ids stored on management objects. DB rows win; config-declared servers and agents fill the gaps."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import AgentsRepository, MCPServerRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.mcp_server.mcp_server_manager import MCPServer


async def mcp_server_display_names(
    prisma_client: PrismaClient,
    server_ids: Sequence[str],
    config_servers: Mapping[str, MCPServer],
) -> Mapping[str, str]:
    """server_id -> alias, falling back to server_name; config-only servers also fall back to their registry name."""
    if not server_ids:
        return MappingProxyType({})
    wanted: Final = frozenset(server_ids)
    where: Final = {"server_id": {"in": tuple(wanted)}}  # mutable-ok: prisma where is a dict
    rows: Final = await MCPServerRepository(prisma_client).table.find_many(where=where)
    from_config: Final = {
        server_id: server.alias or server.server_name or server.name
        for server_id, server in config_servers.items()
        if server_id in wanted
    }
    from_db: Final = {row.server_id: name for row in rows if (name := row.alias or row.server_name)}
    return MappingProxyType({**from_config, **from_db})


async def agent_display_names(
    prisma_client: PrismaClient,
    agent_ids: Sequence[str],
    registry: AgentRegistry,
) -> Mapping[str, str]:
    """agent_id -> agent_name. The registry covers config-declared agents and their legacy ids."""
    if not agent_ids:
        return MappingProxyType({})
    wanted: Final = frozenset(agent_ids)
    where: Final = {"agent_id": {"in": tuple(wanted)}}  # mutable-ok: prisma where is a dict
    rows: Final = await AgentsRepository(prisma_client).table.find_many(where=where)
    from_registry: Final = {
        alias_id: agent.agent_name
        for agent in registry.get_agent_list()
        for alias_id in registry.ids_for_agent(agent.agent_id)
        if alias_id in wanted
    }
    from_db: Final = {row.agent_id: row.agent_name for row in rows}
    return MappingProxyType({**from_registry, **from_db})


async def key_display_names(prisma_client: PrismaClient, tokens: Sequence[str]) -> Mapping[str, str]:
    """token hash -> key_alias for the keys that have one."""
    if not tokens:
        return MappingProxyType({})
    where: Final = {"token": {"in": tuple(frozenset(tokens))}}  # mutable-ok: prisma where is a dict
    rows: Final = await VerificationTokenRepository(prisma_client).table.find_many(where=where)
    return MappingProxyType({row.token: row.key_alias for row in rows if row.key_alias})
