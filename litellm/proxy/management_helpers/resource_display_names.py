"""Display names for ids stored on management objects. DB rows win; config-declared servers and agents fill the gaps.
A vector store declared in the config file keeps its config name, as it does in the vector store list."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.utils import PrismaClient
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.table_repositories import (
    AgentsRepository,
    ManagedVectorStoresRepository,
    MCPServerRepository,
)
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore


async def mcp_server_display_names(
    prisma_client: PrismaClient,
    server_ids: Sequence[str],
    config_servers: Mapping[str, MCPServer],
) -> Mapping[str, str]:
    """server_id -> alias, falling back to server_name; config-only servers also fall back to their registry name."""
    if not server_ids:
        return MappingProxyType({})
    wanted: Final = frozenset(server_ids)
    where: Final = {"server_id": {"in": tuple(wanted)}}
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
    where: Final = {"agent_id": {"in": tuple(wanted)}}
    rows: Final = await AgentsRepository(prisma_client).table.find_many(where=where)
    from_registry: Final = {
        alias_id: agent.agent_name
        for agent in registry.get_agent_list()
        for alias_id in registry.ids_for_agent(agent.agent_id)
        if alias_id in wanted
    }
    from_db: Final = {row.agent_id: row.agent_name for row in rows}
    return MappingProxyType({**from_registry, **from_db})


async def vector_store_display_names(
    prisma_client: PrismaClient,
    vector_store_ids: Sequence[str],
    registered_stores: Sequence[LiteLLM_ManagedVectorStore],
) -> Mapping[str, str]:
    """vector_store_id -> vector_store_name for the stores that have one."""
    wanted: Final = frozenset(vector_store_ids)
    rows: Final = await find_many_in(ManagedVectorStoresRepository(prisma_client).table, "vector_store_id", wanted)
    from_db: Final = {row.vector_store_id: row.vector_store_name for row in rows}
    from_config: Final = {
        store_id: store.get("vector_store_name")
        for store in registered_stores
        if store.get("is_config") and (store_id := store.get("vector_store_id")) is not None and store_id in wanted
    }
    return MappingProxyType({store_id: name for store_id, name in {**from_db, **from_config}.items() if name})


async def key_display_names(prisma_client: PrismaClient, tokens: Sequence[str]) -> Mapping[str, str]:
    """token hash -> key_alias for the keys that have one."""
    if not tokens:
        return MappingProxyType({})
    where: Final = {"token": {"in": tuple(frozenset(tokens))}}
    rows: Final = await VerificationTokenRepository(prisma_client).table.find_many(where=where)
    return MappingProxyType({row.token: row.key_alias for row in rows if row.key_alias})
