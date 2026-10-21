import types
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.management_helpers.resource_display_names import (
    agent_display_names,
    key_display_names,
    mcp_server_display_names,
)
from litellm.types.agents import AgentResponse
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _table(rows=()):
    return types.SimpleNamespace(find_many=AsyncMock(return_value=list(rows)))


def _prisma(**tables):
    return types.SimpleNamespace(db=types.SimpleNamespace(**tables))


def _config_server(server_id: str, name: str, alias: str | None = None, server_name: str | None = None) -> MCPServer:
    return MCPServer(server_id=server_id, name=name, alias=alias, server_name=server_name, transport="http")


def _registry_with(*agents: AgentResponse, legacy_ids: dict[str, str] | None = None) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in agents:
        registry.register_agent(agent)
    registry.config_agent_legacy_ids = MappingProxyType(legacy_ids or {})
    return registry


def _agent(agent_id: str, agent_name: str) -> AgentResponse:
    return AgentResponse(agent_id=agent_id, agent_name=agent_name, agent_card_params={})


@pytest.mark.asyncio
async def test_mcp_db_row_beats_config_entry_for_the_same_server():
    """The DB is authoritative when both sources know a server; the registry may lag behind a rename on another pod."""
    prisma = _prisma(
        litellm_mcpservertable=_table([types.SimpleNamespace(server_id="s1", alias="db-alias", server_name=None)])
    )
    names = await mcp_server_display_names(prisma, ("s1",), {"s1": _config_server("s1", "config-name")})
    assert dict(names) == {"s1": "db-alias"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("alias", "server_name", "expected"),
    [("Alias", "server_name", "Alias"), (None, "server_name", "server_name"), (None, None, "config-name")],
)
async def test_mcp_config_only_server_falls_back_alias_then_server_name_then_name(alias, server_name, expected):
    """Config-declared servers have no DB row, so their registry entry supplies the label."""
    prisma = _prisma(litellm_mcpservertable=_table())
    config = {"s1": _config_server("s1", "config-name", alias=alias, server_name=server_name)}
    names = await mcp_server_display_names(prisma, ("s1",), config)
    assert dict(names) == {"s1": expected}


@pytest.mark.asyncio
async def test_mcp_db_row_without_alias_or_server_name_yields_no_label():
    """A bare DB row must not produce an empty string label; the caller falls back to the id."""
    prisma = _prisma(
        litellm_mcpservertable=_table([types.SimpleNamespace(server_id="s1", alias=None, server_name=None)])
    )
    assert dict(await mcp_server_display_names(prisma, ("s1",), {})) == {}


@pytest.mark.asyncio
async def test_mcp_only_requested_ids_are_returned_and_the_query_is_deduped():
    """Unrequested config servers stay out of the result and repeated ids collapse to one IN filter entry."""
    table = _table([types.SimpleNamespace(server_id="s1", alias="A", server_name=None)])
    prisma = _prisma(litellm_mcpservertable=table)
    config = {"other": _config_server("other", "not-requested")}
    names = await mcp_server_display_names(prisma, ("s1", "s1", "missing"), config)
    assert dict(names) == {"s1": "A"}
    assert sorted(table.find_many.call_args.kwargs["where"]["server_id"]["in"]) == ["missing", "s1"]


@pytest.mark.asyncio
async def test_mcp_empty_ids_skip_the_db():
    table = _table()
    names = await mcp_server_display_names(_prisma(litellm_mcpservertable=table), (), {})
    assert dict(names) == {}
    table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_db_name_beats_registry_name():
    prisma = _prisma(litellm_agentstable=_table([types.SimpleNamespace(agent_id="a1", agent_name="from-db")]))
    registry = _registry_with(_agent("a1", "from-registry"))
    assert dict(await agent_display_names(prisma, ("a1",), registry)) == {"a1": "from-db"}


@pytest.mark.asyncio
async def test_agent_legacy_config_id_resolves_to_the_stable_agent_name():
    """Access groups saved before agent ids were stabilised still carry the legacy hash; it must still get a name."""
    prisma = _prisma(litellm_agentstable=_table())
    registry = _registry_with(_agent("stable-id", "config-agent"), legacy_ids={"legacy-id": "stable-id"})
    names = await agent_display_names(prisma, ("legacy-id", "stable-id", "unknown"), registry)
    assert dict(names) == {"legacy-id": "config-agent", "stable-id": "config-agent"}


@pytest.mark.asyncio
async def test_agent_empty_ids_skip_the_db():
    table = _table()
    names = await agent_display_names(_prisma(litellm_agentstable=table), (), _registry_with())
    assert dict(names) == {}
    table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_key_alias_only_for_keys_that_have_one():
    table = _table(
        [types.SimpleNamespace(token="k1", key_alias="ci-key"), types.SimpleNamespace(token="k2", key_alias=None)]
    )
    names = await key_display_names(_prisma(litellm_verificationtoken=table), ("k1", "k2", "k1"))
    assert dict(names) == {"k1": "ci-key"}
    assert sorted(table.find_many.call_args.kwargs["where"]["token"]["in"]) == ["k1", "k2"]


@pytest.mark.asyncio
async def test_key_empty_ids_skip_the_db():
    table = _table()
    assert dict(await key_display_names(_prisma(litellm_verificationtoken=table), ())) == {}
    table.find_many.assert_not_awaited()
