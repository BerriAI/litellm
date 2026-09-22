from typing import Final

import pytest
from fastapi import HTTPException

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
    AgentAccessGroupCeiling,
    resolve_agent_access_group_ceiling,
)
from litellm.types.agents import AgentResponse

_CARD: Final = {"name": "agent", "url": "http://localhost:9999", "version": "1.0.0"}


def _agent(access_group_ids: list[str] | None) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-1", agent_name="agent", agent_card_params=_CARD, access_group_ids=access_group_ids
    )


def _group(
    group_id: str,
    models: tuple[str, ...] = (),
    mcp_servers: tuple[str, ...] = (),
    agents: tuple[str, ...] = (),
) -> LiteLLM_AccessGroupTable:
    return LiteLLM_AccessGroupTable(
        access_group_id=group_id,
        access_group_name=group_id,
        access_model_names=list(models),
        access_mcp_server_ids=list(mcp_servers),
        access_agent_ids=list(agents),
    )


def _loaders(agent: AgentResponse | None, groups: dict[str, LiteLLM_AccessGroupTable]):
    async def load_agent(agent_id: str) -> tuple[str, ...]:
        return tuple(agent.access_group_ids or ()) if agent is not None else ()

    async def load_group(group_id: str) -> LiteLLM_AccessGroupTable | None:
        return groups.get(group_id)

    return load_agent, load_group


@pytest.mark.asyncio
@pytest.mark.parametrize("access_group_ids", [None, []])
async def test_agent_without_access_groups_has_no_ceiling(access_group_ids: list[str] | None):
    load_agent, load_group = _loaders(_agent(access_group_ids), {"g1": _group("g1", models=("gpt-5",))})

    assert await resolve_agent_access_group_ceiling("agent-1", load_agent, load_group) is None


@pytest.mark.asyncio
async def test_unknown_agent_has_no_ceiling():
    load_agent, load_group = _loaders(None, {})

    assert await resolve_agent_access_group_ceiling("missing", load_agent, load_group) is None


@pytest.mark.asyncio
async def test_ceiling_is_the_union_of_every_attached_group():
    load_agent, load_group = _loaders(
        _agent(["g1", "g2"]),
        {
            "g1": _group("g1", models=("gpt-5",), mcp_servers=("mcp-a",), agents=("agent-b",)),
            "g2": _group("g2", models=("claude-sonnet",), mcp_servers=("mcp-b",), agents=("agent-c",)),
        },
    )

    ceiling: Final = await resolve_agent_access_group_ceiling("agent-1", load_agent, load_group)

    assert ceiling == AgentAccessGroupCeiling(
        access_group_ids=("g1", "g2"),
        models=frozenset({"gpt-5", "claude-sonnet"}),
        mcp_server_ids=frozenset({"mcp-a", "mcp-b"}),
        agent_ids=frozenset({"agent-b", "agent-c"}),
    )


@pytest.mark.asyncio
async def test_unloadable_group_contributes_nothing_but_the_ceiling_still_applies():
    load_agent, load_group = _loaders(_agent(["g1", "gone"]), {"g1": _group("g1", models=("gpt-5",))})

    ceiling: Final = await resolve_agent_access_group_ceiling("agent-1", load_agent, load_group)

    assert ceiling == AgentAccessGroupCeiling(
        access_group_ids=("g1", "gone"),
        models=frozenset({"gpt-5"}),
        mcp_server_ids=frozenset(),
        agent_ids=frozenset(),
    )


@pytest.mark.asyncio
async def test_only_unloadable_groups_is_an_empty_ceiling_not_unrestricted():
    load_agent, load_group = _loaders(_agent(["gone"]), {})

    ceiling: Final = await resolve_agent_access_group_ceiling("agent-1", load_agent, load_group)

    assert ceiling is not None
    assert ceiling.models == frozenset()
    assert ceiling.mcp_server_ids == frozenset()
    assert ceiling.agent_ids == frozenset()


@pytest.mark.asyncio
async def test_default_agent_loader_reads_the_attached_groups_from_the_registry():
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    _, load_group = _loaders(None, {"g1": _group("g1", models=("gpt-5",))})
    global_agent_registry.register_agent(_agent(["g1"]))
    try:
        ceiling: Final = await resolve_agent_access_group_ceiling("agent-1", load_access_group=load_group)
    finally:
        global_agent_registry.deregister_agent("agent")

    assert ceiling == AgentAccessGroupCeiling(
        access_group_ids=("g1",), models=frozenset({"gpt-5"}), mcp_server_ids=frozenset(), agent_ids=frozenset()
    )


@pytest.mark.asyncio
async def test_default_loader_treats_a_missing_group_as_unreadable(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints.auth.agent_access_groups import _load_access_group
    from litellm.proxy.auth import auth_checks

    async def missing_group(**_: object) -> LiteLLM_AccessGroupTable:
        raise HTTPException(status_code=404, detail={"error": "Access group doesn't exist in db."})

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(auth_checks, "get_access_object", missing_group)

    assert await _load_access_group("gone") is None


@pytest.mark.asyncio
async def test_default_loader_returns_nothing_without_a_db(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.agent_endpoints.auth.agent_access_groups import _load_access_group

    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert await _load_access_group("ag-1") is None
