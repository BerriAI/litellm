from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
    AgentAccessGroupCeiling,
    agent_access_group_ids_cache_key,
    load_agent_access_group_ids,
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


@dataclass(frozen=True, slots=True)
class _AgentRow:
    access_group_ids: Sequence[str] | None


class _FakeAgentTable:
    def __init__(self, rows: dict[str, _AgentRow], failing: bool = False) -> None:
        self._rows: Final = rows
        self._failing: Final = failing
        self.reads = 0

    async def find_agent(self, agent_id: str) -> _AgentRow | None:
        self.reads += 1
        if self._failing:
            raise RuntimeError("db down")
        return self._rows.get(agent_id)


async def _registry_snapshot(agent_id: str) -> tuple[str, ...]:
    return ("registry-group",)


@pytest.mark.asyncio
async def test_agent_row_is_read_once_then_served_from_cache():
    cache: Final = DualCache()
    table: Final = _FakeAgentTable({"agent-1": _AgentRow(["g1", "g2"])})

    first: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)
    second: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)

    assert (first, second, table.reads) == (("g1", "g2"), ("g1", "g2"), 1)


@pytest.mark.asyncio
async def test_agent_with_no_row_or_no_groups_caches_an_empty_answer():
    cache: Final = DualCache()
    table: Final = _FakeAgentTable({"bare": _AgentRow(None)})

    bare: Final = await load_agent_access_group_ids("bare", cache, table.find_agent, _registry_snapshot)
    missing: Final = await load_agent_access_group_ids("missing", cache, table.find_agent, _registry_snapshot)
    again: Final = await load_agent_access_group_ids("missing", cache, table.find_agent, _registry_snapshot)

    assert (bare, missing, again, table.reads) == ((), (), (), 2)


@pytest.mark.asyncio
async def test_evicted_cache_entry_picks_up_the_patched_row():
    cache: Final = DualCache()
    rows: Final = {"agent-1": _AgentRow(["g1"])}
    table: Final = _FakeAgentTable(rows)
    await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)

    rows["agent-1"] = _AgentRow(["g2"])
    stale: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)
    await cache.async_delete_cache(key=agent_access_group_ids_cache_key("agent-1"))
    fresh: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)

    assert (stale, fresh) == (("g1",), ("g2",))


@pytest.mark.asyncio
async def test_unreadable_row_falls_back_to_the_registry_without_caching():
    cache: Final = DualCache()
    table: Final = _FakeAgentTable({}, failing=True)

    answer: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)

    assert answer == ("registry-group",)
    assert await cache.async_get_cache(key=agent_access_group_ids_cache_key("agent-1")) is None


@pytest.mark.asyncio
async def test_garbage_in_the_cache_is_treated_as_a_miss():
    cache: Final = DualCache()
    await cache.async_set_cache(key=agent_access_group_ids_cache_key("agent-1"), value={"not": "a list"})
    table: Final = _FakeAgentTable({"agent-1": _AgentRow(["g1"])})

    answer: Final = await load_agent_access_group_ids("agent-1", cache, table.find_agent, _registry_snapshot)

    assert (answer, table.reads) == (("g1",), 1)


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
