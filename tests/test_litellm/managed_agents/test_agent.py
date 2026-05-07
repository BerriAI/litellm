"""
Unit tests for ``litellm.managed_agents.Agent``.

Covers the lifecycle helpers (``from_db_row``, ``create_session``,
``get_session``, ``list_sessions``, ``delete``) end-to-end against the
in-memory Prisma stand-in. The runtime + sandbox here are minimal stubs
since these tests focus on the DB orchestration, not LLM behaviour.
"""

import asyncio
from typing import Any, AsyncIterator, Dict

import pytest

from litellm.managed_agents.agent import Agent
from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import EVENT_TYPE_RUN_FINISHED, Event
from litellm.managed_agents.sandbox.local import LocalSandbox


class _StubRuntime(AgentRuntime):
    """Minimal AgentRuntime that immediately yields run_finished."""

    async def run(
        self,
        prompt: str,
        sandbox,
        session_state: SessionState,
        agent_config: AgentConfig,
    ) -> AsyncIterator[Event]:
        yield Event(
            type=EVENT_TYPE_RUN_FINISHED,
            data={"result": f"echo: {prompt}"},
        )


def _seed_agent_row(db, *, agent_id: str = "agent_1", **overrides: Any):
    """Insert a LiteLLM_Agent row directly into the fake DB."""
    return asyncio.get_event_loop().run_until_complete(
        db.litellm_agent.create(
            data={
                "id": agent_id,
                "name": overrides.get("name", "test-agent"),
                "model": overrides.get("model", "gpt-4o-mini"),
                "user_api_key_hash": overrides.get("user_api_key_hash", "hash-A"),
                "team_id": overrides.get("team_id", None),
                "system_prompt": overrides.get("system_prompt", "you are helpful"),
                "tools_config": overrides.get("tools_config", None),
                "metadata": overrides.get("metadata", {}),
                "default_repos": overrides.get("default_repos", []),
                "default_env_vars": overrides.get("default_env_vars", {}),
            }
        )
    )


@pytest.mark.asyncio
async def test_from_db_row_populates_fields(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={
            "id": "agent_x",
            "name": "alpha",
            "model": "claude-sonnet-4",
            "system_prompt": "hi",
            "user_api_key_hash": "h",
            "tools_config": {"tools": []},
            "metadata": {"k": "v"},
            "default_repos": [{"url": "a", "ref": "main"}],
            "default_env_vars": {"NPM": "1"},
            "team_id": "team-1",
        }
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    assert agent.id == "agent_x"
    assert agent.name == "alpha"
    assert agent.model == "claude-sonnet-4"
    assert agent.system_prompt == "hi"
    assert agent.tools_config == {"tools": []}
    assert agent.metadata == {"k": "v"}
    assert agent.default_repos == [{"url": "a", "ref": "main"}]
    assert agent.default_env_vars == {"NPM": "1"}
    assert agent.team_id == "team-1"
    assert agent.user_api_key_hash == "h"


@pytest.mark.asyncio
async def test_create_session_inserts_row_in_ready(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={
            "id": "agent_1",
            "name": "test",
            "model": "gpt-4o-mini",
            "user_api_key_hash": "h",
            "default_repos": [{"url": "default-repo"}],
            "default_env_vars": {"DEFAULT": "x"},
        }
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )

    session = await agent.create_session(env_vars={"OVERRIDE": "y"})
    assert session.id.startswith("sess_")
    assert session.agent_id == "agent_1"
    assert session.status == "ready"
    # Defaults flow through; overrides win on collisions.
    assert session.env_vars == {"DEFAULT": "x", "OVERRIDE": "y"}
    # repos default carried through (no override).
    assert session.repos == [{"url": "default-repo"}]
    # daemon token returned exactly once at create.
    assert isinstance(session.daemon_token, str) and session.daemon_token

    # Same row should be persisted in the fake DB.
    persisted = await fake_db.db.litellm_agentsession.find_unique(
        where={"id": session.id}
    )
    assert persisted is not None
    assert persisted.status == "ready"


@pytest.mark.asyncio
async def test_create_session_with_repos_replaces_defaults(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={
            "id": "agent_1",
            "name": "x",
            "model": "m",
            "user_api_key_hash": "h",
            "default_repos": [{"url": "default"}],
        }
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    session = await agent.create_session(repos=[{"url": "override"}])
    assert session.repos == [{"url": "override"}]


@pytest.mark.asyncio
async def test_get_session_returns_existing_row(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={"id": "agent_1", "name": "x", "model": "m", "user_api_key_hash": "h"}
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    s1 = await agent.create_session()
    s2 = await agent.get_session(s1.id)
    assert s2.id == s1.id
    assert s2.agent_id == "agent_1"


@pytest.mark.asyncio
async def test_get_session_raises_for_unknown_id(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={"id": "agent_1", "name": "x", "model": "m", "user_api_key_hash": "h"}
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    with pytest.raises(LookupError):
        await agent.get_session("sess_does_not_exist")


@pytest.mark.asyncio
async def test_list_sessions_returns_all_for_agent(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={"id": "agent_1", "name": "x", "model": "m", "user_api_key_hash": "h"}
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    s_a = await agent.create_session()
    s_b = await agent.create_session()

    sessions = await agent.list_sessions()
    ids = {s.id for s in sessions}
    assert ids == {s_a.id, s_b.id}


@pytest.mark.asyncio
async def test_create_session_requires_runtime_and_sandbox(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={"id": "agent_1", "name": "x", "model": "m", "user_api_key_hash": "h"}
    )
    agent = await Agent.from_db_row(row, db=fake_db.db)  # no runtime/sandbox
    with pytest.raises(RuntimeError, match="runtime and sandbox"):
        await agent.create_session()


@pytest.mark.asyncio
async def test_delete_removes_agent_row(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={"id": "agent_1", "name": "x", "model": "m", "user_api_key_hash": "h"}
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    await agent.delete()
    assert await fake_db.db.litellm_agent.find_unique(where={"id": "agent_1"}) is None


@pytest.mark.asyncio
async def test_to_runtime_config_projects_subset(fake_db):
    row = await fake_db.db.litellm_agent.create(
        data={
            "id": "agent_1",
            "name": "x",
            "model": "m",
            "user_api_key_hash": "h",
            "system_prompt": "sp",
            "tools_config": {"tools": []},
            "metadata": {"a": 1},
        }
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=_StubRuntime(), sandbox=LocalSandbox()
    )
    cfg = agent.to_runtime_config()
    assert cfg.name == "x"
    assert cfg.model == "m"
    assert cfg.system_prompt == "sp"
    assert cfg.tools_config == {"tools": []}
    assert cfg.metadata == {"a": 1}
