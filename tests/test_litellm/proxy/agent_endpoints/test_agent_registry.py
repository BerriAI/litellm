"""Unit tests for AgentRegistry DB operations."""

import hashlib
import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry, GrantMigrationResult


def _sample_agent_card_params() -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "Test Agent",
        "description": "desc",
        "url": "http://localhost",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


@pytest.mark.asyncio
async def test_update_agent_in_db_clears_static_headers_and_extra_headers_when_omitted():
    """
    PUT (full-replace) should clear static_headers and extra_headers when omitted.
    Previously, omitting these fields left stale DB values intact.
    """
    registry = AgentRegistry()
    mock_prisma = MagicMock()

    # Simulate existing agent that had headers set
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {},
        "extra_headers": [],
        "object_permission": None,
    }
    updated_agent.object_permission = None

    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    # Agent config WITHOUT static_headers or extra_headers (omitted)
    agent_config = {
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
    }

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent=agent_config,
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    mock_update.assert_awaited_once()
    call_kwargs = mock_update.call_args.kwargs
    update_data = call_kwargs["data"]

    # Should include static_headers and extra_headers with empty defaults
    assert "static_headers" in update_data
    assert update_data["static_headers"] == "{}"
    assert "extra_headers" in update_data
    assert update_data["extra_headers"] == []


@pytest.mark.asyncio
async def test_update_agent_in_db_preserves_explicit_static_headers_and_extra_headers():
    """PUT with explicit values should still work correctly."""
    registry = AgentRegistry()
    mock_prisma = MagicMock()

    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {"Authorization": "Bearer xyz"},
        "extra_headers": ["X-Custom-Header"],
        "object_permission": None,
    }
    updated_agent.object_permission = None

    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    agent_config = {
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {"Authorization": "Bearer xyz"},
        "extra_headers": ["X-Custom-Header"],
    }

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent=agent_config,
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    call_kwargs = mock_update.call_args.kwargs
    update_data = call_kwargs["data"]

    assert update_data["static_headers"] == '{"Authorization": "Bearer xyz"}'
    assert update_data["extra_headers"] == ["X-Custom-Header"]


def test_load_agents_from_db_and_config_retains_previously_loaded_config_agents():
    """
    A DB reload calls this without an explicit agent_config. It resets the
    registry, so it must fall back to the agents remembered from config.yaml
    instead of dropping them.
    """
    registry = AgentRegistry()
    registry.load_agents_from_config(
        [
            {
                "agent_name": "config-agent",
                "agent_card_params": _sample_agent_card_params(),
            }
        ]
    )

    registry.load_agents_from_db_and_config(
        db_agents=[
            {
                "agent_id": "db-id",
                "agent_name": "db-agent",
                "agent_card_params": _sample_agent_card_params(),
            }
        ]
    )

    assert sorted(agent.agent_name for agent in registry.get_agent_list()) == [
        "config-agent",
        "db-agent",
    ]


def test_load_agents_from_db_and_config_skips_incomplete_config_entries():
    """Config entries missing agent_card_params are skipped, not registered half-built."""
    registry = AgentRegistry()
    registry.load_agents_from_config([{"agent_name": "no-card"}])

    registry.load_agents_from_db_and_config(db_agents=None)

    assert registry.get_agent_list() == ()


@pytest.mark.parametrize(
    "db_agent_names",
    [
        ("shared-name", "other-db-agent"),
        ("other-db-agent", "shared-name"),
    ],
    ids=["colliding-db-row-first", "colliding-db-row-last"],
)
def test_load_agents_from_db_and_config_lets_db_rows_win_a_name_collision(db_agent_names):
    """
    A config entry that reuses a DB agent's name must not be registered next to it.

    Name lookups and deregistration both address a single agent, so two entries
    under one name would shadow the DB row for routing and let a single delete
    drop both. Parametrized over both DB row orders because the registry is a
    list and a first-match lookup would otherwise pass on ordering luck.
    """
    registry = AgentRegistry()
    registry.load_agents_from_config(
        [
            {
                "agent_name": "shared-name",
                "agent_card_params": _sample_agent_card_params(),
            },
            {
                "agent_name": "config-only",
                "agent_card_params": _sample_agent_card_params(),
            },
        ]
    )

    registry.load_agents_from_db_and_config(
        db_agents=[
            {
                "agent_id": f"db-id-{name}",
                "agent_name": name,
                "agent_card_params": _sample_agent_card_params(),
            }
            for name in db_agent_names
        ]
    )

    registered = registry.get_agent_list()
    assert sorted(agent.agent_name for agent in registered) == [
        "config-only",
        "other-db-agent",
        "shared-name",
    ]

    shared = registry.get_agent_by_name("shared-name")
    assert shared is not None
    assert shared.agent_id == "db-id-shared-name", "the DB row must win the shared name, not the config entry"

    registry.deregister_agent("shared-name")
    assert [agent.agent_name for agent in registry.get_agent_list() if agent.agent_name == "shared-name"] == [], (
        "deleting the DB agent must not leave a shadowed config duplicate behind"
    )


def test_load_agents_from_config_skips_a_name_already_held_by_a_db_agent():
    """
    The one-agent-per-name rule is enforced by the loader, not by the call order.

    A config load that lands on a registry already holding DB rows must not append a
    colliding entry, otherwise the duplicate is reachable any time the two sources are
    loaded in the other order.
    """
    registry = AgentRegistry()
    registry.load_agents_from_db_and_config(
        db_agents=[
            {
                "agent_id": "db-id",
                "agent_name": "shared-name",
                "agent_card_params": _sample_agent_card_params(),
            }
        ]
    )

    registry.load_agents_from_config(
        [
            {
                "agent_name": "shared-name",
                "agent_card_params": _sample_agent_card_params(),
            }
        ]
    )

    assert [agent.agent_id for agent in registry.get_agent_list()] == ["db-id"]


def test_load_agents_from_config_registers_one_agent_per_name_within_the_config():
    """Two config entries sharing a name collapse to the first, keeping name lookups unambiguous."""
    registry = AgentRegistry()
    registry.load_agents_from_config(
        [
            {"agent_name": "dupe", "agent_card_params": _sample_agent_card_params()},
            {"agent_name": "dupe", "agent_card_params": {**_sample_agent_card_params(), "url": "http://second"}},
        ]
    )

    registered = registry.get_agent_list()
    assert len(registered) == 1
    assert registered[0].agent_card_params["url"] == "http://localhost"


def test_load_agents_from_config_with_an_empty_list_clears_the_remembered_agents():
    """
    An explicitly empty config must forget the previously loaded agents.

    Otherwise the next DB rebuild replays agents the operator removed from config.yaml.
    ``None`` still means "no opinion" and leaves them alone.
    """
    registry = AgentRegistry()
    registry.load_agents_from_config(
        [{"agent_name": "removed-agent", "agent_card_params": _sample_agent_card_params()}]
    )
    assert registry.config_agents != ()

    registry.load_agents_from_config([])
    assert registry.config_agents == ()

    registry.load_agents_from_db_and_config(db_agents=None)
    assert registry.get_agent_list() == (), "a removed config agent must not come back on the next rebuild"


def test_config_agent_id_survives_static_header_secret_rotation():
    """LIT-5144: the id was a hash of the whole entry, so rotating a static_headers secret silently
    re-identified the agent and orphaned every grant pointing at it."""
    base_entry: Final = {
        "agent_name": "rotating-agent",
        "agent_card_params": _sample_agent_card_params(),
        "static_headers": {"x-upstream-token": "token-v1"},
    }
    registry_v1: Final = AgentRegistry()
    registry_v1.load_agents_from_config([base_entry])
    agent_v1: Final = registry_v1.get_agent_by_name("rotating-agent")
    assert agent_v1 is not None

    registry_v2: Final = AgentRegistry()
    registry_v2.load_agents_from_config([{**base_entry, "static_headers": {"x-upstream-token": "token-v2"}}])
    agent_v2: Final = registry_v2.get_agent_by_name("rotating-agent")
    assert agent_v2 is not None

    assert agent_v1.agent_id == agent_v2.agent_id


def test_config_agent_ids_differ_when_only_the_agent_name_differs():
    """Two entries identical except for agent_name must not collapse onto one id."""
    registry: Final = AgentRegistry()
    registry.load_agents_from_config(
        [
            {"agent_name": "agent-a", "agent_card_params": _sample_agent_card_params()},
            {"agent_name": "agent-b", "agent_card_params": _sample_agent_card_params()},
        ]
    )

    ids: Final = {agent.agent_id for agent in registry.get_agent_list()}
    assert len(ids) == 2


def test_legacy_full_entry_hash_still_resolves_the_config_agent():
    """Grants and clients created before LIT-5144 hold the old full-entry hash; it must keep resolving."""
    entry: Final = {
        "agent_name": "legacy-agent",
        "agent_card_params": _sample_agent_card_params(),
        "static_headers": {"x-upstream-token": "token-v1"},
    }
    registry: Final = AgentRegistry()
    registry.load_agents_from_config([entry])
    agent: Final = registry.get_agent_by_name("legacy-agent")
    assert agent is not None

    legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    assert legacy_id != agent.agent_id
    assert registry.config_agent_legacy_ids[legacy_id] == agent.agent_id
    assert legacy_id in registry.ids_for_agent(agent.agent_id)
    assert agent.agent_id in registry.ids_for_agent(agent.agent_id)

    resolved: Final = registry.get_agent_by_id(legacy_id)
    assert resolved is not None
    assert resolved.agent_id == agent.agent_id
    assert registry.get_agent_by_id("nonexistent-id") is None


def test_public_agent_groups_holding_the_legacy_id_still_mark_the_config_agent_public(monkeypatch):
    """LIT-5144: config.yaml written before the fix stores the full-entry hash in
    public_agent_groups; the agent must stay public after its id became name-based."""
    import litellm

    entry: Final = {
        "agent_name": "public-agent",
        "agent_card_params": _sample_agent_card_params(),
    }
    registry: Final = AgentRegistry()
    registry.load_agents_from_config([entry])
    agent: Final = registry.get_agent_by_name("public-agent")
    assert agent is not None
    legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    assert legacy_id != agent.agent_id

    monkeypatch.setattr(litellm, "public_agent_groups", [legacy_id])
    assert [a.agent_id for a in registry.get_public_agent_list()] == [agent.agent_id]

    monkeypatch.setattr(litellm, "public_agent_groups", ["unrelated-id"])
    assert registry.get_public_agent_list() == ()

    monkeypatch.setattr(litellm, "public_agent_groups", None)
    assert registry.get_public_agent_list() == ()


@pytest.mark.asyncio
async def test_migrate_legacy_grant_ids_persists_stable_ids_into_grant_rows():
    """LIT-5144: the startup migration rewrites stored legacy full-entry hashes to the stable
    name id, so a later secret rotation (which re-mints the legacy hash) cannot orphan grants."""
    entry: Final = {
        "agent_name": "migrated-agent",
        "agent_card_params": _sample_agent_card_params(),
        "static_headers": {"x-upstream-token": "token-v1"},
    }
    registry: Final = AgentRegistry()
    registry.load_agents_from_config([entry])
    agent: Final = registry.get_agent_by_name("migrated-agent")
    assert agent is not None
    legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

    row: Final = SimpleNamespace(object_permission_id="op-1", agents=[legacy_id, "unrelated-id", agent.agent_id])
    table: Final = MagicMock()
    table.find_many = AsyncMock(return_value=[row])
    table.update_many = AsyncMock(return_value=1)

    assert await registry.migrate_legacy_grant_ids(table=table) == GrantMigrationResult(rewritten=1, missed=0)
    table.find_many.assert_awaited_once_with(where={"agents": {"has_some": (legacy_id,)}})
    table.update_many.assert_awaited_once_with(
        where={"object_permission_id": "op-1", "agents": {"equals": (legacy_id, "unrelated-id", agent.agent_id)}},
        data={"agents": (agent.agent_id, "unrelated-id")},
    )


@pytest.mark.asyncio
async def test_migrate_legacy_grant_ids_reports_compare_and_swap_misses():
    """A concurrently edited row makes the CAS update affect zero rows; the result must
    surface that as missed so the startup task knows to retry instead of reporting success."""
    entry: Final = {
        "agent_name": "contended-agent",
        "agent_card_params": _sample_agent_card_params(),
        "static_headers": {"x-upstream-token": "token-v1"},
    }
    registry: Final = AgentRegistry()
    registry.load_agents_from_config([entry])
    legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

    row: Final = SimpleNamespace(object_permission_id="op-1", agents=[legacy_id])
    table: Final = MagicMock()
    table.find_many = AsyncMock(return_value=[row])
    table.update_many = AsyncMock(return_value=0)

    assert await registry.migrate_legacy_grant_ids(table=table) == GrantMigrationResult(rewritten=0, missed=1)


@pytest.mark.asyncio
async def test_migrate_legacy_grant_ids_no_ops_without_config_agents():
    """Without config agents there are no legacy hashes to translate, so the DB is never queried."""
    registry: Final = AgentRegistry()
    table: Final = MagicMock()
    table.find_many = AsyncMock()

    assert await registry.migrate_legacy_grant_ids(table=table) == GrantMigrationResult(rewritten=0, missed=0)
    table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_agent_in_db_raises_when_row_deleted_mid_update():
    """Prisma's update returns None when the row vanished between read and write. Without a
    guard the code dereferences None and reports an opaque AttributeError instead of the id."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()
    mock_prisma.db.litellm_agentstable.update = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Error updating agent in DB") as exc_info:
        await registry.update_agent_in_db(
            agent_id="agent-123",
            agent={
                "agent_name": "Updated Agent",
                "agent_card_params": _sample_agent_card_params(),
                "litellm_params": {},
            },
            prisma_client=mock_prisma,
            updated_by="test-user",
        )

    assert str(exc_info.value) == "Error updating agent in DB: Agent not found, passed agent_id=agent-123"


@pytest.mark.asyncio
async def test_patch_agent_in_db_raises_when_row_deleted_mid_update():
    """Same race on PATCH: the existing row is read, then deleted before the update lands."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()
    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value={"agent_id": "agent-123", "agent_name": "Old Agent", "object_permission_id": None}
    )
    mock_prisma.db.litellm_agentstable.update = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Error patching agent in DB") as exc_info:
        await registry.patch_agent_in_db(
            agent_id="agent-123",
            agent={"agent_name": "Patched Agent"},
            prisma_client=mock_prisma,
            updated_by="test-user",
        )

    assert str(exc_info.value) == "Error patching agent in DB: Agent not found, passed agent_id=agent-123"


@pytest.mark.asyncio
async def test_delete_agent_from_db_raises_when_row_already_gone():
    """Prisma's delete returns None for a missing row, which dict() cannot consume."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()
    mock_prisma.db.litellm_agentstable.delete = AsyncMock(return_value=None)

    with pytest.raises(Exception, match="Error deleting agent from DB") as exc_info:
        await registry.delete_agent_from_db(agent_id="agent-123", prisma_client=mock_prisma)

    assert str(exc_info.value) == "Error deleting agent from DB: Agent not found, passed agent_id=agent-123"
