"""Unit tests for AgentRegistry DB operations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry


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

    assert registry.get_agent_list() == []


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
    assert registry.get_agent_list() == [], "a removed config agent must not come back on the next rebuild"
