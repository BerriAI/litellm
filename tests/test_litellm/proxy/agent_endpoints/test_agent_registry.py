"""Unit tests for AgentRegistry DB operations."""

import hashlib
import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.proxy.agent_endpoints.agent_registry import (
    AgentRegistry,
    GrantMigrationResult,
    _restore_redacted_litellm_params,
    redact_sensitive_agent_litellm_params,
)

# Obviously-fake stand-ins for a real AWS credential pair (LIT-6736 regression
# fixtures) -- never a real key shape, and must never appear in any response.
SENTINEL_AWS_ACCESS_KEY_ID: Final = "AKIATESTSENTINEL0000"
SENTINEL_AWS_SECRET_ACCESS_KEY: Final = "test-sentinel-do-not-use-secret-value"


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
    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

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
    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

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
    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(litellm_params={}, object_permission_id=None)
    )
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


# ---------- LIT-6736: agent litellm_params secret redaction ----------


def test_redact_sensitive_agent_litellm_params_masks_secrets_keeps_the_rest():
    """The sentinel secret must never appear in the redacted output; non-secret
    keys (model reference, is_public) must survive untouched."""
    redacted = redact_sensitive_agent_litellm_params(
        {
            "aws_access_key_id": SENTINEL_AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY,
            "model": "bedrock/agentcore/my-agent",
            "is_public": True,
        }
    )

    assert SENTINEL_AWS_ACCESS_KEY_ID not in json.dumps(redacted)
    assert SENTINEL_AWS_SECRET_ACCESS_KEY not in json.dumps(redacted)
    assert redacted["aws_access_key_id"] == REDACTED_BY_LITELM_STRING
    assert redacted["aws_secret_access_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["model"] == "bedrock/agentcore/my-agent"
    assert redacted["is_public"] is True


def test_redact_sensitive_agent_litellm_params_recurses_into_nested_dicts():
    """A secret nested one level down (e.g. a per-provider sub-config) must
    also be redacted, not just top-level keys."""
    redacted = redact_sensitive_agent_litellm_params(
        {"provider_config": {"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "region": "us-east-1"}}
    )

    assert SENTINEL_AWS_SECRET_ACCESS_KEY not in json.dumps(redacted)
    assert redacted["provider_config"]["aws_secret_access_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["provider_config"]["region"] == "us-east-1"


def test_redact_sensitive_agent_litellm_params_handles_none_and_json_string():
    assert redact_sensitive_agent_litellm_params(None) is None

    serialized = json.dumps({"api_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "model": "gpt-4"})
    redacted = redact_sensitive_agent_litellm_params(serialized)

    assert SENTINEL_AWS_SECRET_ACCESS_KEY not in redacted
    assert json.loads(redacted)["api_key"] == REDACTED_BY_LITELM_STRING
    assert json.loads(redacted)["model"] == "gpt-4"


def test_redact_sensitive_agent_litellm_params_recurses_into_lists_of_dicts():
    """A secret nested inside a list of provider sub-configs (a shape a
    non-sensitively-named key can legitimately hold) must also be redacted,
    not silently returned as-is."""
    redacted = redact_sensitive_agent_litellm_params(
        {
            "provider_configs": [
                {"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "region": "us-east-1"},
                {"aws_secret_access_key": "other-" + SENTINEL_AWS_SECRET_ACCESS_KEY, "region": "us-west-2"},
            ]
        }
    )

    assert SENTINEL_AWS_SECRET_ACCESS_KEY not in json.dumps(redacted)
    assert redacted["provider_configs"][0]["aws_secret_access_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["provider_configs"][0]["region"] == "us-east-1"
    assert redacted["provider_configs"][1]["aws_secret_access_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["provider_configs"][1]["region"] == "us-west-2"


def test_redact_sensitive_agent_litellm_params_redacts_secrets_inside_model_list():
    """The exact shape flagged in review: litellm_params.model_list, where each
    entry carries its own nested litellm_params with a provider credential."""
    redacted = redact_sensitive_agent_litellm_params(
        {
            "model_list": [
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"api_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "model": "gpt-4"},
                },
                {
                    "model_name": "claude",
                    "litellm_params": {
                        "aws_secret_access_key": "other-" + SENTINEL_AWS_SECRET_ACCESS_KEY,
                        "model": "bedrock/claude",
                    },
                },
            ]
        }
    )

    assert SENTINEL_AWS_SECRET_ACCESS_KEY not in json.dumps(redacted)
    assert redacted["model_list"][0]["litellm_params"]["api_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["model_list"][0]["litellm_params"]["model"] == "gpt-4"
    assert redacted["model_list"][1]["litellm_params"]["aws_secret_access_key"] == REDACTED_BY_LITELM_STRING
    assert redacted["model_list"][1]["litellm_params"]["model"] == "bedrock/claude"


def test_restore_redacted_litellm_params_preserves_secret_inside_model_list():
    """The write-side counterpart: a caller echoing the model_list back
    unchanged, with its nested secret masked, while editing a sibling
    top-level field, must not corrupt the stored per-deployment credential."""
    existing = {
        "agent_name": "my-agent",
        "model_list": [
            {
                "model_name": "gpt-4",
                "litellm_params": {"api_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "model": "gpt-4"},
            },
        ],
    }
    incoming = {
        "agent_name": "my-agent-renamed",
        "model_list": [
            {
                "model_name": "gpt-4",
                "litellm_params": {"api_key": REDACTED_BY_LITELM_STRING, "model": "gpt-4"},
            },
        ],
    }

    restored = _restore_redacted_litellm_params(incoming, existing)

    assert SENTINEL_AWS_SECRET_ACCESS_KEY == restored["model_list"][0]["litellm_params"]["api_key"]
    assert restored["agent_name"] == "my-agent-renamed"


def test_restore_redacted_litellm_params_does_not_misassign_across_reordered_list_entries():
    """If the model_list entries no longer match up (e.g. reordered, or the
    entry actually changed), position-based restoration must not attach one
    entry's credential to a different entry -- the caller's own value (even
    the literal marker) is used instead of guessing."""
    existing = {
        "model_list": [
            {"model_name": "gpt-4", "litellm_params": {"api_key": SENTINEL_AWS_SECRET_ACCESS_KEY}},
            {"model_name": "claude", "litellm_params": {"api_key": "other-" + SENTINEL_AWS_SECRET_ACCESS_KEY}},
        ],
    }
    incoming = {
        "model_list": [
            # Same index (0) now holds what used to be at index 1's entry.
            {"model_name": "claude", "litellm_params": {"api_key": REDACTED_BY_LITELM_STRING}},
        ],
    }

    restored = _restore_redacted_litellm_params(incoming, existing)

    # Must NOT have pulled index 0's ("gpt-4") credential onto "claude": either
    # dropped entirely (nothing to safely restore from) or left as the
    # caller's own value, never another entry's real secret.
    assert restored["model_list"][0]["litellm_params"].get("api_key") != SENTINEL_AWS_SECRET_ACCESS_KEY


def test_restore_redacted_litellm_params_recovers_a_whole_subtree_collapsed_by_the_depth_cap():
    """Past the read-side recursion depth cap, a whole nested subtree is
    collapsed to the flat REDACTED_BY_LITELM marker rather than a dict/list.
    If the caller echoes that flat marker back unchanged, the whole
    subtree -- not just the literal marker string -- must be restored."""
    existing_subtree = {"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY, "region": "us-east-1"}
    incoming = {"provider_config": REDACTED_BY_LITELM_STRING}
    existing = {"provider_config": existing_subtree}

    restored = _restore_redacted_litellm_params(incoming, existing)

    assert restored["provider_config"] == existing_subtree


def test_redact_sensitive_agent_litellm_params_does_not_reinterpret_plain_string_values_as_json():
    """A plain non-JSON string value (most string leaves) must pass through
    unchanged rather than failing to parse and getting redacted."""
    redacted = redact_sensitive_agent_litellm_params({"model": "bedrock/agentcore/my-agent", "is_public": True})

    assert redacted["model"] == "bedrock/agentcore/my-agent"
    assert redacted["is_public"] is True


@pytest.mark.asyncio
async def test_add_agent_to_db_drops_a_sentinel_value_instead_of_storing_the_placeholder():
    """A create has nothing stored to restore behind a redaction marker, so a
    sensitive key submitted as the literal marker is dropped rather than
    persisted as the placeholder string itself."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()
    created_agent = MagicMock()
    created_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    created_agent.object_permission = None
    mock_create = AsyncMock(return_value=created_agent)
    mock_prisma.db.litellm_agentstable.create = mock_create

    await registry.add_agent_to_db(
        agent={
            "agent_name": "Test Agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {
                "aws_secret_access_key": REDACTED_BY_LITELM_STRING,
                "model": "bedrock/agentcore/my-agent",
            },
        },
        prisma_client=mock_prisma,
        created_by="test-user",
    )

    stored_params: Final = json.loads(mock_create.call_args.kwargs["data"]["litellm_params"])
    assert "aws_secret_access_key" not in stored_params
    assert stored_params["model"] == "bedrock/agentcore/my-agent"


@pytest.mark.asyncio
async def test_update_agent_in_db_preserves_secret_when_echoed_back_redacted():
    """PUT round-trips the GET response, which shows the secret redacted. Saving
    an unrelated field change must not overwrite the real stored credential
    with the redaction marker."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            litellm_params={
                "aws_access_key_id": SENTINEL_AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY,
                "model": "bedrock/agentcore/my-agent",
            },
            object_permission_id=None,
        )
    )
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Renamed Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    updated_agent.object_permission = None
    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent={
            "agent_name": "Renamed Agent",
            "agent_card_params": _sample_agent_card_params(),
            # The UI round-tripped the redacted secret and the untouched
            # access key id verbatim; only agent_name actually changed.
            "litellm_params": {
                "aws_access_key_id": SENTINEL_AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": REDACTED_BY_LITELM_STRING,
                "model": "bedrock/agentcore/my-agent",
            },
        },
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    stored_params: Final = json.loads(mock_update.call_args.kwargs["data"]["litellm_params"])
    assert stored_params["aws_secret_access_key"] == SENTINEL_AWS_SECRET_ACCESS_KEY
    assert stored_params["aws_access_key_id"] == SENTINEL_AWS_ACCESS_KEY_ID
    assert stored_params["model"] == "bedrock/agentcore/my-agent"


@pytest.mark.asyncio
async def test_update_agent_in_db_preserves_secret_when_key_omitted_entirely():
    """Omitting the sensitive key altogether must fall back to the stored
    value too, not just an explicit redaction-marker round-trip."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            litellm_params={"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY},
            object_permission_id=None,
        )
    )
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    updated_agent.object_permission = None
    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent={
            "agent_name": "Test Agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {"model": "bedrock/agentcore/my-agent"},
        },
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    stored_params: Final = json.loads(mock_update.call_args.kwargs["data"]["litellm_params"])
    assert stored_params["aws_secret_access_key"] == SENTINEL_AWS_SECRET_ACCESS_KEY


@pytest.mark.asyncio
async def test_update_agent_in_db_preserves_secret_nested_under_a_non_sensitive_key():
    """A secret nested inside a dict held by a non-sensitively-named key
    (e.g. a per-provider sub-config) must also survive an echoed-back
    redaction marker, not just top-level secret keys."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            litellm_params={
                "provider_config": {
                    "aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY,
                    "region": "us-east-1",
                }
            },
            object_permission_id=None,
        )
    )
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    updated_agent.object_permission = None
    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent={
            "agent_name": "Test Agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {
                # The GET response redacted the nested secret; the caller
                # round-trips it verbatim while changing nothing.
                "provider_config": {
                    "aws_secret_access_key": REDACTED_BY_LITELM_STRING,
                    "region": "us-west-2",
                }
            },
        },
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    stored_params: Final = json.loads(mock_update.call_args.kwargs["data"]["litellm_params"])
    assert stored_params["provider_config"]["aws_secret_access_key"] == SENTINEL_AWS_SECRET_ACCESS_KEY
    assert stored_params["provider_config"]["region"] == "us-west-2"


@pytest.mark.asyncio
async def test_update_agent_in_db_clears_secret_on_explicit_empty_value():
    """An explicit empty string is a deliberate clear, distinct from an omitted
    key or the redaction marker, and must actually clear the stored secret."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            litellm_params={"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY},
            object_permission_id=None,
        )
    )
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    updated_agent.object_permission = None
    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent={
            "agent_name": "Test Agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {"aws_secret_access_key": ""},
        },
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    stored_params: Final = json.loads(mock_update.call_args.kwargs["data"]["litellm_params"])
    assert stored_params["aws_secret_access_key"] == ""


@pytest.mark.asyncio
async def test_patch_agent_in_db_preserves_secret_when_litellm_params_omitted():
    """A PATCH that only renames the agent must not touch (let alone drop) the
    stored litellm_params secret."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value={
            "agent_id": "agent-123",
            "agent_name": "Old Name",
            "litellm_params": {"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY},
            "object_permission_id": None,
        }
    )
    patched_agent = MagicMock()
    patched_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "New Name",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {"aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY},
        "object_permission": None,
    }
    patched_agent.object_permission = None
    mock_update = AsyncMock(return_value=patched_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.patch_agent_in_db(
        agent_id="agent-123",
        agent={"agent_name": "New Name"},
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    update_data: Final = mock_update.call_args.kwargs["data"]
    assert "litellm_params" not in update_data


@pytest.mark.asyncio
async def test_patch_agent_in_db_preserves_secret_when_echoed_back_redacted():
    """A PATCH that includes litellm_params (e.g. to flip an unrelated flag)
    with the secret round-tripped as the redaction marker must not clobber
    the stored credential."""
    registry: Final = AgentRegistry()
    mock_prisma: Final = MagicMock()

    mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(
        return_value={
            "agent_id": "agent-123",
            "agent_name": "Test Agent",
            "litellm_params": {
                "aws_secret_access_key": SENTINEL_AWS_SECRET_ACCESS_KEY,
                "is_public": False,
            },
            "object_permission_id": None,
        }
    )
    patched_agent = MagicMock()
    patched_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "object_permission": None,
    }
    patched_agent.object_permission = None
    mock_update = AsyncMock(return_value=patched_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    await registry.patch_agent_in_db(
        agent_id="agent-123",
        agent={
            "litellm_params": {
                "aws_secret_access_key": REDACTED_BY_LITELM_STRING,
                "is_public": True,
            }
        },
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    stored_params: Final = json.loads(mock_update.call_args.kwargs["data"]["litellm_params"])
    assert stored_params["aws_secret_access_key"] == SENTINEL_AWS_SECRET_ACCESS_KEY
    assert stored_params["is_public"] is True
