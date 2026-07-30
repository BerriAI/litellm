"""
Unit tests for PolicyRegistry - config vs DB provenance of in-memory policies.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.policy_engine.policy_registry import PolicyRegistry


def _make_row(policy_name, guardrails_add, version_status="production", policy_id="pid-1"):
    row = MagicMock()
    row.policy_id = policy_id
    row.policy_name = policy_name
    row.version_number = 1
    row.version_status = version_status
    row.parent_version_id = None
    row.is_latest = True
    row.published_at = None
    row.production_at = None
    row.inherit = None
    row.description = None
    row.guardrails_add = guardrails_add
    row.guardrails_remove = []
    row.condition = None
    row.pipeline = None
    row.created_at = datetime.now(timezone.utc)
    row.updated_at = datetime.now(timezone.utc)
    row.created_by = None
    row.updated_by = None
    return row


def _prisma_with_rows(*rows):
    prisma = MagicMock()
    prisma.db.litellm_policytable.find_many = AsyncMock(
        side_effect=lambda where=None, order=None: [
            row
            for row in rows
            if where is None
            or (
                row.version_status == where.get("version_status")
                if isinstance(where.get("version_status"), str)
                else row.version_status in where.get("version_status", {}).get("in", [])
            )
        ]
    )
    return prisma


class TestConfigPoliciesSurviveDbSync:
    """A DB-connected proxy must keep enforcing policies defined in config.yaml."""

    @pytest.mark.asyncio
    async def test_config_policy_survives_sync(self):
        registry = PolicyRegistry()
        registry.load_policies({"config-policy": {"guardrails": {"add": ["tooling"]}}})

        await registry.sync_policies_from_db(_prisma_with_rows(_make_row("db-policy", ["pii"])))

        config_policy = registry.get_policy("config-policy")
        assert config_policy is not None
        assert config_policy.guardrails.get_add() == ["tooling"]
        assert registry.get_policy_source("config-policy") == "config"

        db_policy = registry.get_policy("db-policy")
        assert db_policy is not None
        assert db_policy.guardrails.get_add() == ["pii"]
        assert registry.get_policy_source("db-policy") == "db"

    @pytest.mark.asyncio
    async def test_config_policy_survives_repeated_syncs(self):
        registry = PolicyRegistry()
        registry.load_policies({"config-policy": {"guardrails": {"add": ["tooling"]}}})
        prisma = _prisma_with_rows(_make_row("db-policy", ["pii"]))

        await registry.sync_policies_from_db(prisma)
        await registry.sync_policies_from_db(prisma)

        assert registry.has_policy("config-policy")
        assert registry.get_policy_names().count("db-policy") == 1

    @pytest.mark.asyncio
    async def test_db_policy_shadows_config_policy_of_same_name(self):
        registry = PolicyRegistry()
        registry.load_policies({"shared": {"guardrails": {"add": ["from-config"]}}})

        await registry.sync_policies_from_db(_prisma_with_rows(_make_row("shared", ["from-db"])))

        policy = registry.get_policy("shared")
        assert policy is not None
        assert policy.guardrails.get_add() == ["from-db"]
        assert registry.get_policy_source("shared") == "db"
        assert "shared" not in registry.get_config_policies()


class TestGetConfigPolicies:
    def test_reports_config_policies(self):
        registry = PolicyRegistry()
        registry.load_policies({"config-policy": {"description": "d", "guardrails": {"add": ["tooling"]}}})

        config_policies = registry.get_config_policies()

        assert list(config_policies) == ["config-policy"]
        assert config_policies["config-policy"].description == "d"

    def test_empty_before_any_config_is_loaded(self):
        assert PolicyRegistry().get_config_policies() == {}

    def test_source_is_none_for_unknown_policy(self):
        assert PolicyRegistry().get_policy_source("nope") is None
