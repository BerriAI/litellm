"""
Unit tests for policy_engine/policy_endpoints.py list endpoints.

Regression tests for issue #35255: config-defined policies and attachments must be
returned by the list endpoints (marked definition_location="config"), DB rows must keep
their exact shape, and the endpoints must not 500 when no database is connected.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.policy_engine.policy_endpoints as policy_endpoints
from litellm.proxy.policy_engine.attachment_registry import AttachmentRegistry
from litellm.proxy.policy_engine.policy_registry import PolicyRegistry


def _make_policy_row(
    policy_id="uuid-1",
    policy_name="db-policy",
    version_status="production",
    guardrails_add=None,
):
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
    row.description = "db description"
    row.guardrails_add = guardrails_add or []
    row.guardrails_remove = []
    row.condition = None
    row.pipeline = None
    row.created_at = datetime.now(timezone.utc)
    row.updated_at = datetime.now(timezone.utc)
    row.created_by = "admin"
    row.updated_by = "admin"
    return row


def _make_attachment_row(attachment_id="att-1", policy_name="db-policy", scope="*"):
    row = MagicMock()
    row.attachment_id = attachment_id
    row.policy_name = policy_name
    row.scope = scope
    row.teams = []
    row.keys = []
    row.models = []
    row.tags = []
    row.created_at = datetime.now(timezone.utc)
    row.updated_at = datetime.now(timezone.utc)
    row.created_by = "admin"
    row.updated_by = "admin"
    return row


@pytest.fixture
def policy_registry(monkeypatch):
    registry = PolicyRegistry()
    monkeypatch.setattr(policy_endpoints, "get_policy_registry", lambda: registry)
    return registry


@pytest.fixture
def attachment_registry(monkeypatch):
    registry = AttachmentRegistry()
    monkeypatch.setattr(policy_endpoints, "get_attachment_registry", lambda: registry)
    return registry


def _set_prisma(monkeypatch, prisma):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)


class TestListPoliciesIncludesConfig:
    @pytest.mark.asyncio
    async def test_returns_config_policies_without_prisma(self, policy_registry, monkeypatch):
        _set_prisma(monkeypatch, None)
        policy_registry.load_policies(
            {"config-policy": {"description": "from config", "guardrails": {"add": ["tooling"]}}}
        )

        response = await policy_endpoints.list_policies()

        assert response.total_count == 1
        entry = response.policies[0]
        assert entry.policy_name == "config-policy"
        assert entry.policy_id == "config-policy"
        assert entry.definition_location == "config"
        assert entry.version_status == "production"
        assert entry.guardrails_add == ["tooling"]
        assert entry.description == "from config"
        assert entry.created_at is None

    @pytest.mark.asyncio
    async def test_merges_db_rows_with_config_and_keeps_db_row_shape(self, policy_registry, monkeypatch):
        row = _make_policy_row(policy_id="uuid-1", policy_name="db-policy", guardrails_add=["db-guard"])
        prisma = MagicMock()
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[row])
        _set_prisma(monkeypatch, prisma)
        policy_registry.load_policies({"config-policy": {"guardrails": {"add": ["tooling"]}}})

        response = await policy_endpoints.list_policies()

        assert response.total_count == 2
        db_entry = next(p for p in response.policies if p.policy_name == "db-policy")
        assert db_entry.definition_location == "db"
        assert db_entry.policy_id == "uuid-1"
        assert db_entry.guardrails_add == ["db-guard"]
        assert db_entry.description == "db description"
        assert db_entry.created_at == row.created_at
        assert db_entry.created_by == "admin"
        config_entry = next(p for p in response.policies if p.policy_name == "config-policy")
        assert config_entry.definition_location == "config"

    @pytest.mark.asyncio
    async def test_db_policy_shadows_config_policy_with_same_name(self, policy_registry, monkeypatch):
        row = _make_policy_row(policy_id="uuid-1", policy_name="shared-name", guardrails_add=["db-guard"])
        prisma = MagicMock()
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[row])
        _set_prisma(monkeypatch, prisma)
        policy_registry.load_policies({"shared-name": {"guardrails": {"add": ["config-guard"]}}})

        response = await policy_endpoints.list_policies()

        assert response.total_count == 1
        assert response.policies[0].definition_location == "db"
        assert response.policies[0].guardrails_add == ["db-guard"]

    @pytest.mark.asyncio
    async def test_draft_db_policy_does_not_hide_enforced_config_policy(self, policy_registry, monkeypatch):
        """
        Runtime sync only lets production DB versions override a config policy,
        so a draft or published DB version sharing the name must not suppress
        the config entry: the config version is still the one being enforced,
        and hiding it makes the list API disagree with actual enforcement.
        """
        row = _make_policy_row(
            policy_id="uuid-1", policy_name="shared-name", version_status="draft", guardrails_add=["db-guard"]
        )
        prisma = MagicMock()
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[row])
        _set_prisma(monkeypatch, prisma)
        policy_registry.load_policies({"shared-name": {"guardrails": {"add": ["config-guard"]}}})

        response = await policy_endpoints.list_policies()

        assert response.total_count == 2
        config_entry = next(p for p in response.policies if p.definition_location == "config")
        assert config_entry.policy_name == "shared-name"
        assert config_entry.version_status == "production"
        assert config_entry.guardrails_add == ["config-guard"]
        db_entry = next(p for p in response.policies if p.definition_location == "db")
        assert db_entry.version_status == "draft"

    @pytest.mark.asyncio
    async def test_stale_registry_provenance_does_not_hide_config_policy(self, policy_registry, monkeypatch):
        """
        Another proxy instance can delete or demote the production DB override
        between registry syncs. The endpoint's fresh DB query is the source of
        truth for conflicts; stale in-memory provenance from the last sync must
        not suppress the config entry once no production override exists.
        """
        policy_registry.load_policies({"shared-name": {"guardrails": {"add": ["config-guard"]}}})
        production_row = _make_policy_row(policy_id="uuid-1", policy_name="shared-name", guardrails_add=["db-guard"])
        sync_prisma = MagicMock()
        sync_prisma.db.litellm_policytable.find_many = AsyncMock(side_effect=[[production_row], []])
        await policy_registry.sync_policies_from_db(sync_prisma)
        assert policy_registry.get_source("shared-name") == "db"

        fresh_prisma = MagicMock()
        fresh_prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[])
        _set_prisma(monkeypatch, fresh_prisma)

        response = await policy_endpoints.list_policies()

        assert response.total_count == 1
        entry = response.policies[0]
        assert entry.policy_name == "shared-name"
        assert entry.definition_location == "config"
        assert entry.guardrails_add == ["config-guard"]

    @pytest.mark.asyncio
    async def test_version_status_filter_excludes_config_policies(self, policy_registry, monkeypatch):
        row = _make_policy_row(policy_id="uuid-1", policy_name="db-policy", version_status="draft")
        prisma = MagicMock()
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[row])
        _set_prisma(monkeypatch, prisma)
        policy_registry.load_policies({"config-policy": {"guardrails": {"add": ["tooling"]}}})

        response = await policy_endpoints.list_policies(version_status="draft")

        assert response.total_count == 1
        assert response.policies[0].policy_name == "db-policy"
        assert response.policies[0].definition_location == "db"

    @pytest.mark.asyncio
    async def test_production_filter_includes_config_policies(self, policy_registry, monkeypatch):
        _set_prisma(monkeypatch, None)
        policy_registry.load_policies({"config-policy": {"guardrails": {"add": ["tooling"]}}})

        response = await policy_endpoints.list_policies(version_status="production")

        assert response.total_count == 1
        assert response.policies[0].definition_location == "config"


class TestListAttachmentsIncludesConfig:
    @pytest.mark.asyncio
    async def test_returns_config_attachments_without_prisma(self, attachment_registry, monkeypatch):
        _set_prisma(monkeypatch, None)
        attachment_registry.load_attachments([{"policy": "config-policy", "scope": "*"}])

        response = await policy_endpoints.list_policy_attachments()

        assert response.total_count == 1
        entry = response.attachments[0]
        assert entry.attachment_id == "config-0"
        assert entry.policy_name == "config-policy"
        assert entry.scope == "*"
        assert entry.definition_location == "config"
        assert entry.created_at is None

    @pytest.mark.asyncio
    async def test_merges_db_attachments_with_config_and_keeps_db_row_shape(self, attachment_registry, monkeypatch):
        row = _make_attachment_row(attachment_id="att-1", policy_name="db-policy")
        prisma = MagicMock()
        prisma.db.litellm_policyattachmenttable.find_many = AsyncMock(return_value=[row])
        _set_prisma(monkeypatch, prisma)
        attachment_registry.load_attachments([{"policy": "config-policy", "scope": "*"}])

        response = await policy_endpoints.list_policy_attachments()

        assert response.total_count == 2
        db_entry = next(a for a in response.attachments if a.policy_name == "db-policy")
        assert db_entry.attachment_id == "att-1"
        assert db_entry.definition_location == "db"
        assert db_entry.created_at == row.created_at
        assert db_entry.created_by == "admin"
        config_entry = next(a for a in response.attachments if a.policy_name == "config-policy")
        assert config_entry.attachment_id == "config-0"
        assert config_entry.definition_location == "config"
