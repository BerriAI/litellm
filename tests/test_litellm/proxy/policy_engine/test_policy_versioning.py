"""
Unit tests for policy versioning: registry behavior, status transitions, and version CRUD.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.policy_engine.policy_registry import (
    PolicyRegistry,
    _row_to_policy_db_response,
    get_policy_registry,
)
from litellm.types.proxy.policy_engine import (
    PolicyCreateRequest,
    PolicyDBResponse,
    PolicyUpdateRequest,
)


def _make_row(
    policy_id="pid-1",
    policy_name="test-policy",
    version_number=1,
    version_status="production",
    parent_version_id=None,
    is_latest=True,
    published_at=None,
    production_at=None,
    inherit=None,
    description="desc",
    guardrails_add=None,
    guardrails_remove=None,
    condition=None,
    pipeline=None,
    created_at=None,
    updated_at=None,
    created_by=None,
    updated_by=None,
):
    row = MagicMock()
    row.policy_id = policy_id
    row.policy_name = policy_name
    row.version_number = version_number
    row.version_status = version_status
    row.parent_version_id = parent_version_id
    row.is_latest = is_latest
    row.published_at = published_at
    row.production_at = production_at
    row.inherit = inherit
    row.description = description
    row.guardrails_add = guardrails_add or []
    row.guardrails_remove = guardrails_remove or []
    row.condition = condition
    row.pipeline = pipeline
    row.created_at = created_at or datetime.now(timezone.utc)
    row.updated_at = updated_at or datetime.now(timezone.utc)
    row.created_by = created_by
    row.updated_by = updated_by
    return row


class TestRowToPolicyDBResponse:
    """Test _row_to_policy_db_response includes all version fields."""

    def test_includes_version_fields(self):
        row = _make_row(
            version_number=2,
            version_status="draft",
            parent_version_id="pid-0",
            is_latest=True,
            published_at=None,
            production_at=None,
        )
        resp = _row_to_policy_db_response(row)
        assert isinstance(resp, PolicyDBResponse)
        assert resp.policy_id == "pid-1"
        assert resp.policy_name == "test-policy"
        assert resp.version_number == 2
        assert resp.version_status == "draft"
        assert resp.parent_version_id == "pid-0"
        assert resp.is_latest is True
        assert resp.published_at is None
        assert resp.production_at is None

    def test_backward_compat_missing_version_attrs(self):
        row = _make_row()
        del row.version_number
        del row.version_status
        del row.parent_version_id
        del row.is_latest
        del row.published_at
        del row.production_at
        resp = _row_to_policy_db_response(row)
        assert resp.version_number == 1
        assert resp.version_status == "production"
        assert resp.parent_version_id is None
        assert resp.is_latest is True


class TestSyncPoliciesFromDbProductionOnly:
    """Test that sync_policies_from_db only loads production versions."""

    @pytest.mark.asyncio
    async def test_get_all_policies_with_version_status_calls_find_many_with_where(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        prod_row = _make_row(policy_id="prod-1", version_status="production")
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[prod_row])

        result = await registry.get_all_policies_from_db(
            prisma, version_status="production"
        )

        assert len(result) == 1
        assert result[0].version_status == "production"
        prisma.db.litellm_policytable.find_many.assert_called_once()
        call_kw = prisma.db.litellm_policytable.find_many.call_args[1]
        assert call_kw.get("where") == {"version_status": "production"}

    @pytest.mark.asyncio
    async def test_sync_policies_from_db_only_loads_production(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        prod_row = _make_row(
            policy_id="prod-1",
            policy_name="foo",
            version_status="production",
            guardrails_add=["g1"],
        )
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[prod_row])

        await registry.sync_policies_from_db(prisma)

        assert registry.has_policy("foo")
        policy = registry.get_policy("foo")
        assert policy is not None
        assert policy.guardrails.add == ["g1"]
        # find_many was called with version_status=production (via get_all_policies_from_db)
        find_many_calls = prisma.db.litellm_policytable.find_many.call_args_list
        assert len(find_many_calls) >= 1
        assert find_many_calls[0][1].get("where") == {"version_status": "production"}


class TestUpdatePolicyDraftOnly:
    """Test that update_policy_in_db only allows draft versions."""

    @pytest.mark.asyncio
    async def test_update_production_raises(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        prod_row = _make_row(policy_id="pid-1", version_status="production")
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=prod_row)

        with pytest.raises(Exception) as exc_info:
            await registry.update_policy_in_db(
                policy_id="pid-1",
                policy_request=PolicyUpdateRequest(description="new"),
                prisma_client=prisma,
            )
        assert "Only draft" in str(exc_info.value) or "draft" in str(exc_info.value).lower()
        prisma.db.litellm_policytable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_draft_succeeds_and_does_not_update_registry(self):
        registry = PolicyRegistry()
        registry.add_policy("test-policy", MagicMock())  # in-memory state
        prisma = MagicMock()
        draft_row = _make_row(
            policy_id="draft-1",
            policy_name="test-policy",
            version_status="draft",
            description="old",
        )
        updated_row = _make_row(
            policy_id="draft-1",
            policy_name="test-policy",
            version_status="draft",
            description="new",
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=draft_row)
        prisma.db.litellm_policytable.update = AsyncMock(return_value=updated_row)

        result = await registry.update_policy_in_db(
            policy_id="draft-1",
            policy_request=PolicyUpdateRequest(description="new"),
            prisma_client=prisma,
        )

        assert result.description == "new"
        prisma.db.litellm_policytable.update.assert_called_once()
        # Registry still has old in-memory policy (drafts are not in registry; we don't add)
        assert registry.has_policy("test-policy")


class TestDeletePolicyFromDb:
    """Test delete_policy_from_db removes production from registry and returns warning."""

    @pytest.mark.asyncio
    async def test_delete_production_removes_from_registry_and_returns_warning(self):
        registry = PolicyRegistry()
        registry.add_policy("deleted-policy", MagicMock())
        prisma = MagicMock()
        prod_row = _make_row(
            policy_id="prod-1",
            policy_name="deleted-policy",
            version_status="production",
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=prod_row)
        prisma.db.litellm_policytable.delete = AsyncMock()

        result = await registry.delete_policy_from_db(
            policy_id="prod-1",
            prisma_client=prisma,
        )

        assert result["message"]
        assert "warning" in result
        assert "Production" in result["warning"] or "production" in result["warning"]
        assert not registry.has_policy("deleted-policy")

    @pytest.mark.asyncio
    async def test_delete_draft_does_not_remove_from_registry_no_warning(self):
        registry = PolicyRegistry()
        registry.add_policy("my-policy", MagicMock())
        prisma = MagicMock()
        draft_row = _make_row(
            policy_id="draft-1",
            policy_name="my-policy",
            version_status="draft",
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=draft_row)
        prisma.db.litellm_policytable.delete = AsyncMock()

        result = await registry.delete_policy_from_db(
            policy_id="draft-1",
            prisma_client=prisma,
        )

        assert "warning" not in result
        assert registry.has_policy("my-policy")


class TestCreateNewVersion:
    """Test create_new_version copies all fields and sets draft."""

    @pytest.mark.asyncio
    async def test_create_new_version_from_production_increments_version(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        prod = _make_row(
            policy_id="prod-1",
            policy_name="foo",
            version_number=1,
            version_status="production",
            guardrails_add=["g1"],
            description="base",
            inherit=None,
            pipeline={"mode": "pre_call", "steps": []},
        )
        # find_first for production
        prisma.db.litellm_policytable.find_first = AsyncMock(return_value=prod)
        # find_first for latest version number
        prisma.db.litellm_policytable.find_first.side_effect = [
            prod,  # production lookup
            prod,  # latest version_number lookup
        ]
        # update_many for is_latest=False
        prisma.db.litellm_policytable.update_many = AsyncMock()
        new_row = _make_row(
            policy_id="new-id",
            policy_name="foo",
            version_number=2,
            version_status="draft",
            parent_version_id="prod-1",
            is_latest=True,
            guardrails_add=["g1"],
            description="base",
            pipeline={"mode": "pre_call", "steps": []},
        )
        prisma.db.litellm_policytable.create = AsyncMock(return_value=new_row)

        result = await registry.create_new_version(
            policy_name="foo",
            prisma_client=prisma,
            source_policy_id=None,
            created_by="user",
        )

        assert result.version_number == 2
        assert result.version_status == "draft"
        assert result.parent_version_id == "prod-1"
        assert result.guardrails_add == ["g1"]
        assert result.description == "base"
        create_call = prisma.db.litellm_policytable.create.call_args[1]["data"]
        assert create_call["version_number"] == 2
        assert create_call["version_status"] == "draft"
        assert create_call["parent_version_id"] == "prod-1"
        assert create_call["guardrails_add"] == ["g1"]


class TestUpdateVersionStatus:
    """Test status transitions: valid succeed, invalid return error."""

    @pytest.mark.asyncio
    async def test_draft_to_published_sets_published_at(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        draft = _make_row(policy_id="d-1", version_status="draft")
        updated = _make_row(
            policy_id="d-1",
            version_status="published",
            published_at=datetime.now(timezone.utc),
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=draft)
        prisma.db.litellm_policytable.update = AsyncMock(return_value=updated)

        result = await registry.update_version_status(
            policy_id="d-1",
            new_status="published",
            prisma_client=prisma,
        )

        assert result.version_status == "published"
        update_data = prisma.db.litellm_policytable.update.call_args[1]["data"]
        assert update_data["version_status"] == "published"
        assert "published_at" in update_data

    @pytest.mark.asyncio
    async def test_draft_to_production_raises(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        draft = _make_row(policy_id="d-1", version_status="draft")
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=draft)

        with pytest.raises(Exception) as exc_info:
            await registry.update_version_status(
                policy_id="d-1",
                new_status="production",
                prisma_client=prisma,
            )
        assert "publish" in str(exc_info.value).lower() or "draft" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_published_to_production_demotes_old_and_updates_registry(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        published_row = _make_row(
            policy_id="pub-1",
            policy_name="foo",
            version_status="published",
        )
        updated_row = _make_row(
            policy_id="pub-1",
            policy_name="foo",
            version_status="production",
            production_at=datetime.now(timezone.utc),
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=published_row)
        prisma.db.litellm_policytable.update_many = AsyncMock()
        prisma.db.litellm_policytable.update = AsyncMock(return_value=updated_row)

        result = await registry.update_version_status(
            policy_id="pub-1",
            new_status="production",
            prisma_client=prisma,
        )

        assert result.version_status == "production"
        # update_many should have been called to demote current production
        assert prisma.db.litellm_policytable.update_many.called
        # Registry should have been updated with new production
        assert registry.has_policy("foo")


class TestCompareVersions:
    """Test compare_versions returns correct field diffs."""

    @pytest.mark.asyncio
    async def test_compare_versions_returns_diffs(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        a = _make_row(
            policy_id="a",
            policy_name="p",
            description="desc A",
            guardrails_add=["g1"],
        )
        b = _make_row(
            policy_id="b",
            policy_name="p",
            description="desc B",
            guardrails_add=["g1", "g2"],
        )
        prisma.db.litellm_policytable.find_unique = AsyncMock(side_effect=[a, b])

        result = await registry.compare_versions(
            policy_id_a="a",
            policy_id_b="b",
            prisma_client=prisma,
        )

        assert result.version_a.policy_id == "a"
        assert result.version_b.policy_id == "b"
        assert "description" in result.field_diffs
        assert result.field_diffs["description"]["version_a"] == "desc A"
        assert result.field_diffs["description"]["version_b"] == "desc B"
        assert "guardrails_add" in result.field_diffs


class TestResolveGuardrailsProductionOnly:
    """Test that resolve_guardrails_from_db uses only production versions."""

    @pytest.mark.asyncio
    async def test_resolve_guardrails_calls_get_all_with_production_filter(self):
        registry = PolicyRegistry()
        prisma = MagicMock()
        prod_row = _make_row(
            policy_name="base",
            version_status="production",
            guardrails_add=["g1"],
        )
        prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[prod_row])

        result = await registry.resolve_guardrails_from_db(
            policy_name="base",
            prisma_client=prisma,
        )

        assert "g1" in result
        call_kw = prisma.db.litellm_policytable.find_many.call_args[1]
        assert call_kw.get("where") == {"version_status": "production"}


class TestGetPolicyRegistrySingleton:
    """Test get_policy_registry returns same instance."""

    def test_returns_singleton(self):
        a = get_policy_registry()
        b = get_policy_registry()
        assert a is b
