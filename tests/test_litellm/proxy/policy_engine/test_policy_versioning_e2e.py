"""
Integration-style tests for policy versioning: full lifecycle with mocked DB.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.policy_engine.policy_registry import PolicyRegistry
from litellm.types.proxy.policy_engine import (PolicyCreateRequest,
                                               PolicyUpdateRequest)


def _make_row(
    policy_id,
    policy_name,
    version_number=1,
    version_status="production",
    parent_version_id=None,
    is_latest=True,
    published_at=None,
    production_at=None,
    inherit=None,
    description="",
    guardrails_add=None,
    guardrails_remove=None,
    condition=None,
    pipeline=None,
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
    row.created_at = datetime.now(timezone.utc)
    row.updated_at = datetime.now(timezone.utc)
    row.created_by = None
    row.updated_by = None
    return row


@pytest.mark.asyncio
async def test_full_lifecycle_create_draft_edit_publish_promote():
    """
    Full lifecycle: create policy -> create draft version -> edit draft ->
    publish -> promote to production -> verify old version demoted ->
    verify in-memory updated.
    """
    registry = PolicyRegistry()
    prisma = MagicMock()
    now = datetime.now(timezone.utc)

    # 1) Create initial policy (v1 production)
    create_data = {}
    created_v1 = _make_row(
        policy_id="v1-id",
        policy_name="lifecycle-policy",
        version_number=1,
        version_status="production",
        production_at=now,
        guardrails_add=["g1"],
        description="Initial",
    )

    async def create_impl(data=None, **kwargs):
        create_data.update(kwargs.get("data", data or {}))
        return created_v1

    prisma.db.litellm_policytable.create = AsyncMock(side_effect=create_impl)
    req = PolicyCreateRequest(
        policy_name="lifecycle-policy",
        description="Initial",
        guardrails_add=["g1"],
    )
    created = await registry.add_policy_to_db(req, prisma, created_by="user")
    assert created.version_number == 1
    assert created.version_status == "production"
    assert registry.has_policy("lifecycle-policy")

    # 2) Create new draft version (v2)
    v2_row = _make_row(
        policy_id="v2-id",
        policy_name="lifecycle-policy",
        version_number=2,
        version_status="draft",
        parent_version_id="v1-id",
        is_latest=True,
        guardrails_add=["g1", "g2"],
        description="Draft v2",
    )
    prisma.db.litellm_policytable.find_first = AsyncMock(return_value=created_v1)
    prisma.db.litellm_policytable.update_many = AsyncMock()
    prisma.db.litellm_policytable.create = AsyncMock(return_value=v2_row)

    draft_v2 = await registry.create_new_version(
        policy_name="lifecycle-policy",
        prisma_client=prisma,
        source_policy_id=None,
        created_by="user",
    )
    assert draft_v2.version_number == 2
    assert draft_v2.version_status == "draft"
    assert draft_v2.parent_version_id == "v1-id"
    # In-memory still has v1 (only production is in registry)
    assert registry.has_policy("lifecycle-policy")
    policy = registry.get_policy("lifecycle-policy")
    assert policy.guardrails.add == ["g1"]  # still v1

    # 3) Edit draft v2
    v2_updated_row = _make_row(
        policy_id="v2-id",
        policy_name="lifecycle-policy",
        version_number=2,
        version_status="draft",
        guardrails_add=["g1", "g2", "g3"],
        description="Draft v2 edited",
    )
    prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=v2_row)
    prisma.db.litellm_policytable.update = AsyncMock(return_value=v2_updated_row)

    updated_draft = await registry.update_policy_in_db(
        policy_id="v2-id",
        policy_request=PolicyUpdateRequest(
            description="Draft v2 edited",
            guardrails_add=["g1", "g2", "g3"],
        ),
        prisma_client=prisma,
        updated_by="user",
    )
    assert updated_draft.description == "Draft v2 edited"
    assert updated_draft.guardrails_add == ["g1", "g2", "g3"]

    # 4) Publish v2 (draft -> published)
    v2_published = _make_row(
        policy_id="v2-id",
        policy_name="lifecycle-policy",
        version_number=2,
        version_status="published",
        published_at=now,
        guardrails_add=["g1", "g2", "g3"],
        description="Draft v2 edited",
    )
    prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=v2_updated_row)
    prisma.db.litellm_policytable.update = AsyncMock(return_value=v2_published)

    published = await registry.update_version_status(
        policy_id="v2-id",
        new_status="published",
        prisma_client=prisma,
        updated_by="user",
    )
    assert published.version_status == "published"

    # 5) Promote v2 to production (demote v1 to published, update registry)
    prisma.db.litellm_policytable.find_unique = AsyncMock(return_value=v2_published)
    prisma.db.litellm_policytable.update_many = AsyncMock()
    v2_production = _make_row(
        policy_id="v2-id",
        policy_name="lifecycle-policy",
        version_number=2,
        version_status="production",
        production_at=now,
        guardrails_add=["g1", "g2", "g3"],
        description="Draft v2 edited",
    )
    prisma.db.litellm_policytable.update = AsyncMock(return_value=v2_production)

    prod = await registry.update_version_status(
        policy_id="v2-id",
        new_status="production",
        prisma_client=prisma,
        updated_by="user",
    )
    assert prod.version_status == "production"
    # In-memory registry should now have v2 content
    assert registry.has_policy("lifecycle-policy")
    policy = registry.get_policy("lifecycle-policy")
    assert policy.guardrails.add == ["g1", "g2", "g3"]


@pytest.mark.asyncio
async def test_attachments_resolve_against_production_after_promotion():
    """
    After promoting a new version to production, resolve_guardrails_from_db
    returns guardrails from the new production version (inheritance resolves
    against production).
    """
    registry = PolicyRegistry()
    prisma = MagicMock()
    # Simulate only production versions loaded for resolution
    prod_row = _make_row(
        policy_id="prod-1",
        policy_name="att-policy",
        version_status="production",
        guardrails_add=["ga", "gb"],
    )
    prisma.db.litellm_policytable.find_many = AsyncMock(return_value=[prod_row])

    resolved = await registry.resolve_guardrails_from_db(
        policy_name="att-policy",
        prisma_client=prisma,
    )
    assert "ga" in resolved
    assert "gb" in resolved
    call_kw = prisma.db.litellm_policytable.find_many.call_args[1]
    assert call_kw.get("where") == {"version_status": "production"}
