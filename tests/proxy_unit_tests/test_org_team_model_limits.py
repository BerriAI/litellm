"""
Unit tests for _check_org_team_limits() in team_endpoints.py.

Tests the fix for empty/None team models bypassing org model restrictions
(PR #20846).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    NewTeamRequest,
    SpecialModelNames,
    UpdateTeamRequest,
)
from litellm.proxy.management_endpoints.team_endpoints import _check_org_team_limits
from litellm.proxy.utils import PrismaClient


def _make_org_table(
    organization_id: str = "test-org-123",
    models: list = None,
    litellm_budget_table: LiteLLM_BudgetTable = None,
) -> LiteLLM_OrganizationTable:
    """Create a minimal LiteLLM_OrganizationTable for testing."""
    return LiteLLM_OrganizationTable(
        organization_id=organization_id,
        budget_id="test-budget-id",
        spend=0.0,
        models=models or [],
        created_by="test-user",
        updated_by="test-user",
        litellm_budget_table=litellm_budget_table,
    )


def _make_prisma_client() -> MagicMock:
    """Create a mock PrismaClient that won't be used for model validation tests."""
    mock = MagicMock(spec=PrismaClient)
    mock.db = MagicMock()
    mock.db.litellm_teamtable = MagicMock()
    mock.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    return mock


@pytest.mark.asyncio
async def test_empty_team_models_inherits_org_models():
    """
    When team has empty models ([]) and org has model restrictions,
    team should inherit org's models to prevent bypassing restrictions.
    """
    org_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    org_table = _make_org_table(models=org_models)
    data = NewTeamRequest(
        team_alias="test-team",
        organization_id="test-org-123",
        models=[],
        members_with_roles=[],
    )
    prisma_client = _make_prisma_client()

    await _check_org_team_limits(
        org_table=org_table,
        data=data,
        prisma_client=prisma_client,
    )

    assert data.models == org_models


@pytest.mark.asyncio
async def test_none_team_models_inherits_org_models():
    """
    When team has None models and org has model restrictions,
    team should inherit org's models to prevent bypassing restrictions.
    """
    org_models = ["gpt-4", "gpt-3.5-turbo"]
    org_table = _make_org_table(models=org_models)
    data = UpdateTeamRequest(
        team_id="test-team-id",
        organization_id="test-org-123",
        models=None,
    )
    prisma_client = _make_prisma_client()

    await _check_org_team_limits(
        org_table=org_table,
        data=data,
        prisma_client=prisma_client,
    )

    assert data.models == org_models


@pytest.mark.asyncio
async def test_valid_team_models_passes():
    """
    When team models are a subset of org's allowed models, validation passes.
    """
    org_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    org_table = _make_org_table(models=org_models)
    team_models = ["gpt-4", "gpt-3.5-turbo"]
    data = NewTeamRequest(
        team_alias="test-team",
        organization_id="test-org-123",
        models=team_models.copy(),
        members_with_roles=[],
    )
    prisma_client = _make_prisma_client()

    await _check_org_team_limits(
        org_table=org_table,
        data=data,
        prisma_client=prisma_client,
    )

    assert data.models == team_models


@pytest.mark.asyncio
async def test_invalid_team_models_raises_httpexception():
    """
    When team has a model not in org's allowed list, raises HTTPException.
    """
    org_models = ["gpt-4", "gpt-3.5-turbo"]
    org_table = _make_org_table(models=org_models)
    data = NewTeamRequest(
        team_alias="test-team",
        organization_id="test-org-123",
        models=["gpt-4", "claude-3-opus"],  # claude-3-opus not in org
        members_with_roles=[],
    )
    prisma_client = _make_prisma_client()

    with pytest.raises(HTTPException) as exc_info:
        await _check_org_team_limits(
            org_table=org_table,
            data=data,
            prisma_client=prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert "claude-3-opus" in str(exc_info.value.detail)
    assert "not in organization's allowed models" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_org_with_all_proxy_models_skips_validation():
    """
    When org has 'all-proxy-models', team model validation is skipped.
    """
    org_models = [SpecialModelNames.all_proxy_models.value]
    org_table = _make_org_table(models=org_models)
    data = NewTeamRequest(
        team_alias="test-team",
        organization_id="test-org-123",
        models=[],  # Empty - would normally inherit, but validation is skipped
        members_with_roles=[],
    )
    prisma_client = _make_prisma_client()

    await _check_org_team_limits(
        org_table=org_table,
        data=data,
        prisma_client=prisma_client,
    )

    # Models remain empty - no inheritance when all-proxy-models
    assert data.models == []


@pytest.mark.asyncio
async def test_org_with_no_models_skips_validation():
    """
    When org has no model restrictions (empty models list), validation is skipped.
    """
    org_table = _make_org_table(models=[])
    data = NewTeamRequest(
        team_alias="test-team",
        organization_id="test-org-123",
        models=["gpt-4", "any-model"],  # Would fail if org had restrictions
        members_with_roles=[],
    )
    prisma_client = _make_prisma_client()

    await _check_org_team_limits(
        org_table=org_table,
        data=data,
        prisma_client=prisma_client,
    )

    # No exception, models unchanged
    assert data.models == ["gpt-4", "any-model"]
