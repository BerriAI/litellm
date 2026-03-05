"""
Unit tests for team-scoped model overrides.

Tests cover:
- compute_effective_team_models (union logic)
- can_team_access_model with overrides (runtime enforcement)
- default_models ⊆ team.models validation
- member models ⊆ team.models validation
- _validate_key_models_against_effective_team_models (key creation)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    compute_effective_team_models,
)


# ── compute_effective_team_models ────────────────────────────────────────────


class TestComputeEffectiveTeamModels:
    def test_union_of_defaults_and_member(self):
        result = compute_effective_team_models(
            team_default_models=["gpt-4o"],
            team_member_models=["claude-sonnet"],
        )
        assert set(result) == {"gpt-4o", "claude-sonnet"}

    def test_deduplicates(self):
        result = compute_effective_team_models(
            team_default_models=["gpt-4o", "claude-sonnet"],
            team_member_models=["claude-sonnet"],
        )
        assert sorted(result) == sorted(["gpt-4o", "claude-sonnet"])

    def test_none_defaults(self):
        result = compute_effective_team_models(
            team_default_models=None,
            team_member_models=["claude-sonnet"],
        )
        assert result == ["claude-sonnet"]

    def test_none_member(self):
        result = compute_effective_team_models(
            team_default_models=["gpt-4o"],
            team_member_models=None,
        )
        assert result == ["gpt-4o"]

    def test_both_none(self):
        result = compute_effective_team_models(
            team_default_models=None,
            team_member_models=None,
        )
        assert result == []

    def test_both_empty(self):
        result = compute_effective_team_models(
            team_default_models=[],
            team_member_models=[],
        )
        assert result == []


# ── can_team_access_model (runtime check) ────────────────────────────────────


class TestCanTeamAccessModelOverrides:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_allowed_model_in_defaults(self):
        """Model in default_models should be allowed."""
        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o", "gpt-4o-mini", "claude-sonnet"],
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=["gpt-4o"],
            team_member_models=None,
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        result = await can_team_access_model(
            model="gpt-4o",
            team_object=team_object,
            llm_router=mock_router,
            team_model_aliases=None,
            valid_token=valid_token,
        )
        assert result is True

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_allowed_model_in_member_override(self):
        """Model in member models should be allowed."""
        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o", "gpt-4o-mini", "claude-sonnet"],
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=["gpt-4o"],
            team_member_models=["claude-sonnet"],
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        result = await can_team_access_model(
            model="claude-sonnet",
            team_object=team_object,
            llm_router=mock_router,
            team_model_aliases=None,
            valid_token=valid_token,
        )
        assert result is True

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_blocked_model_not_in_effective(self):
        """Model in team.models but NOT in effective models should be blocked."""

        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o", "gpt-4o-mini", "claude-sonnet"],
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=["gpt-4o"],
            team_member_models=["claude-sonnet"],
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        with pytest.raises(Exception):
            await can_team_access_model(
                model="gpt-4o-mini",
                team_object=team_object,
                llm_router=mock_router,
                team_model_aliases=None,
                valid_token=valid_token,
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_effective_models_intersected_with_team_models(self):
        """Even if default_models has out-of-bounds model, runtime should block it."""
        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o"],  # team only allows gpt-4o
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=[
                "gpt-4o",
                "claude-sonnet",
            ],  # claude-sonnet is out of bounds
            team_member_models=None,
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        # claude-sonnet should be blocked even though it's in default_models
        with pytest.raises(Exception):
            await can_team_access_model(
                model="claude-sonnet",
                team_object=team_object,
                llm_router=mock_router,
                team_model_aliases=None,
                valid_token=valid_token,
            )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "false"})
    async def test_flag_off_uses_team_models(self):
        """When flag is off, should use team.models as before."""
        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o", "gpt-4o-mini"],
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=["gpt-4o"],
            team_member_models=None,
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        # gpt-4o-mini should be allowed because flag is off, team.models is used
        result = await can_team_access_model(
            model="gpt-4o-mini",
            team_object=team_object,
            llm_router=mock_router,
            team_model_aliases=None,
            valid_token=valid_token,
        )
        assert result is True

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_no_overrides_configured_uses_team_models(self):
        """Team with no default_models/member_models uses team.models unchanged."""
        team_object = LiteLLM_TeamTable(
            team_id="team-1",
            models=["gpt-4o", "gpt-4o-mini"],
        )
        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="team-1",
            team_default_models=None,
            team_member_models=None,
        )
        mock_router = MagicMock()
        mock_router.get_model_group_info.return_value = None

        result = await can_team_access_model(
            model="gpt-4o-mini",
            team_object=team_object,
            llm_router=mock_router,
            team_model_aliases=None,
            valid_token=valid_token,
        )
        assert result is True


# ── _validate_key_models_against_effective_team_models ───────────────────────


class TestValidateKeyModelsAgainstEffective:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_empty_data_models_gets_effective(self):
        """Key with no models should inherit effective models."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _validate_key_models_against_effective_team_models,
        )

        mock_prisma = MagicMock()
        mock_membership = MagicMock()
        mock_membership.models = ["claude-sonnet"]
        mock_prisma.db.litellm_teammembership.find_unique = AsyncMock(
            return_value=mock_membership
        )

        team_table = MagicMock()
        team_table.default_models = ["gpt-4o"]
        team_table.models = ["gpt-4o", "claude-sonnet", "gpt-4o-mini"]

        data = MagicMock()
        data.models = []

        await _validate_key_models_against_effective_team_models(
            team_id="team-1",
            user_id="user-1",
            data=data,
            team_table=team_table,
            prisma_client=mock_prisma,
        )

        assert set(data.models) == {"gpt-4o", "claude-sonnet"}

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_effective_models_capped_to_team_models(self):
        """Key effective models should be intersected with team.models."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _validate_key_models_against_effective_team_models,
        )

        mock_prisma = MagicMock()
        mock_membership = MagicMock()
        mock_membership.models = ["claude-sonnet"]  # out-of-bounds
        mock_prisma.db.litellm_teammembership.find_unique = AsyncMock(
            return_value=mock_membership
        )

        team_table = MagicMock()
        team_table.default_models = ["gpt-4o"]
        team_table.models = ["gpt-4o"]  # team only allows gpt-4o

        data = MagicMock()
        data.models = []

        await _validate_key_models_against_effective_team_models(
            team_id="team-1",
            user_id="user-1",
            data=data,
            team_table=team_table,
            prisma_client=mock_prisma,
        )

        # claude-sonnet should be capped out
        assert data.models == ["gpt-4o"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_disallowed_model_in_key_raises(self):
        """Key requesting model outside effective set should raise 403."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _validate_key_models_against_effective_team_models,
        )

        mock_prisma = MagicMock()
        mock_membership = MagicMock()
        mock_membership.models = ["claude-sonnet"]
        mock_prisma.db.litellm_teammembership.find_unique = AsyncMock(
            return_value=mock_membership
        )

        team_table = MagicMock()
        team_table.default_models = ["gpt-4o"]
        team_table.models = ["gpt-4o", "claude-sonnet", "gpt-4o-mini"]

        data = MagicMock()
        data.models = ["gpt-4o-mini"]  # not in effective models

        with pytest.raises(HTTPException) as exc_info:
            await _validate_key_models_against_effective_team_models(
                team_id="team-1",
                user_id="user-1",
                data=data,
                team_table=team_table,
                prisma_client=mock_prisma,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "true"})
    async def test_no_overrides_skips_validation(self):
        """Teams without default_models or member models skip override validation."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _validate_key_models_against_effective_team_models,
        )

        mock_prisma = MagicMock()
        mock_membership = MagicMock()
        mock_membership.models = []
        mock_prisma.db.litellm_teammembership.find_unique = AsyncMock(
            return_value=mock_membership
        )

        team_table = MagicMock()
        team_table.default_models = []
        team_table.models = ["gpt-4o", "gpt-4o-mini"]

        data = MagicMock()
        data.models = ["gpt-4o-mini"]

        # Should return without modifying data.models
        await _validate_key_models_against_effective_team_models(
            team_id="team-1",
            user_id="user-1",
            data=data,
            team_table=team_table,
            prisma_client=mock_prisma,
        )

        assert data.models == ["gpt-4o-mini"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"LITELLM_TEAM_MODEL_OVERRIDES": "false"})
    async def test_flag_off_skips_entirely(self):
        """When flag is off, validation is skipped entirely."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _validate_key_models_against_effective_team_models,
        )

        mock_prisma = MagicMock()
        team_table = MagicMock()

        data = MagicMock()
        data.models = ["anything"]

        await _validate_key_models_against_effective_team_models(
            team_id="team-1",
            user_id="user-1",
            data=data,
            team_table=team_table,
            prisma_client=mock_prisma,
        )

        # Should be unchanged — no validation happened
        assert data.models == ["anything"]
