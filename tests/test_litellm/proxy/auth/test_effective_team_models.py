"""
Tests for the per-user team model overrides feature.

Covers:
- get_effective_team_models() helper
- can_team_access_model() with effective_models parameter
- all-team-models resolution via get_key_models()
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    get_effective_team_models,
)

FLAG_PATH = "litellm.constants.LITELLM_TEAM_MODEL_OVERRIDES_ENABLED"


# ---------------------------------------------------------------------------
# get_effective_team_models
# ---------------------------------------------------------------------------


class TestGetEffectiveTeamModelsFeatureDisabled:
    """When the feature flag is off, behaviour is unchanged."""

    @patch(FLAG_PATH, "false")
    def test_returns_team_models_when_flag_off(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3"],
            team_member_models=["o1-pro"],
            team_metadata={"team_default_models": ["gpt-4"]},
        )
        assert result == ["gpt-4", "claude-3"]

    @patch(FLAG_PATH, "false")
    def test_returns_team_object_models_when_flag_off(self):
        team = MagicMock(spec=LiteLLM_TeamTable)
        team.models = ["gpt-4"]
        result = get_effective_team_models(team_object=team)
        assert result == ["gpt-4"]


class TestGetEffectiveTeamModelsNoDefaultsConfigured:
    """When flag is on but team_default_models is not in metadata."""

    @patch(FLAG_PATH, "true")
    def test_returns_team_models_no_metadata_key(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3"],
            team_member_models=["o1-pro"],
            team_metadata={},
        )
        assert result == ["gpt-4", "claude-3"]

    @patch(FLAG_PATH, "true")
    def test_returns_team_models_none_metadata(self):
        result = get_effective_team_models(
            team_models=["gpt-4"],
            team_metadata=None,
        )
        assert result == ["gpt-4"]

    @patch(FLAG_PATH, "true")
    def test_returns_team_models_invalid_type(self):
        result = get_effective_team_models(
            team_models=["gpt-4"],
            team_metadata={"team_default_models": "not-a-list"},
        )
        assert result == ["gpt-4"]


class TestGetEffectiveTeamModelsDefaultsPlusOverrides:
    """When feature is on and team_default_models is configured."""

    @patch(FLAG_PATH, "true")
    def test_union_of_defaults_and_member(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=["o1-pro"],
            team_metadata={"team_default_models": ["gpt-4", "claude-3"]},
        )
        assert result == ["gpt-4", "claude-3", "o1-pro"]

    @patch(FLAG_PATH, "true")
    def test_empty_defaults_member_only(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=["o1-pro"],
            team_metadata={"team_default_models": []},
        )
        assert result == ["o1-pro"]

    @patch(FLAG_PATH, "true")
    def test_no_overrides_defaults_only(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=[],
            team_metadata={"team_default_models": ["gpt-4", "claude-3"]},
        )
        assert result == ["gpt-4", "claude-3"]

    @patch(FLAG_PATH, "true")
    def test_no_overrides_none_member_models(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=None,
            team_metadata={"team_default_models": ["gpt-4", "claude-3"]},
        )
        assert result == ["gpt-4", "claude-3"]

    @patch(FLAG_PATH, "true")
    def test_deduplication(self):
        result = get_effective_team_models(
            team_models=["gpt-4", "claude-3"],
            team_member_models=["gpt-4", "o1-pro"],
            team_metadata={"team_default_models": ["gpt-4", "claude-3"]},
        )
        assert result == ["gpt-4", "claude-3", "o1-pro"]

    @patch(FLAG_PATH, "true")
    def test_team_object_metadata(self):
        team = MagicMock(spec=LiteLLM_TeamTable)
        team.models = ["gpt-4", "claude-3", "o1-pro"]
        team.metadata = {"team_default_models": ["gpt-4"]}
        result = get_effective_team_models(
            team_object=team,
            team_member_models=["o1-pro"],
        )
        assert result == ["gpt-4", "o1-pro"]


# ---------------------------------------------------------------------------
# can_team_access_model with effective_models
# ---------------------------------------------------------------------------


class TestCanTeamAccessModelEffective:
    @pytest.mark.asyncio
    async def test_effective_models_used_when_provided(self):
        """When effective_models is passed, it should be used instead of team_object.models."""
        team = MagicMock(spec=LiteLLM_TeamTable)
        team.models = ["gpt-4"]  # team has gpt-4 only
        team.team_id = "test-team"
        team.access_group_ids = None

        # o1-pro is not in team.models but IS in effective_models
        router = MagicMock()
        router.get_model_group_info.return_value = None

        result = await can_team_access_model(
            model="o1-pro",
            team_object=team,
            llm_router=router,
            effective_models=["gpt-4", "o1-pro"],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_without_effective_models_uses_team(self):
        """Without effective_models, falls back to team_object.models."""
        team = MagicMock(spec=LiteLLM_TeamTable)
        team.models = ["gpt-4", "claude-3"]
        team.team_id = "test-team"
        team.access_group_ids = None

        router = MagicMock()
        router.get_model_group_info.return_value = None

        result = await can_team_access_model(
            model="gpt-4",
            team_object=team,
            llm_router=router,
        )
        assert result is True


# ---------------------------------------------------------------------------
# all-team-models resolution
# ---------------------------------------------------------------------------


class TestAllTeamModelsResolution:
    @patch(FLAG_PATH, "true")
    def test_all_team_models_resolves_to_effective(self):
        from litellm.proxy.auth.model_checks import get_key_models

        user_api_key_dict = UserAPIKeyAuth(
            models=["all-team-models"],
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=["o1-pro"],
            team_metadata={"team_default_models": ["gpt-4"]},
        )
        result = get_key_models(
            user_api_key_dict=user_api_key_dict,
            proxy_model_list=[],
            model_access_groups={},
        )
        assert result == ["gpt-4", "o1-pro"]

    @patch(FLAG_PATH, "false")
    def test_all_team_models_resolves_unchanged_when_disabled(self):
        from litellm.proxy.auth.model_checks import get_key_models

        user_api_key_dict = UserAPIKeyAuth(
            models=["all-team-models"],
            team_models=["gpt-4", "claude-3", "o1-pro"],
            team_member_models=["o1-pro"],
            team_metadata={"team_default_models": ["gpt-4"]},
        )
        result = get_key_models(
            user_api_key_dict=user_api_key_dict,
            proxy_model_list=[],
            model_access_groups={},
        )
        assert result == ["gpt-4", "claude-3", "o1-pro"]
