import sys
import os
import pytest

# Add the parent directory to the system path to import litellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import litellm
from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_TeamTable
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    get_effective_team_models,
)


@pytest.mark.asyncio
async def test_get_effective_team_models():
    original_flag = litellm.team_model_overrides_enabled
    try:
        litellm.team_model_overrides_enabled = True

        # Case 1: No overrides, should return team.models
        team = LiteLLM_TeamTable(team_id="t1", models=["m1"])
        assert get_effective_team_models(team) == ["m1"]

        # Case 2: Team defaults exist
        team.default_models = ["d1"]
        assert set(get_effective_team_models(team)) == {"d1"}

        # Case 3: Team defaults + Member overrides
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        assert set(get_effective_team_models(team, token)) == {"d1", "mo1"}

        # Case 4: No team object (should use token values if available)
        token.team_default_models = ["td1"]
        assert set(get_effective_team_models(None, token)) == {"td1", "mo1"}

        # Case 5: Feature disabled
        litellm.team_model_overrides_enabled = False
        assert get_effective_team_models(team, token) == ["m1"]
    finally:
        litellm.team_model_overrides_enabled = original_flag


@pytest.mark.asyncio
async def test_can_team_access_model_with_overrides():
    original_flag = litellm.team_model_overrides_enabled
    try:
        litellm.team_model_overrides_enabled = True

        # Team has access to m1, but we override it to only d1
        team = LiteLLM_TeamTable(team_id="t1", models=["m1"], default_models=["d1"])

        # Should NOT have access to m1 anymore if overrides are on
        with pytest.raises(Exception):
            await can_team_access_model(model="m1", team_object=team, llm_router=None)

        # Should have access to d1
        assert (
            await can_team_access_model(model="d1", team_object=team, llm_router=None)
            is True
        )

        # Guest member has extra access to g1
        token = UserAPIKeyAuth(team_member_models=["g1"])
        assert (
            await can_team_access_model(
                model="g1", team_object=team, llm_router=None, valid_token=token
            )
            is True
        )
        assert (
            await can_team_access_model(
                model="d1", team_object=team, llm_router=None, valid_token=token
            )
            is True
        )

        # Should NOT have access to m1
        with pytest.raises(Exception):
            await can_team_access_model(
                model="m1", team_object=team, llm_router=None, valid_token=token
            )
    finally:
        litellm.team_model_overrides_enabled = original_flag
