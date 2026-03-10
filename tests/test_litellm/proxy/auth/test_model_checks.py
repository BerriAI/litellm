from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.proxy.auth.handle_jwt import JWTAuthManager


def test_get_team_models_for_all_models_and_team_only_models():
    from litellm.proxy.auth.model_checks import get_team_models

    team_models = ["all-proxy-models", "team-only-model", "team-only-model-2"]
    proxy_model_list = ["model1", "model2", "model3"]
    model_access_groups = {}
    include_model_access_groups = False

    result = get_team_models(
        team_models, proxy_model_list, model_access_groups, include_model_access_groups
    )
    combined_models = team_models + proxy_model_list
    assert set(result) == set(combined_models)


@pytest.mark.parametrize(
    "key_models,team_models,proxy_model_list,model_list,expected",
    [
        (
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"],
            [],
            [],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"]
        ),
        (
            [],
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"],
            [],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"]
        ),
        (
            [],
            [],
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-haiku-20241022"]
        ),
    ],
)
def test_get_complete_model_list_order(key_models, team_models, proxy_model_list, model_list, expected):
    """
    Test that get_complete_model_list preserves order
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    assert get_complete_model_list(
        proxy_model_list=proxy_model_list,
        key_models=key_models,
        team_models=team_models,
        user_model=None,
        infer_model_from_keys=False,
        llm_router=Router(model_list=model_list),
    ) == expected


def test_get_complete_model_list_byok_wildcard_expansion():
    """
    Test that wildcard models (e.g., openai/*) are expanded when the router has
    no deployment for them - BYOK case where team has openai/* but proxy has
    no openai config.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    # Router with empty model_list - no openai/* deployment (BYOK scenario)
    result = get_complete_model_list(
        key_models=[],
        team_models=["openai/*"],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=Router(model_list=[]),
    )
    # Should expand openai/* to actual OpenAI models
    assert len(result) > 0
    assert all(m.startswith("openai/") for m in result)
    assert "openai/*" not in result
