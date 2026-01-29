from unittest.mock import AsyncMock, MagicMock, patch

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


def test_wildcard_patterns_removed_from_model_list_with_router():
    """
    Test that wildcard patterns like 'anthropic/*' are removed from the model
    list when llm_router is present.

    Previously, models_to_remove.add(model) was only called in the
    llm_router is None branch, causing wildcard patterns to appear as
    literal model names in the /v1/models response.
    """
    from litellm.proxy.auth.model_checks import _get_wildcard_models

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [
        {"litellm_params": {"model": "anthropic/claude-3-haiku-20240307"}}
    ]

    unique_models = ["anthropic/*", "gpt-4"]

    with patch(
        "litellm.proxy.auth.model_checks.get_known_models_from_wildcard",
        return_value=["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-sonnet-20241022"],
    ):
        result = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=mock_router,
        )

    # The wildcard pattern should be removed from unique_models
    assert "anthropic/*" not in unique_models
    # Explicit models should remain
    assert "gpt-4" in unique_models
    # Expanded models should be in the result
    assert "anthropic/claude-3-haiku-20240307" in result
    assert "anthropic/claude-3-5-sonnet-20241022" in result


def test_wildcard_expansion_deduplicates_across_deployments():
    """
    Test that when multiple wildcard deployments exist for the same provider
    (e.g. several 'anthropic/*' entries with different credentials), the
    expanded model list does not contain duplicates.
    """
    from litellm.proxy.auth.model_checks import _get_wildcard_models

    mock_router = MagicMock()
    # Simulate two deployments for anthropic/* (e.g. different team credentials)
    mock_router.get_model_list.return_value = [
        {"litellm_params": {"model": "anthropic/team1-key"}},
        {"litellm_params": {"model": "anthropic/team2-key"}},
    ]

    unique_models = ["anthropic/*"]
    expanded = ["anthropic/claude-3-haiku-20240307", "anthropic/claude-3-5-sonnet-20241022"]

    with patch(
        "litellm.proxy.auth.model_checks.get_known_models_from_wildcard",
        return_value=expanded,
    ):
        result = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=mock_router,
        )

    # Each model should appear exactly once despite two deployments
    assert result.count("anthropic/claude-3-haiku-20240307") == 1
    assert result.count("anthropic/claude-3-5-sonnet-20241022") == 1


def test_wildcard_expansion_deduplicates_against_explicit_models():
    """
    Test that models already present in unique_models (with or without
    provider prefix) are not duplicated by wildcard expansion.
    """
    from litellm.proxy.auth.model_checks import _get_wildcard_models

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [
        {"litellm_params": {"model": "openai/some-key"}}
    ]

    # gpt-4 is explicitly configured (without prefix)
    unique_models = ["openai/*", "gpt-4"]

    with patch(
        "litellm.proxy.auth.model_checks.get_known_models_from_wildcard",
        return_value=["openai/gpt-4", "openai/gpt-4o", "openai/gpt-3.5-turbo"],
    ):
        result = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=mock_router,
        )

    # openai/gpt-4 should NOT appear since gpt-4 is already in unique_models
    assert "openai/gpt-4" not in result
    # Other expanded models should appear
    assert "openai/gpt-4o" in result
    assert "openai/gpt-3.5-turbo" in result
