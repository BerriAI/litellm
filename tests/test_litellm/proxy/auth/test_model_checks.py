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


def test_is_model_deprecated_filters_past_dates():
    """
    Test that _is_model_deprecated returns True for models whose
    deprecation_date is in the past.
    """
    from litellm.proxy.auth.model_checks import _is_model_deprecated

    mock_cost = {
        "claude-3-5-sonnet-20240620": {
            "deprecation_date": "2025-06-01",
            "litellm_provider": "anthropic",
        },
        "claude-3-5-sonnet-20241022": {
            "deprecation_date": "2099-12-31",
            "litellm_provider": "anthropic",
        },
        "gpt-4-current": {
            "litellm_provider": "openai",
        },
    }

    with patch("litellm.proxy.auth.model_checks.litellm") as mock_litellm:
        mock_litellm.model_cost = mock_cost

        # Past deprecation date -> deprecated
        assert _is_model_deprecated("claude-3-5-sonnet-20240620") is True
        # Also works with provider prefix (bare name lookup)
        assert _is_model_deprecated("anthropic/claude-3-5-sonnet-20240620") is True

        # Future deprecation date -> not deprecated
        assert _is_model_deprecated("claude-3-5-sonnet-20241022") is False
        assert _is_model_deprecated("anthropic/claude-3-5-sonnet-20241022") is False

        # No deprecation_date field -> not deprecated
        assert _is_model_deprecated("gpt-4-current") is False

        # Unknown model -> not deprecated
        assert _is_model_deprecated("nonexistent-model") is False


def test_wildcard_expansion_filters_deprecated_models():
    """
    Test that _get_wildcard_models excludes models with a past
    deprecation_date from the expanded wildcard list.
    """
    from litellm.proxy.auth.model_checks import _get_wildcard_models

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [
        {"litellm_params": {"model": "anthropic/some-key"}}
    ]

    unique_models = ["anthropic/*"]

    expanded = [
        "anthropic/claude-3-haiku-20240307",      # no deprecation_date
        "anthropic/claude-3-5-sonnet-20240620",    # deprecated (past date)
        "anthropic/claude-3-5-sonnet-20241022",    # not deprecated (future date)
    ]

    mock_cost = {
        "claude-3-5-sonnet-20240620": {
            "deprecation_date": "2025-06-01",
            "litellm_provider": "anthropic",
        },
        "claude-3-5-sonnet-20241022": {
            "deprecation_date": "2099-12-31",
            "litellm_provider": "anthropic",
        },
    }

    with patch(
        "litellm.proxy.auth.model_checks.get_known_models_from_wildcard",
        return_value=expanded,
    ), patch("litellm.proxy.auth.model_checks.litellm") as mock_litellm:
        mock_litellm.model_cost = mock_cost

        result = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=mock_router,
        )

    # Deprecated model should be filtered out
    assert "anthropic/claude-3-5-sonnet-20240620" not in result
    # Non-deprecated models should remain
    assert "anthropic/claude-3-haiku-20240307" in result
    assert "anthropic/claude-3-5-sonnet-20241022" in result
