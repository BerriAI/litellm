from unittest.mock import AsyncMock, patch, MagicMock

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


@pytest.mark.parametrize(
    "wildcard_model,litellm_params,expected_models",
    [
        # Test case 1: litellm_params is None (the main bug fixed in https://github.com/BerriAI/litellm/pull/14125)
        (
            "openai/*",
            None,
            ["openai/gpt-4", "openai/gpt-3.5-turbo", "openai/gpt-4o"]  # Mock static models
        ),
        # Test case 2: litellm_params provided (existing functionality)
        (
            "anthropic/*", 
            MagicMock(model="anthropic/claude-3-haiku"),
            ["anthropic/claude-3-haiku", "anthropic/claude-3-sonnet"]  # Mock static models
        ),
    ],
)
def test_get_known_models_from_wildcard_with_none_params(wildcard_model, litellm_params, expected_models):
    """
    Test that get_known_models_from_wildcard correctly handles the case when litellm_params is None.
    This tests the fix for the bug where wildcard models were not being expanded when litellm_params was None.
    """
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    
    # Mock the get_provider_models function to return our expected models without the provider prefix
    with patch("litellm.proxy.auth.model_checks.get_provider_models") as mock_get_provider_models:
        if wildcard_model.startswith("openai/"):
            mock_get_provider_models.return_value = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
        elif wildcard_model.startswith("anthropic/"):
            mock_get_provider_models.return_value = ["claude-3-haiku", "claude-3-sonnet"]
        
        result = get_known_models_from_wildcard(wildcard_model, litellm_params)
        
        # Verify the result contains the expected models with the correct provider prefix
        assert result == expected_models
        assert len(result) > 0, "Should return expanded models, not empty list"


def test_get_wildcard_models_with_router_fallback():
    """
    Test that _get_wildcard_models falls back to direct expansion when router lookup fails.
    This tests the fix for the router fallback logic.
    """
    from litellm.proxy.auth.model_checks import _get_wildcard_models
    
    # Create a mock router that returns None for wildcard model lookup
    mock_router = MagicMock()
    mock_router.get_model_list.return_value = None  # Simulate router lookup failure
    
    unique_models = ["openai/*", "anthropic/*"]
    
    # Mock the get_known_models_from_wildcard to return expanded models
    with patch("litellm.proxy.auth.model_checks.get_known_models_from_wildcard") as mock_expand:
        mock_expand.side_effect = [
            ["openai/gpt-4", "openai/gpt-3.5-turbo"],  # For openai/*
            ["anthropic/claude-3-haiku", "anthropic/claude-3-sonnet"]  # For anthropic/*
        ]
        
        result = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=mock_router
        )
        
        # Verify that the wildcard models were expanded despite router lookup failure
        expected_models = [
            "openai/gpt-4", "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku", "anthropic/claude-3-sonnet"
        ]
        assert result == expected_models
        
        # Verify router was called but fallback was used
        mock_router.get_model_list.assert_called()
        
        # Verify direct expansion was called for both models
        assert mock_expand.call_count == 2


def test_get_provider_models_fallback_to_static():
    """
    Test that get_provider_models falls back to static model list when API calls fail.
    This tests the error handling and fallback logic.
    """
    from litellm.proxy.auth.model_checks import get_provider_models
    import litellm
    
    # Mock the static models list
    mock_static_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
    
    with patch("litellm.models_by_provider", {"openai": mock_static_models}):
        # Test case 1: get_valid_models returns empty list
        with patch("litellm.proxy.auth.model_checks.get_valid_models") as mock_get_valid:
            mock_get_valid.return_value = []  # Simulate API failure returning empty
            
            result = get_provider_models("openai")
            
            # Should fall back to static model list
            assert result == mock_static_models
        
        # Test case 2: get_valid_models raises exception
        with patch("litellm.proxy.auth.model_checks.get_valid_models") as mock_get_valid:
            mock_get_valid.side_effect = Exception("API Error")  # Simulate API exception
            
            result = get_provider_models("openai")
            
            # Should fall back to static model list
            assert result == mock_static_models


def test_wildcard_expansion_integration():
    """
    Integration test for the complete wildcard expansion flow.
    This tests the full flow from configuration to expanded model list.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    
    # Mock the static models
    mock_openai_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
    
    with patch("litellm.models_by_provider", {"openai": mock_openai_models}):
        with patch("litellm.proxy.auth.model_checks.get_valid_models") as mock_get_valid:
            # Simulate API failure to test fallback
            mock_get_valid.return_value = []
            
            result = get_complete_model_list(
                key_models=[],
                team_models=[],
                proxy_model_list=["openai/*"],  # Wildcard model in proxy config
                user_model=None,
                infer_model_from_keys=False,
                return_wildcard_routes=False,  # Should expand, not return wildcard
                llm_router=None  # No router, should use direct expansion
            )
            
            # Should return expanded models with provider prefix
            expected = ["openai/gpt-4", "openai/gpt-3.5-turbo", "openai/gpt-4o"]
            assert result == expected
            assert "openai/*" not in result, "Should not contain the original wildcard"
