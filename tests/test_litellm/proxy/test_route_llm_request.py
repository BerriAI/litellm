import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.route_llm_request import route_request


@pytest.mark.parametrize(
    "route_type",
    [
        "atext_completion",
        "acompletion",
        "aembedding",
        "aimage_generation",
        "aspeech",
        "atranscription",
        "amoderation",
        "arerank",
    ],
)
@pytest.mark.asyncio
async def test_route_request_dynamic_credentials(route_type):
    data = {
        "model": "openai/gpt-4o-mini-2024-07-18",
        "api_key": "my-bad-key",
        "api_base": "https://api.openai.com/v1 ",
    }
    llm_router = MagicMock()
    # Ensure that the dynamic method exists on the llm_router mock.
    getattr(llm_router, route_type).return_value = "fake_response"

    response = await route_request(data, llm_router, None, route_type)
    # Optionally verify the response if needed:
    assert response == "fake_response"
    # Now assert that the dynamic method was called once with the expected kwargs.
    getattr(llm_router, route_type).assert_called_once_with(**data)


@pytest.mark.asyncio
async def test_route_request_no_model_required():
    """Test route types that don't require model parameter"""
    test_cases = [
        "amoderation",
        "aget_responses",
        "adelete_responses",
        "avector_store_create",
        "avector_store_search",
    ]

    for route_type in test_cases:
        # Test data without model parameter
        data = {"input": "test input", "api_key": "test-key"}

        llm_router = MagicMock()
        getattr(llm_router, route_type).return_value = "fake_response"

        response = await route_request(data, llm_router, None, route_type)

        # Verify response
        assert response == "fake_response"
        # Verify the method was called with correct parameters
        getattr(llm_router, route_type).assert_called_once_with(**data)

        # Reset mock for next iteration
        llm_router.reset_mock()


@pytest.mark.asyncio
async def test_route_request_no_model_required_with_router_settings():
    """Test route types that don't require model parameter with router settings"""
    test_cases = [
        "amoderation",
        "aget_responses",
        "adelete_responses",
        "avector_store_create",
        "avector_store_search",
    ]

    for route_type in test_cases:
        # Test data with model parameter (it will be ignored for these route types)
        data = {
            "input": "test input",
            "model": "test-model",  # Include dummy model to avoid KeyError
        }

        llm_router = MagicMock()
        # Set up router settings
        llm_router.router_general_settings.pass_through_all_models = False
        llm_router.default_deployment = None
        llm_router.pattern_router.patterns = []
        llm_router.model_names = []  # Empty model names list
        llm_router.get_model_ids.return_value = []  # Empty model IDs
        llm_router.model_group_alias = None  # No model group alias

        # Mock the async route call
        getattr(llm_router, route_type).return_value = "fake_response"

        # Run the request
        response = await route_request(data, llm_router, None, route_type)

        # Assert the mocked method was called with expected input
        assert response == "fake_response"
        getattr(llm_router, route_type).assert_called_once_with(**data)

        # Reset the mock for the next route
        llm_router.reset_mock()


@pytest.mark.asyncio
async def test_route_request_no_model_required_with_router_settings_and_no_router():
    """Test route types that don't require model parameter with router settings and no router"""
    from unittest.mock import patch

    import litellm
    from litellm.proxy.route_llm_request import route_request

    data = {
        "model": "my-model-id",
        "api_key": "my-api-key",
        "messages": [{"role": "user", "content": "what llm are you"}],
    }

    with patch.object(
        litellm, "acompletion", return_value="fake_response"
    ) as mock_completion:
        await route_request(data, None, "gpt-3.5-turbo", "acompletion")

        mock_completion.assert_called_once_with(**data)


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override():
    """
    Test that route_request handles router_settings_override by merging settings into kwargs
    instead of creating a new Router (which is expensive and was the old behavior).
    """
    # Mock data with router_settings_override containing per-request settings
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["gpt-4"]}],
            "num_retries": 5,
            "timeout": 30,
            "model_group_retry_policy": {"gpt-3.5-turbo": {"RateLimitErrorRetries": 3}},
            # These settings should be ignored (not in per_request_settings list)
            "routing_strategy": "least-busy",
            "model_group_alias": {"alias": "real_model"},
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "success"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    # Verify the router method was called with merged settings
    call_kwargs = llm_router.acompletion.call_args[1]
    assert call_kwargs["fallbacks"] == [{"gpt-3.5-turbo": ["gpt-4"]}]
    assert call_kwargs["num_retries"] == 5
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["model_group_retry_policy"] == {"gpt-3.5-turbo": {"RateLimitErrorRetries": 3}}
    # Verify unsupported settings were NOT merged
    assert "routing_strategy" not in call_kwargs
    assert "model_group_alias" not in call_kwargs
    # Verify router_settings_override was removed from data
    assert "router_settings_override" not in call_kwargs


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override_no_router():
    """
    Test that router_settings_override works when no router is provided,
    falling back to litellm module directly.
    """
    import litellm

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["gpt-4"]}],
            "num_retries": 3,
        },
    }

    # Use MagicMock explicitly to avoid auto-AsyncMock behavior in Python 3.12+
    mock_completion = MagicMock(return_value="success")
    original_acompletion = litellm.acompletion
    litellm.acompletion = mock_completion

    try:
        response = await route_request(data, None, None, "acompletion")

        assert response == "success"
        # Verify litellm.acompletion was called with merged settings
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["fallbacks"] == [{"gpt-3.5-turbo": ["gpt-4"]}]
        assert call_kwargs["num_retries"] == 3
    finally:
        litellm.acompletion = original_acompletion


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override_preserves_existing():
    """
    Test that router_settings_override does not override settings already in the request.
    Request-level settings take precedence over key/team settings.
    """
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "num_retries": 10,  # Request-level setting
        "router_settings_override": {
            "num_retries": 3,  # Key/team setting - should NOT override
            "timeout": 30,  # Key/team setting - should be applied
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "success"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    call_kwargs = llm_router.acompletion.call_args[1]
    # Request-level num_retries should take precedence
    assert call_kwargs["num_retries"] == 10
    # Key/team timeout should be applied since not in request
    assert call_kwargs["timeout"] == 30
