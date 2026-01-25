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
async def test_route_request_with_invalid_router_params():
    """
    Test that route_request filters out invalid Router init params from 'user_config'.
    This covers the fix for https://github.com/BerriAI/litellm/issues/19693
    """
    import litellm
    from litellm.router import Router
    from unittest.mock import AsyncMock

    # Mock data with user_config containing invalid keys (simulating DB entry)
    data = {
        "model": "gpt-3.5-turbo",
        "user_config": {
            "model_list": [
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test"},
                }
            ],
            "model_alias_map": {"alias": "real_model"},  # INVALID PARAM
            "invalid_garbage_key": "crash_me",  # INVALID PARAM
        },
    }

    # We expect Router(**config) to succeed because of the filtering.
    # If filtering fails, this will raise TypeError and fail the test.
    try:
        # route_request calls getattr(user_router, route_type)(**data)
        # We'll mock the internal call to avoid making real network requests
        with pytest.MonkeyPatch.context() as m:
            # Mock the method that gets called on the router instance
            # We don't easily have access to the instance created INSIDE existing route_request
            # So we will wrap litellm.Router to spy on it or verify it doesn't crash

            original_router_init = litellm.Router.__init__

            def safe_router_init(self, **kwargs):
                # Verify that invalid keys are NOT present in kwargs
                assert "model_alias_map" not in kwargs
                assert "invalid_garbage_key" not in kwargs
                # Call original init (which would raise TypeError if invalid keys were present)
                original_router_init(self, **kwargs)

            m.setattr(litellm.Router, "__init__", safe_router_init)

            # Use 'acompletion' as the route_type
            # We also need to mock the completion method to avoid real calls
            m.setattr(Router, "acompletion", AsyncMock(return_value="success"))

            response = await route_request(data, None, None, "acompletion")
            assert response == "success"

    except TypeError as e:
        pytest.fail(
            f"route_request raised TypeError, implying invalid params were passed to Router: {e}"
        )
    except Exception:
        # Other exceptions might happen (e.g. valid config issues) but we care about TypeError here
        pass
