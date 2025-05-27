import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

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
    test_cases = ["amoderation", "aget_responses", "adelete_responses"]

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
    test_cases = ["amoderation", "aget_responses", "adelete_responses"]

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
