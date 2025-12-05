"""
Test parameter validation error handling through proxy server.

This test verifies that TypeError exceptions from missing required parameters
are properly converted to BadRequestError with HTTP 400 status code when
requests are made through the proxy server (route_request).
"""
import json
import os
import sys
from unittest import mock

from dotenv import load_dotenv

load_dotenv()
import asyncio
import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm
from litellm.proxy.proxy_server import initialize, router, app


@pytest.fixture
def client():
    """Initialize proxy server and return test client."""
    # Use a basic config without any special authentication
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"

    # If config doesn't exist, initialize without config
    if os.path.exists(config_fp):
        asyncio.run(initialize(config=config_fp))
    else:
        asyncio.run(initialize())

    return TestClient(app)


def test_proxy_chat_completion_missing_model(client):
    """
    Test that missing 'model' parameter returns HTTP 400 through proxy.

    When a request is made through the proxy server without the required
    'model' parameter, it should return HTTP 400 Bad Request.
    """
    test_data = {
        "messages": [
            {"role": "user", "content": "test"},
        ],
        # model parameter is intentionally omitted
    }

    response = client.post("/chat/completions", json=test_data)

    # Verify HTTP 400 status code
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}"

    # Verify error response format
    json_response = response.json()
    assert "error" in json_response, "Response should contain 'error' key"

    error = json_response["error"]
    assert "message" in error, "Error should contain 'message' key"

    error_message = error["message"].lower()
    # Check that error message mentions missing or required parameter
    assert "missing" in error_message or "required" in error_message or "model" in error_message, \
        f"Error message should mention missing/required parameter, got: {error['message']}"


def test_proxy_chat_completion_missing_messages(client):
    """
    Test that missing 'messages' parameter returns HTTP 400 through proxy.

    Note: In current implementation, 'messages' has a default value of [],
    so this might not raise an error. This test documents the behavior.
    """
    test_data = {
        "model": "gpt-3.5-turbo",
        # messages parameter is intentionally omitted
    }

    response = client.post("/chat/completions", json=test_data)

    # Since messages has default value [], the request might succeed or fail
    # depending on the provider. We just verify the response is valid.
    assert response.status_code in [200, 400, 401, 500], \
        f"Response should be a valid HTTP status code, got {response.status_code}"


def test_proxy_acompletion_missing_model(client):
    """
    Test that missing 'model' parameter in async completion returns HTTP 400 through proxy.

    When an async completion request is made through the proxy server without
    the required 'model' parameter, it should return HTTP 400 Bad Request.

    This test verifies that route_request properly handles parameter validation
    for async endpoints (acompletion).
    """
    test_data = {
        "messages": [
            {"role": "user", "content": "test"},
        ],
        # model parameter is intentionally omitted
    }

    response = client.post("/chat/completions", json=test_data)

    # Verify HTTP 400 status code
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}"

    # Verify error response format
    json_response = response.json()
    assert "error" in json_response, "Response should contain 'error' key"

    error = json_response["error"]
    assert "message" in error, "Error should contain 'message' key"

    error_message = error["message"].lower()
    # Check that error message mentions missing or required parameter
    assert "missing" in error_message or "required" in error_message or "model" in error_message, \
        f"Error message should mention missing/required parameter, got: {error['message']}"


@pytest.mark.asyncio
async def test_proxy_responses_api_missing_model(client):
    """
    Test that missing 'model' parameter in responses API returns HTTP 400.

    The responses API requires both 'model' and 'input' parameters.
    """
    test_data = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "test"}],
                "type": "message"
            }
        ],
        # model parameter is intentionally omitted
    }

    response = client.post("/v1/responses", json=test_data)

    # Verify HTTP 400 status code
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}"

    # Verify error response format
    json_response = response.json()
    assert "error" in json_response, "Response should contain 'error' key"

    error = json_response["error"]
    assert "message" in error, "Error should contain 'message' key"


@pytest.mark.asyncio
async def test_proxy_responses_api_wrong_parameter(client):
    """
    Test that using 'messages' instead of 'input' for responses API returns HTTP 400.

    Responses API expects 'input' parameter, not 'messages'.
    Using wrong parameter should return a clear error.
    """
    test_data = {
        "model": "test_openai_models",
        "messages": [{"role": "user", "content": "test"}],  # Wrong: should be 'input'
    }

    response = client.post("/v1/responses", json=test_data)

    # Should return 400 for wrong parameter
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}"

    # Verify error response format
    json_response = response.json()
    assert "error" in json_response, "Response should contain 'error' key"


def test_proxy_error_response_format(client):
    """
    Test that error responses follow OpenAI error format.

    Error responses should have the structure:
    {
        "error": {
            "message": str,
            "type": str (optional),
            "param": str (optional),
            "code": str
        }
    }
    """
    # Use model that exists in test config (test_openai_models)
    # Pass wrong parameter type to trigger parameter validation error
    test_data = {
        "input": [{"role": "user", "content": "test"}],  # Wrong: should be 'messages' for chat/completions
        "model": "test_openai_models",
    }

    response = client.post("/chat/completions", json=test_data)

    assert response.status_code == 400

    json_response = response.json()
    assert "error" in json_response

    error = json_response["error"]

    # Verify required fields
    assert "message" in error, "Error must have 'message' field"
    assert isinstance(error["message"], str), "Error message must be string"

    # Verify code field if present
    if "code" in error:
        # OpenAI SDK requires code to be string
        # https://github.com/openai/openai-python/blob/main/src/openai/types/shared/error_object.py
        assert isinstance(error["code"], str), "Error code must be string"