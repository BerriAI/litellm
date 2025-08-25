import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.llms.custom_httpx.http_handler import HTTPHandler

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

import litellm
from litellm.passthrough.main import llm_passthrough_route


def test_llm_passthrough_route():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(
        client.client,
        "send",
        return_value=MagicMock(status_code=200, json={"message": "Hello, world!"}),
    ) as mock_post:
        response = llm_passthrough_route(
            model="vllm/anthropic.claude-3-5-sonnet-20240620-v1:0",
            endpoint="v1/chat/completions",
            method="POST",
            request_url="http://localhost:8000/v1/chat/completions",
            api_base="http://localhost:8090",
            json={
                "model": "my-custom-model",
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
            client=client,
        )

        mock_post.call_args.kwargs[
            "request"
        ].url == "http://localhost:8090/v1/chat/completions"

        assert response.status_code == 200
        assert response.json == {"message": "Hello, world!"}


def test_bedrock_application_inference_profile_url_encoding():
    client = HTTPHandler()
    
    mock_provider_config = MagicMock()
    mock_provider_config.get_complete_url.return_value = (
        httpx.URL("https://bedrock-runtime.us-east-1.amazonaws.com/model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd/converse"),
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    mock_provider_config.get_api_key.return_value = "test-key"
    mock_provider_config.validate_environment.return_value = {}
    mock_provider_config.sign_request.return_value = ({}, None)
    mock_provider_config.is_streaming_request.return_value = False

    with patch("litellm.utils.ProviderConfigManager.get_provider_passthrough_config", return_value=mock_provider_config), \
         patch("litellm.litellm_core_utils.get_litellm_params.get_litellm_params", return_value={}), \
         patch("litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider", return_value=("test-model", "bedrock", "test-key", "test-base")), \
         patch.object(client.client, "send", return_value=MagicMock(status_code=200)) as mock_send, \
         patch.object(client.client, "build_request") as mock_build_request:
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()
        
        response = llm_passthrough_route(
            model="arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd",
            endpoint="model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd/converse",
            method="POST",
            custom_llm_provider="bedrock",
            client=client,
            litellm_logging_obj=mock_logging_obj,
        )

        # Verify that build_request was called with the encoded URL
        mock_build_request.assert_called_once()
        call_args = mock_build_request.call_args
        
        # The URL should have the application-inference-profile ID encoded
        actual_url = str(call_args.kwargs["url"])
        assert "application-inference-profile%2Fr742sbn2zckd" in actual_url
        assert response.status_code == 200


def test_bedrock_non_application_inference_profile_no_encoding():
    client = HTTPHandler()
    
    # Mock the provider config and its methods
    mock_provider_config = MagicMock()
    mock_provider_config.get_complete_url.return_value = (
        httpx.URL("https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/converse"),
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    mock_provider_config.get_api_key.return_value = "test-key"
    mock_provider_config.validate_environment.return_value = {}
    mock_provider_config.sign_request.return_value = ({}, None)
    mock_provider_config.is_streaming_request.return_value = False

    with patch("litellm.utils.ProviderConfigManager.get_provider_passthrough_config", return_value=mock_provider_config), \
         patch("litellm.litellm_core_utils.get_litellm_params.get_litellm_params", return_value={}), \
         patch("litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider", return_value=("test-model", "bedrock", "test-key", "test-base")), \
         patch.object(client.client, "send", return_value=MagicMock(status_code=200)) as mock_send, \
         patch.object(client.client, "build_request") as mock_build_request:
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()
        
        response = llm_passthrough_route(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            endpoint="model/anthropic.claude-3-sonnet-20240229-v1:0/converse",
            method="POST",
            custom_llm_provider="bedrock",
            client=client,
            litellm_logging_obj=mock_logging_obj,
        )

        # Verify that build_request was called with the original URL (no encoding)
        mock_build_request.assert_called_once()
        call_args = mock_build_request.call_args
        
        # The URL should NOT have application-inference-profile encoding
        actual_url = str(call_args.kwargs["url"])
        assert "application-inference-profile%2F" not in actual_url
        assert "anthropic.claude-3-sonnet-20240229-v1:0" in actual_url
        assert response.status_code == 200


def test_update_stream_param_based_on_request_body():
    """
    Test _update_stream_param_based_on_request_body handles stream parameter correctly.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )

    # Test 1: stream in request body should take precedence
    parsed_body = {"stream": True, "model": "test-model"}
    result = HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
        parsed_body=parsed_body, stream=False
    )
    assert result is True
    
    # Test 2: no stream in request body should return original stream param
    parsed_body = {"model": "test-model"}
    result = HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
        parsed_body=parsed_body, stream=False
    )
    assert result is False
    
    # Test 3: stream=False in request body should return False
    parsed_body = {"stream": False, "model": "test-model"}
    result = HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
        parsed_body=parsed_body, stream=True
    )
    assert result is False
    
    # Test 4: no stream param provided, no stream in body
    parsed_body = {"model": "test-model"}
    result = HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
        parsed_body=parsed_body, stream=None
    )
    assert result is None