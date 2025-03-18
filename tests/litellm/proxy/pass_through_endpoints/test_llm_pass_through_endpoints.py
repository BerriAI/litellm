import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    BaseOpenAIPassThroughHandler,
    RouteChecks,
    create_pass_through_route,
)


class TestBaseOpenAIPassThroughHandler:

    def test_join_url_paths(self):
        print("\nTesting _join_url_paths method...")

        # Test joining base URL with no path and a path
        base_url = httpx.URL("https://api.example.com")
        path = "/v1/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with no path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test joining base URL with path and another path
        base_url = httpx.URL("https://api.example.com/v1")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with path not starting with slash
        base_url = httpx.URL("https://api.example.com/v1")
        path = "chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Path without leading slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with base URL having trailing slash
        base_url = httpx.URL("https://api.example.com/v1/")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with trailing slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

    def test_append_openai_beta_header(self):
        print("\nTesting _append_openai_beta_header method...")

        # Create mock requests with different paths
        assistants_request = MagicMock(spec=Request)
        assistants_request.url = MagicMock()
        assistants_request.url.path = "/v1/threads/thread_123456/messages"

        non_assistants_request = MagicMock(spec=Request)
        non_assistants_request.url = MagicMock()
        non_assistants_request.url.path = "/v1/chat/completions"

        headers = {"authorization": "Bearer test_key"}

        # Test with assistants API request
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistants_request
        )
        print(f"Assistants API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"

        # Test with non-assistants API request
        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, non_assistants_request
        )
        print(f"Non-assistants API request: Headers: {result}")
        assert "OpenAI-Beta" not in result

        # Test with assistant in the path
        assistant_request = MagicMock(spec=Request)
        assistant_request.url = MagicMock()
        assistant_request.url.path = "/v1/assistants/asst_123456"

        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistant_request
        )
        print(f"Assistant API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"

    def test_assemble_headers(self):
        print("\nTesting _assemble_headers method...")

        # Mock request
        mock_request = MagicMock(spec=Request)
        api_key = "test_api_key"

        # Patch the _append_openai_beta_header method to avoid testing it again
        with patch.object(
            BaseOpenAIPassThroughHandler,
            "_append_openai_beta_header",
            return_value={
                "authorization": "Bearer test_api_key",
                "api-key": "test_api_key",
                "test-header": "value",
            },
        ):
            result = BaseOpenAIPassThroughHandler._assemble_headers(
                api_key, mock_request
            )
            print(f"Assembled headers: {result}")
            assert result["authorization"] == "Bearer test_api_key"
            assert result["api-key"] == "test_api_key"
            assert result["test-header"] == "value"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
    )
    async def test_base_openai_pass_through_handler(self, mock_create_pass_through):
        print("\nTesting _base_openai_pass_through_handler method...")

        # Mock dependencies
        mock_request = MagicMock(spec=Request)
        mock_request.query_params = {"model": "gpt-4"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock the endpoint function returned by create_pass_through_route
        mock_endpoint_func = MagicMock()
        mock_endpoint_func.return_value = {"result": "success"}
        mock_create_pass_through.return_value = mock_endpoint_func

        print("Testing standard endpoint pass-through...")
        # Test with standard endpoint
        result = await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
            base_target_url="https://api.openai.com",
            api_key="test_api_key",
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        )

        # Verify the result
        print(f"Result from handler: {result}")
        assert result == {"result": "success"}

        # Verify create_pass_through_route was called with correct parameters
        call_args = mock_create_pass_through.call_args[1]
        print(
            f"create_pass_through_route called with endpoint: {call_args['endpoint']}"
        )
        print(f"create_pass_through_route called with target: {call_args['target']}")
        assert call_args["endpoint"] == "/chat/completions"
        assert call_args["target"] == "https://api.openai.com/v1/chat/completions"

        # Verify endpoint_func was called with correct parameters
        print("Verifying endpoint_func call parameters...")
        call_kwargs = mock_endpoint_func.call_args[1]
        print(f"stream parameter: {call_kwargs['stream']}")
        print(f"query_params: {call_kwargs['query_params']}")
        assert call_kwargs["stream"] is False
        assert call_kwargs["query_params"] == {"model": "gpt-4"}
