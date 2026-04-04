import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

import litellm
from litellm.passthrough.main import allm_passthrough_route, llm_passthrough_route


def test_llm_passthrough_route():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(
        client.client,
        "send",
        return_value=MagicMock(status_code=200, json={"message": "Hello, world!"}),
    ) as mock_post:
        response = llm_passthrough_route(
            model="vllm/anthropic.claude-haiku-4-5-20251001-v1:0",
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
        httpx.URL(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd/converse"
        ),
        "https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    mock_provider_config.get_api_key.return_value = "test-key"
    mock_provider_config.validate_environment.return_value = {}
    mock_provider_config.sign_request.return_value = ({}, None)
    mock_provider_config.is_streaming_request.return_value = False

    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_passthrough_config",
        return_value=mock_provider_config,
    ), patch(
        "litellm.litellm_core_utils.get_litellm_params.get_litellm_params",
        return_value={},
    ), patch(
        "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
        return_value=("test-model", "bedrock", "test-key", "test-base"),
    ), patch.object(
        client.client, "send", return_value=MagicMock(status_code=200)
    ) as mock_send, patch.object(
        client.client, "build_request"
    ) as mock_build_request:

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
        httpx.URL(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/converse"
        ),
        "https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    mock_provider_config.get_api_key.return_value = "test-key"
    mock_provider_config.validate_environment.return_value = {}
    mock_provider_config.sign_request.return_value = ({}, None)
    mock_provider_config.is_streaming_request.return_value = False

    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_passthrough_config",
        return_value=mock_provider_config,
    ), patch(
        "litellm.litellm_core_utils.get_litellm_params.get_litellm_params",
        return_value={},
    ), patch(
        "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
        return_value=("test-model", "bedrock", "test-key", "test-base"),
    ), patch.object(
        client.client, "send", return_value=MagicMock(status_code=200)
    ) as mock_send, patch.object(
        client.client, "build_request"
    ) as mock_build_request:

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


@pytest.fixture
def mock_request():
    """Create a mock request with headers"""
    from typing import Optional

    class QueryParams:
        def __init__(self):
            self._dict = {}

        def __iter__(self):
            return iter(self._dict)

        def items(self):
            return self._dict.items()

    class MockRequest:
        def __init__(
            self, headers=None, method="POST", request_body: Optional[dict] = None
        ):
            self.headers = headers or {}
            self.query_params = QueryParams()
            self.method = method
            self.request_body = request_body or {}
            # Add url attribute that the actual code expects
            self.url = "http://localhost:8000/test"

        async def body(self) -> bytes:
            return bytes(json.dumps(self.request_body), "utf-8")

    return MockRequest


@pytest.fixture
def mock_user_api_key_dict():
    """Create a mock user API key dictionary"""
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        end_user_id="test-user",
    )


@pytest.mark.asyncio
async def test_pass_through_request_stream_param_override(
    mock_request, mock_user_api_key_dict
):
    """
    Test that when stream=None is passed as parameter but stream=True
    is in request body, the request body value takes precedence and
    the eventual POST request uses streaming.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    # Create request body with stream=True
    request_body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello, world"}],
        "stream": True,  # This should override the function parameter
    }

    # Create a mock streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    # Mock the streaming response behavior
    async def mock_aiter_bytes():
        yield b'data: {"content": "Hello"}\n\n'
        yield b'data: {"content": "World"}\n\n'
        yield b"data: [DONE]\n\n"

    mock_response.aiter_bytes = mock_aiter_bytes

    # Create mocks for the async client
    mock_async_client = AsyncMock()
    mock_request_obj = AsyncMock()

    # Mock build_request to return a request object (it's a sync method)
    mock_async_client.build_request = Mock(return_value=mock_request_obj)

    # Mock send to return the streaming response
    mock_async_client.send.return_value = mock_response

    # Mock get_async_httpx_client to return our mock client
    mock_client_obj = Mock()
    mock_client_obj.client = mock_async_client

    # Create the request
    request = mock_request(headers={}, method="POST", request_body=request_body)

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_client_obj,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        return_value=request_body,  # Return the request body unchanged
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new=AsyncMock(),  # Mock the success handler
    ):
        # Call pass_through_request with stream=False parameter
        response = await pass_through_request(
            request=request,
            target="https://api.anthropic.com/v1/messages",
            custom_headers={"Authorization": "Bearer test-key"},
            user_api_key_dict=mock_user_api_key_dict,
            stream=None,  # This should be overridden by request body
        )

        # Verify that build_request was called (indicating streaming path)
        mock_async_client.build_request.assert_called_once_with(
            "POST",
            httpx.URL("https://api.anthropic.com/v1/messages"),
            json=request_body,
            params={},
            headers={"Authorization": "Bearer test-key"},
        )

        # Verify that send was called with stream=True
        mock_async_client.send.assert_called_once_with(
            mock_request_obj,
            stream=True,  # This proves that stream=True from request body was used
        )

        # Verify that the non-streaming request method was NOT called
        mock_async_client.request.assert_not_called()

        # Verify response is a StreamingResponse
        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_pass_through_request_stream_param_no_override(
    mock_request, mock_user_api_key_dict
):
    """
    Test that when stream=False is passed as parameter and no stream
    is in request body, the function parameter is used and
    the eventual request uses non-streaming.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    # Create request body without stream parameter
    request_body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello, world"}],
        # No stream parameter - should use function parameter stream=False
    }

    # Create a mock non-streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response._content = b'{"response": "Hello world"}'

    async def mock_aread():
        return mock_response._content

    mock_response.aread = mock_aread

    # Create mocks for the async client
    mock_async_client = AsyncMock()

    # Mock request to return the non-streaming response
    mock_async_client.request.return_value = mock_response

    # Mock get_async_httpx_client to return our mock client
    mock_client_obj = Mock()
    mock_client_obj.client = mock_async_client

    # Create the request
    request = mock_request(headers={}, method="POST", request_body=request_body)

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=mock_client_obj,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
        return_value=request_body,  # Return the request body unchanged
    ), patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
        new=AsyncMock(),  # Mock the success handler
    ):
        # Call pass_through_request with stream=False parameter
        response = await pass_through_request(
            request=request,
            target="https://api.anthropic.com/v1/messages",
            custom_headers={"Authorization": "Bearer test-key"},
            user_api_key_dict=mock_user_api_key_dict,
            stream=False,  # Should be used since no stream in request body
        )

        # Verify that build_request was NOT called (no streaming path)
        mock_async_client.build_request.assert_not_called()

        # Verify that send was NOT called (no streaming path)
        mock_async_client.send.assert_not_called()

        # Verify that the non-streaming request method WAS called
        mock_async_client.request.assert_called_once_with(
            method="POST",
            url=httpx.URL("https://api.anthropic.com/v1/messages"),
            headers={"Authorization": "Bearer test-key"},
            params={},
            json=request_body,
        )

        # Verify response is a regular Response (not StreamingResponse)
        from fastapi.responses import Response, StreamingResponse

        assert not isinstance(response, StreamingResponse)
        assert isinstance(response, Response)
        assert response.status_code == 200


def test_azure_with_custom_api_base_and_key():
    """
    Test that llm_passthrough_route correctly handles Azure OpenAI
    with custom api_base and api_key.
    """
    client = HTTPHandler()

    # Mock the provider config and its methods
    mock_provider_config = MagicMock()
    mock_provider_config.get_complete_url.return_value = (
        httpx.URL(
            "https://my-custom-base/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-01"
        ),
        "https://my-custom-base",
    )
    mock_provider_config.get_api_key.return_value = "my-custom-key"
    mock_provider_config.validate_environment.return_value = {
        "api-key": "my-custom-key"
    }
    mock_provider_config.sign_request.return_value = (
        {"api-key": "my-custom-key"},
        None,
    )
    mock_provider_config.is_streaming_request.return_value = False

    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_passthrough_config",
        return_value=mock_provider_config,
    ), patch(
        "litellm.litellm_core_utils.get_litellm_params.get_litellm_params",
        return_value={},
    ), patch(
        "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
        return_value=("gpt-4.1", "azure", "my-custom-key", "https://my-custom-base"),
    ), patch.object(
        client.client,
        "send",
        return_value=MagicMock(
            status_code=200, json=lambda: {"id": "chatcmpl-123", "choices": []}
        ),
    ) as mock_send, patch.object(
        client.client, "build_request"
    ) as mock_build_request:

        # Mock logging object
        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()

        response = llm_passthrough_route(
            model="azure/gpt-4.1",
            endpoint="openai/deployments/gpt-4.1/chat/completions",
            method="POST",
            custom_llm_provider="azure",
            api_base="https://my-custom-base",
            api_key="my-custom-key",
            json={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
            client=client,
            litellm_logging_obj=mock_logging_obj,
        )

        # Verify that build_request was called with the correct parameters
        mock_build_request.assert_called_once()
        call_args = mock_build_request.call_args

        # Verify the URL contains the custom base
        actual_url = str(call_args.kwargs["url"])
        assert "my-custom-base" in actual_url
        assert "gpt-4.1" in actual_url

        # Verify the headers contain the custom API key
        headers = call_args.kwargs["headers"]
        assert headers["api-key"] == "my-custom-key"

        # Verify the model in JSON body is updated
        json_body = call_args.kwargs["json"]
        assert json_body["model"] == "gpt-4.1"

        assert response.status_code == 200  # type: ignore[union-attr]


def test_content_param_forwarded_to_build_request():
    """
    Regression test: the `content` parameter passed to llm_passthrough_route
    must be forwarded to build_request instead of silently dropped.
    When content is provided and signed_json_body is None, build_request should
    receive content=<value> and data=None, json=None.
    """
    client = HTTPHandler()

    mock_provider_config = MagicMock()
    mock_provider_config.get_complete_url.return_value = (
        httpx.URL("https://my-azure.openai.azure.com/openai/deployments/gpt-4/chat/completions"),
        "https://my-azure.openai.azure.com",
    )
    mock_provider_config.get_api_key.return_value = "test-key"
    mock_provider_config.validate_environment.return_value = {"api-key": "test-key"}
    # sign_request returns (headers, None) — no signed body, so content should be used
    mock_provider_config.sign_request.return_value = ({"api-key": "test-key"}, None)
    mock_provider_config.is_streaming_request.return_value = False

    raw_content = b'{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}'

    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_passthrough_config",
        return_value=mock_provider_config,
    ), patch(
        "litellm.litellm_core_utils.get_litellm_params.get_litellm_params",
        return_value={},
    ), patch(
        "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
        return_value=("gpt-4", "azure", "test-key", "https://my-azure.openai.azure.com"),
    ), patch.object(
        client.client, "send", return_value=MagicMock(status_code=200)
    ), patch.object(
        client.client, "build_request"
    ) as mock_build_request:

        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()

        llm_passthrough_route(
            model="azure/gpt-4",
            endpoint="openai/deployments/gpt-4/chat/completions",
            method="POST",
            custom_llm_provider="azure",
            content=raw_content,
            data=None,
            json=None,
            client=client,
            litellm_logging_obj=mock_logging_obj,
        )

        mock_build_request.assert_called_once()
        call_kwargs = mock_build_request.call_args.kwargs
        # content must be forwarded (not dropped)
        assert call_kwargs["content"] == raw_content
        # data and json must be None when content is provided
        assert call_kwargs["data"] is None
        assert call_kwargs["json"] is None


def _make_429_streaming_response() -> MagicMock:
    """Build a mock httpx.Response that looks like a streaming 429 from Azure."""
    error_body = json.dumps(
        {"error": {"code": "429", "message": "Rate limit exceeded. Retry after 10 seconds."}}
    ).encode()

    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 429
    mock.headers = httpx.Headers({"content-type": "application/json"})

    def _raise_for_status():
        request = httpx.Request(
            "POST",
            "https://my-azure.openai.azure.com/openai/deployments/gpt-4/responses",
        )
        raise httpx.HTTPStatusError(
            message="429 Too Many Requests",
            request=request,
            response=httpx.Response(
                status_code=429,
                content=error_body,
                request=request,
            ),
        )

    mock.raise_for_status = _raise_for_status

    async def _aiter_bytes():
        yield error_body

    mock.aiter_bytes = _aiter_bytes
    return mock


@pytest.mark.asyncio
async def test_allm_passthrough_route_429_streaming_raises():
    """
    Regression test: Azure 429 during streaming must raise HTTPStatusError,
    not be silently forwarded as raw bytes under HTTP 200.

    Before the fix, _async_streaming() would yield the 429 error JSON as
    chunks and allm_passthrough_route returned an async generator.  The
    caller (azure_proxy_route) wrapped it in StreamingResponse(status_code=200),
    so the client saw HTTP 200 + unparseable SSE body → silent task_complete(null).

    After the fix, raise_for_status() fires inside _async_streaming() before
    any chunks are yielded, so the exception propagates all the way up.
    """
    mock_provider_config = MagicMock()
    mock_provider_config.get_complete_url.return_value = (
        httpx.URL(
            "https://my-azure.openai.azure.com/openai/deployments/gpt-4/responses"
        ),
        "https://my-azure.openai.azure.com",
    )
    mock_provider_config.get_api_key.return_value = "fake-azure-key"
    mock_provider_config.validate_environment.return_value = {"api-key": "fake-azure-key"}
    mock_provider_config.sign_request.return_value = ({"api-key": "fake-azure-key"}, None)
    mock_provider_config.is_streaming_request.return_value = True

    mock_429_response = _make_429_streaming_response()

    async_client = AsyncHTTPHandler()
    mock_send = AsyncMock(return_value=mock_429_response)
    mock_build_request = MagicMock(return_value=MagicMock())

    mock_logging_obj = MagicMock()
    mock_logging_obj.update_environment_variables = MagicMock()
    mock_logging_obj.async_flush_passthrough_collected_chunks = AsyncMock()

    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_passthrough_config",
        return_value=mock_provider_config,
    ), patch(
        "litellm.litellm_core_utils.get_litellm_params.get_litellm_params",
        return_value={},
    ), patch(
        "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
        return_value=(
            "gpt-4",
            "azure",
            "fake-azure-key",
            "https://my-azure.openai.azure.com",
        ),
    ), patch.object(
        async_client.client, "send", mock_send
    ), patch.object(
        async_client.client, "build_request", mock_build_request
    ):
        result = await allm_passthrough_route(
            model="azure/gpt-4",
            endpoint="openai/deployments/gpt-4/responses",
            method="POST",
            custom_llm_provider="azure",
            api_base="https://my-azure.openai.azure.com",
            api_key="fake-azure-key",
            json={"model": "gpt-4", "input": "hello", "stream": True},
            client=async_client,
            litellm_logging_obj=mock_logging_obj,
        )

        # result is an async generator — consuming it must raise, not silently yield error bytes
        chunks = []
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async for chunk in result:  # type: ignore[union-attr]
                chunks.append(chunk)

    assert exc_info.value.response.status_code == 429
    assert len(chunks) == 0, "No chunks should be yielded before the 429 raises"
