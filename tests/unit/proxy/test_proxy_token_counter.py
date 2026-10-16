# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import asyncio
import time
from functools import partial
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from dotenv import load_dotenv
from pydantic import JsonValue

load_dotenv()

# this file is to test litellm/proxy


from fastapi import HTTPException, Request

import litellm
from litellm import Router
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.proxy.common_utils.token_count_settings import UNAVAILABLE_FALLBACK_TTL_SECONDS, TokenCountSettings
from litellm.proxy.proxy_server import _provider_or_fallback_token_count, _try_provider_token_count, token_counter
from litellm.types.utils import TokenCountResponse



@pytest.mark.asyncio
async def test_vLLM_token_counting():
    """
    Test Token counter for vLLM models
    - User passes model="special-alias"
    - token_counter should infer that special_alias -> maps to wolfram/miquliz-120b-v2.0
    -> token counter should use hugging face tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "special-alias",
                "litellm_params": {
                    "model": "openai/wolfram/miquliz-120b-v2.0",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the default tokenizer
    assert response.model_used == "wolfram/miquliz-120b-v2.0"


@pytest.mark.asyncio
async def test_token_counting_model_not_in_model_list():
    """
    Test Token counter - when a model is not in model_list
    -> should use the default OpenAI tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the OpenAI tokenizer
    assert response.model_used == "special-alias"


@pytest.mark.asyncio
async def test_gpt_token_counting():
    """
    Test Token counter
    -> should work for gpt-4
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the OpenAI tokenizer
    assert response.request_model == "gpt-4"


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_endpoint():
    """
    Test /v1/messages/count_tokens endpoint with Anthropic model
    - Should return response in Anthropic format: {"input_tokens": <count>}
    - Should work as wrapper around internal token_counter function
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(request, call_endpoint=False):
        assert (
            call_endpoint == True
        ), "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "claude-3-sonnet-20240229"
        assert request.messages == [{"role": "user", "content": "Hello Claude!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=15,
            request_model="claude-3-sonnet-20240229",
            model_used="claude-3-sonnet-20240229",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.token_counter
    proxy_server.token_counter = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 15
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Anthropic endpoint test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_with_non_anthropic_model():
    """
    Test /v1/messages/count_tokens endpoint with non-Anthropic model (GPT-4)
    - Should still work and return Anthropic format
    - Should call internal token_counter with from_anthropic_endpoint=True
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello GPT!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(request, call_endpoint=True):
        assert (
            call_endpoint == True
        ), "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "gpt-4"
        assert request.messages == [{"role": "user", "content": "Hello GPT!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=12,
            request_model="gpt-4",
            model_used="gpt-4",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.token_counter
    proxy_server.token_counter = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 12
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Non-Anthropic model test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_internal_token_counter_anthropic_provider_detection():
    """
    Test that the internal token_counter correctly detects Anthropic providers
    and handles the from_anthropic_endpoint flag appropriately
    """

    # Test with Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-test",
                "litellm_params": {
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "api_key": "test-key",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with is_direct_request=False (simulating call from Anthropic endpoint)
    # The provider counter is engaged: its API error surfaces instead of a local count
    with pytest.raises(ProxyException) as exc_info:
        await token_counter(
            request=TokenCountRequest(
                model="claude-test",
                messages=[{"role": "user", "content": "hello"}],
            ),
            call_endpoint=True,
        )
    assert exc_info.value.type == "token_counting_error"

    # Test with non-Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-test",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with is_direct_request=False but non-Anthropic provider
    response = await token_counter(
        request=TokenCountRequest(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ),
        call_endpoint=True,
    )

    print("Non-Anthropic provider test response:", response)

    # Verify response structure
    assert response.request_model == "gpt-test"
    assert response.model_used == "gpt-4"
    assert response.total_tokens > 0
    assert response.tokenizer_type == "openai_tokenizer"  # Should use LiteLLM tokenizer


@pytest.mark.asyncio
async def test_anthropic_endpoint_error_handling():
    """
    Test error handling in the /v1/messages/count_tokens endpoint
    """
    from unittest.mock import MagicMock

    from fastapi import HTTPException, Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_user_api_key_dict = MagicMock()

    # Test missing model parameter
    mock_request_data = {
        "messages": [{"role": "user", "content": "Hello!"}]
        # Missing "model" key
    }

    async def mock_read_request_body(request):
        return mock_request_data

    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    try:
        # Should raise HTTPException for missing model
        with pytest.raises(HTTPException) as exc_info:
            await count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "model parameter is required" in str(exc_info.value.detail)

        print("✅ Error handling test passed!")

    finally:
        anthropic_endpoints.read_request_body = original_read_request_body


@pytest.mark.asyncio
async def test_factory_anthropic_endpoint_calls_anthropic_counter():
    """Test that /v1/messages/count_tokens with Anthropic model uses Anthropic counter."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock router to return Anthropic deployment
        with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
            mock_router.model_list = [
                {
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            ]

            # Mock the async method properly
            mock_router.async_get_available_deployment = AsyncMock(
                return_value={
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            )

            # Set ANTHROPIC_API_KEY for the test
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 42

                # Verify that Anthropic handler was called
                mock_handler.handle_count_tokens_request.assert_called_once()


@pytest.mark.asyncio
async def test_factory_gpt4_endpoint_does_not_call_anthropic_counter():
    """Test that /v1/messages/count_tokens with GPT-4 does NOT use Anthropic counter."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 50

            # Mock router to return GPT-4 deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.model_list = [
                    {
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "openai/gpt-4"},
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "openai/gpt-4"},
                        "model_info": {},
                    }
                )

                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 50

                # Verify that Anthropic handler was NOT called
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_normal_token_counter_endpoint_does_not_call_anthropic():
    """Test that /utils/token_counter does NOT use Anthropic counter even with Anthropic model."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler,
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 35

            # Mock router to return Anthropic deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.model_list = [
                    {
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {
                            "model": "anthropic/claude-3-5-sonnet-20241022"
                        },
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {
                            "model": "anthropic/claude-3-5-sonnet-20241022"
                        },
                        "model_info": {},
                    }
                )

                client = TestClient(app)

                response = client.post(
                    "/utils/token_counter",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["total_tokens"] == 35

                # Verify that Anthropic handler was NOT called (since call_endpoint=False)
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_registration():
    """Test that the new factory pattern correctly provides counters."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    # Test Anthropic ModelInfo provides token counter
    anthropic_model_info = AnthropicModelInfo()
    counter = anthropic_model_info.get_token_counter()
    assert counter is not None

    # Create test deployments
    anthropic_deployment = {
        "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}
    }

    non_anthropic_deployment = {"litellm_params": {"model": "openai/gpt-4"}}

    # Test Anthropic counter supports provider
    assert counter.should_use_token_counting_api(custom_llm_provider="anthropic")
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test non-Anthropic provider
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test None deployment
    assert not counter.should_use_token_counting_api(custom_llm_provider=None)


@pytest.mark.asyncio
async def test_bedrock_count_tokens_endpoint():
    """
    Test that Bedrock CountTokens endpoint correctly extracts model from request body.
    """
    from litellm.router import Router

    # Mock the Bedrock CountTokens handler
    async def mock_count_tokens_handler(request_data, litellm_params, resolved_model):
        # Verify the correct model was resolved
        assert resolved_model == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["messages"] == [{"role": "user", "content": "Hello!"}]

        return {"input_tokens": 25}

    # Set up router with Bedrock model
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-bedrock",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test the mock handler directly to verify correct parameter extraction
    request_data = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    # Test the mock handler directly to verify correct parameter extraction
    await mock_count_tokens_handler(
        request_data, {}, "anthropic.claude-3-sonnet-20240229-v1:0"
    )


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_token_counting():
    """
    Unit test for Vertex AI Anthropic token counting with mocked API calls.

    This tests the token counting implementation for Vertex AI partner models
    without making actual API calls. Mocks at the handler level to test the full flow.
    """
    from unittest.mock import patch

    # Mock the Vertex AI partner models token counter response
    mock_token_response = {
        "input_tokens": 15,
        "tokenizer_used": "vertex_ai_partner_models",
    }

    llm_router = Router(
        model_list=[
            {
                "model_name": "vertex_ai/claude-3-5-sonnet-20241022",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-5-sonnet-20241022",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Mock the lower level handler method
    with patch(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler.VertexAIPartnerModelsTokenCounter.handle_count_tokens_request"
    ) as mock_handle_count_tokens:
        mock_handle_count_tokens.return_value = mock_token_response

        # Test with messages format and call_endpoint=True
        response = await token_counter(
            request=TokenCountRequest(
                model="vertex_ai/claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello Claude on Vertex AI! How are you?",
                    }
                ],
            ),
            call_endpoint=True,
        )

        # Validate that handle_count_tokens_request was called
        assert mock_handle_count_tokens.called

        # Verify the call arguments
        call_args = mock_handle_count_tokens.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert "messages" in call_args.kwargs["request_data"]
        assert (
            call_args.kwargs["request_data"]["messages"][0]["content"]
            == "Hello Claude on Vertex AI! How are you?"
        )

        # Validate response structure
        assert response.model_used == "claude-3-5-sonnet-20241022"
        assert response.request_model == "vertex_ai/claude-3-5-sonnet-20241022"
        assert response.total_tokens == 15
        assert response.tokenizer_type == "vertex_ai_partner_models"

        # Validate original response contains input_tokens
        assert response.original_response is not None
        assert "input_tokens" in response.original_response
        assert response.original_response["input_tokens"] == 15


@pytest.mark.parametrize("vertex_location", ["global", "us-central1"])
def test_vertex_ai_partner_models_token_counting_endpoint(vertex_location):
    """
    Test that the VertexAIPartnerModelsTokenCounter builds the correct endpoint URL
    for different vertex locations, including the special 'global' location.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
        VertexAIPartnerModelsTokenCounter,
    )

    endpoint = VertexAIPartnerModelsTokenCounter()._build_count_tokens_endpoint(
        model="claude-3-5-sonnet-20241022",
        project_id="test-project",
        vertex_location=vertex_location,
        api_base=None,
    )
    if vertex_location == "global":
        assert endpoint.startswith("https://aiplatform.googleapis.com")
    else:
        assert endpoint.startswith(
            f"https://{vertex_location}-aiplatform.googleapis.com"
        )


class _FailingRuntimeHandler(BedrockCountTokensHandler):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def handle_count_tokens_request(
        self,
        request_data: dict[str, object],
        litellm_params: dict[str, object],
        resolved_model: str,
        client: AsyncHTTPHandler | None = None,
    ) -> dict[str, JsonValue]:
        raise self._error


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_bedrock_error():
    """
    Test that BedrockTokenCounter properly returns error response when BedrockError is raised.
    Verifies that the status code and error message are preserved.
    """
    counter = BedrockTokenCounter(
        runtime_handler=_FailingRuntimeHandler(BedrockError(status_code=429, message="Rate limit exceeded"))
    )

    result = await counter.count_tokens(
        model_to_use="anthropic.claude-3-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="bedrock/anthropic.claude-3-sonnet",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 429
    assert "Rate limit exceeded" in result.error_message
    assert result.tokenizer_type == "bedrock_api"
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_generic_exception():
    """
    Test that BedrockTokenCounter returns error response with 500 status for generic exceptions.
    """
    counter = BedrockTokenCounter(runtime_handler=_FailingRuntimeHandler(Exception("Unexpected error")))

    result = await counter.count_tokens(
        model_to_use="anthropic.claude-3-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="bedrock/anthropic.claude-3-sonnet",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 500
    assert "Unexpected error" in result.error_message


@pytest.mark.asyncio
async def test_bedrock_handler_httpx_error_status_code_propagation():
    """
    Test that BedrockCountTokensHandler properly extracts status code from httpx.HTTPStatusError.
    """
    handler = BedrockCountTokensHandler()

    # Create a mock httpx response with 403 status
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden - Invalid credentials"

    # Create HTTPStatusError
    http_error = httpx.HTTPStatusError(
        message="Client error '403 Forbidden'",
        request=MagicMock(),
        response=mock_response,
    )

    with patch.object(handler, "validate_count_tokens_request"):
        with patch.object(handler, "_get_aws_region_name", return_value="us-west-2"):
            with patch.object(
                handler, "transform_anthropic_to_bedrock_count_tokens", return_value={}
            ):
                with patch.object(
                    handler,
                    "get_bedrock_count_tokens_endpoint",
                    return_value="https://example.com",
                ):
                    with patch.object(
                        handler, "_sign_request", return_value=({}, "{}")
                    ):
                        with patch(
                            "litellm.llms.bedrock.count_tokens.handler.get_async_httpx_client"
                        ) as mock_client:
                            mock_async_client = AsyncMock()
                            mock_async_client.post = AsyncMock(side_effect=http_error)
                            mock_client.return_value = mock_async_client

                            with pytest.raises(BedrockError) as exc_info:
                                await handler.handle_count_tokens_request(
                                    request_data={
                                        "model": "test",
                                        "messages": [
                                            {"role": "user", "content": "hello"}
                                        ],
                                    },
                                    litellm_params={},
                                    resolved_model="anthropic.claude-3-sonnet",
                                )

                            assert exc_info.value.status_code == 403
                            # Message should be the raw response text
                            assert (
                                exc_info.value.message
                                == "Forbidden - Invalid credentials"
                            )


@pytest.mark.asyncio
async def test_token_counter_httpx_status_error_raises_proxy_exception():
    """
    When provider_counter.count_tokens() raises httpx.HTTPStatusError,
    the token_counter endpoint should catch it and raise a ProxyException
    with the upstream status code and error message.
    """

    upstream_status = 429
    upstream_message = "Rate limit exceeded"
    response = httpx.Response(
        status_code=upstream_status,
        request=httpx.Request("POST", "https://provider.example.com/count"),
    )
    http_error = httpx.HTTPStatusError(
        message=upstream_message,
        request=response.request,
        response=response,
    )

    mock_counter = MagicMock()
    mock_counter.should_use_token_counting_api.return_value = True
    mock_counter.count_tokens = AsyncMock(side_effect=http_error)

    # Save originals
    original_get_provider_token_counter = (
        litellm.proxy.proxy_server._get_provider_token_counter
    )
    original_router = litellm.proxy.proxy_server.llm_router

    try:

        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "claude-4-6-sonnet", "vertex_ai")

        litellm.proxy.proxy_server._get_provider_token_counter = (
            mock_get_provider_token_counter
        )

        mock_router = MagicMock()
        mock_router.async_get_available_deployment = AsyncMock(
            return_value={
                "litellm_params": {
                    "model": "vertex_ai/claude-4-6-sonnet",
                    "api_key": "fake-key",
                },
                "model_info": {},
            }
        )
        litellm.proxy.proxy_server.llm_router = mock_router

        with pytest.raises(ProxyException) as exc_info:
            await token_counter(
                request=TokenCountRequest(
                    model="claude-4-6-sonnet",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                call_endpoint=True,
            )

        assert exc_info.value.code == str(upstream_status)
        assert upstream_message in exc_info.value.message
        assert exc_info.value.type == "token_counting_error"
        assert exc_info.value.param == "model"
    finally:
        litellm.proxy.proxy_server._get_provider_token_counter = (
            original_get_provider_token_counter
        )
        litellm.proxy.proxy_server.llm_router = original_router


@pytest.mark.asyncio
async def test_proxy_token_counter_error_raises_exception_when_disabled():
    """
    Test that proxy token_counter raises ProxyException when disable_token_counter=True
    and provider returns an error response.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = MagicMock()
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = (
        litellm.proxy.proxy_server._get_provider_token_counter
    )

    try:
        litellm.disable_token_counter = True

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = (
            mock_get_provider_token_counter
        )

        with pytest.raises(ProxyException) as exc_info:
            await token_counter(
                request=TokenCountRequest(
                    model="claude-bedrock",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                call_endpoint=True,
            )

        assert exc_info.value.code == "429"
        assert "Rate limit exceeded" in exc_info.value.message
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = (
            original_get_provider_token_counter
        )


@pytest.mark.asyncio
async def test_proxy_token_counter_error_raises_when_enabled():
    """
    Test that proxy token_counter raises ProxyException when disable_token_counter=False
    and the provider returns an error response, never falling back to the local tokenizer.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = MagicMock()
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = (
        litellm.proxy.proxy_server._get_provider_token_counter
    )

    try:
        litellm.disable_token_counter = False

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = (
            mock_get_provider_token_counter
        )

        with patch("litellm.token_counter") as mock_local_counter:
            with pytest.raises(ProxyException) as exc_info:
                await token_counter(
                    request=TokenCountRequest(
                        model="claude-bedrock",
                        messages=[{"role": "user", "content": "hello"}],
                    ),
                    call_endpoint=True,
                )

        assert exc_info.value.code == "429"
        assert "Rate limit exceeded" in exc_info.value.message
        assert exc_info.value.type == "token_counting_error"
        mock_local_counter.assert_not_called()
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = (
            original_get_provider_token_counter
        )


@pytest.mark.asyncio
async def test_anthropic_endpoint_returns_anthropic_error_format():
    """
    Test that /v1/messages/count_tokens returns errors in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with Bedrock-style error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message='{"detail":{"message":"Input is too long for requested model."}}',
            type="token_counting_error",
            param="model",
            code=400,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        # Verify HTTP status code is correct
        assert exc_info.value.status_code == 400

        # Verify error is in Anthropic format
        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "invalid_request_error"
        assert detail["error"]["message"] == "Input is too long for requested model."
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_403_permission_error_format():
    """
    Test that 403 errors are returned as permission_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with 403 error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message='{"Message":"Bearer Token has expired"}',
            type="token_counting_error",
            param="model",
            code=403,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 403

        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "permission_error"
        assert detail["error"]["message"] == "Bearer Token has expired"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_429_rate_limit_error_format():
    """
    Test that 429 errors are returned as rate_limit_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with 429 error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message="Rate limit exceeded",
            type="token_counting_error",
            param="model",
            code=429,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 429

        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "rate_limit_error"
        assert detail["error"]["message"] == "Rate limit exceeded"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


class _ScriptedCounter(BaseTokenCounter):
    def __init__(self, replies: dict[str, TokenCountResponse]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, object, object, object]] = []

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return True

    async def count_tokens(
        self,
        model_to_use: str,
        messages,
        contents,
        deployment=None,
        request_model: str = "",
        tools=None,
        system=None,
    ) -> TokenCountResponse | None:
        self.calls.append((model_to_use, messages, tools, system))
        return self._replies[model_to_use]


def _provider_reply(model: str, total_tokens: int = 0, status_code: int | None = None) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=total_tokens,
        request_model="sonnet-5-5",
        model_used=model,
        tokenizer_type="bedrock_api",
        error=status_code is not None,
        error_message=None if status_code is None else f"scripted {status_code} for {model}",
        status_code=status_code,
    )


_COUNT_REQUEST = TokenCountRequest(
    model="sonnet-5-5",
    messages=[{"role": "user", "content": "hello"}],
    tools=[{"name": "get_quote", "input_schema": {"type": "object"}}],
    system="be brief",
)


async def _ladder(
    counter: _ScriptedCounter,
    settings: TokenCountSettings,
    local_estimate_allowed: bool = True,
    deployment_id: str = "sonnet-deployment",
    unavailable_fallbacks: InMemoryCache | None = None,
) -> TokenCountResponse | None:
    return await _provider_or_fallback_token_count(
        count_with=partial(
            _try_provider_token_count,
            counter,
            "bedrock",
            messages=_COUNT_REQUEST.messages,
            contents=_COUNT_REQUEST.contents,
            deployment=None,
            request_model=_COUNT_REQUEST.model,
            tools=_COUNT_REQUEST.tools,
            system=_COUNT_REQUEST.system,
        ),
        model_to_use="us.anthropic.claude-sonnet-5-5",
        settings=settings,
        local_estimate_allowed=local_estimate_allowed,
        deployment_id=deployment_id,
        unavailable_fallbacks=InMemoryCache() if unavailable_fallbacks is None else unavailable_fallbacks,
    )


@pytest.mark.asyncio
async def test_provider_count_is_tagged_provider_and_never_tries_the_fallback():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=447)})

    result = await _ladder(counter, TokenCountSettings(framing=10, fallback_model="opus"))

    assert result is not None
    assert (result.total_tokens, result.count_source) == (447, "provider")
    assert [call[0] for call in counter.calls] == ["us.anthropic.claude-sonnet-5-5"]


@pytest.mark.parametrize("status_code", [400, 404])
@pytest.mark.asyncio
async def test_uncountable_model_is_counted_with_the_fallback_model_on_the_same_request(status_code: int):
    counter = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447),
        }
    )

    result = await _ladder(counter, TokenCountSettings(framing=None, fallback_model="us.anthropic.claude-opus-5-5"))

    assert result is not None
    assert (result.total_tokens, result.count_source, result.model_used) == (
        447,
        "fallback_model",
        "us.anthropic.claude-opus-5-5",
    )
    assert [call[0] for call in counter.calls] == ["us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-opus-5-5"]
    assert counter.calls[0][1:] == counter.calls[1][1:]


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@pytest.mark.asyncio
async def test_errors_other_than_cannot_count_raise_without_a_fallback(status_code: int):
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, fallback_model="us.anthropic.claude-opus-5-5"))

    assert raised.value.code == str(status_code)
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_unavailable_fallback_model_keeps_the_requested_model_error_unchanged():
    counter = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404),
        }
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, fallback_model="us.anthropic.claude-opus-5-5"))

    assert (raised.value.code, raised.value.message) == ("404", "scripted 404 for sonnet")


_SONNET_WITH_UNAVAILABLE_OPUS: Final = {
    "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404),
    "us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404),
}
_OPUS_FALLBACK: Final = TokenCountSettings(framing=None, fallback_model="us.anthropic.claude-opus-5-5")


async def _ladder_error(counter: _ScriptedCounter, deployment_id: str, unavailable_fallbacks: InMemoryCache) -> str:
    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, _OPUS_FALLBACK, deployment_id=deployment_id, unavailable_fallbacks=unavailable_fallbacks)
    return f"{raised.value.code} {raised.value.message}"


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_unavailable_fallback_is_skipped_for_that_deployment_until_the_ttl_passes():
    clock: Final = _ManualClock()
    unavailable: Final = InMemoryCache(default_ttl=UNAVAILABLE_FALLBACK_TTL_SECONDS, clock=clock)
    counter: Final = _ScriptedCounter(_SONNET_WITH_UNAVAILABLE_OPUS)

    errors: Final = [await _ladder_error(counter, "us-west-2-sonnet", unavailable) for _ in range(2)]
    clock.advance(UNAVAILABLE_FALLBACK_TTL_SECONDS + 1)
    after_ttl: Final = await _ladder_error(counter, "us-west-2-sonnet", unavailable)

    assert errors == ["404 scripted 404 for sonnet", "404 scripted 404 for sonnet"]
    assert after_ttl == errors[0]
    assert [call[0] for call in counter.calls] == [
        "us.anthropic.claude-sonnet-5-5",
        "us.anthropic.claude-opus-5-5",
        "us.anthropic.claude-sonnet-5-5",
        "us.anthropic.claude-sonnet-5-5",
        "us.anthropic.claude-opus-5-5",
    ]


@pytest.mark.asyncio
async def test_unavailable_fallback_on_one_deployment_still_runs_on_another():
    unavailable: Final = InMemoryCache()
    counter: Final = _ScriptedCounter(_SONNET_WITH_UNAVAILABLE_OPUS)

    await _ladder_error(counter, "us-west-2-sonnet", unavailable)
    await _ladder_error(counter, "us-east-1-sonnet", unavailable)

    assert [call[0] for call in counter.calls].count("us.anthropic.claude-opus-5-5") == 2


@pytest.mark.parametrize("status_code", [400, 403, 429, 500])
@pytest.mark.asyncio
async def test_fallback_failure_answers_with_the_fallback_error_and_its_status(status_code: int):
    counter = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code),
        }
    )

    unavailable: Final = InMemoryCache()

    for _ in range(2):
        with pytest.raises(ProxyException) as raised:
            await _ladder(
                counter,
                TokenCountSettings(framing=10, fallback_model="us.anthropic.claude-opus-5-5"),
                unavailable_fallbacks=unavailable,
            )
        assert (raised.value.code, raised.value.message) == (str(status_code), f"scripted {status_code} for opus")

    assert [call[0] for call in counter.calls].count("us.anthropic.claude-opus-5-5") == 2


@pytest.mark.asyncio
async def test_uncountable_model_without_fallback_or_framing_raises():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, fallback_model=None))

    assert raised.value.code == "404"


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_falls_through_to_the_local_estimate():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    assert await _ladder(counter, TokenCountSettings(framing=0, fallback_model=None)) is None


@pytest.mark.asyncio
async def test_local_estimate_adds_the_configured_framing_once_per_request(monkeypatch: pytest.MonkeyPatch):
    def deployment(name: str, model_info: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": f"sk-unused-{name}"},
            "model_info": model_info,
        }

    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[deployment("framed", {"token_count_framing": 100}), deployment("plain", {})]),
    )
    messages: Final = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}]

    plain: Final = await token_counter(request=TokenCountRequest(model="plain", messages=messages))
    framed: Final = await token_counter(request=TokenCountRequest(model="framed", messages=messages))
    framed_prompt: Final = await token_counter(request=TokenCountRequest(model="framed", prompt="hello there"))

    assert plain.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages)
    assert framed.total_tokens == plain.total_tokens + 100
    assert (plain.count_source, framed.count_source) == ("local_estimate", "local_estimate")
    assert framed_prompt.total_tokens == litellm.token_counter(model="gpt-5.4-mini", text="hello there")


@pytest.mark.asyncio
async def test_slow_cost_map_lookup_does_not_stall_other_requests_on_the_worker(monkeypatch: pytest.MonkeyPatch):
    real_get_model_info: Final = litellm.get_model_info

    def slow_get_model_info(model: str, *args: object, **kwargs: object):
        if model == "slow-lookup-model":
            time.sleep(0.5)
        return real_get_model_info(model, *args, **kwargs)

    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "get_model_info", slow_get_model_info)
    counting: Final = asyncio.create_task(
        token_counter(request=TokenCountRequest(model="slow-lookup-model", messages=[{"role": "user", "content": "hi"}]))
    )
    started: Final = time.monotonic()
    await asyncio.sleep(0.05)
    other_request_wait: Final = time.monotonic() - started
    await counting

    assert other_request_wait < 0.4


@pytest.mark.asyncio
async def test_uncountable_model_keeps_the_provider_error_when_local_counting_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    counter: Final = _ScriptedCounter(
        {"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)}
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "kimi",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {"token_count_framing": 78},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    monkeypatch.setattr(litellm, "disable_token_counter", True)

    with pytest.raises(ProxyException) as raised:
        await token_counter(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
        )

    assert raised.value.code == "400"
    assert "scripted 400 for kimi" in raised.value.message


@pytest.mark.parametrize(("model_info", "counted_models"), [
    ({}, ["us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-opus-5-5"]),
    ({"token_count_fallback_model": ""}, ["us.anthropic.claude-sonnet-5-5"]),
])
@pytest.mark.asyncio
async def test_empty_configured_fallback_model_turns_off_the_shipped_fallback(
    monkeypatch: pytest.MonkeyPatch, model_info: dict[str, object], counted_models: list[str]
):
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404),
        }
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5"},
                    "model_info": model_info,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    with pytest.raises(ProxyException) as raised:
        await token_counter(
            request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
        )

    assert (raised.value.code, raised.value.message) == ("404", "scripted 404 for sonnet")
    assert [call[0] for call in counter.calls] == counted_models


@pytest.mark.asyncio
async def test_framing_still_applies_while_the_deployment_is_cooling_down(monkeypatch: pytest.MonkeyPatch):
    router: Final = Router(
        model_list=[
            {
                "model_name": "framed",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-unused-framed"},
                "model_info": {"id": "framed-deployment", "token_count_framing": 100},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    messages: Final = [{"role": "user", "content": "hello there"}]
    warm: Final = await token_counter(request=TokenCountRequest(model="framed", messages=messages))

    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="framed-deployment", original_exception=Exception("rate limited"), exception_status=429, cooldown_time=60
    )
    cooling: Final = await token_counter(request=TokenCountRequest(model="framed", messages=messages))

    assert warm.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages) + 100
    assert cooling.total_tokens - 100 == litellm.token_counter(model="framed", messages=messages)
    assert cooling.count_source == "local_estimate"
