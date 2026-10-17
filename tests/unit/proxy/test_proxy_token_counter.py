# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import asyncio
import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Final, NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from pydantic import BaseModel, ConfigDict, JsonValue

load_dotenv()

# this file is to test litellm/proxy


from fastapi import Request
from fastapi.responses import JSONResponse

import litellm
from litellm import Router
from litellm.llms.anthropic.count_tokens import handler as anthropic_count_tokens_handler
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm._logging import verbose_proxy_logger
from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler
from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import VertexAIPartnerModelsTokenCounter
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request
from litellm.proxy.common_utils.token_count_settings import (
    ProviderCountFailure,
    TokenCountModel,
    TokenCountSettings,
    cost_map_entry,
)
from litellm.proxy.proxy_server import (
    _estimate_or_provider_error,
    _provider_token_count,
    _try_provider_token_count,
    count_request_tokens,
    token_counter,
)
from litellm.types.utils import SelectTokenizerResponse, TokenCountMessagesFormat, TokenCountResponse


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

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the default tokenizer
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

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the OpenAI tokenizer
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

    assert response.tokenizer_type == "openai_tokenizer"  # SHOULD use the OpenAI tokenizer
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
    mock_request.headers = {}
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
    async def mock_token_counter(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
    ):
        assert call_endpoint == True, "Should be called with call_endpoint=True for Anthropic endpoint"
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

    original_token_counter = proxy_server.count_request_tokens
    proxy_server.count_request_tokens = mock_token_counter

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
        proxy_server.count_request_tokens = original_token_counter


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
    mock_request.headers = {}
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
    async def mock_token_counter(
        request,
        call_endpoint=True,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
    ):
        assert call_endpoint == True, "Should be called with call_endpoint=True for Anthropic endpoint"
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

    original_token_counter = proxy_server.count_request_tokens
    proxy_server.count_request_tokens = mock_token_counter

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
        proxy_server.count_request_tokens = original_token_counter


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

    with pytest.raises(ProxyException) as missing_key:
        await token_counter(
            request=TokenCountRequest(
                model="gpt-test",
                messages=[{"role": "user", "content": "hello"}],
            ),
            call_endpoint=True,
        )
    assert missing_key.value.code == "401"
    assert "No OpenAI API key is set for model gpt-test" in missing_key.value.message


@pytest.mark.asyncio
async def test_anthropic_endpoint_error_handling():
    """
    Test error handling in the /v1/messages/count_tokens endpoint
    """
    from unittest.mock import MagicMock

    from fastapi import Request

    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.scope = {}
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
        response = await count_tokens(mock_request, mock_user_api_key_dict)

        assert response.status_code == 400
        assert json.loads(response.body) == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "model parameter is required"},
        }

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
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

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
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

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
                        "litellm_params": {"model": "databricks/gpt-4"},
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "databricks/gpt-4"},
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
    mock_handler.handle_count_tokens_request = AsyncMock(return_value={"input_tokens": 42})

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
    anthropic_deployment = {"litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}}

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
                "litellm_params": {"model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"},
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
    await mock_count_tokens_handler(request_data, {}, "anthropic.claude-3-sonnet-20240229-v1:0")


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
        assert call_args.kwargs["request_data"]["messages"][0]["content"] == "Hello Claude on Vertex AI! How are you?"

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
        assert endpoint.startswith(f"https://{vertex_location}-aiplatform.googleapis.com")


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
            with patch.object(handler, "transform_anthropic_to_bedrock_count_tokens", return_value={}):
                with patch.object(
                    handler,
                    "get_bedrock_count_tokens_endpoint",
                    return_value="https://example.com",
                ):
                    with patch.object(handler, "_sign_request", return_value=({}, "{}")):
                        with patch("litellm.llms.bedrock.count_tokens.handler.get_async_httpx_client") as mock_client:
                            mock_async_client = AsyncMock()
                            mock_async_client.post = AsyncMock(side_effect=http_error)
                            mock_client.return_value = mock_async_client

                            with pytest.raises(BedrockError) as exc_info:
                                await handler.handle_count_tokens_request(
                                    request_data={
                                        "model": "test",
                                        "messages": [{"role": "user", "content": "hello"}],
                                    },
                                    litellm_params={},
                                    resolved_model="anthropic.claude-3-sonnet",
                                )

                            assert exc_info.value.status_code == 403
                            # Message should be the raw response text
                            assert exc_info.value.message == "Forbidden - Invalid credentials"


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
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter
    original_router = litellm.proxy.proxy_server.llm_router

    try:

        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "claude-4-6-sonnet", "vertex_ai")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

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
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter
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
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = True

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

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
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


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
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = False

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

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
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_returns_anthropic_error_format():
    """
    Test that /v1/messages/count_tokens returns errors in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with Bedrock-style error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
    ):
        raise ProxyException(
            message='{"detail":{"message":"Input is too long for requested model."}}',
            type="token_counting_error",
            param="model",
            code=400,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "invalid_request_error"
        assert detail["error"]["message"] == "Input is too long for requested model."
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_403_permission_error_format():
    """
    Test that 403 errors are returned as permission_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with 403 error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
    ):
        raise ProxyException(
            message='{"Message":"Bearer Token has expired"}',
            type="token_counting_error",
            param="model",
            code=403,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 403
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "permission_error"
        assert detail["error"]["message"] == "Bearer Token has expired"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_429_rate_limit_error_format():
    """
    Test that 429 errors are returned as rate_limit_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints.read_request_body
    anthropic_endpoints.read_request_body = mock_read_request_body

    original_token_counter = proxy_server.count_request_tokens

    # Mock token_counter to raise ProxyException with 429 error
    async def mock_token_counter_error(
        request,
        call_endpoint=False,
        messages_format="openai",
        output_format=None,
        context_management=None,
        request_litellm_params=None,
    ):
        raise ProxyException(
            message="Rate limit exceeded",
            type="token_counting_error",
            param="model",
            code=429,
        )

    proxy_server.count_request_tokens = mock_token_counter_error

    try:
        response: Final[object] = await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 429
        detail = json.loads(bytes(response.body))
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "rate_limit_error"
        assert detail["error"]["message"] == "Rate limit exceeded"
    finally:
        anthropic_endpoints.read_request_body = original_read_request_body
        proxy_server.count_request_tokens = original_token_counter


class _ScriptedCounter(BaseTokenCounter):
    def __init__(self, replies: dict[str, TokenCountResponse], burst: int = 1) -> None:
        self._replies = replies
        self._burst = burst
        self._burst_arrived = asyncio.Event()
        self.calls: list[tuple[str, object, object, object]] = []
        self.regions: list[object] = []

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return True

    async def count_tokens(
        self,
        model_to_use: str,
        messages: object,
        contents: object,
        deployment: object = None,
        request_model: str = "",
        tools: object = None,
        system: object = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: object = None,
    ) -> TokenCountResponse | None:
        self.calls.append((model_to_use, messages, tools, system))
        self.regions.append(None if deployment is None else deployment["litellm_params"].get("aws_region_name"))
        if len(self.calls) >= self._burst:
            self._burst_arrived.set()
        await self._burst_arrived.wait()
        return self._replies[model_to_use]


def _provider_reply(
    model: str, total_tokens: int = 0, status_code: int | None = None, rejected_before_provider_call: bool = False
) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=total_tokens,
        request_model="sonnet-5-5",
        model_used=model,
        tokenizer_type="bedrock_api",
        error=status_code is not None,
        error_message=None if status_code is None else f"scripted {status_code} for {model}",
        status_code=status_code,
        rejected_before_provider_call=rejected_before_provider_call,
    )


_COUNT_REQUEST = TokenCountRequest(
    model="sonnet-5-5",
    messages=[{"role": "user", "content": "hello"}],
    tools=[{"name": "get_quote", "input_schema": {"type": "object"}}],
    system="be brief",
)


_SONNET_DEPLOYMENT: Final = MappingProxyType(
    {"litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5", "aws_region_name": "us-west-2"}}
)
_OPUS_IN_US_EAST_1: Final = TokenCountModel(model="us.anthropic.claude-opus-5-5", region="us-east-1")


async def _ladder(
    counter: _ScriptedCounter, settings: TokenCountSettings, local_estimate_allowed: bool = True
) -> TokenCountResponse | None:
    return await _provider_token_count(
        count_with=partial(
            _try_provider_token_count,
            counter,
            "bedrock",
            messages=_COUNT_REQUEST.messages,
            contents=_COUNT_REQUEST.contents,
            deployment=_SONNET_DEPLOYMENT,
            request_model=_COUNT_REQUEST.model,
            tools=_COUNT_REQUEST.tools,
            system=_COUNT_REQUEST.system,
        ),
        model_to_use="us.anthropic.claude-sonnet-5-5",
        settings=settings,
        local_estimate_allowed=local_estimate_allowed,
    )


@pytest.mark.asyncio
async def test_model_without_token_count_model_counts_itself_in_the_deployment_region():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=447)})

    result = await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

    assert result is not None
    assert (result.total_tokens, result.count_source) == (447, "provider")
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-sonnet-5-5"], ["us-west-2"])


@pytest.mark.parametrize(
    ("count_model", "region"),
    [
        (_OPUS_IN_US_EAST_1, "us-east-1"),
        (TokenCountModel(model="us.anthropic.claude-opus-5-5", region=None), "us-west-2"),
    ],
)
@pytest.mark.asyncio
async def test_token_count_model_counts_the_same_request_in_its_own_region_without_trying_the_model(
    count_model: TokenCountModel, region: str
):
    counter = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})

    result = await _ladder(counter, TokenCountSettings(framing=None, count_model=count_model))

    assert result is not None
    assert (result.total_tokens, result.count_source, result.model_used) == (
        447,
        "token_count_model",
        "us.anthropic.claude-opus-5-5",
    )
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-opus-5-5"], [region])
    assert counter.calls[0][1:] == (_COUNT_REQUEST.messages, _COUNT_REQUEST.tools, _COUNT_REQUEST.system)
    assert _SONNET_DEPLOYMENT["litellm_params"]["aws_region_name"] == "us-west-2"


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@pytest.mark.asyncio
async def test_errors_other_than_cannot_count_raise_even_with_framing(status_code: int):
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

    assert (raised.value.code, raised.value.message) == (str(status_code), f"scripted {status_code} for sonnet")


@pytest.mark.parametrize("status_code", [400, 403, 404, 500])
@pytest.mark.asyncio
async def test_token_count_model_failure_names_its_region_so_the_user_can_enable_it(status_code: int):
    counter = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, count_model=_OPUS_IN_US_EAST_1))

    assert raised.value.code == str(status_code)
    assert raised.value.message.startswith(f"scripted {status_code} for opus")
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in raised.value.message
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_token_count_model_without_a_region_answers_its_error_unchanged():
    counter = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(
            counter,
            TokenCountSettings(
                framing=None, count_model=TokenCountModel(model="us.anthropic.claude-opus-5-5", region=None)
            ),
        )

    assert (raised.value.code, raised.value.message) == ("404", "scripted 404 for opus")


@pytest.mark.asyncio
async def test_uncountable_model_without_framing_raises():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, count_model=None))

    assert raised.value.code == "404"


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_falls_through_to_the_local_estimate():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=0, count_model=None))

    assert isinstance(result, ProviderCountFailure)
    assert len(counter.calls) == 1


@pytest.mark.parametrize("status_code", [400, 404])
@pytest.mark.asyncio
async def test_token_count_model_error_stands_even_with_framing(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=78, count_model=_OPUS_IN_US_EAST_1))

    assert raised.value.code == str(status_code)
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in raised.value.message


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_raises_when_local_counting_is_disabled():
    counter = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=None), local_estimate_allowed=False)

    assert raised.value.code == "400"


@pytest.mark.asyncio
async def test_request_the_counter_rejects_before_calling_the_provider_answers_400_without_framing():
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply(
                "sonnet", status_code=400, rejected_before_provider_call=True
            )
        }
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=100, count_model=None))

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for sonnet")


@pytest.mark.parametrize("model", ["openai/gpt-5.4-mini", "gemini/gemini-3.8-flash"])
@pytest.mark.asyncio
async def test_framing_keeps_the_400_for_blocks_the_provider_counter_cannot_convert(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "framed",
                    "litellm_params": {"model": model, "api_key": "sk-unused-framed"},
                    "model_info": {"token_count_framing": 100},
                }
            ]
        ),
    )
    search_result: Final = {"type": "search_result", "source": "https://example.com", "title": "Notes", "content": []}

    with pytest.raises(ProxyException) as raised:
        await count_request_tokens(
            request=TokenCountRequest(
                model="framed",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Summarize."}, search_result]}],
            ),
            call_endpoint=True,
            messages_format="anthropic",
        )

    assert raised.value.code == "400"
    assert "search_result" in raised.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unreadable_body",
    [
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "What is new?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_01",
                                "name": "web_search",
                                "input": {"query": "new"},
                            },
                            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01", "content": []},
                        ],
                    },
                ]
            },
            id="web_search_blocks",
        ),
        pytest.param(
            {
                "messages": [{"role": "user", "content": "What is new?"}],
                "tools": [{"name": "search", "input_schema": {"type": "object", "properties": {"query": True}}}],
            },
            id="boolean_subschema_tool",
        ),
        pytest.param(
            {
                "messages": [{"role": "user", "content": "What is new?"}],
                "tools": [
                    {"name": "search", "input_schema": {"type": "object", "properties": {"tags": {"type": "array"}}}}
                ],
            },
            id="array_schema_without_items",
        ),
        pytest.param(
            {"messages": [], "contents": [{"role": "user", "parts": [{"text": "What is new?"}]}]},
            id="google_contents",
        ),
        pytest.param(
            {
                "messages": [{"role": "user", "content": "Describe this"}],
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Describe this"},
                            {"fileData": {"mimeType": "image/png", "fileUri": "gs://b/c.png"}},
                        ],
                    }
                ],
            },
            id="google_contents_with_media_the_translation_dropped",
        ),
    ],
)
async def test_framed_estimate_that_cannot_read_the_body_answers_the_provider_error(
    monkeypatch: pytest.MonkeyPatch, unreadable_body: dict[str, list[dict[str, object]]]
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    with pytest.raises(ProxyException) as raised:
        await token_counter(request=TokenCountRequest(model="kimi", **unreadable_body), call_endpoint=True)
    readable: Final = await token_counter(
        request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
        call_endpoint=True,
    )

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for kimi")
    assert readable.count_source == "local_estimate"


@pytest.mark.asyncio
async def test_framed_estimate_counts_google_contents_like_the_same_text_sent_as_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi(monkeypatch)
    google_request: Final = _google_count_tokens_request(
        "kimi", {"contents": [{"role": "user", "parts": [{"text": "What is new in the release notes?"}]}]}
    )

    from_google: Final = await token_counter(request=google_request, call_endpoint=True)
    from_messages: Final = await token_counter(
        request=TokenCountRequest(model="kimi", messages=google_request.messages), call_endpoint=True
    )

    assert from_google.count_source == "local_estimate"
    assert from_google.total_tokens == from_messages.total_tokens


@pytest.mark.asyncio
async def test_framed_deployment_whose_custom_tokenizer_cannot_load_answers_the_provider_error(
    monkeypatch: pytest.MonkeyPatch,
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "kimi",
                    "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                    "model_info": {"custom_tokenizer": {"identifier": "moonshotai/kimi-k3", "revision": "main"}},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    with pytest.raises(ProxyException) as raised:
        await token_counter(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )

    assert (raised.value.code, raised.value.message) == ("400", "scripted 400 for kimi")


def _framed_kimis_with_custom_tokenizers(
    monkeypatch: pytest.MonkeyPatch, tokenizers: Mapping[str, tuple[str, str]], burst: int = 1
) -> None:
    counter: Final = _ScriptedCounter(
        {
            "us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400),
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=12),
        },
        burst=burst,
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-5-5"},
                    "model_info": {"token_count_model": {}},
                },
                *(
                    {
                        "model_name": name,
                        "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"},
                        "model_info": {
                            "custom_tokenizer": {"identifier": identifier, "revision": revision, "auth_token": None}
                        },
                    }
                    for name, (identifier, revision) in tokenizers.items()
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )


def _framed_kimi_with_custom_tokenizer(monkeypatch: pytest.MonkeyPatch, identifier: str) -> None:
    _framed_kimis_with_custom_tokenizers(monkeypatch, {"kimi": (identifier, "main")})


async def _count_status(model: str = "kimi") -> str:
    try:
        response: Final = await token_counter(
            request=TokenCountRequest(model=model, messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )
    except ProxyException as e:
        return e.code
    return response.count_source or ""


async def _timed_count_status(model: str) -> tuple[str, float]:
    started: Final = time.monotonic()
    status: Final = await _count_status(model)
    return status, time.monotonic() - started


def _tokenizer_that_panics_while_encoding() -> SelectTokenizerResponse:
    template: Final = Tokenizer(WordLevel({"[UNK]": 0, "<s>": 1}, unk_token="[UNK]"))
    template.pre_tokenizer = Whitespace()
    template.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    spec: Final = json.loads(template.to_str())
    return litellm.utils.create_tokenizer(
        json.dumps({**spec, "post_processor": {**spec["post_processor"], "special_tokens": {}}})
    )


@pytest.mark.asyncio
async def test_hanging_tokenizer_download_answers_the_provider_error_after_the_wait_with_one_download_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/hanging-tokenizer")
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def hanging_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)
    try:
        burst: Final = await asyncio.wait_for(asyncio.gather(*(_count_status() for _ in range(60))), 8)
        later_started: Final = time.monotonic()
        later: Final = await asyncio.wait_for(_count_status(), 8)
        later_seconds: Final = time.monotonic() - later_started
    finally:
        hub_released.set()

    assert set(burst) == {"400"} and later == "400"
    assert later_seconds < 0.8
    assert len(downloads) == 1


@pytest.mark.asyncio
async def test_framed_count_waits_for_a_tokenizer_download_that_finishes_within_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()

    def slow_download(model: str, custom_tokenizer: object = None):
        download_started.set()
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", slow_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    count: Final = asyncio.create_task(_count_status())
    try:
        await asyncio.to_thread(download_started.wait, 10)
    finally:
        hub_released.set()

    assert await asyncio.wait_for(count, 10) == "local_estimate"


@pytest.mark.asyncio
async def test_failed_tokenizer_download_is_reused_until_the_retry_window_passes(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/flaky-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_down_once)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e9)
    within_window: Final = [await _count_status() for _ in range(3)]
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)
    after_window: Final = await _count_status()

    assert within_window == ["400", "400", "400"]
    assert after_window == "local_estimate"
    assert len(downloads) == 2


@pytest.mark.asyncio
async def test_failed_tokenizer_load_is_reused_for_the_window_after_it_fails(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-failing-tokenizer")
    downloads: Final[list[str]] = []

    def slow_failing_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        time.sleep(1.0)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", slow_failing_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 0.5)

    statuses: Final = [await _count_status() for _ in range(2)]

    assert statuses == ["400", "400"]
    assert len(downloads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_tokenizer",
    [("lit-tests/tokenizer-b", "main"), ("lit-tests/tokenizer-a", "v2")],
    ids=["identifier", "revision"],
)
async def test_framed_deployments_with_different_custom_tokenizers_each_load_their_own(
    monkeypatch: pytest.MonkeyPatch, second_tokenizer: tuple[str, str]
):
    _framed_kimis_with_custom_tokenizers(
        monkeypatch, {"kimi-a": ("lit-tests/tokenizer-a", "main"), "kimi-b": second_tokenizer}
    )
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    loaded: Final[list[object]] = []

    def load(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        loaded.append(custom_tokenizer)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", load)

    statuses: Final = [await _count_status(name) for name in ("kimi-a", "kimi-b")]

    assert statuses == ["local_estimate", "local_estimate"]
    assert loaded == [
        {"identifier": "lit-tests/tokenizer-a", "revision": "main", "auth_token": None},
        {"identifier": second_tokenizer[0], "revision": second_tokenizer[1], "auth_token": None},
    ]


@pytest.mark.asyncio
async def test_tokenizer_loads_past_the_cap_answer_the_provider_error_without_stalling_other_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizers: Final = {f"kimi-{n}": (f"lit-tests/unreachable-tokenizer-{n}", "main") for n in range(5)}
    _framed_kimis_with_custom_tokenizers(monkeypatch, tokenizers, burst=len(tokenizers))
    hub_released: Final = threading.Event()
    downloads: Final[list[object]] = []

    def hanging_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(custom_tokenizer)
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 2)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 2)
    try:
        framed: Final = await asyncio.wait_for(asyncio.gather(*(_timed_count_status(name) for name in tokenizers)), 8)
        unrelated: Final = await asyncio.wait_for(_count_status("sonnet"), 5)
    finally:
        hub_released.set()
    await asyncio.sleep(0.3)

    assert [status for status, _ in framed] == ["400"] * 5
    assert sum(seconds < 1.0 for _, seconds in framed) == 3
    assert unrelated == "provider"
    assert len(downloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("panics_while", ["loading", "counting"])
async def test_framed_count_answers_the_provider_error_when_the_tokenizer_panics(
    monkeypatch: pytest.MonkeyPatch, panics_while: str
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, f"lit-tests/panicking-tokenizer-{panics_while}")
    panicking: Final = _tokenizer_that_panics_while_encoding()

    def load(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        if panics_while == "loading":
            panicking["tokenizer"].encode("hello world")
        return panicking

    monkeypatch.setattr(litellm.utils, "select_tokenizer", load)

    assert await _count_status() == "400"


def _framed_kimi(monkeypatch: pytest.MonkeyPatch) -> None:
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )


@pytest.mark.asyncio
async def test_local_count_loads_the_tokenizer_again_after_a_failed_load(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/local-route-tokenizer")
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_down_once)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e9)
    request: Final = TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}])

    with pytest.raises(OSError, match="hub unreachable"):
        await token_counter(request=request)
    retried: Final = await token_counter(request=request)

    assert retried.count_source == "local_estimate" and retried.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reuse_seconds", "tokenizer_types"),
    [(1e9, ("openai_tokenizer", "openai_tokenizer")), (1e-9, ("openai_tokenizer", "huggingface_tokenizer"))],
    ids=["within_the_window", "after_the_window"],
)
async def test_framed_count_loads_the_tokenizer_again_only_after_the_window_of_a_load_that_fell_back(
    monkeypatch: pytest.MonkeyPatch, reuse_seconds: float, tokenizer_types: tuple[str, str]
):
    _framed_kimi(monkeypatch)
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    hub_tokenizer: Final = litellm.utils.create_tokenizer(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).to_str()
    )
    loads: Final[list[str]] = []

    def hub_up_on_the_second_load(model: str, custom_tokenizer: object = None):
        loads.append(model)
        return real_select_tokenizer(model="gpt-5.4-mini") if len(loads) == 1 else hub_tokenizer

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hub_up_on_the_second_load)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", reuse_seconds)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 1)
    request: Final = TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}])

    first: Final = await token_counter(request=request, call_endpoint=True)
    second: Final = await token_counter(request=request, call_endpoint=True)

    assert (first.tokenizer_type, second.tokenizer_type) == tokenizer_types


async def _fork_guard_traceback_frames() -> int:
    with pytest.raises(ForkedAfterNativeRuntimeStarted) as raised:
        await token_counter(
            request=TokenCountRequest(model="kimi", messages=[{"role": "user", "content": "What is new?"}]),
            call_endpoint=True,
        )
    return len(traceback.extract_tb(raised.value.__traceback__))


@pytest.mark.asyncio
async def test_fork_guard_error_while_loading_the_tokenizer_is_raised_to_every_count_sharing_the_load(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi(monkeypatch)

    def forked_load(model: str, custom_tokenizer: object = None) -> NoReturn:
        raise ForkedAfterNativeRuntimeStarted("native runtime started before fork")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", forked_load)

    frames: Final = [await _fork_guard_traceback_frames() for _ in range(3)]

    assert len(set(frames)) == 1


@pytest.mark.asyncio
async def test_swallowed_estimate_error_is_logged_by_type_without_the_request_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "kimi", "litellm_params": {"model": "bedrock/us.moonshotai.kimi-k3"}}]),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name), pytest.raises(ProxyException):
            await token_counter(
                request=TokenCountRequest(
                    model="kimi",
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok", "tool_calls": "private-marker-in-the-prompt"},
                    ],
                ),
                call_endpoint=True,
            )
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)

    warnings: Final = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("count error 400" in message and "ValueError" in message for message in warnings)
    assert not any("private-marker-in-the-prompt" in message for message in warnings)


@pytest.mark.asyncio
async def test_fork_guard_error_in_the_estimate_is_never_answered_as_the_provider_error():
    async def forked_estimate() -> NoReturn:
        raise ForkedAfterNativeRuntimeStarted("native runtime started before fork")

    with pytest.raises(ForkedAfterNativeRuntimeStarted):
        await _estimate_or_provider_error(
            forked_estimate(), provider_failure=ProviderCountFailure(status_code=400, message="scripted 400")
        )


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
async def test_framing_comes_from_the_row_the_deployment_provider_or_base_model_names(monkeypatch: pytest.MonkeyPatch):
    def deployment(name: str, litellm_params: dict[str, object], model_info: dict[str, object]) -> dict[str, object]:
        return {
            "model_name": name,
            "litellm_params": {**litellm_params, "api_key": "unused", "api_base": "https://unused.invalid"},
            "model_info": model_info,
        }

    kimi: Final = "databricks/system.ai.kimi-k3"
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                deployment("prefixed", {"model": kimi}, {}),
                deployment("split", {"model": "system.ai.kimi-k3", "custom_llm_provider": "databricks"}, {}),
                deployment("aliased", {"model": "gpt-5.4-mini", "custom_llm_provider": "openai"}, {"base_model": kimi}),
            ]
        ),
    )
    messages: Final = [{"role": "user", "content": "hello there"}]
    framing: Final = cost_map_entry(kimi)["token_count_framing"]

    prefixed: Final = await token_counter(request=TokenCountRequest(model="prefixed", messages=messages))
    split: Final = await token_counter(request=TokenCountRequest(model="split", messages=messages))
    aliased: Final = await token_counter(request=TokenCountRequest(model="aliased", messages=messages))

    assert prefixed.total_tokens == litellm.token_counter(model="system.ai.kimi-k3", messages=messages) + framing
    assert split.total_tokens == prefixed.total_tokens
    assert aliased.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages) + framing


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
        token_counter(
            request=TokenCountRequest(model="slow-lookup-model", messages=[{"role": "user", "content": "hi"}])
        )
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
    counter: Final = _ScriptedCounter({"us.moonshotai.kimi-k3": _provider_reply("kimi", status_code=400)})
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


@pytest.mark.parametrize(
    ("model_info", "counted_with"),
    [
        ({}, ("us.anthropic.claude-opus-5-5", "us-east-1", 447)),
        (
            {"token_count_model": {"model": "us.anthropic.claude-opus-5-5", "region": "ap-southeast-4"}},
            ("us.anthropic.claude-opus-5-5", "ap-southeast-4", 447),
        ),
        ({"token_count_model": {}}, ("us.anthropic.claude-sonnet-5-5", "us-west-2", 300)),
    ],
)
@pytest.mark.asyncio
async def test_configured_token_count_model_overrides_the_shipped_one_and_an_empty_one_counts_the_model_itself(
    monkeypatch: pytest.MonkeyPatch, model_info: dict[str, object], counted_with: tuple[str, str, int]
):
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=300),
            "us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447),
        }
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "bedrock/us.anthropic.claude-sonnet-5-5",
                        "aws_region_name": "us-west-2",
                    },
                    "model_info": model_info,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    counted: Final = await token_counter(
        request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
    )

    assert (counter.calls[0][0], counter.regions[0], counted.total_tokens) == counted_with
    assert len(counter.calls) == 1


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
        model_id="framed-deployment",
        original_exception=Exception("rate limited"),
        exception_status=429,
        cooldown_time=60,
    )
    cooling: Final = await token_counter(request=TokenCountRequest(model="framed", messages=messages))

    assert warm.total_tokens == litellm.token_counter(model="gpt-5.4-mini", messages=messages) + 100
    assert cooling.total_tokens - 100 == litellm.token_counter(model="framed", messages=messages)
    assert cooling.count_source == "local_estimate"


_GEMINI_TOOL_OUTPUT: Final = "line of a long file that the agent read earlier. " * 180


class _GeminiCountTokensPeer:
    def __init__(self) -> None:
        self.request_bodies: tuple[bytes, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_bodies = (*self.request_bodies, request.content)
        return httpx.Response(200, json={"totalTokens": 41}, request=request)


def _agent_turns_with_tool_results(rounds: int) -> list[dict[str, JsonValue]]:
    return [
        *(
            turn
            for index in range(rounds)
            for turn in (
                {"role": "user", "content": f"Read file {index}."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"toolu_{index}", "content": _GEMINI_TOOL_OUTPUT}
                    ],
                },
            )
        ),  # comprehension-ok: flattens each round into its three turns
        {"role": "user", "content": "Summarize what you read."},
    ]


_READ_FILE_TOOL: Final[dict[str, JsonValue]] = {
    "name": "read_file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_CLEAR_TOOL_USES: Final[dict[str, JsonValue]] = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 500},
            "keep": {"type": "tool_uses", "value": 1},
        }
    ]
}
_INVALID_COMPACT_EDIT: Final[dict[str, JsonValue]] = {
    "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
}


def _gemini_deployment(model_name: str, litellm_params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key", **litellm_params},
    }


async def _count_through_the_messages_endpoint(
    router: Router, request_options: Mapping[str, JsonValue]
) -> tuple[object, tuple[bytes, ...]]:
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server
    from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler

    peer: Final = _GeminiCountTokensPeer()
    request_body: Final[dict[str, JsonValue]] = {
        "messages": _agent_turns_with_tool_results(3),
        "tools": [_READ_FILE_TOOL],
        "context_management": _CLEAR_TOOL_USES,
        **request_options,
    }
    mock_request: Final = MagicMock(spec=Request)
    mock_request.headers = {}

    with (
        patch.object(proxy_server, "llm_router", router),
        patch.object(anthropic_endpoints, "read_request_body", AsyncMock(return_value=request_body)),
        patch.object(
            gemini_count_tokens_handler,
            "get_async_httpx_client",
            return_value=AsyncHTTPHandler(transport=httpx.MockTransport(peer)),
        ),
    ):
        response: Final = await anthropic_count_tokens(mock_request, MagicMock())
    return response, peer.request_bodies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "requested_model",
        "additional_drop_params",
        "router_additional_drop_params",
        "request_options",
        "expected_tool_outputs_sent",
    ),
    (
        ("gemini-agent", None, None, {}, 1),
        ("gemini-agent", ["context_management"], None, {}, 3),
        ("gemini-agent", None, None, {"additional_drop_params": ["context_management"]}, 3),
        ("gemini-agent", ["context_management"], None, {"additional_drop_params": []}, 1),
        ("gemini-agent", None, ["context_management"], {}, 3),
        ("gemini-agent", None, ["context_management"], {"additional_drop_params": []}, 1),
        ("gemini/gemini-2.5-flash", None, None, {}, 1),
        ("gemini/gemini-2.5-flash", ["context_management"], None, {}, 3),
        ("gemini/gemini-2.5-flash", None, ["context_management"], {}, 3),
        ("gemini/gemini-2.5-flash", None, None, {"additional_drop_params": ["context_management"]}, 3),
        ("gemini/gemini-2.5-flash", ["context_management"], None, {"additional_drop_params": []}, 1),
    ),
    ids=(
        "edits-applied",
        "deployment-drops-context-management",
        "request-drops-context-management",
        "request-overrides-the-deployment-drop",
        "router-default-drops-context-management",
        "request-overrides-the-router-default-drop",
        "litellm-model-name-edits-applied",
        "litellm-model-name-deployment-drops-context-management",
        "litellm-model-name-router-default-drops-context-management",
        "litellm-model-name-request-drops-context-management",
        "litellm-model-name-request-overrides-the-deployment-drop",
    ),
)
async def test_gemini_messages_count_tokens_applies_the_context_edits_inference_applies(
    requested_model: str,
    additional_drop_params: list[str] | None,
    router_additional_drop_params: list[str] | None,
    request_options: dict[str, JsonValue],
    expected_tool_outputs_sent: int,
) -> None:
    router: Final = Router(
        model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": additional_drop_params})],
        default_litellm_params={"additional_drop_params": router_additional_drop_params},
    )
    response, sent_bodies = await _count_through_the_messages_endpoint(
        router, {**request_options, "model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert len(sent_bodies) == 1
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "model_list"),
    (
        (
            "gemini-agent",
            [
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key", "order": 1},
                },
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "order": 2,
                        "additional_drop_params": ["context_management"],
                    },
                },
            ],
        ),
        (
            "gemini-agent-deployment-id",
            [
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "additional_drop_params": ["context_management"],
                    },
                    "model_info": {"id": "gemini-agent-deployment-id"},
                },
            ],
        ),
        (
            "gemini-agent-deployment-id",
            [
                {"model_name": "*", "litellm_params": {"model": "gemini/*", "api_key": "test-key"}},
                {
                    "model_name": "gemini-agent",
                    "litellm_params": {
                        "model": "gemini/gemini-2.5-flash",
                        "api_key": "test-key",
                        "additional_drop_params": ["context_management"],
                    },
                    "model_info": {"id": "gemini-agent-deployment-id"},
                },
            ],
        ),
    ),
    ids=("group-member-drops", "deployment-id-drops", "deployment-id-drops-beside-a-wildcard"),
)
async def test_gemini_messages_count_tokens_keeps_tool_results_when_a_routable_deployment_drops_context_management(
    requested_model: str, model_list: list[dict[str, JsonValue]]
) -> None:
    response, sent_bodies = await _count_through_the_messages_endpoint(
        Router(model_list=model_list), {"model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "make_router", "request_options", "expected_tool_outputs_sent"),
    (
        (
            "unrouted-model",
            lambda: Router(model_list=[_gemini_deployment("gemini-agent", {})], default_fallbacks=["gemini-agent"]),
            {},
            1,
        ),
        (
            "unrouted-model",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": ["context_management"]})],
                default_fallbacks=["gemini-agent"],
            ),
            {},
            3,
        ),
        (
            "unrouted-model",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {})],
                default_fallbacks=["gemini-agent"],
                default_litellm_params={"additional_drop_params": ["context_management"]},
            ),
            {},
            3,
        ),
        (
            "unrouted-model",
            lambda: Router(model_list=[_gemini_deployment("gemini-agent", {})], default_fallbacks=["gemini-agent"]),
            {"additional_drop_params": ["context_management"]},
            3,
        ),
        (
            "flash-alias",
            lambda: Router(
                model_list=[_gemini_deployment("gemini-agent", {"additional_drop_params": ["context_management"]})],
                model_group_alias={"flash-alias": "gemini/gemini-2.5-flash"},
            ),
            {},
            3,
        ),
    ),
    ids=(
        "default-fallback-edits-applied",
        "default-fallback-deployment-drops-context-management",
        "default-fallback-router-default-drops-context-management",
        "default-fallback-request-drops-context-management",
        "alias-to-a-litellm-model-deployment-drops-context-management",
    ),
)
async def test_gemini_messages_count_tokens_reads_the_opt_out_of_the_deployment_the_router_falls_back_to(
    requested_model: str,
    make_router: Callable[[], Router],
    request_options: dict[str, JsonValue],
    expected_tool_outputs_sent: int,
) -> None:
    response, sent_bodies = await _count_through_the_messages_endpoint(
        make_router(), {**request_options, "model": requested_model}
    )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "completion_model", "expected_tool_outputs_sent"),
    (("gemini-agent", "gemini-dropping", 3), ("gemini-dropping", "gemini-agent", 1)),
    ids=("completion-model-drops-context-management", "completion-model-keeps-context-management"),
)
async def test_gemini_messages_count_tokens_reads_the_opt_out_of_the_completion_model_inference_sends_to(
    requested_model: str, completion_model: str, expected_tool_outputs_sent: int
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            _gemini_deployment("gemini-agent", {}),
            _gemini_deployment("gemini-dropping", {"additional_drop_params": ["context_management"]}),
        ]
    )
    with patch.object(proxy_server, "general_settings", {"completion_model": completion_model}):
        response, sent_bodies = await _count_through_the_messages_endpoint(router, {"model": requested_model})

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == expected_tool_outputs_sent


class _GeminiPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str = ""


class _GeminiContent(BaseModel):
    model_config = ConfigDict(frozen=True)
    parts: tuple[_GeminiPart, ...] = ()


class _GeminiCountTokensBody(BaseModel):
    model_config = ConfigDict(frozen=True)
    contents: tuple[_GeminiContent, ...] = ()


def _word_count(content: _GeminiContent) -> int:
    return sum(len(part.text.split()) for part in content.parts)


def _gemini_word_counting_peer(request: httpx.Request) -> httpx.Response:
    body: Final = _GeminiCountTokensBody.model_validate_json(request.content)
    total: Final = sum(_word_count(content) for content in body.contents)
    return httpx.Response(200, json={"totalTokens": total}, request=request)


def _unavailable_peer(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": {"message": "unavailable"}}, request=request)


def _gemini_router(drop_params: bool = False) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-counted",
                "litellm_params": {
                    "model": "gemini/gemini-3.8-flash",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                    "drop_params": drop_params,
                },
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "router_defaults", "request_options", "litellm_drop_params"),
    (
        ({"drop_params": True}, {}, {}, False),
        ({}, {"drop_params": True}, {}, False),
        ({}, {}, {"drop_params": True}, False),
        ({}, {}, {}, True),
    ),
    ids=("deployment-drop-params", "router-default-drop-params", "request-drop-params", "litellm-drop-params"),
)
async def test_gemini_messages_count_tokens_counts_the_unedited_body_when_drop_params_drops_an_invalid_edit(
    deployment_params: dict[str, JsonValue],
    router_defaults: dict[str, JsonValue],
    request_options: dict[str, JsonValue],
    litellm_drop_params: bool,
) -> None:
    router: Final = Router(
        model_list=[_gemini_deployment("gemini-agent", deployment_params)], default_litellm_params=router_defaults
    )
    with patch.object(litellm, "drop_params", litellm_drop_params):
        response, sent_bodies = await _count_through_the_messages_endpoint(
            router, {**request_options, "model": "gemini-agent", "context_management": _INVALID_COMPACT_EDIT}
        )

    assert response == {"input_tokens": 41, "litellm_count_source": "provider"}
    assert sent_bodies[0].decode().count(_GEMINI_TOOL_OUTPUT) == 3


@pytest.mark.asyncio
async def test_gemini_messages_count_tokens_rejects_an_invalid_edit_without_drop_params() -> None:
    with patch.object(litellm, "drop_params", False):
        response, sent_bodies = await _count_through_the_messages_endpoint(
            Router(model_list=[_gemini_deployment("gemini-agent", {})]),
            {"model": "gemini-agent", "context_management": _INVALID_COMPACT_EDIT},
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert sent_bodies == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["hello there", "The quick brown fox jumps over the lazy dog. " * 50], ids=["short", "long"]
)
async def test_call_endpoint_counts_a_prompt_like_the_same_text_sent_as_a_user_message(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        from_prompt: Final = await token_counter(
            request=TokenCountRequest(model="gemini-counted", prompt=text), call_endpoint=True
        )
        from_message: Final = await token_counter(
            request=TokenCountRequest(model="gemini-counted", messages=[{"role": "user", "content": text}]),
            call_endpoint=True,
        )

    assert from_prompt.error is False
    assert from_prompt.total_tokens == from_message.total_tokens == len(text.split())


class _PartnerMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str = ""


class _PartnerCountRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: tuple[_PartnerMessage, ...] = ()


async def _vertex_partner_word_count(
    model: str, request_data: JsonValue, litellm_params: JsonValue
) -> dict[str, JsonValue]:
    body: Final = _PartnerCountRequest.model_validate(request_data)
    return {"input_tokens": sum(len(message.content.split()) for message in body.messages)}


@pytest.mark.asyncio
async def test_call_endpoint_counts_a_prompt_on_a_vertex_ai_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "vertex-counted",
                "litellm_params": {
                    "model": "vertex_ai/claude-sonnet-5-5",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    text: Final = "The quick brown fox jumps over the lazy dog."

    with patch.object(
        VertexAIPartnerModelsTokenCounter, "handle_count_tokens_request", side_effect=_vertex_partner_word_count
    ):
        from_prompt: Final = await token_counter(
            request=TokenCountRequest(model="vertex-counted", prompt=text), call_endpoint=True
        )

    assert (from_prompt.error, from_prompt.total_tokens) == (False, 9)


_FOX: Final = "The quick brown fox jumps over the lazy dog."


_CONVERSATIONS: Final = (
    {"messages": [{"role": "user", "content": _FOX}]},
    {"contents": [{"role": "user", "parts": [{"text": _FOX}]}]},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", _CONVERSATIONS, ids=["messages", "contents"])
@pytest.mark.parametrize("call_endpoint", [True, False], ids=["provider", "local"])
async def test_prompt_next_to_a_conversation_answers_400_without_drop_params(
    conversation: dict[str, JsonValue], call_endpoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with (
        patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client),
        pytest.raises(ProxyException) as rejected,
    ):
        await token_counter(
            request=TokenCountRequest(model="gemini-counted", prompt="hi", **conversation), call_endpoint=call_endpoint
        )

    assert (rejected.value.code, rejected.value.param) == ("400", "prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation", "call_endpoint", "global_drop_params", "deployment_drop_params"),
    [
        (_CONVERSATIONS[0], True, False, True),
        (_CONVERSATIONS[1], True, False, True),
        (_CONVERSATIONS[0], False, False, True),
        (_CONVERSATIONS[0], False, True, False),
    ],
    ids=[
        "provider_messages",
        "provider_contents",
        "local_messages",
        "local_messages_global_setting",
    ],
)
async def test_drop_params_counts_a_conversation_without_the_prompt(
    conversation: dict[str, JsonValue],
    call_endpoint: bool,
    global_drop_params: bool,
    deployment_drop_params: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router(drop_params=deployment_drop_params))
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        with_prompt: Final = await token_counter(
            request=TokenCountRequest(model="gemini-counted", prompt="hi", **conversation), call_endpoint=call_endpoint
        )
        conversation_only: Final = await token_counter(
            request=TokenCountRequest(model="gemini-counted", **conversation), call_endpoint=call_endpoint
        )

    assert with_prompt.error is False
    assert with_prompt.total_tokens == conversation_only.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_conversation", [{"messages": []}, {"contents": []}], ids=["messages", "contents"])
async def test_call_endpoint_prompt_next_to_an_empty_conversation_counts_the_prompt(
    empty_conversation: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_gemini_word_counting_peer))

    with patch.object(gemini_count_tokens_handler, "get_async_httpx_client", return_value=client):
        counted: Final = await token_counter(
            request=TokenCountRequest(model="gemini-counted", prompt=_FOX, **empty_conversation), call_endpoint=True
        )

    assert (counted.error, counted.total_tokens) == (False, 9)


@pytest.mark.asyncio
async def test_local_count_of_a_prompt_next_to_empty_messages_counts_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())

    with_empty: Final = await token_counter(request=TokenCountRequest(model="gemini-counted", prompt=_FOX, messages=[]))
    alone: Final = await token_counter(request=TokenCountRequest(model="gemini-counted", prompt=_FOX))

    assert with_empty.error is False
    assert with_empty.total_tokens == alone.total_tokens > 0


_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_fields",
    [{"system": "You answer in one short sentence. " * 20}, {"tools": [_WEATHER_TOOL]}],
    ids=["system", "tools"],
)
@pytest.mark.parametrize("empty_conversation", [{}, {"messages": []}], ids=["no_messages", "empty_messages"])
async def test_local_count_of_a_prompt_with_system_or_tools_counts_them_like_the_message_form(
    chat_fields: dict[str, JsonValue], empty_conversation: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", _gemini_router())

    from_prompt: Final = await token_counter(
        request=TokenCountRequest(model="gemini-counted", prompt=_FOX, **empty_conversation, **chat_fields)
    )
    from_message: Final = await token_counter(
        request=TokenCountRequest(model="gemini-counted", messages=[{"role": "user", "content": _FOX}], **chat_fields)
    )
    prompt_alone: Final = await token_counter(request=TokenCountRequest(model="gemini-counted", prompt=_FOX))

    assert from_prompt.error is False
    assert from_prompt.total_tokens == from_message.total_tokens > prompt_alone.total_tokens


@pytest.mark.asyncio
async def test_call_endpoint_prompt_keeps_the_local_count_on_a_counter_that_needs_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic-counted",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_unavailable_peer))
    request: Final = TokenCountRequest(model="anthropic-counted", prompt="The quick brown fox jumps over the lazy dog.")

    with patch.object(anthropic_count_tokens_handler, "get_async_httpx_client", return_value=client):
        with_endpoint: Final = await token_counter(request=request, call_endpoint=True)
    local: Final = await token_counter(request=request, call_endpoint=False)

    assert (with_endpoint.error, with_endpoint.total_tokens, with_endpoint.tokenizer_type) == (
        False,
        local.total_tokens,
        local.tokenizer_type,
    )
