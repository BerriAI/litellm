# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import asyncio
import base64
import hashlib
import itertools
import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Coroutine, Iterator, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest
import respx
import tiktoken
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, JsonValue
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

load_dotenv()

# this file is to test litellm/proxy


from fastapi import Request
from fastapi.responses import JSONResponse

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.anthropic.count_tokens import handler as anthropic_count_tokens_handler
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.bedrock.count_tokens.kimi_k3 import (
    KIMI_K3_RANKS_SHA256,
    KimiK3Encodings,
    KimiK3TokenizerSource,
    kimi_k3_template_tokens,
    kimi_k3_tokens,
    load_kimi_k3_encoding,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.count_tokens import handler as gemini_count_tokens_handler
from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import VertexAIPartnerModelsTokenCounter
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.proxy.common_utils.token_count_settings import (
    ProviderCountFailure,
    TokenCountModel,
    TokenCountSettings,
    cost_map_entry,
)
from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request
from litellm.proxy.proxy_server import (
    _estimate_or_provider_error,
    _provider_token_count,
    _try_provider_token_count,
    count_request_tokens,
    token_counter,
)
from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from litellm.types.utils import SelectTokenizerResponse, TokenCountMessagesFormat, TokenCountResponse


def _hugging_face_offline() -> NoReturn:
    raise OSError("unit tests never download the Kimi K3 tokenizer")


@pytest.fixture(autouse=True)
def _kimi_k3_tokenizer_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "_KIMI_K3_ENCODINGS",
        KimiK3Encodings(
            source=KimiK3TokenizerSource(ranks_path=_hugging_face_offline, sha256=KIMI_K3_RANKS_SHA256),
            retry_after_seconds=60,
        ),
    )


@pytest.fixture
def bedrock_sonnet_5_5_mapped_as_on_main() -> Iterator[None]:
    entries: Final = {
        name: {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "token_count_model": {"model": count_model, "region": "us-east-1"},
        }
        for name, count_model in (
            ("anthropic.claude-sonnet-5-5", "anthropic.claude-opus-5-5"),
            ("us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-opus-5-5"),
        )
    }
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(litellm, "model_cost", {**litellm.model_cost, **entries})
        litellm.utils._invalidate_model_cost_lowercase_map()
        yield
    litellm.utils._invalidate_model_cost_lowercase_map()


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
    async def mock_token_counter(request, call_endpoint=False, messages_format="openai", output_format=None):
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
    async def mock_token_counter(request, call_endpoint=True, messages_format="openai", output_format=None):
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
@pytest.mark.parametrize("status_code", (400, 401, 403, 404, 429, 500))
@pytest.mark.parametrize(
    ("model", "count_url"),
    (
        ("anthropic/claude-sonnet-5-5", "https://api.anthropic.com/v1/messages/count_tokens"),
        ("openai/gpt-6.1-sol", "https://api.openai.com/v1/responses/input_tokens"),
    ),
)
async def test_provider_errors_return_labelled_local_estimates(
    monkeypatch: pytest.MonkeyPatch, status_code: int, model: str, count_url: str
) -> None:
    messages: Final = [
        {"role": "user", "content": "Count the whole request locally when the count API is unavailable."}
    ]
    router: Final = Router(
        model_list=[{"model_name": "counted", "litellm_params": {"model": model, "api_key": "test-count-key"}}]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "disable_token_counter", False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    with respx.mock(assert_all_called=True) as upstream:
        upstream.post(count_url).mock(
            return_value=httpx.Response(
                status_code, json={"type": "error", "error": {"type": "api_error", "message": "Count unavailable"}}
            )
        )
        counted: Final = await token_counter(
            request=TokenCountRequest(model="counted", messages=messages), call_endpoint=True
        )

    assert counted.total_tokens == litellm.token_counter(model=model.split("/", 1)[1], messages=messages)
    assert counted.count_source == "local_estimate"
    assert counted.error is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part",
    (
        {"fileData": {"mimeType": "image/png", "fileUri": "gs://bucket/image.png"}},
        {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
        {"text": "Reasoning about the request", "thought": True},
    ),
)
async def test_google_provider_errors_preserve_legacy_translated_message_estimates(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, JsonValue]
) -> None:
    model: Final = "openai/gpt-6.1-sol"
    request: Final = _google_count_tokens_request(
        "counted", {"contents": [{"role": "user", "parts": [{"text": "Describe this"}, part]}]}
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(model_list=[{"model_name": "counted", "litellm_params": {"model": model, "api_key": "test-key"}}]),
    )
    monkeypatch.setattr(litellm, "disable_token_counter", False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    with respx.mock(assert_all_called=True) as upstream:
        upstream.post("https://api.openai.com/v1/responses/input_tokens").mock(
            return_value=httpx.Response(400, json={"error": {"message": "Count unavailable"}})
        )
        counted: Final = await token_counter(request=request, call_endpoint=True)
    assert counted.total_tokens == litellm.token_counter(model="gpt-6.1-sol", messages=request.messages)
    assert counted.count_source == "local_estimate"
    assert counted.error is False


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
    async def mock_token_counter_error(request, call_endpoint=False, messages_format="openai", output_format=None):
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
    async def mock_token_counter_error(request, call_endpoint=False, messages_format="openai", output_format=None):
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
    async def mock_token_counter_error(request, call_endpoint=False, messages_format="openai", output_format=None):
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
) -> TokenCountResponse | ProviderCountFailure | None:
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
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=447)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

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
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=None, count_model=count_model))

    assert result is not None
    assert (result.total_tokens, result.count_source, result.model_used) == (
        447,
        "token_count_model",
        "us.anthropic.claude-opus-5-5",
    )
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-opus-5-5"], [region])
    assert counter.calls[0][1:] == (_COUNT_REQUEST.messages, _COUNT_REQUEST.tools, _COUNT_REQUEST.system)
    assert _SONNET_DEPLOYMENT["litellm_params"]["aws_region_name"] == "us-west-2"


@pytest.mark.parametrize("status_code", (400, 401, 403, 404, 429, 500))
@pytest.mark.parametrize("framing", (None, 0, 10))
@pytest.mark.asyncio
async def test_provider_count_errors_preserve_legacy_local_fallback(status_code: int, framing: int | None):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code)}
    )

    result: Final = await _ladder(counter, TokenCountSettings(framing=framing, count_model=None))

    assert isinstance(result, ProviderCountFailure)
    assert (result.status_code, result.message) == (status_code, f"scripted {status_code} for sonnet")


@pytest.mark.parametrize("status_code", [400, 403, 404, 500])
@pytest.mark.asyncio
async def test_token_count_model_failure_names_its_region_so_the_user_can_enable_it(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    result: Final = await _ladder(counter, TokenCountSettings(framing=None, count_model=_OPUS_IN_US_EAST_1))

    assert isinstance(result, ProviderCountFailure)
    assert result.status_code == status_code
    assert result.message.startswith(f"scripted {status_code} for opus")
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in result.message
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_token_count_model_without_a_region_preserves_its_error_for_a_failed_local_estimate():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404)})

    result: Final = await _ladder(
        counter,
        TokenCountSettings(framing=None, count_model=TokenCountModel(model="us.anthropic.claude-opus-5-5", region=None)),
    )

    assert isinstance(result, ProviderCountFailure)
    assert (result.status_code, result.message) == (404, "scripted 404 for opus")


@pytest.mark.asyncio
async def test_uncountable_model_without_framing_preserves_legacy_local_fallback():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=None, count_model=None))

    assert isinstance(result, ProviderCountFailure)
    assert (result.status_code, result.message) == (404, "scripted 404 for sonnet")


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_falls_through_to_the_local_estimate():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=0, count_model=None))

    assert isinstance(result, ProviderCountFailure)
    assert len(counter.calls) == 1


@pytest.mark.parametrize("status_code", [400, 404])
@pytest.mark.asyncio
async def test_token_count_model_failure_allows_a_framed_local_estimate(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    result: Final = await _ladder(counter, TokenCountSettings(framing=78, count_model=_OPUS_IN_US_EAST_1))

    assert isinstance(result, ProviderCountFailure)
    assert result.status_code == status_code
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in result.message


@pytest.mark.asyncio
async def test_uncountable_model_with_framing_raises_when_local_counting_is_disabled():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=None), local_estimate_allowed=False)

    assert raised.value.code == "400"


@pytest.mark.asyncio
async def test_rejected_provider_conversion_still_allows_a_local_estimate():
    counter: Final = _ScriptedCounter(
        {
            "us.anthropic.claude-sonnet-5-5": _provider_reply(
                "sonnet", status_code=400, rejected_before_provider_call=True
            )
        }
    )

    result: Final = await _ladder(counter, TokenCountSettings(framing=100, count_model=None))

    assert isinstance(result, ProviderCountFailure)
    assert (result.status_code, result.message) == (400, "scripted 400 for sonnet")


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
@pytest.mark.parametrize(
    "contents",
    [
        pytest.param([{"role": "user", "parts": [{"text": "What is new in the release notes?"}]}], id="text"),
        pytest.param(
            [
                {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "c2ln"}
                    ],
                },
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "get_weather", "response": {"forecast": "Sun"}}}],
                },
            ],
            id="sdk_agent_loop",
        ),
        pytest.param(
            [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!", "thoughtSignature": "c2ln"}]},
                {"role": "user", "parts": [{"text": "What is new?"}]},
            ],
            id="sdk_chat_with_a_thought_signature",
        ),
        pytest.param([], id="empty"),
    ],
)
async def test_framed_estimate_counts_google_contents_like_the_same_conversation_sent_as_messages(
    monkeypatch: pytest.MonkeyPatch, contents: list[dict[str, object]]
):
    _framed_kimi(monkeypatch)
    google_request: Final = _google_count_tokens_request("kimi", {"contents": contents})

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


class _SkippableClockLoop(asyncio.SelectorEventLoop):
    skipped_seconds: float = 0.0

    def time(self) -> float:
        return super().time() + self.skipped_seconds

    def skip(self, seconds: float) -> None:
        self.skipped_seconds += seconds


def _run_on_skippable_clock(
    scenario: Callable[[_SkippableClockLoop], Coroutine[object, object, tuple[str, ...]]],
) -> tuple[str, ...]:
    loop: Final = _SkippableClockLoop()
    try:
        return loop.run_until_complete(scenario(loop))
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


async def _until_finished(tasks: tuple[asyncio.Future[str], ...], count: int) -> None:
    for finished in itertools.islice(asyncio.as_completed(tasks), count):
        await finished


def _tokenizer_that_panics_while_encoding() -> SelectTokenizerResponse:
    template: Final = Tokenizer(WordLevel({"[UNK]": 0, "<s>": 1}, unk_token="[UNK]"))
    template.pre_tokenizer = Whitespace()
    template.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    spec: Final = json.loads(template.to_str())
    return litellm.utils.create_tokenizer(
        json.dumps({**spec, "post_processor": {**spec["post_processor"], "special_tokens": {}}})
    )


def test_hanging_tokenizer_download_answers_the_provider_error_after_the_wait_with_one_download_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/hanging-tokenizer")
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def hanging_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        download_started.set()
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", hanging_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)

    async def burst_past_the_wait_then_one_more(loop: _SkippableClockLoop) -> tuple[str, ...]:
        try:
            burst: Final = asyncio.gather(*(_count_status() for _ in range(60)))
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(31)
            answered: Final = await asyncio.wait_for(burst, 8)
            later: Final = await asyncio.wait_for(_count_status(), 8)
        finally:
            hub_released.set()
        return (*answered, later)

    assert set(_run_on_skippable_clock(burst_past_the_wait_then_one_more)) == {"400"}
    assert len(downloads) == 1


@pytest.mark.asyncio
async def test_framed_count_waits_for_a_tokenizer_download_that_finishes_within_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-tokenizer")
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()

    def slow_download(model: str, custom_tokenizer: object = None):
        download_started.set()
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", slow_download)
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
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", hub_down_once)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e9)
    within_window: Final = [await _count_status() for _ in range(3)]
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 1e-9)
    after_window: Final = await _count_status()

    assert within_window == ["400", "400", "400"]
    assert after_window == "local_estimate"
    assert len(downloads) == 2


def test_failed_tokenizer_load_is_reused_for_the_window_after_it_fails(monkeypatch: pytest.MonkeyPatch):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/slow-failing-tokenizer")
    download_started: Final = threading.Event()
    hub_released: Final = threading.Event()
    downloads: Final[list[str]] = []

    def slow_failing_download(model: str, custom_tokenizer: object = None) -> NoReturn:
        downloads.append(model)
        download_started.set()
        hub_released.wait(10)
        raise OSError("hub unreachable")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", slow_failing_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 1000)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_REUSE_SECONDS", 60)

    async def fail_after_longer_than_the_window_then_retry(loop: _SkippableClockLoop) -> tuple[str, ...]:
        first: Final = asyncio.ensure_future(_count_status())
        try:
            await asyncio.to_thread(download_started.wait, 10)
            loop.skip(120)
        finally:
            hub_released.set()
        return await asyncio.wait_for(first, 8), await asyncio.wait_for(_count_status(), 8)

    assert _run_on_skippable_clock(fail_after_longer_than_the_window_then_retry) == ("400", "400")
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
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    loaded: Final[list[object]] = []

    def load(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        loaded.append(custom_tokenizer)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", load)

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
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    hub_released: Final = threading.Event()
    downloads: Final[list[object]] = []

    def held_download(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        downloads.append(custom_tokenizer)
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", held_download)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_MAX_RUNNING", 2)
    monkeypatch.setattr(litellm.proxy.proxy_server, "TOKENIZER_LOAD_WAIT_SECONDS", 30)
    monkeypatch.setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 2)
    framed: Final = tuple(asyncio.ensure_future(_count_status(name)) for name in tokenizers)
    try:
        await asyncio.wait_for(_until_finished(framed, 3), 8)
        unrelated: Final = await asyncio.wait_for(_count_status("sonnet"), 5)
    finally:
        hub_released.set()
    statuses: Final = await asyncio.wait_for(asyncio.gather(*framed), 8)

    assert sorted(statuses) == ["400"] * 3 + ["local_estimate"] * 2
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

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", load)

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
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    downloads: Final[list[str]] = []

    def hub_down_once(model: str, custom_tokenizer: object = None):
        downloads.append(model)
        if len(downloads) == 1:
            raise OSError("hub unreachable")
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", hub_down_once)
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
    real_select_tokenizer: Final = litellm.utils._select_tokenizer
    hub_tokenizer: Final = litellm.utils.create_tokenizer(
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).to_str()
    )
    loads: Final[list[str]] = []

    def hub_up_on_the_second_load(model: str, custom_tokenizer: object = None):
        loads.append(model)
        return real_select_tokenizer(model="gpt-5.4-mini") if len(loads) == 1 else hub_tokenizer

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", hub_up_on_the_second_load)
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

    monkeypatch.setattr(litellm.utils, "_select_tokenizer", forked_load)

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


_KIMI_WEATHER: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _byte_level_kimi_k3(path: Path) -> KimiK3TokenizerSource:
    path.write_bytes(b"\n".join(base64.b64encode(bytes((value,))) + b" %d" % value for value in range(256)))
    return KimiK3TokenizerSource(ranks_path=lambda: str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _available_kimi_k3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tiktoken.Encoding:
    source: Final = _byte_level_kimi_k3(tmp_path / "tiktoken.model")
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_KIMI_K3_ENCODINGS", KimiK3Encodings(source=source, retry_after_seconds=60)
    )
    encoding: Final = load_kimi_k3_encoding(source)
    assert encoding is not None
    return encoding


def _shipped_kimi_k3_framing() -> int:
    framing: Final = cost_map_entry("bedrock/us.moonshotai.kimi-k3")["token_count_framing"]
    assert isinstance(framing, int)
    return framing


@pytest.mark.asyncio
async def test_bedrock_kimi_k3_is_counted_with_its_tokenizer_plus_the_shipped_framing_in_both_request_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _framed_kimi(monkeypatch)
    encoding: Final = _available_kimi_k3(monkeypatch, tmp_path)
    question: Final = {"role": "user", "content": "Weather in Paris?"}
    openai_messages: Final = [
        {"role": "system", "content": "Be terse."},
        question,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sun"},
    ]

    openai: Final = await count_request_tokens(
        request=TokenCountRequest(model="kimi", messages=openai_messages, tools=[_KIMI_WEATHER]), call_endpoint=True
    )
    anthropic: Final = await count_request_tokens(
        request=TokenCountRequest(
            model="kimi",
            system="Be terse.",
            messages=[
                question,
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Paris"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Sun"}]},
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": _KIMI_WEATHER["function"]["parameters"],
                }
            ],
        ),
        call_endpoint=True,
        messages_format="anthropic",
    )
    exact: Final = kimi_k3_tokens(encoding, "us.moonshotai.kimi-k3", openai_messages, [_KIMI_WEATHER])

    assert exact is not None
    assert (openai.total_tokens, openai.tokenizer_type, openai.count_source) == (
        exact + _shipped_kimi_k3_framing(),
        "huggingface_tokenizer",
        "local_estimate",
    )
    assert (anthropic.total_tokens, anthropic.tokenizer_type) == (openai.total_tokens, "huggingface_tokenizer")


@pytest.mark.asyncio
async def test_bedrock_kimi_k3_deployment_with_a_custom_tokenizer_keeps_counting_with_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _framed_kimi_with_custom_tokenizer(monkeypatch, "lit-tests/kimi-custom-tokenizer")
    _available_kimi_k3(monkeypatch, tmp_path)
    custom: Final = litellm.utils._select_tokenizer(model="gpt-5.4-mini")
    monkeypatch.setattr(litellm.utils, "_select_tokenizer", lambda model, custom_tokenizer=None: custom)
    messages: Final = [{"role": "user", "content": "What is new?"}]

    counted: Final = await token_counter(request=TokenCountRequest(model="kimi", messages=messages), call_endpoint=True)

    assert counted.total_tokens == (
        litellm.token_counter(model=counted.model_used, messages=messages, custom_tokenizer=custom)
        + kimi_k3_template_tokens(len(messages), False)
        + _shipped_kimi_k3_framing()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "tools", "template_tokens"),
    [
        pytest.param([{"role": "user", "content": "What is new?"}], None, 14, id="one_message"),
        pytest.param(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "Sunny."},
                {"role": "user", "content": "And Oslo?"},
            ],
            [_KIMI_WEATHER],
            14 + 3 * 13 + 40,
            id="turns_and_tools",
        ),
    ],
)
async def test_bedrock_kimi_k3_without_its_tokenizer_adds_the_template_offsets_to_the_local_estimate(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    template_tokens: int,
):
    _framed_kimi(monkeypatch)

    counted: Final = await token_counter(
        request=TokenCountRequest(model="kimi", messages=messages, tools=tools), call_endpoint=True
    )
    estimate: Final = litellm.token_counter(
        model=counted.model_used,
        messages=messages,
        tools=tools,
        custom_tokenizer=litellm.utils._select_tokenizer(model=counted.model_used),
    )

    assert counted.total_tokens == estimate + template_tokens + _shipped_kimi_k3_framing()


@pytest.mark.asyncio
async def test_slow_cost_map_lookup_does_not_stall_other_requests_on_the_worker(monkeypatch: pytest.MonkeyPatch):
    real_get_model_info: Final = litellm.get_model_info
    lookup_started: Final = threading.Event()
    other_request_ran: Final = threading.Event()
    released_by_the_other_request: Final[list[bool]] = []

    def slow_get_model_info(model: str, *args: object, **kwargs: object):
        if model == "slow-lookup-model":
            lookup_started.set()
            released_by_the_other_request.append(other_request_ran.wait(5))
        return real_get_model_info(model, *args, **kwargs)

    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "get_model_info", slow_get_model_info)
    counting: Final = asyncio.create_task(
        token_counter(
            request=TokenCountRequest(model="slow-lookup-model", messages=[{"role": "user", "content": "hi"}])
        )
    )
    await asyncio.to_thread(lookup_started.wait, 5)
    other_request_ran.set()
    await counting

    assert set(released_by_the_other_request) == {True}


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
@pytest.mark.usefixtures("bedrock_sonnet_5_5_mapped_as_on_main")
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


@pytest.mark.parametrize(
    ("aws_region_name", "counted_in"),
    [("eu-central-1", "eu-central-1"), ("us-west-2", "us-east-1")],
)
@pytest.mark.usefixtures("bedrock_sonnet_5_5_mapped_as_on_main")
@pytest.mark.asyncio
async def test_shipped_token_count_model_keeps_the_prompt_in_the_deployment_geography(
    monkeypatch: pytest.MonkeyPatch, aws_region_name: str, counted_in: str
):
    counter: Final = _ScriptedCounter({"anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "bedrock/anthropic.claude-sonnet-5-5",
                        "aws_region_name": aws_region_name,
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm.proxy.proxy_server, "_get_provider_token_counter", lambda deployment, model: (counter, model, "bedrock")
    )

    await token_counter(
        request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]), call_endpoint=True
    )

    assert (counter.calls[0][0], counter.regions[0]) == ("anthropic.claude-opus-5-5", counted_in)


class _SlowVault:
    def sync_read_secret(self, secret_name: str) -> None:
        time.sleep(0.5)


async def _longest_stall(done: asyncio.Event) -> float:
    if done.is_set():
        return 0.0
    started: Final = time.monotonic()
    await asyncio.sleep(0.01)
    return max(time.monotonic() - started, await _longest_stall(done))


@pytest.mark.usefixtures("bedrock_sonnet_5_5_mapped_as_on_main")
@pytest.mark.asyncio
async def test_region_lookup_for_a_shipped_count_model_leaves_the_event_loop_free(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    counter: Final = _ScriptedCounter({"anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})

    def bedrock_counter(deployment: object, model: str) -> tuple[_ScriptedCounter, str, str]:
        return counter, model, "bedrock"

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[{"model_name": "sonnet", "litellm_params": {"model": "bedrock/anthropic.claude-sonnet-5-5"}}]
        ),
    )
    monkeypatch.setattr(proxy_server, "_get_provider_token_counter", bedrock_counter)
    monkeypatch.setattr(litellm, "secret_manager_client", _SlowVault())
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.HASHICORP_VAULT)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    done: Final = asyncio.Event()

    async def count() -> TokenCountResponse:
        try:
            return await token_counter(
                request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]),
                call_endpoint=True,
            )
        finally:
            done.set()

    started: Final = time.monotonic()
    stall, response = await asyncio.gather(_longest_stall(done), count())

    assert time.monotonic() - started >= 1.0
    assert stall < 0.4
    assert response.total_tokens == 447


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
@pytest.mark.parametrize(
    ("conversation", "call_endpoint"),
    ((_CONVERSATIONS[0], True), (_CONVERSATIONS[1], True), (_CONVERSATIONS[0], False)),
    ids=("provider_messages", "provider_contents", "local_messages"),
)
@pytest.mark.parametrize(
    ("global_drop_params", "deployment_drop_params"),
    ((False, False), (False, True), (True, False), (True, True)),
    ids=("no_drop", "deployment_drop", "global_drop", "both_drop"),
)
async def test_conversation_takes_precedence_over_prompt_regardless_of_drop_params(
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
