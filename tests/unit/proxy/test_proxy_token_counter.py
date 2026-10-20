# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import asyncio
import base64
import hashlib
import itertools
import json
import logging
import threading
import traceback
from collections.abc import Callable, Coroutine, Mapping
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
from starlette.types import Message

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
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
from litellm.llms.openai.responses.count_tokens import handler as openai_count_tokens_http
from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import VertexAIPartnerModelsTokenCounter
from litellm.proxy._types import ProxyException, TokenCountRequest, UserAPIKeyAuth
from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.proxy.common_utils.token_count_settings import (
    ProviderCountFailure,
    TokenCountModel,
    TokenCountSettings,
    cost_map_entry,
    resolve_token_count_settings,
)
from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request
from litellm.proxy.proxy_server import (
    _estimate_or_provider_error,
    _provider_token_count,
    _try_provider_token_count,
    count_request_tokens,
    token_counter,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from litellm.types.utils import (
    ADDRESSED_RESPONSE_ID_FIELD,
    SelectTokenizerResponse,
    SpecialEnums,
    TokenCountMessagesFormat,
    TokenCountResponse,
)


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
async def test_token_counter_leaves_the_routers_deployment_without_the_credentials_it_resolves(tmp_path, monkeypatch):
    """token_counter reads the router's own deployment, so loading the named credential into it would leave the
    token read from the file on the shared deployment for every later request."""
    from litellm.types.utils import CredentialItem

    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token_file = tmp_path / "token"
    token_file.write_text("sa-token-from-file")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="mistral-sa",
                credential_values={"api_key": f"oidc/file/{token_file}"},
                credential_info={},
            )
        ],
    )
    llm_router = Router(
        model_list=[
            {
                "model_name": "devstral",
                "litellm_params": {"model": "mistral/devstral-latest", "litellm_credential_name": "mistral-sa"},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(model="devstral", messages=[{"role": "user", "content": "hello"}])
    )

    assert response.total_tokens > 0
    stored = (await llm_router.async_get_available_deployment(model="devstral", request_kwargs={}))["litellm_params"]
    assert "api_key" not in stored
    assert stored["litellm_credential_name"] == "mistral-sa"


@pytest.mark.asyncio
@pytest.mark.parametrize("call_endpoint", [False, True], ids=["local tokenizer", "provider count api"])
async def test_token_counter_reads_an_oidc_api_key_only_when_it_calls_the_provider(
    tmp_path, monkeypatch, call_endpoint
):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": f"oidc/file/{tmp_path / 'missing-token'}",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    request = TokenCountRequest(model="claude", messages=[{"role": "user", "content": "hello"}])

    if not call_endpoint:
        assert (await token_counter(request=request, call_endpoint=call_endpoint)).total_tokens > 0
        return
    with pytest.raises(ProxyException) as excinfo:
        await token_counter(request=request, call_endpoint=call_endpoint)
    assert excinfo.value.code == "401"
    assert str(tmp_path) not in excinfo.value.message


@pytest.mark.asyncio
async def test_native_openai_responses_count_sends_the_token_an_oidc_api_key_points_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token_file: Final = tmp_path / "token"
    token_file.write_text("sa-token-from-file")
    monkeypatch.setattr(
        litellm.proxy.proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "gpt-oidc",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": f"oidc/file/{token_file}",
                        "api_base": "https://count.test/v1",
                    },
                }
            ]
        ),
    )
    authorizations: Final[list[str | None]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 9}, request=request)

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="gpt-oidc", messages=[{"role": "user", "content": "hello"}]),
            call_endpoint=True,
            responses_request={"model": "gpt-oidc", "input": "hello"},
        )

    assert (result.error, result.total_tokens) == (False, 9)
    assert authorizations == ["Bearer sa-token-from-file"]


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


def _json_request(body: Mapping[str, object], user_agent: str) -> Request:
    payload: Final = json.dumps(body).encode()

    async def receive() -> Message:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages/count_tokens",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"user-agent", user_agent.encode())],
        },
        receive,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_drop_params", "user_agent"),
    (({"drop_params": True}, "python-httpx/0.28.1"), ({}, "claude-cli/2.1.0 (external, cli)")),
)
async def test_anthropic_count_tokens_drops_a_malformed_tool_result_text_when_drop_params_applies(
    monkeypatch: pytest.MonkeyPatch, request_drop_params: dict[str, bool], user_agent: str
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    body: Final = {
        "model": "gpt-tool-result",
        **request_drop_params,
        "messages": [
            {"role": "user", "content": "run the build"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_log", "name": "bash", "input": {"cmd": "make"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_log",
                        "content": [{"type": "text", "text": "build ok"}, {"type": "text", "text": None}],
                    }
                ],
            },
        ],
    }
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-tool-result", "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-test"}}
        ]
    )
    request: Final = _json_request(body, user_agent)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        route: Final = respx.post("https://api.openai.com/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 9})
        )
        response: Final = await anthropic_count_tokens(request, UserAPIKeyAuth())

    assert response["input_tokens"] == 9
    sent: Final = json.loads(route.calls.last.request.content)
    assert {"type": "function_call_output", "call_id": "toolu_log", "output": "build ok"} in sent["input"]


def _build_log_messages(log_text: JsonValue) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": "run the build"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_log", "name": "bash", "input": {"cmd": "make"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_log",
                    "content": [{"type": "text", "text": "build ok"}, {"type": "text", "text": log_text}],
                }
            ],
        },
    ]


async def _count_on_a_model_without_a_count_api(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object], router_defaults: dict[str, JsonValue] | None = None
) -> JSONResponse | dict[str, JsonValue]:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "mistral-tool-result",
                "litellm_params": {"model": "mistral/mistral-small-latest", "api_key": "sk-test"},
            }
        ],
        default_litellm_params=router_defaults,
    )
    request: Final = _json_request(body, "python-httpx/0.28.1")
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    with respx.mock:
        return await anthropic_count_tokens(request, UserAPIKeyAuth())


@pytest.mark.asyncio
@pytest.mark.parametrize("log_text", (None, 5))
async def test_count_tokens_without_a_count_api_rejects_a_non_string_tool_result_text(
    monkeypatch: pytest.MonkeyPatch, log_text: JsonValue
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch, {"model": "mistral-tool-result", "messages": _build_log_messages(log_text)}
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(bytes(response.body))["error"] == {
        "type": "invalid_request_error",
        "message": "tool_result toolu_log content.1.text: Input should be a valid string",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("mistral-tool-result", "model-not-in-the-router"))
async def test_count_tokens_without_a_count_api_counts_a_non_string_tool_result_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch, {"model": model, "drop_params": True, "messages": _build_log_messages(None)}
    )

    assert isinstance(response, dict)
    assert isinstance(response["input_tokens"], int) and response["input_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("body_drop_params", "expected_status"), (({}, None), ({"drop_params": False}, 400)))
async def test_count_tokens_honors_router_default_drop_params_unless_the_body_overrides_it(
    monkeypatch: pytest.MonkeyPatch, body_drop_params: dict[str, JsonValue], expected_status: int | None
) -> None:
    response: Final = await _count_on_a_model_without_a_count_api(
        monkeypatch,
        {"model": "mistral-tool-result", "messages": _build_log_messages(None), **body_drop_params},
        router_defaults={"drop_params": True},
    )

    match response:
        case JSONResponse():
            assert response.status_code == expected_status
        case _:
            assert expected_status is None
            assert isinstance(response["input_tokens"], int) and response["input_tokens"] > 0


@pytest.mark.asyncio
async def test_anthropic_count_tokens_on_a_claude_deployment_drops_a_malformed_tool_result_text_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-native-drop",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5-5",
                    "api_key": "sk-ant-test",
                    "drop_params": True,
                },
            }
        ]
    )
    request: Final = _json_request(
        {"model": "claude-native-drop", "messages": _build_log_messages(5)}, "python-httpx/0.28.1"
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        route: Final = respx.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        response: Final = await anthropic_count_tokens(request, UserAPIKeyAuth())

    assert response["input_tokens"] == 7
    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["messages"][2]["content"][0]["content"] == [{"type": "text", "text": "build ok"}]


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
        chat_params: object = None,
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
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", total_tokens=447)})

    result: Final = await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

    assert result is not None
    assert (result.total_tokens, result.count_source) == (447, "provider")
    assert ([call[0] for call in counter.calls], counter.regions) == (["us.anthropic.claude-sonnet-5-5"], ["us-west-2"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_host", "expected_tokens"),
    (
        ("bedrock", "bedrock-runtime.us-east-1.amazonaws.com", 193),
        ("bedrock_mantle", "bedrock-mantle.us-east-1.api.aws", 271),
    ),
)
async def test_token_count_model_provider_selects_the_endpoint_and_preserves_the_public_model(
    provider: str, expected_host: str, expected_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    model: Final = "anthropic.claude-opus-5-5"

    def peer(request: httpx.Request) -> httpx.Response:
        assert request.url.host == expected_host
        if provider == "bedrock_mantle":
            assert json.loads(request.content)["model"] == model
            return httpx.Response(200, json={"input_tokens": expected_tokens})
        assert request.url.path == f"/model/{model}/count-tokens"
        return httpx.Response(200, json={"inputTokens": expected_tokens})

    settings: Final = resolve_token_count_settings(
        configured={},
        default={"token_count_model": {"model": model, "litellm_provider": provider, "region": "us-east-1"}},
        deployment_region=lambda: "us-west-2",
    )
    counter: Final = BedrockTokenCounter(client=AsyncHTTPHandler(transport=httpx.MockTransport(peer)))
    result: Final = await _provider_token_count(
        count_with=partial(
            _try_provider_token_count,
            counter,
            "bedrock",
            messages=_COUNT_REQUEST.messages,
            contents=_COUNT_REQUEST.contents,
            request_model=_COUNT_REQUEST.model,
            deployment={
                "litellm_params": {
                    "aws_access_key_id": "AKIATESTACCESSKEY",
                    "aws_secret_access_key": "test-secret",
                    "aws_region_name": "us-west-2",
                }
            },
            tools=_COUNT_REQUEST.tools,
            system=_COUNT_REQUEST.system,
        ),
        model_to_use="us.anthropic.claude-sonnet-5-5",
        settings=settings,
        local_estimate_allowed=False,
    )

    assert isinstance(result, TokenCountResponse)
    assert (result.total_tokens, result.model_used, result.count_source) == (
        expected_tokens,
        model,
        "token_count_model",
    )


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


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@pytest.mark.asyncio
async def test_errors_other_than_cannot_count_raise_even_with_framing(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=10, count_model=None))

    assert (raised.value.code, raised.value.message) == (str(status_code), f"scripted {status_code} for sonnet")


@pytest.mark.parametrize("status_code", [400, 403, 404, 500])
@pytest.mark.asyncio
async def test_token_count_model_failure_names_its_region_so_the_user_can_enable_it(status_code: int):
    counter: Final = _ScriptedCounter(
        {"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=status_code)}
    )

    with pytest.raises(ProxyException) as raised:
        await _ladder(counter, TokenCountSettings(framing=None, count_model=_OPUS_IN_US_EAST_1))

    assert raised.value.code == str(status_code)
    assert raised.value.message.startswith(f"scripted {status_code} for opus")
    assert "us.anthropic.claude-opus-5-5 in us-east-1" in raised.value.message
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_token_count_model_without_a_region_answers_its_error_unchanged():
    counter: Final = _ScriptedCounter({"us.anthropic.claude-opus-5-5": _provider_reply("opus", status_code=404)})

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
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=404)})

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
    counter: Final = _ScriptedCounter({"us.anthropic.claude-sonnet-5-5": _provider_reply("sonnet", status_code=400)})

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
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "What is new?"},
                    {"role": "assistant", "content": "Weighing the release notes"},
                ],
                "contents": [
                    {"role": "user", "parts": [{"text": "What is new?"}]},
                    {"role": "model", "parts": [{"text": "Weighing the release notes", "thought": True}]},
                ],
            },
            id="google_contents_with_a_thought_the_translation_counts_as_text",
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

    monkeypatch.setattr(litellm.utils, "select_tokenizer", hanging_download)
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

    monkeypatch.setattr(litellm.utils, "select_tokenizer", slow_failing_download)
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
    real_select_tokenizer: Final = litellm.utils.select_tokenizer
    hub_released: Final = threading.Event()
    downloads: Final[list[object]] = []

    def held_download(model: str, custom_tokenizer: object = None) -> SelectTokenizerResponse:
        downloads.append(custom_tokenizer)
        hub_released.wait(10)
        return real_select_tokenizer(model="gpt-5.4-mini")

    monkeypatch.setattr(litellm.utils, "select_tokenizer", held_download)
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
    custom: Final = litellm.utils.select_tokenizer(model="gpt-5.4-mini")
    monkeypatch.setattr(litellm.utils, "select_tokenizer", lambda model, custom_tokenizer=None: custom)
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
        custom_tokenizer=litellm.utils.select_tokenizer(model=counted.model_used),
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
    assert await asyncio.to_thread(lookup_started.wait, 5)
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
        ({}, ("bedrock_mantle/us.anthropic.claude-opus-5-5", "us-east-1", 447)),
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
            "bedrock_mantle/us.anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447),
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
@pytest.mark.asyncio
async def test_shipped_token_count_model_keeps_the_prompt_in_the_deployment_geography(
    monkeypatch: pytest.MonkeyPatch, aws_region_name: str, counted_in: str
):
    counter: Final = _ScriptedCounter({"bedrock_mantle/anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})
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

    assert (counter.calls[0][0], counter.regions[0]) == ("bedrock_mantle/anthropic.claude-opus-5-5", counted_in)


_LOOP_PROGRESS_TIMEOUT_SECONDS: Final = 5


class _VaultWaitingForLoopProgress:
    def __init__(
        self,
        lookup_started: threading.Event,
        loop_progressed: threading.Event,
        loop_progressed_during_lookup: threading.Event,
    ) -> None:
        self._lookup_started = lookup_started
        self._loop_progressed = loop_progressed
        self._loop_progressed_during_lookup = loop_progressed_during_lookup

    def sync_read_secret(self, secret_name: str) -> None:
        self._lookup_started.set()
        if self._loop_progressed.wait(_LOOP_PROGRESS_TIMEOUT_SECONDS):
            self._loop_progressed_during_lookup.set()


@pytest.mark.asyncio
async def test_region_lookup_for_a_shipped_count_model_leaves_the_event_loop_free(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    counter: Final = _ScriptedCounter({"bedrock_mantle/anthropic.claude-opus-5-5": _provider_reply("opus", total_tokens=447)})

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
    lookup_started: Final = threading.Event()
    loop_progressed: Final = threading.Event()
    loop_progressed_during_lookup: Final = threading.Event()
    monkeypatch.setattr(
        litellm,
        "secret_manager_client",
        _VaultWaitingForLoopProgress(lookup_started, loop_progressed, loop_progressed_during_lookup),
    )
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.HASHICORP_VAULT)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))

    async def count() -> TokenCountResponse:
        return await token_counter(
            request=TokenCountRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}]),
            call_endpoint=True,
        )

    async def prove_loop_progress() -> None:
        assert await asyncio.to_thread(lookup_started.wait, _LOOP_PROGRESS_TIMEOUT_SECONDS)
        loop_progressed.set()

    response, _ = await asyncio.gather(count(), prove_loop_progress())

    assert loop_progressed_during_lookup.is_set()
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
        ({}, {}, {"drop_params": "true"}, False),
        ({}, {}, {}, True),
    ),
    ids=(
        "deployment-drop-params",
        "router-default-drop-params",
        "request-drop-params",
        "request-drop-params-string",
        "litellm-drop-params",
    ),
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


_PINNED_COUNTS: Final = MappingProxyType({"deployment-a.test": 11, "deployment-b.test": 22})


def _pinned_count_peer(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("previous_response_id") != "resp_original":
        return httpx.Response(400, json={"error": {"message": "previous_response_id was not decoded"}}, request=request)
    return httpx.Response(200, json={"input_tokens": _PINNED_COUNTS[request.url.host]}, request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", ["deployment-a", "deployment-b"])
async def test_responses_count_goes_to_the_deployment_that_served_the_previous_response(
    monkeypatch: pytest.MonkeyPatch, pinned: str
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "api_base": f"https://{identity}.test/v1",
                },
                "model_info": {"id": identity},
            }
            for identity in ("deployment-a", "deployment-b")
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_pinned_count_peer))
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        "openai", pinned, "resp_original"
    )

    with patch.object(openai_count_tokens_http, "get_async_httpx_client", return_value=client):
        counts: Final = [
            (
                await count_request_tokens(
                    request=TokenCountRequest(model="gpt-group", messages=[{"role": "user", "content": "again"}]),
                    call_endpoint=True,
                    responses_request={
                        "model": "gpt-group",
                        "input": "again",
                        "previous_response_id": previous_response_id,
                    },
                )
            ).total_tokens
            for _ in range(8)
        ]

    assert counts == [_PINNED_COUNTS[f"{pinned}.test"]] * 8


def _managed_file_id(provider_file_id: str, model_id: str) -> str:
    unified: Final = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/pdf", "3f0c9a", "gpt-files", provider_file_id, model_id
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


@pytest.mark.asyncio
@pytest.mark.parametrize(("with_mapping", "expected_file_id"), [(False, "file-uploaded"), (True, "file-on-a")])
async def test_responses_count_sends_openai_the_provider_file_id_inference_sends(
    monkeypatch: pytest.MonkeyPatch, with_mapping: bool, expected_file_id: str
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-files",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key", "api_base": "https://count.test/v1"},
                "model_info": {"id": "deployment-a"},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    managed_id: Final = _managed_file_id("file-uploaded", "deployment-a")
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 49}, request=request)

    with patch.object(
        openai_count_tokens_http,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="gpt-files", messages=[{"role": "user", "content": "Summarize"}]),
            call_endpoint=True,
            responses_request={
                "model": "gpt-files",
                "input": [{"role": "user", "content": [{"type": "input_file", "file_id": managed_id}]}],
                **({"model_file_id_mapping": {managed_id: {"deployment-a": "file-on-a"}}} if with_mapping else {}),
            },
        )

    assert (result.error, result.total_tokens) == (False, 49)
    (body,) = bodies
    assert body["input"] == [{"role": "user", "content": [{"type": "input_file", "file_id": expected_file_id}]}]
    assert "model_file_id_mapping" not in body


_ANSWER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_responses_count_on_claude_counts_the_chat_request_responses_inference_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-counted",
                "litellm_params": {
                    "model": "anthropic/claude-opus-5-5",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"input_tokens": 77}, request=request)

    responses_tools: Final[list[dict[str, JsonValue]]] = [
        {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        {"type": "web_search"},
        {"type": "file_search", "vector_store_ids": ["vs_docs"]},
    ]
    with patch.object(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(
                model="claude-counted", messages=[{"role": "user", "content": "Weather?"}], tools=responses_tools
            ),
            call_endpoint=True,
            responses_request={
                "model": "claude-counted",
                "input": "Weather?",
                "tools": responses_tools,
                "tool_choice": "auto",
                "text": {"format": {"type": "json_schema", "name": "answer", "schema": _ANSWER_SCHEMA}},
                "reasoning": {"effort": "low"},
            },
        )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    tools: Final = body["tools"]
    assert isinstance(tools, list)
    tool_names: Final = [tool["name"] for tool in tools if isinstance(tool, dict)]
    assert sorted(tool_names) == ["get_weather", "litellm_file_search", "web_search"]
    assert any(isinstance(tool, dict) and str(tool.get("type")).startswith("web_search") for tool in tools)
    assert body["tool_choice"] == {"type": "auto"}
    output_config: Final = body["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"] == {"type": "json_schema", "schema": _ANSWER_SCHEMA}
    assert "thinking" in body


_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {}},
}


async def _claude_count(
    monkeypatch: pytest.MonkeyPatch,
    deployment_params: Mapping[str, JsonValue],
    responses_request: Mapping[str, object],
) -> tuple[TokenCountResponse, list[dict[str, JsonValue]]]:
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-counted",
                "litellm_params": {
                    "model": "anthropic/claude-zeta-9",
                    "api_key": "test-key",
                    "api_base": "https://count-tokens.test",
                    **deployment_params,
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", router)
    bodies: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: collects what the mock peer receives

    def count_peer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"input_tokens": 77}, request=request)

    with patch.object(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        return_value=AsyncHTTPHandler(transport=httpx.MockTransport(count_peer)),
    ):
        result: Final = await count_request_tokens(
            request=TokenCountRequest(model="claude-counted", messages=[{"role": "user", "content": "Weather?"}]),
            call_endpoint=True,
            responses_request={"model": "claude-counted", **responses_request},
        )
    return result, bodies


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "request_params"),
    [({"drop_params": True}, {}), ({}, {"drop_params": True})],
    ids=["deployment", "request"],
)
async def test_responses_count_on_claude_downgrades_forced_tool_choice_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
    deployment_params: Mapping[str, JsonValue],
    request_params: Mapping[str, JsonValue],
) -> None:
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_forced_tool_use": False})

    result, bodies = await _claude_count(
        monkeypatch,
        deployment_params,
        {"input": "Weather?", "tools": [_WEATHER_TOOL], "tool_choice": "required", **request_params},
    )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    assert body["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_responses_count_on_claude_rejects_forced_tool_choice_without_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_forced_tool_use": False})

    with pytest.raises(ProxyException) as rejected:
        await _claude_count(
            monkeypatch, {}, {"input": "Weather?", "tools": [_WEATHER_TOOL], "tool_choice": "required"}
        )

    assert (rejected.value.code, "forced tool use" in rejected.value.message) == ("400", True)


def _texts(message: JsonValue) -> list[str]:
    assert isinstance(message, dict)
    content: Final = message["content"]
    if isinstance(content, str):
        return [content]
    assert isinstance(content, list)
    return [str(block["text"]) for block in content if isinstance(block, dict) and "text" in block]


@pytest.mark.asyncio
async def test_responses_count_on_claude_counts_the_turns_previous_response_id_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_turn: Final = {
        "request_id": "chatcmpl-stored",
        "call_type": "aresponses",
        "session_id": "session-1",
        "proxy_server_request": {"input": "Who won in 1998?", "model": "claude-counted"},
        "response": {
            "id": "chatcmpl-stored",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The Bulls.",
                        "reasoning_content": "Chicago beat Utah in six games.",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        "status": "success",
    }

    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
        return_value=[stored_turn],
    ) as spend_logs:
        result, bodies = await _claude_count(
            monkeypatch,
            {},
            {
                "input": "And in 1999?",
                "previous_response_id": "chatcmpl-stored",
                ADDRESSED_RESPONSE_ID_FIELD: "resp_client_form",
            },
        )

    assert (result.error, result.total_tokens) == (False, 77)
    (body,) = bodies
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    assert list(itertools.chain.from_iterable(map(_texts, messages))) == [
        "Who won in 1998?",
        "The Bulls.",
        "And in 1999?",
    ]
    assert [sorted(message) for message in messages if isinstance(message, dict)] == [["content", "role"]] * 3
    spend_logs.assert_awaited_once_with("chatcmpl-stored", ("chatcmpl-stored", "resp_client_form"))
