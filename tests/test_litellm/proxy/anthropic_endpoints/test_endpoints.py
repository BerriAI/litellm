"""
Test for anthropic_endpoints/endpoints.py, focusing on handling dictionary objects in streaming responses
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


class TestAnthropicEndpoints(unittest.TestCase):
    @patch("litellm.litellm_core_utils.safe_json_dumps.safe_dumps")
    @pytest.mark.asyncio
    async def test_async_data_generator_anthropic_dict_handling(self, mock_safe_dumps):
        """Test async_data_generator_anthropic handles dictionary chunks properly"""
        # Setup
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"type": "message_start", "message": {"id": "msg_123"}},
            "text chunk data",
            {"type": "content_block_delta", "delta": {"text": "more data"}},
            "text chunk data again",
        ]

        mock_user_api_key_dict = MagicMock()
        mock_request_data = {}
        mock_proxy_logging_obj = MagicMock()
        mock_proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
            side_effect=lambda **kwargs: kwargs["response"]
        )

        # Configure safe_dumps to return a properly formatted JSON string
        mock_safe_dumps.side_effect = lambda chunk: json.dumps(chunk)

        # Execute
        result = [
            chunk
            async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
                request_data=mock_request_data,
                proxy_logging_obj=mock_proxy_logging_obj,
            )
        ]

        # Verify
        expected_result = [
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n\n',
            "text chunk data",
            'data: {"type": "content_block_delta", "delta": {"text": "more data"}}\n\n',
            "text chunk data again",
        ]

        self.assertEqual(result, expected_result)

        # Assert safe_dumps was called for dictionary objects
        mock_safe_dumps.assert_any_call({"type": "message_start", "message": {"id": "msg_123"}})
        mock_safe_dumps.assert_any_call({"type": "content_block_delta", "delta": {"text": "more data"}})
        assert mock_safe_dumps.call_count == 2  # Called twice, once for each dict object


class TestBlockedResponseUsage:
    """Blocked responses report the blocked LLM response's real usage."""

    def test_uses_original_response_usage(self):
        from litellm.proxy.anthropic_endpoints.endpoints import _blocked_response_usage

        # original_response is the AnthropicMessagesResponse the LLM produced
        # before the guardrail blocked it; its usage is real.
        original = {"usage": {"input_tokens": 31, "output_tokens": 9}}
        assert _blocked_response_usage(original) == {
            "input_tokens": 31,
            "output_tokens": 9,
        }

    def test_zero_usage_when_no_original_response(self):
        from litellm.proxy.anthropic_endpoints.endpoints import _blocked_response_usage

        # Pre-call blocks never invoked the LLM -> nothing consumed.
        assert _blocked_response_usage(None) == {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_blocked_endpoint_response_carries_original_usage(self):
        """The /v1/messages block handler reports the blocked response's real
        usage, carried on ModifyResponseException.original_response."""
        from unittest.mock import AsyncMock, MagicMock

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.integrations.custom_guardrail import ModifyResponseException

        exc = ModifyResponseException(
            message="blocked by guardrail",
            model="claude-3-5-sonnet-20240620",
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            guardrail_name="rubrik",
            original_response={"usage": {"input_tokens": 12, "output_tokens": 5}},
        )

        with (
            patch.object(ep, "_read_request_body", new=AsyncMock(return_value={})),
            patch.object(
                ep.ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=MagicMock(),
                user_api_key_dict=MagicMock(),
            )

        assert response["content"][0]["text"] == "blocked by guardrail"
        assert response["usage"] == {"input_tokens": 12, "output_tokens": 5}
        mock_logging.post_call_failure_hook.assert_awaited_once()


class TestEventLoggingBatchEndpoint:
    """Test the stubbed event logging batch endpoint"""

    def test_event_logging_batch_endpoint_exists(self):
        """Test that the event_logging_batch endpoint exists and returns 200"""
        from fastapi import FastAPI

        from litellm.proxy.anthropic_endpoints.endpoints import router

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.post("/api/event_logging/batch", json={"events": []})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStripTotalTokens(unittest.TestCase):
    """Cover ``_strip_total_tokens_from_anthropic_response``.

    The Anthropic /v1/messages spec does not define ``usage.total_tokens``.
    LiteLLM injects it internally; the helper must remove it from the wire
    response so the non-streaming path matches the streaming SSE shape and
    direct Anthropic API responses.
    """

    def test_strips_total_tokens_when_present(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {
            "id": "msg_123",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        _strip_total_tokens_from_anthropic_response(response)
        assert "total_tokens" not in response["usage"]
        assert response["usage"]["input_tokens"] == 100
        assert response["usage"]["output_tokens"] == 50
        assert response["usage"]["cache_read_input_tokens"] == 0

    def test_no_op_when_total_tokens_absent(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        _strip_total_tokens_from_anthropic_response(response)
        assert response["usage"] == {"input_tokens": 100, "output_tokens": 50}

    def test_no_op_when_usage_missing(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        response = {"id": "msg_123"}
        _strip_total_tokens_from_anthropic_response(response)
        assert response == {"id": "msg_123"}

    def test_no_op_on_non_dict_response(self):
        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        # Streaming responses (StreamingResponse, async iterators) are not dicts.
        # The helper must not raise or attempt to mutate them.
        for value in (None, "stream", 42, [{"usage": {"total_tokens": 1}}]):
            _strip_total_tokens_from_anthropic_response(value)  # no raise

    def test_strips_total_tokens_on_pydantic_model_with_dict_usage(self):
        """Greptile P1 on #30382: helper must not silently no-op when the
        response is a Pydantic-shaped object whose `usage` attribute is a
        plain dict (the common case for objects wrapping raw upstream JSON).
        """
        from types import SimpleNamespace

        from litellm.proxy.anthropic_endpoints.endpoints import (
            _strip_total_tokens_from_anthropic_response,
        )

        # SimpleNamespace mimics the .usage attribute access pattern; the
        # helper's contract: if .usage is dict-shaped, strip total_tokens.
        response = SimpleNamespace(usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        _strip_total_tokens_from_anthropic_response(response)
        assert "total_tokens" not in response.usage
        assert response.usage == {"input_tokens": 100, "output_tokens": 50}


class TestStripTotalTokensFeatureFlag(unittest.TestCase):
    """The strip is gated behind `litellm.strip_anthropic_total_tokens`.

    Default off (backward compat). Greptile P1 on #30382 required a
    user-controlled flag so existing clients reading the LiteLLM-shaped
    `usage.total_tokens` continue to work after this PR lands.
    """

    def test_flag_defaults_off(self):
        import litellm

        assert litellm.strip_anthropic_total_tokens is False


class TestCountTokensErrorHandling:
    """`/v1/messages/count_tokens` must not leak internal errors on bad input."""

    @pytest.mark.asyncio
    async def test_malformed_messages_returns_400_without_reflection(self):
        """A malformed messages payload should map to 400, not a reflected 500.

        `messages=[123]` fails TokenCountRequest validation; previously the
        terminal `except Exception` echoed the raw internal error at status 500.
        It must now be a 400 whose body carries no internal exception text.
        """
        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [123],
            }

        with patch.object(ep, "_read_request_body", new=mock_read_request_body):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail)
        assert "validation error" not in detail.lower()
        assert "TokenCountRequest" not in detail
        assert "pydantic" not in detail.lower()

    @pytest.mark.asyncio
    async def test_malformed_content_block_returns_400_without_reflection(self):
        """A content block that breaks the tokenizer must surface as 400.

        The internal proxy token_counter now raises HTTPException(400) for such
        input; the wrapper's `except HTTPException: raise` passes it through so
        the client never sees the raw ValueError text.
        """
        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        setattr(proxy_server, "llm_router", None)

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": [123]}],
            }

        with patch.object(ep, "_read_request_body", new=mock_read_request_body):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail)
        assert "subscriptable" not in detail
        assert "Traceback" not in detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tokenizer_error",
        [
            AttributeError("'NoneType' object has no attribute 'get'"),
            KeyError("type"),
            IndexError("list index out of range"),
        ],
    )
    async def test_tokenizer_attribute_error_returns_400_not_500(self, tokenizer_error):
        """Non-(ValueError/TypeError) tokenizer failures must also map to 400.

        AttributeError/KeyError/IndexError raised deep inside the tokenizer
        used to fall through to the terminal `except Exception` and surface as
        a 500. They must now be a 400 whose body carries no internal exception
        text.
        """
        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
            }

        async def mock_token_counter(request, call_endpoint=False):
            raise tokenizer_error

        with (
            patch.object(ep, "_read_request_body", new=mock_read_request_body),
            patch.object(proxy_server, "token_counter", new=mock_token_counter),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail)
        assert "NoneType" not in detail
        assert "attribute" not in detail
        assert "Internal server error" not in detail

    @pytest.mark.asyncio
    async def test_wellformed_request_still_counts(self):
        """A valid request must still return the Anthropic-shaped token count."""
        from fastapi import Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.types.utils import TokenCountResponse

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hello Claude!"}],
            }

        async def mock_token_counter(request, call_endpoint=False):
            return TokenCountResponse(
                total_tokens=15,
                request_model="claude-3-sonnet-20240229",
                model_used="claude-3-sonnet-20240229",
                tokenizer_type="openai_tokenizer",
            )

        with (
            patch.object(ep, "_read_request_body", new=mock_read_request_body),
            patch.object(proxy_server, "token_counter", new=mock_token_counter),
        ):
            response = await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert response == {"input_tokens": 15}

    @pytest.mark.asyncio
    async def test_oversized_payload_returns_400(self):
        """A payload exceeding the token-counting size cap must surface as 400.

        The internal token_counter raises HTTPException(400) before offloading
        the count; the wrapper's `except HTTPException: raise` passes it through.
        """
        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        setattr(proxy_server, "llm_router", None)

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "a" * 200}],
            }

        with (
            patch.object(ep, "_read_request_body", new=mock_read_request_body),
            patch.object(proxy_server, "TOKEN_COUNTER_MAX_REQUEST_CHARS", 100),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == {"error": "Request payload too large for token counting"}

    @pytest.mark.asyncio
    async def test_saturated_concurrency_returns_429(self):
        """A saturated tokenization concurrency bound must surface as 429.

        The internal token_counter raises ProxyRateLimitError (an HTTPException
        subclass) when the bound cannot be acquired; the wrapper's
        `except HTTPException: raise` passes the 429 through untouched.
        """
        import asyncio

        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        setattr(proxy_server, "llm_router", None)

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
            }

        with (
            patch.object(ep, "_read_request_body", new=mock_read_request_body),
            patch.object(proxy_server, "_token_count_semaphore", asyncio.Semaphore(0)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == {"error": "Too many concurrent token counting requests. Please retry later."}

    @pytest.mark.asyncio
    async def test_unexpected_internal_error_returns_generic_500(self):
        """An unexpected internal failure must map to a generic 500.

        The terminal `except Exception` handler must not reflect the internal
        exception text back to the client.
        """
        from fastapi import HTTPException, Request

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        mock_request = MagicMock(spec=Request)
        mock_user_api_key_dict = MagicMock()

        async def mock_read_request_body(request):
            return {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
            }

        async def mock_token_counter(request, call_endpoint=False):
            raise RuntimeError("secret internal failure detail")

        with (
            patch.object(ep, "_read_request_body", new=mock_read_request_body),
            patch.object(proxy_server, "token_counter", new=mock_token_counter),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ep.count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == {"error": "Internal server error"}
        assert "secret internal failure detail" not in str(exc_info.value.detail)
