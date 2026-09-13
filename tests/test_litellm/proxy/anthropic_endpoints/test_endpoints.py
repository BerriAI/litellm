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


class TestProxyExceptionAnthropicEnvelope:
    @pytest.mark.asyncio
    async def test_anthropic_response_maps_proxy_exception_to_anthropic_envelope(self):
        """LIT-6468: a 400 ProxyException from request validation must surface as
        Anthropic's documented {"type": "error", "error": {...}} envelope with the
        original status and message, not the OpenAI {"error": {...}} envelope."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        exc = ProxyException(
            message="Invalid type for 'metadata': expected an object, but got a string instead.",
            type=ProxyErrorTypes.bad_request_error,
            param="metadata",
            code=400,
        )
        request = MagicMock()
        request.headers = {"x-request-id": "req_test_6468"}

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
                request=request,
                user_api_key_dict=MagicMock(),
            )

        assert response.status_code == 400
        body = json.loads(response.body)
        assert body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid type for 'metadata': expected an object, but got a string instead.",
            },
            "request_id": "req_test_6468",
        }
        mock_logging.post_call_failure_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anthropic_response_maps_429_to_rate_limit_error(self):
        """The Anthropic error type follows the status code (429 -> rate_limit_error),
        and a code-less exception falls back to 500 api_error."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        from litellm.proxy._types import ProxyException

        request = MagicMock()
        request.headers = {}

        response = ep._anthropic_error_json_response(
            ProxyException(message="Rate limit exceeded", type="rate_limit_error", param=None, code=429),
            request,
        )
        assert response.status_code == 429
        assert json.loads(response.body)["error"]["type"] == "rate_limit_error"

        fallback = ep._anthropic_error_json_response(
            ProxyException(message="boom", type="None", param=None, code=None),
            request,
        )
        assert fallback.status_code == 500
        assert json.loads(fallback.body)["error"]["type"] == "api_error"


class TestHttpExceptionDictDetail:
    @pytest.mark.asyncio
    async def test_anthropic_response_serializes_dict_detail_http_exception(self):
        """LIT-6466 + LIT-6468: a post_call guardrail's HTTPException(detail=<dict>)
        must surface as Anthropic's {"type": "error", "error": {...}} envelope with
        the guardrail's clean message plus provider_specific_fields, not the str()
        of the exception and not the OpenAI envelope."""
        from fastapi import HTTPException

        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        detail = {
            "error": "Content blocked: keyword 'kumquat' detected",
            "keyword": "kumquat",
            "guardrail": "keyword-block",
        }
        exc = HTTPException(status_code=400, detail=detail)
        request = MagicMock()
        request.headers = {}

        with (
            patch.object(ep, "_read_request_body", new=AsyncMock(return_value={})),  # test-quality-ok: endpoint reads the body via a module function; no injection seam
            patch.object(  # test-quality-ok: the guardrail raise happens deep inside this call; the test targets the endpoint's except block
                ep.ProxyBaseLLMRequestProcessing,
                "base_process_llm_request",
                new=AsyncMock(side_effect=exc),
            ),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,  # test-quality-ok: module global imported at call time; no injection seam
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 400
        body = json.loads(response.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "Content blocked: keyword 'kumquat' detected"
        assert "{'error'" not in body["error"]["message"]
        assert body["error"]["provider_specific_fields"] == detail
        mock_logging.post_call_failure_hook.assert_awaited_once()


class TestFailureHookRequestData:
    @pytest.mark.asyncio
    async def test_failure_hook_gets_post_setup_data_with_logging_obj(self):
        """Request setup replaces the processor's data dict (adding the logging
        object the failure hook needs to lift token usage from); the exception
        handler must pass that replaced dict, not the raw request body dict."""
        import litellm.proxy.anthropic_endpoints.endpoints as ep
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy._types import UserAPIKeyAuth

        captured = {}

        async def fake_process(self, **kwargs):
            self.data = {**self.data, "litellm_logging_obj": "logging-obj-sentinel"}
            captured["processor_data"] = self.data
            raise RuntimeError("provider timeout")

        request = MagicMock()
        request.headers = {}

        with (
            patch.object(ep, "_read_request_body", new=AsyncMock(return_value={"model": "claude-sonnet"})),
            patch.object(ep.ProxyBaseLLMRequestProcessing, "base_process_llm_request", new=fake_process),
            patch.object(proxy_server, "proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()
            response = await ep.anthropic_response(
                fastapi_response=MagicMock(),
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert response.status_code == 500
        assert json.loads(response.body)["error"]["message"] == "provider timeout"

        hook_request_data = mock_logging.post_call_failure_hook.await_args.kwargs["request_data"]
        assert hook_request_data is captured["processor_data"]
        assert hook_request_data["litellm_logging_obj"] == "logging-obj-sentinel"


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
