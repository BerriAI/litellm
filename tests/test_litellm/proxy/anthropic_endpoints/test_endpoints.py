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
        mock_safe_dumps.assert_any_call(
            {"type": "message_start", "message": {"id": "msg_123"}}
        )
        mock_safe_dumps.assert_any_call(
            {"type": "content_block_delta", "delta": {"text": "more data"}}
        )
        assert (
            mock_safe_dumps.call_count == 2
        )  # Called twice, once for each dict object


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
        response = SimpleNamespace(
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        )
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
