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


class TestAutoPromptCaching:
    """
    Tests for automatic cache_control injection at the /v1/messages endpoint level.
    Detects Claude Code via User-Agent and injects cache_control into the last
    content block of the last message.

    Related issue: https://github.com/BerriAI/litellm/issues/20418
    """

    def _make_mock_request(self, user_agent: str = "") -> MagicMock:
        """Create a mock FastAPI Request with the given User-Agent."""
        mock_request = MagicMock()
        mock_request.headers = {"user-agent": user_agent}
        return mock_request

    def test_claude_code_client_detection(self):
        """Test that Claude Code is detected from User-Agent header."""
        from litellm.proxy.anthropic_endpoints.endpoints import _is_claude_code_client

        mock_request = self._make_mock_request("claude-code/1.0.0")
        assert _is_claude_code_client(mock_request) is True

        mock_request = self._make_mock_request("Claude-Code/2.0")
        assert _is_claude_code_client(mock_request) is True

        mock_request = self._make_mock_request("Mozilla/5.0")
        assert _is_claude_code_client(mock_request) is False

        mock_request = self._make_mock_request("")
        assert _is_claude_code_client(mock_request) is False

    def test_cache_control_injected_for_claude_code_list_content(self):
        """Test that cache_control is injected into last content block for Claude Code."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First block"},
                        {"type": "text", "text": "Second block"},
                    ],
                },
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )
        last_block = result["messages"][-1]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}
        # First block should NOT have cache_control
        first_block = result["messages"][-1]["content"][0]
        assert "cache_control" not in first_block

    def test_cache_control_injected_for_string_content(self):
        """Test that string content is converted to list format with cache_control."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {
            "messages": [
                {"role": "user", "content": "Hello, analyze this."},
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )
        last_msg = result["messages"][-1]
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 1
        block = last_msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "Hello, analyze this."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_not_double_injected(self):
        """Test that cache_control is NOT injected when already present in messages."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "block with cache",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "no cache here"},
                    ],
                },
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )
        # Should NOT inject because cache_control already exists
        last_block = result["messages"][-1]["content"][-1]
        assert "cache_control" not in last_block

    def test_cache_control_not_injected_for_non_claude_code(self):
        """Test that cache_control is NOT injected for non-Claude Code clients."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("Mozilla/5.0")
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )
        last_block = result["messages"][-1]["content"][-1]
        assert "cache_control" not in last_block

    def test_cache_control_disabled_by_config(self):
        """Test that auto_prompt_caching can be disabled via general_settings."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request,
            data=data,
            general_settings={"auto_prompt_caching": False},
        )
        last_block = result["messages"][-1]["content"][-1]
        assert "cache_control" not in last_block

    def test_cache_control_with_empty_messages(self):
        """Test that empty messages list doesn't cause errors."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {"messages": []}
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )
        assert result["messages"] == []

    def test_cache_control_with_multi_turn_claude_code_messages(self):
        """Test cache_control injection with typical Claude Code multi-turn messages."""
        from litellm.proxy.anthropic_endpoints.endpoints import (
            maybe_inject_auto_prompt_caching,
        )

        mock_request = self._make_mock_request("claude-code/1.0.0")
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "System prompt"},
                        {"type": "text", "text": "Code block 1"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I understand."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "tool result"},
                        {"type": "text", "text": "Continue analyzing."},
                    ],
                },
            ],
        }
        result = maybe_inject_auto_prompt_caching(
            request=mock_request, data=data, general_settings={}
        )

        # Count cache_control occurrences - should be exactly 1
        cache_count = 0
        for msg in result["messages"]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cache_count += 1

        assert cache_count == 1
        # Should be on the very last block of the very last message
        last_block = result["messages"][-1]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}
        assert last_block["text"] == "Continue analyzing."
