"""
Test that an empty ``tools`` array is not forwarded to the upstream provider on
the chat/completions path.

An empty tools array is invalid per the OpenAI spec ("provide at least one tool
or omit the field entirely") and strict OpenAI-compatible backends (e.g. vLLM)
reject it with a 422. ``litellm.completion`` must normalize ``tools: []`` to no
tools so neither ``tools`` nor the now-meaningless ``tool_choice`` is sent.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm


def _mock_chat_response() -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    mock_response.json.return_value = payload
    mock_response.text = json.dumps(payload)
    return mock_response


def _request_body_from_post(mock_post: MagicMock) -> dict:
    mock_post.assert_called()
    kwargs = mock_post.call_args.kwargs
    raw = kwargs.get("data")
    if raw is not None:
        return json.loads(raw)
    return kwargs.get("json", {})


class TestHostedVLLMEmptyTools:
    @patch("litellm.llms.custom_httpx.llm_http_handler._get_httpx_client")
    def test_empty_tools_array_is_omitted(self, mock_get_httpx_client):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_chat_response()
        mock_get_httpx_client.return_value = mock_client

        litellm.completion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://test-vllm.example.com/v1",
            tools=[],
            tool_choice="auto",
        )

        body = _request_body_from_post(mock_client.post)
        assert "tools" not in body
        # tool_choice is meaningless without tools and must not leak either.
        assert "tool_choice" not in body

    @patch("litellm.llms.custom_httpx.llm_http_handler._get_httpx_client")
    def test_non_empty_tools_are_preserved(self, mock_get_httpx_client):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_chat_response()
        mock_get_httpx_client.return_value = mock_client

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        litellm.completion(
            model="hosted_vllm/test-model",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://test-vllm.example.com/v1",
            tools=tools,
        )

        body = _request_body_from_post(mock_client.post)
        assert "tools" in body
        assert len(body["tools"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
