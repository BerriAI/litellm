"""
End-to-end tests for the command_code provider through litellm.completion.

Fully mocked at the HTTP client layer - no real network calls.
"""

import json
import os
import sys
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

MODEL = "command_code/deepseek/deepseek-v4-flash"

EVENTS = (
    json.dumps({"type": "reasoning-delta", "text": "thinking"})
    + "\n"
    + json.dumps({"type": "text-delta", "text": "Hello "})
    + "\n"
    + json.dumps({"type": "text-delta", "text": "world"})
    + "\n"
    + json.dumps(
        {
            "type": "finish",
            "finishReason": "stop",
            "totalUsage": {
                "inputTokens": 7,
                "outputTokens": 2,
                "inputTokenDetails": {"cacheReadTokens": 1, "cacheWriteTokens": 0},
            },
        }
    )
    + "\n"
)

TOOL_EVENTS = (
    json.dumps({"type": "text-delta", "text": "calling"})
    + "\n"
    + json.dumps(
        {
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "get_weather",
            "input": {"city": "sf"},
        }
    )
    + "\n"
    + json.dumps({"type": "finish", "finishReason": "tool-calls"})
    + "\n"
)


def _mock_response(payload: str, url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=payload.encode("utf-8"),
        request=httpx.Request("POST", url),
    )


class TestSyncCompletion:
    def test_non_streaming_completion(self):
        captured = {}

        def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["data"] = json.loads(kwargs.get("data"))
            captured["headers"] = kwargs.get("headers")
            return _mock_response(EVENTS, url)

        with patch.object(HTTPHandler, "post", fake_post):
            response = litellm.completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi"},
                ],
                api_key="sk-test",
            )

        assert captured["url"] == "https://api.commandcode.ai/alpha/generate"
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        assert "x-command-code-version" in captured["headers"]

        params = captured["data"]["params"]
        assert params["model"] == "deepseek/deepseek-v4-flash"
        assert params["system"] == "be nice"
        assert params["stream"] is True
        assert captured["data"]["config"]["workingDir"] == "/tmp"

        assert response.choices[0].message.content == "Hello world"
        assert response.choices[0].message.reasoning_content == "thinking"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 8  # 7 input + 1 cache read
        assert response.usage.completion_tokens == 2
        assert response.usage.cache_read_input_tokens == 1

    def test_streaming_completion(self):
        def fake_post(self, url, **kwargs):
            return _mock_response(EVENTS, url)

        with patch.object(HTTPHandler, "post", fake_post):
            stream = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
                stream=True,
            )
            content, reasoning, finish = [], [], None
            for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    reasoning.append(delta.reasoning_content)
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason

        assert "".join(content) == "Hello world"
        assert "".join(reasoning) == "thinking"
        assert finish == "stop"

    def test_streaming_tool_call(self):
        def fake_post(self, url, **kwargs):
            return _mock_response(TOOL_EVENTS, url)

        with patch.object(HTTPHandler, "post", fake_post):
            stream = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "weather in sf?"}],
                api_key="sk-test",
                stream=True,
            )
            tool_names, finish = [], None
            for chunk in stream:
                delta = chunk.choices[0].delta
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    tool_names.append(tool_call.function.name)
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason

        assert tool_names == ["get_weather"]
        assert finish == "tool_calls"


class TestAsyncCompletion:
    @pytest.mark.asyncio
    async def test_async_non_streaming_completion(self):
        async def fake_post(self, url, **kwargs):
            return _mock_response(EVENTS, url)

        with patch.object(AsyncHTTPHandler, "post", fake_post):
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-test",
            )

        assert response.choices[0].message.content == "Hello world"
        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_async_streaming_completion(self):
        async def fake_post(self, url, **kwargs):
            return _mock_response(TOOL_EVENTS, url)

        with patch.object(AsyncHTTPHandler, "post", fake_post):
            stream = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "weather in sf?"}],
                api_key="sk-test",
                stream=True,
            )
            content, tool_names, finish = [], [], None
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content.append(delta.content)
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    tool_names.append(tool_call.function.name)
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason

        assert "".join(content) == "calling"
        assert tool_names == ["get_weather"]
        assert finish == "tool_calls"
