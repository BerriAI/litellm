"""
Tests for the unified Shell tool in the OpenAI Responses API.

Covers:
  1. Mock: request body sent to OpenAI includes shell tool unchanged
  2. Mock: container_reference environment passes through
  3. Mock: shell + function tools coexist in same request
  4. Live: OpenAI non-streaming call with shell tool
  5. Live: OpenAI streaming call with shell tool
"""

import json
import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SHELL_TOOL = {"type": "shell", "environment": {"type": "container_auto"}}

SHELL_TOOL_WITH_CONTAINER_REF = {
    "type": "shell",
    "environment": {
        "type": "container_reference",
        "container_id": "ctr_abc123",
    },
}

MOCK_OPENAI_RESPONSE = {
    "id": "resp_shell_test_001",
    "object": "response",
    "created_at": 1734366691,
    "status": "completed",
    "model": "gpt-4.1",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Here are the files in the current directory.",
                    "annotations": [],
                }
            ],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": None,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [SHELL_TOOL],
    "top_p": None,
    "max_output_tokens": None,
    "previous_response_id": None,
    "reasoning": None,
    "truncation": None,
    "user": None,
}


class _MockHTTPResponse:
    """Minimal httpx.Response stand-in for mocked POST calls."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = httpx.Headers({"content-type": "application/json"})

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# Mock tests — verify request body (no API key needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_shell_tool_request_body_passthrough():
    """
    Verify that when calling litellm.aresponses() with an OpenAI model
    and a shell tool, the tool is included in the POST body unchanged.
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _MockHTTPResponse(MOCK_OPENAI_RESPONSE)

        await litellm.aresponses(
            model="openai/gpt-4.1",
            input="List files in the current directory",
            tools=[SHELL_TOOL],
            tool_choice="auto",
            max_output_tokens=128,
        )

        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]

        assert "tools" in request_body, "Request body must contain 'tools'"
        tools_sent = request_body["tools"]
        assert len(tools_sent) == 1
        assert tools_sent[0]["type"] == "shell"
        assert tools_sent[0]["environment"]["type"] == "container_auto"


@pytest.mark.asyncio
async def test_openai_shell_tool_with_container_reference():
    """
    Verify that container_reference environment config passes through intact.
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _MockHTTPResponse(MOCK_OPENAI_RESPONSE)

        await litellm.aresponses(
            model="openai/gpt-4.1",
            input="Run python --version",
            tools=[SHELL_TOOL_WITH_CONTAINER_REF],
            tool_choice="auto",
        )

        request_body = mock_post.call_args.kwargs["json"]
        tool_sent = request_body["tools"][0]

        assert tool_sent["type"] == "shell"
        assert tool_sent["environment"]["type"] == "container_reference"
        assert tool_sent["environment"]["container_id"] == "ctr_abc123"


@pytest.mark.asyncio
async def test_openai_shell_tool_mixed_with_function_tools():
    """
    Verify that shell tools can coexist with function tools in the same request.
    """
    function_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {}},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _MockHTTPResponse(MOCK_OPENAI_RESPONSE)

        await litellm.aresponses(
            model="openai/gpt-4.1",
            input="What's the weather? Also list files.",
            tools=[function_tool, SHELL_TOOL],
            tool_choice="auto",
        )

        request_body = mock_post.call_args.kwargs["json"]
        tools_sent = request_body["tools"]

        assert len(tools_sent) == 2
        tool_types = {t["type"] for t in tools_sent}
        assert "function" in tool_types
        assert "shell" in tool_types


# ---------------------------------------------------------------------------
# Live E2E tests — require OPENAI_API_KEY (skipped without it)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live OpenAI test",
)
@pytest.mark.asyncio
async def test_openai_shell_tool_live_non_streaming():
    """
    Live call to OpenAI Responses API with a shell tool (non-streaming).
    The model should accept the tool and return a valid response.
    """
    try:
        response = await litellm.aresponses(
            model="openai/gpt-5.2",
            input="Run: echo 'hello from shell tool test'",
            tools=[SHELL_TOOL],
            tool_choice="auto",
            max_output_tokens=256,
        )
    except litellm.BadRequestError as e:
        if "shell" in str(e).lower():
            pytest.skip("Model does not support shell tool")
        raise

    assert response is not None
    resp_dict = dict(response) if not isinstance(response, dict) else response
    assert resp_dict.get("id") is not None
    assert resp_dict.get("status") is not None
    print("Live non-streaming response:", json.dumps(resp_dict, indent=2, default=str))


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live OpenAI test",
)
@pytest.mark.asyncio
async def test_openai_shell_tool_live_streaming():
    """
    Live streaming call to OpenAI Responses API with a shell tool.
    Iterates the event stream and validates at least one event is received.
    """
    try:
        stream = await litellm.aresponses(
            model="openai/gpt-5.2",
            input="Run: echo 'streaming shell test'",
            tools=[SHELL_TOOL],
            tool_choice="auto",
            max_output_tokens=256,
            stream=True,
        )
    except litellm.BadRequestError as e:
        if "shell" in str(e).lower():
            pytest.skip("Model does not support shell tool")
        raise

    event_count = 0
    event_types_seen = []

    async for event in stream:
        event_count += 1
        event_type = getattr(event, "type", None) or (
            event.get("type") if isinstance(event, dict) else None
        )
        if event_type:
            event_types_seen.append(event_type)

    assert event_count > 0, "Should receive at least one streaming event"
    print(f"Streaming: received {event_count} events, types: {set(event_types_seen)}")
