"""
Tests for the unified Shell tool → provider-specific mapping.

Covers:
  1. Live: Anthropic non-streaming call with shell tool (shell → bash_20250124)
  2. Live: Anthropic streaming call with shell tool
  3. Live: Anthropic shell + function tools coexist
  4. Unit: Bedrock shell → bash_20250124 mapping at the transformation layer
  5. Unit: Vertex AI shell → code_execution mapping
  6. Unit: Unsupported provider raises clear error
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm

SHELL_TOOL = {"type": "shell", "environment": {"type": "container_auto"}}


# ---------------------------------------------------------------------------
# Live E2E tests — require ANTHROPIC_API_KEY (skipped without it)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic test",
)
@pytest.mark.asyncio
async def test_anthropic_shell_tool_live_non_streaming():
    """
    Live call to Anthropic via litellm.aresponses() with a shell tool.
    The unified 'shell' tool should be mapped to bash_20250124 and the
    model should accept it and return a valid response.
    """
    response = await litellm.aresponses(
        model="anthropic/claude-sonnet-4-5-20250929",
        input="Run: echo 'hello from litellm shell test'",
        tools=[SHELL_TOOL],
        max_output_tokens=256,
    )

    assert response is not None
    resp_dict = dict(response) if not isinstance(response, dict) else response
    assert resp_dict.get("id") is not None
    assert resp_dict.get("status") is not None
    print("Live non-streaming response:", json.dumps(resp_dict, indent=2, default=str))


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic test",
)
@pytest.mark.asyncio
async def test_anthropic_shell_tool_live_streaming():
    """
    Live streaming call to Anthropic via litellm.aresponses() with a shell tool.
    Iterates the event stream and validates at least one event is received.
    """
    stream = await litellm.aresponses(
        model="anthropic/claude-sonnet-4-5-20250929",
        input="Run: echo 'streaming shell test'",
        tools=[SHELL_TOOL],
        max_output_tokens=256,
        stream=True,
    )

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


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic test",
)
@pytest.mark.asyncio
async def test_anthropic_shell_tool_mixed_with_function_tools_live():
    """
    Live call to Anthropic with both shell and function tools.
    Verifies the model accepts both tool types in a single request.
    """
    function_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }

    response = await litellm.aresponses(
        model="anthropic/claude-sonnet-4-5-20250929",
        input="What's the weather in NYC? Also run: echo hello",
        tools=[function_tool, SHELL_TOOL],
        max_output_tokens=256,
    )

    assert response is not None
    resp_dict = dict(response) if not isinstance(response, dict) else response
    assert resp_dict.get("id") is not None
    print("Live mixed tools response:", json.dumps(resp_dict, indent=2, default=str))


# ---------------------------------------------------------------------------
# Unit tests — transformation layer (no API key needed)
# ---------------------------------------------------------------------------


def test_bedrock_shell_tool_maps_to_bash():
    """
    Verify that the Chat Completion bridge maps shell → bash_20250124
    when the provider is 'bedrock'.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[SHELL_TOOL],
        custom_llm_provider="bedrock",
    )

    assert len(result_tools) == 1
    assert result_tools[0]["type"] == "bash_20250124"
    assert result_tools[0]["name"] == "bash"


def test_vertex_ai_shell_tool_maps_to_code_execution():
    """
    Verify that the Chat Completion bridge maps shell → code_execution
    when the provider is 'vertex_ai'.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[SHELL_TOOL],
        custom_llm_provider="vertex_ai",
    )

    assert len(result_tools) == 1
    assert "code_execution" in result_tools[0]
    assert result_tools[0]["code_execution"] == {}


def test_gemini_shell_tool_maps_to_code_execution():
    """
    Verify that the Chat Completion bridge maps shell → code_execution
    when the provider is 'gemini'.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[SHELL_TOOL],
        custom_llm_provider="gemini",
    )

    assert len(result_tools) == 1
    assert "code_execution" in result_tools[0]


def test_unsupported_provider_maps_to_litellm_shell_function():
    """
    Verify that passing a shell tool for an unsupported provider
    produces a synthetic ``_litellm_shell`` function tool (sandbox fallback).
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    result_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[SHELL_TOOL],
        custom_llm_provider="cohere",
    )

    assert len(result_tools) == 1
    assert result_tools[0]["type"] == "function"
    assert result_tools[0]["function"]["name"] == "_litellm_shell"
    assert "command" in result_tools[0]["function"]["parameters"]["properties"]
