import asyncio
import copy
import json
import os
import sys
from datetime import datetime
from unittest.mock import Mock

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.bedrock.common_utils import (
    ensure_bedrock_anthropic_messages_tool_names,
    normalize_tool_input_schema_types_for_bedrock_invoke,
    remove_custom_field_from_tools,
)
from litellm.constants import BEDROCK_MIN_THINKING_BUDGET_TOKENS
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
    AmazonAnthropicClaudeMessagesStreamDecoder,
)


@pytest.mark.asyncio
async def test_bedrock_sse_wrapper_encodes_dict_chunks():
    """Verify that `bedrock_sse_wrapper` converts dictionary chunks to properly formatted Server-Sent Events and forwards non-dict chunks unchanged."""

    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _dummy_stream():  # type: ignore[return-type]
        yield {"type": "message_delta", "text": "hello"}
        yield b"raw-bytes"

    # Collect all chunks returned by the wrapper
    collected: list[bytes] = []
    async for chunk in cfg.bedrock_sse_wrapper(
        _dummy_stream(),
        litellm_logging_obj=LiteLLMLoggingObj(
            model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[
                {"role": "user", "content": "Hello, can you tell me a short joke?"}
            ],
            stream=True,
            call_type="chat",
            start_time=datetime.now(),
            litellm_call_id="test_bedrock_sse_wrapper_encodes_dict_chunks",
            function_id="test_bedrock_sse_wrapper_encodes_dict_chunks",
        ),
        request_body={},
    ):
        collected.append(chunk)

    assert collected, "No chunks returned from wrapper"

    # First chunk should be SSE encoded
    first_chunk = collected[0]
    assert first_chunk.startswith(b"event: message_delta\n"), first_chunk
    assert first_chunk.endswith(b"\n\n"), first_chunk
    # Ensure the JSON payload is present in the SSE data line
    assert b'"hello"' in first_chunk  # payload contains the text

    # Second chunk should be forwarded unchanged
    assert collected[1] == b"raw-bytes"


@pytest.mark.asyncio
async def test_bedrock_sse_wrapper_keeps_usage_in_message_start_and_message_delta():
    """Regression test: usage should be available on both message_start and message_delta SSE events."""

    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _dummy_stream():  # type: ignore[return-type]
        yield {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                },
            },
        }
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 3,
                "output_tokens": 8,
            },
        }
        yield {
            "type": "message_stop",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 1562,
                "cache_read_input_tokens": 32392,
            },
        }

    collected: list[bytes] = []
    async for chunk in cfg.bedrock_sse_wrapper(
        _dummy_stream(),
        litellm_logging_obj=LiteLLMLoggingObj(
            model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            call_type="chat",
            start_time=datetime.now(),
            litellm_call_id="test_bedrock_sse_wrapper_keeps_usage_in_both_events",
            function_id="test_bedrock_sse_wrapper_keeps_usage_in_both_events",
        ),
        request_body={},
    ):
        collected.append(chunk)

    start_chunk = next(c for c in collected if b"event: message_start\n" in c)
    delta_chunk = next(c for c in collected if b"event: message_delta\n" in c)

    start_json = json.loads(start_chunk.decode("utf-8").split("data: ", 1)[1].strip())
    delta_json = json.loads(delta_chunk.decode("utf-8").split("data: ", 1)[1].strip())

    assert "usage" in start_json["message"]
    assert start_json["message"]["usage"]["input_tokens"] == 3

    assert "usage" in delta_json
    assert delta_json["usage"]["cache_creation_input_tokens"] == 1562
    assert delta_json["usage"]["cache_read_input_tokens"] == 32392
    assert delta_json["usage"]["input_tokens"] == 3
    assert delta_json["usage"]["output_tokens"] == 8


def test_chunk_parser_usage_transformation():
    """Ensure Bedrock invocation metrics are transformed to Anthropic usage keys."""

    decoder = AmazonAnthropicClaudeMessagesStreamDecoder(
        model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0"
    )

    chunk = {
        "type": "message_delta",
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 10,
            "outputTokenCount": 5,
        },
    }

    parsed = decoder._chunk_parser(chunk.copy())  # use copy to avoid side-effects

    # The invocation metrics key should be removed and replaced by `usage`
    assert "amazon-bedrock-invocationMetrics" not in parsed
    assert "usage" in parsed
    assert parsed["usage"]["input_tokens"] == 10
    assert parsed["usage"]["output_tokens"] == 5


def test_remove_ttl_from_cache_control():
    """Ensure ttl field is removed from cache_control in messages."""

    cfg = AmazonAnthropicClaudeMessagesConfig()

    # Test case 1: Message with cache_control containing ttl
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ]
    }

    cfg._remove_ttl_from_cache_control(request)

    # Verify ttl is removed but cache_control remains
    assert "cache_control" in request["messages"][0]["content"][0]
    assert "ttl" not in request["messages"][0]["content"][0]["cache_control"]
    assert request["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"

    # Test case 2: Message with multiple content items
    request2 = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    },
                    {
                        "type": "text",
                        "text": "World",
                        "cache_control": {"type": "ephemeral", "ttl": "2h"},
                    },
                ],
            }
        ]
    }

    cfg._remove_ttl_from_cache_control(request2)

    # Verify ttl is removed from all items
    for item in request2["messages"][0]["content"]:
        if "cache_control" in item:
            assert "ttl" not in item["cache_control"]

    # Test case 3: Message without ttl (should remain unchanged)
    request3 = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    }

    cfg._remove_ttl_from_cache_control(request3)

    # Verify cache_control is unchanged
    assert request3["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"

    # Test case 4: Empty messages (should not raise error)
    request4 = {"messages": []}
    cfg._remove_ttl_from_cache_control(request4)
    assert request4 == {"messages": []}

    # Test case 5: Request without messages key (should not raise error)
    request5 = {}
    cfg._remove_ttl_from_cache_control(request5)
    assert request5 == {}


def test_remove_custom_field_from_tools():
    """
    Ensure the `custom` field is stripped from every tool definition.

    Claude Code v2.1.69+ sends `custom: {defer_loading: true}` on tool
    objects.  Bedrock does not accept this extra field and returns
    "Extra inputs are not permitted".

    Ref: https://github.com/BerriAI/litellm/issues/22847
    """

    # Case 1: tool with `custom` field should have it removed
    request = {
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
                "custom": {"defer_loading": True},
            },
            {
                "name": "Write",
                "description": "Write a file",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
    }

    remove_custom_field_from_tools(request)

    for tool in request["tools"]:
        assert "custom" not in tool, f"Tool {tool['name']} still has 'custom' field"
    # Other fields should be preserved
    assert request["tools"][0]["name"] == "Read"
    assert request["tools"][1]["name"] == "Write"

    # Case 2: request without tools key (should not raise error)
    request2 = {"messages": [{"role": "user", "content": "hi"}]}
    remove_custom_field_from_tools(request2)
    assert "tools" not in request2

    # Case 3: empty tools list (should not raise error)
    request3 = {"tools": []}
    remove_custom_field_from_tools(request3)
    assert request3["tools"] == []

    # Case 4: tools with None value (should not raise error)
    request4 = {"tools": None}
    remove_custom_field_from_tools(request4)
    assert request4["tools"] is None


def test_normalize_tool_input_schema_types_for_bedrock_invoke():
    """
    Claude Code sends ``input_schema.type: \"custom\"`` for custom tools.
    Bedrock Invoke rejects this; it requires JSON Schema ``type: \"object\"``.
    """

    request = {
        "tools": [
            {
                "name": "Agent",
                "type": "custom",
                "description": "subagent",
                "input_schema": {
                    "type": "custom",
                    "additionalProperties": False,
                    "properties": {
                        "nested": {
                            "type": "custom",
                            "properties": {"x": {"type": "string"}},
                        }
                    },
                    "required": ["nested"],
                },
            },
            {
                "name": "Read",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
    }

    normalize_tool_input_schema_types_for_bedrock_invoke(request)

    agent_tool = request["tools"][0]
    assert agent_tool["type"] == "custom"
    assert agent_tool["input_schema"]["type"] == "object"
    assert agent_tool["input_schema"]["properties"]["nested"]["type"] == "object"
    assert request["tools"][1]["input_schema"]["type"] == "object"

    request2 = {"messages": []}
    normalize_tool_input_schema_types_for_bedrock_invoke(request2)
    assert request2 == {"messages": []}


def test_ensure_bedrock_anthropic_messages_tool_names():
    request = {
        "tools": [
            {"input_schema": {"type": "object", "properties": {}}},
            {"name": "", "input_schema": {"type": "object", "properties": {}}},
            {"name": "  ", "input_schema": {"type": "object", "properties": {}}},
            {"name": "KeepMe", "input_schema": {"type": "object", "properties": {}}},
        ]
    }
    ensure_bedrock_anthropic_messages_tool_names(request)
    assert request["tools"][0]["name"] == "litellm_unnamed_tool_0"
    assert request["tools"][1]["name"] == "litellm_unnamed_tool_1"
    assert request["tools"][2]["name"] == "litellm_unnamed_tool_2"
    assert request["tools"][3]["name"] == "KeepMe"


def test_bedrock_invoke_messages_transform_adds_name_when_tool_missing_name():
    """Bedrock requires tools.0.custom.name when the payload is schema-only."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    optional_params = {
        "max_tokens": 128,
        "tools": [
            {
                "input_schema": {
                    "type": "object",
                    "properties": {"questions": {"type": "array"}},
                    "required": ["questions"],
                },
            }
        ],
        "stream": False,
    }
    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=copy.deepcopy(optional_params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["tools"][0]["name"] == "litellm_unnamed_tool_0"


def test_bedrock_invoke_messages_skips_thinking_injection_when_already_enabled():
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    optional_params = {
        "max_tokens": 32000,
        "stream": False,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }
    result = cfg.transform_anthropic_messages_request(
        model="global.anthropic.claude-sonnet-4-6-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=copy.deepcopy(optional_params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    # Claude 4.6/4.7 reject ``thinking.type=enabled``; legacy ``enabled`` is
    # translated to ``adaptive`` (budget_tokens => output_config.effort) and the
    # pre-4.6 ``interleaved-thinking-2025-05-14`` beta must not be attached.
    assert result["thinking"]["type"] == "adaptive"
    betas = result.get("anthropic_beta") or []
    assert "interleaved-thinking-2025-05-14" not in betas


def test_bedrock_invoke_messages_transform_converts_custom_tool_schema_type_to_object():
    """
    End-to-end: AmazonAnthropicClaudeMessagesConfig must emit Bedrock Invoke bodies
    where every ``input_schema`` uses JSON Schema types (``object``), not Anthropic
    ``type: \"custom\"`` (root and nested).
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    tools = [
        {
            "name": "Agent",
            "type": "custom",
            "description": "Subagent",
            "input_schema": {
                "type": "custom",
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string"},
                    "nested": {
                        "type": "custom",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                },
                "required": ["prompt"],
            },
        }
    ]
    optional_params = {
        "max_tokens": 256,
        "tools": copy.deepcopy(tools),
        "stream": False,
    }
    messages = [{"role": "user", "content": "hi"}]

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "tools" in result
    schema = result["tools"][0]["input_schema"]
    assert schema["type"] == "object"
    assert schema["properties"]["nested"]["type"] == "object"
    # Tool discriminator stays Anthropic-side; only input_schema is normalized
    assert result["tools"][0]["type"] == "custom"


def test_remove_ttl_from_cache_control_processes_tools():
    """
    Ensure _remove_ttl_from_cache_control also sanitizes cache_control on tools.

    Without this, tools keep unsupported ttl values while system/messages have
    them stripped, causing TTL ordering violations on Bedrock.
    """

    cfg = AmazonAnthropicClaudeMessagesConfig()

    # Tools with ttl should have it stripped for non-Claude-4.5 models
    request = {
        "tools": [
            {
                "name": "get_weather",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            {
                "name": "get_time",
                "input_schema": {"type": "object"},
            },
        ],
        "system": [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [],
    }

    cfg._remove_ttl_from_cache_control(
        request, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # Tool ttl should be stripped
    assert "ttl" not in request["tools"][0]["cache_control"]
    assert request["tools"][0]["cache_control"]["type"] == "ephemeral"
    # Tool without cache_control should be unchanged
    assert "cache_control" not in request["tools"][1]
    # System ttl should also be stripped
    assert "ttl" not in request["system"][0]["cache_control"]


def test_remove_ttl_from_cache_control_preserves_tools_ttl_for_claude_4_5():
    """
    For Claude 4.5+ models, ttl in ["5m", "1h"] should be preserved on tools,
    just like it is for system and messages.
    """

    cfg = AmazonAnthropicClaudeMessagesConfig()

    request = {
        "tools": [
            {
                "name": "get_weather",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "system": [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }

    cfg._remove_ttl_from_cache_control(
        request, model="us.anthropic.claude-sonnet-4-5-20250514-v1:0"
    )

    # Both tools and system should preserve ttl for Claude 4.5
    assert request["tools"][0]["cache_control"]["ttl"] == "1h"
    assert request["system"][0]["cache_control"]["ttl"] == "1h"


def test_remove_scope_from_cache_control():
    """Ensure scope field is removed from cache_control for Bedrock (not supported)."""

    cfg = AmazonAnthropicClaudeMessagesConfig()

    # Test case 1: System with cache_control containing scope
    request = {
        "system": [
            {
                "type": "text",
                "text": "You are an AI assistant.",
                "cache_control": {
                    "type": "ephemeral",
                    "scope": "global",
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {
                            "type": "ephemeral",
                            "scope": "global",
                        },
                    }
                ],
            }
        ],
    }

    cfg._remove_ttl_from_cache_control(request)

    # Verify scope is removed from system
    assert "scope" not in request["system"][0]["cache_control"]
    assert request["system"][0]["cache_control"]["type"] == "ephemeral"

    # Verify scope is removed from messages
    assert "scope" not in request["messages"][0]["content"][0]["cache_control"]
    assert request["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"


def test_bedrock_messages_strips_output_config():
    """
    Ensure output_config is stripped from the request before sending to
    Bedrock Invoke, which doesn't support this Anthropic-specific parameter.

    Regression test for: https://github.com/BerriAI/litellm/issues/22797
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "high",
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        "output_config" not in result
    ), "output_config should be stripped — Bedrock Invoke rejects it"
    # Other params should be preserved
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_strips_output_config_with_output_format():
    """
    When both output_config and output_format are present, both should be
    stripped (output_format is converted to inline schema, output_config
    is simply dropped).
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {"effort": "low"},
        "output_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "output_config" not in result
    assert "output_format" not in result


def test_bedrock_messages_strips_context_management():
    """
    Ensure context_management is stripped from the request before sending to
    Bedrock Invoke, which doesn't support this Anthropic-specific parameter.

    Claude Code sends context_management on every request; leaving it in the body
    causes a 400 "context_management: Extra inputs are not permitted" from Bedrock.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        "context_management" not in result
    ), "context_management should be stripped — Bedrock Invoke rejects it"
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_allowlist_filters_anthropic_only_fields():
    """
    Bedrock Invoke rejects any top-level body field it doesn't recognize with
    "Extra inputs are not permitted". Defend against that by filtering the
    outgoing body to a Bedrock-supported allowlist — catches Anthropic-only
    extensions (speed, mcp_servers, container, ...) and any future additions
    Claude Code starts sending before we learn about them.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "temperature": 0.5,
        "speed": "fast",
        "mcp_servers": [{"type": "url", "url": "https://example.com"}],
        "container": {"skills": []},
        "inference_geo": "us",
        "output_config": {"effort": "low"},
        "context_management": {"edits": []},
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    for bad in (
        "speed",
        "mcp_servers",
        "container",
        "inference_geo",
        "output_config",
        "context_management",
        "model",
        "stream",
    ):
        assert bad not in result, f"{bad} should be stripped by the allowlist"

    # Supported fields pass through.
    assert result["max_tokens"] == 4096
    assert result["temperature"] == 0.5
    assert result["anthropic_version"] == cfg.DEFAULT_BEDROCK_ANTHROPIC_API_VERSION
    # Every surviving key is in the allowlist.
    assert set(result).issubset(cfg.BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS)


def test_bedrock_messages_filters_user_provided_unsupported_beta_header():
    """
    In proxy deployments the client (e.g. Claude Code) doesn't know the backend
    is Bedrock and may send Anthropic-direct beta headers Bedrock can't handle.
    All betas must go through the provider mapping, not just auto-injected ones
    — otherwise Bedrock 400s on the unsupported value.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {"max_tokens": 128}
    # `advisor-tool-2026-03-01` has no bedrock mapping entry → must be dropped.
    # `context-1m-2025-08-07` does → must pass through.
    headers = {
        "anthropic-beta": "advisor-tool-2026-03-01,context-1m-2025-08-07",
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )

    betas = result.get("anthropic_beta") or []
    assert (
        "advisor-tool-2026-03-01" not in betas
    ), "user-provided beta not in the Bedrock mapping must be dropped"
    assert (
        "context-1m-2025-08-07" in betas
    ), "user-provided beta that IS in the Bedrock mapping should survive"


def test_bedrock_messages_renames_user_provided_aliased_beta_header():
    """
    Bedrock's config maps `advanced-tool-use-2025-11-20` to
    `tool-search-tool-2025-10-19`. User-provided betas must go through the
    rename too, not be forwarded under their Anthropic-direct spelling.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {"max_tokens": 128}
    headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )

    betas = result.get("anthropic_beta") or []
    assert (
        "advanced-tool-use-2025-11-20" not in betas
    ), "Anthropic-direct spelling should be rewritten, not forwarded verbatim"
    assert (
        "tool-search-tool-2025-10-19" in betas
    ), "user-provided beta should be renamed to the Bedrock-side spelling"


@pytest.mark.asyncio
async def test_promote_message_stop_usage_preserves_message_delta_output_tokens():
    """
    Bedrock unified /messages streaming can send full usage on message_delta and a
    conflicting smaller usage on message_stop (e.g. output_tokens 9 vs 12).
    _promote_message_stop_usage must not replace message_delta output_tokens.
    """
    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _stream():  # type: ignore[return-type]
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 10553,
                "cache_read_input_tokens": 25490,
                "output_tokens": 12,
            },
        }
        yield {
            "type": "message_stop",
            "usage": {"input_tokens": 3, "output_tokens": 9},
        }

    merged: list[dict] = []
    async for chunk in cfg._promote_message_stop_usage(_stream()):
        if isinstance(chunk, dict):
            merged.append(chunk)

    assert len(merged) >= 1
    delta_out = merged[0]
    assert delta_out["type"] == "message_delta"
    assert delta_out["usage"]["output_tokens"] == 12
    assert delta_out["usage"]["cache_creation_input_tokens"] == 10553
    assert delta_out["usage"]["cache_read_input_tokens"] == 25490
    assert delta_out["usage"]["input_tokens"] == 3


@pytest.mark.asyncio
async def test_promote_message_start_cache_when_message_stop_omits_cache_fields():
    """
    GovCloud / some Bedrock streams put cache_read only on message_start; delta and
    stop repeat uncached input_tokens only. Merging start cache onto message_delta
    avoids inconsistent usage and negative input costs (LIT-2411).
    """
    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _stream():  # type: ignore[return-type]
        yield {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 22167,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0,
                    },
                    "output_tokens": 4,
                },
            },
        }
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 10, "output_tokens": 181},
        }
        yield {"type": "message_stop", "usage": {"input_tokens": 10, "output_tokens": 181}}

    merged: list[dict] = []
    async for chunk in cfg._promote_message_stop_usage(_stream()):
        if isinstance(chunk, dict):
            merged.append(chunk)

    delta_chunks = [c for c in merged if c.get("type") == "message_delta"]
    assert len(delta_chunks) == 1
    u = delta_chunks[0]["usage"]
    assert u["input_tokens"] == 10
    assert u["output_tokens"] == 181
    assert u["cache_read_input_tokens"] == 22167
    assert u["cache_creation_input_tokens"] == 0


@pytest.mark.asyncio
async def test_unified_bedrock_messages_cache_on_start_only_never_negative_cost():
    """
    Regression guard for LIT-2411:
    If cache usage is present only on message_start (and omitted from
    message_delta/message_stop), final reconstructed usage + cost must still
    be consistent and non-negative.
    """
    from litellm import completion_cost
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _stream():  # type: ignore[return-type]
        yield {
            "type": "message_start",
            "message": {
                "id": "msg_bdrk_01WuFzkDbE9KWgiWakMRNKcA",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 22167,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0,
                    },
                    "output_tokens": 4,
                },
            },
        }
        yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello from regression test"},
        }
        yield {"type": "content_block_stop", "index": 0}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 181, "input_tokens": 10},
        }
        yield {"type": "message_stop", "usage": {"input_tokens": 10, "output_tokens": 181}}

    logging_obj = LiteLLMLoggingObj(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        call_type="chat",
        start_time=datetime.now(),
        litellm_call_id="test_cache_on_start_only_never_negative_cost",
        function_id="test_cache_on_start_only_never_negative_cost",
    )

    collected: list[bytes] = []
    async for sse in cfg.bedrock_sse_wrapper(
        completion_stream=_stream(),
        litellm_logging_obj=logging_obj,
        request_body={"model": "anthropic.claude-3-5-sonnet-20240620-v1:0"},
    ):
        collected.append(sse)

    built = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=collected,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        litellm_logging_obj=Mock(),
    )
    assert built.usage is not None
    assert built.usage.prompt_tokens == 22177
    assert built.usage.completion_tokens == 181
    assert built.usage.cache_creation_input_tokens == 0
    assert built.usage.cache_read_input_tokens == 22167

    cost = completion_cost(
        completion_response=built,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        custom_llm_provider="bedrock",
    )
    assert cost > 0
    assert cost == pytest.approx(0.0093951, rel=0, abs=1e-9)


@pytest.mark.asyncio
async def test_unified_bedrock_messages_sse_usage_and_cost_claude_sonnet_46():
    """
    End-to-end for Bedrock Invoke Anthropic Messages (unified) streaming path:
    dict chunks -> _promote_message_stop_usage -> bedrock_sse_wrapper SSE bytes ->
    same logging reconstruction as Anthropic /messages. Ensures token counts and
    completion_cost match model_prices for us.anthropic.claude-sonnet-4-6.
    """
    from litellm import completion_cost
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _stream():  # type: ignore[return-type]
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 10553,
                "cache_read_input_tokens": 25490,
                "output_tokens": 12,
            },
        }
        yield {
            "type": "message_stop",
            "usage": {"input_tokens": 3, "output_tokens": 9},
        }

    logging_obj = LiteLLMLoggingObj(
        model="bedrock/us.anthropic.claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        call_type="chat",
        start_time=datetime.now(),
        litellm_call_id="test_unified_bedrock_messages_sse_cost",
        function_id="test_unified_bedrock_messages_sse_cost",
    )

    collected: list[bytes] = []
    async for sse in cfg.bedrock_sse_wrapper(
        completion_stream=_stream(),
        litellm_logging_obj=logging_obj,
        request_body={"model": "us.anthropic.claude-sonnet-4-6"},
    ):
        collected.append(sse)

    built = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=collected,
        model="us.anthropic.claude-sonnet-4-6",
        litellm_logging_obj=Mock(),
    )
    assert built.usage is not None
    assert built.usage.completion_tokens == 12
    assert built.usage.prompt_tokens == 36046
    assert built.usage.total_tokens == 36058
    assert built.usage.cache_creation_input_tokens == 10553
    assert built.usage.cache_read_input_tokens == 25490

    cost = completion_cost(
        completion_response=built,
        model="bedrock/us.anthropic.claude-sonnet-4-6",
        custom_llm_provider="bedrock",
    )
    assert cost == pytest.approx(0.052150725, rel=0, abs=1e-9)
