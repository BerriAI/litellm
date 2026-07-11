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
from litellm.constants import (
    BEDROCK_MIN_THINKING_BUDGET_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
    AmazonAnthropicClaudeMessagesStreamDecoder,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so adaptive-thinking detection reads this
    branch's ``supports_adaptive_thinking`` flags, which the network-fetched
    ``main`` copy lacks until merge."""
    import litellm

    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


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
async def test_bedrock_sse_wrapper_appends_error_event_when_stream_truncates_mid_tool_use():
    """
    Regression test for LIT-3724: Bedrock invoke streams that go silent
    mid tool_use (no content_block_stop / message_delta / message_stop)
    used to be closed as a successful SSE stream, handing clients
    unterminated tool-call JSON with HTTP 200. The stream must now end
    with an Anthropic-protocol `error` SSE event.
    """
    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _truncated_stream():
        yield {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3, "output_tokens": 1}}}
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tooluse_1", "name": "write", "input": {}},
        }
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "/builder/docs/QUAL'},
        }

    collected: list[bytes] = []
    async for chunk in cfg.bedrock_sse_wrapper(
        _truncated_stream(),
        litellm_logging_obj=LiteLLMLoggingObj(
            model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "write the file"}],
            stream=True,
            call_type="chat",
            start_time=datetime.now(),
            litellm_call_id="test_bedrock_sse_wrapper_truncated_tool_use",
            function_id="test_bedrock_sse_wrapper_truncated_tool_use",
        ),
        request_body={},
    ):
        collected.append(chunk)

    assert len(collected) == 4
    error_event = collected[-1].decode()
    assert error_event.startswith("event: error\n")
    error_payload = json.loads(error_event.split("data: ", 1)[1])
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_bedrock_sse_wrapper_no_error_event_when_stream_ends_with_message_stop():
    cfg = AmazonAnthropicClaudeMessagesConfig()

    async def _complete_stream():
        yield {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3, "output_tokens": 1}}}
        yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}
        yield {"type": "message_stop"}

    collected: list[bytes] = []
    async for chunk in cfg.bedrock_sse_wrapper(
        _complete_stream(),
        litellm_logging_obj=LiteLLMLoggingObj(
            model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="chat",
            start_time=datetime.now(),
            litellm_call_id="test_bedrock_sse_wrapper_complete_stream",
            function_id="test_bedrock_sse_wrapper_complete_stream",
        ),
        request_body={},
    ):
        collected.append(chunk)

    assert len(collected) == 6
    assert collected[-1].startswith(b"event: message_stop\n")
    assert not any(chunk.startswith(b"event: error\n") for chunk in collected)


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


def test_bedrock_invoke_messages_skips_thinking_injection_when_already_enabled(
    local_model_cost_map,
):
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


def test_remove_ttl_from_cache_control_processes_tools(local_model_cost_map):
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


def test_remove_ttl_from_cache_control_preserves_tools_ttl_for_claude_4_5(local_model_cost_map):
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
        request, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
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
    Ensure output_config is stripped from the request for models that do not
    support it.

    Regression test for: https://github.com/BerriAI/litellm/issues/22797
    """
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "high",
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=False,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert "output_config" not in result, (
        "output_config should be stripped for models that don't support it"
    )
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_preserves_output_config_for_claude_4_6():
    """
    Ensure output_config is preserved for models that support it on Bedrock Invoke.
    """
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "high",
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-6-v1",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert "output_config" in result, (
        "output_config should be preserved for supported models"
    )
    assert result["output_config"] == {"effort": "high"}
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_checks_output_config_support_with_bedrock_provider():
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "high",
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ) as mock_supports_factory:
        result = cfg.transform_anthropic_messages_request(
            model="us.anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    mock_supports_factory.assert_called_with(
        model="us.anthropic.claude-opus-4-7",
        custom_llm_provider="bedrock",
        key="supports_output_config",
    )
    assert result["output_config"] == {"effort": "high"}


def test_bedrock_messages_forwards_output_config():
    """Bedrock Invoke /v1/messages forwards ``output_config`` for supported models."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "high",
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert result.get("output_config") == {"effort": "high"}
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_forwards_output_config_with_output_format():
    """``output_config`` is forwarded; ``output_format`` is converted to inline schema."""
    from unittest.mock import patch

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

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert result.get("output_config") == {"effort": "low"}
    assert "output_format" not in result


def test_bedrock_messages_converts_output_config_format_to_inline_schema():
    """``output_config.format`` is consumed so Bedrock does not see an unknown nested key."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "xhigh",
            "format": {"type": "json_schema", "schema": schema},
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert result.get("output_config") == {"effort": "xhigh"}
    assert "output_format" not in result
    last_content = result["messages"][0]["content"]
    assert json.loads(last_content[-1]["text"]) == schema


@pytest.mark.parametrize(
    "model,expected_effort",
    [
        ("anthropic.claude-opus-4-5-20251101-v1:0", "high"),
        ("anthropic.claude-opus-4-6-v1", "max"),
        ("anthropic.claude-opus-4-7", "xhigh"),
    ],
)
def test_bedrock_messages_normalizes_output_config_effort_for_opus(
    model, expected_effort
):
    """Bedrock /v1/messages accepts ``xhigh`` and forwards the provider-safe effort."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            anthropic_messages_optional_request_params={
                "max_tokens": 4096,
                "output_config": {"effort": "xhigh"},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert result.get("output_config") == {"effort": expected_effort}


def test_bedrock_messages_does_not_mutate_callers_messages_when_embedding_schema():
    """Inline-schema embedding must not mutate the caller's ``messages`` list,
    message dicts, or content list."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    caller_content = [{"type": "text", "text": "Hello"}]
    caller_message = {"role": "user", "content": caller_content}
    caller_messages = [caller_message]
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    optional_params = {
        "max_tokens": 4096,
        "output_config": {
            "effort": "xhigh",
            "format": {"type": "json_schema", "schema": schema},
        },
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=caller_messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert caller_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert caller_message == {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }
    assert caller_content == [{"type": "text", "text": "Hello"}]
    last_content = result["messages"][-1]["content"]
    assert json.loads(last_content[-1]["text"]) == schema


def test_bedrock_messages_does_not_mutate_callers_output_config():
    """`pop_bedrock_invoke_output_config_format` / effort normalization must not
    leak into the caller's ``optional_params`` dict."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    caller_output_config = {
        "effort": "xhigh",
        "format": {"type": "json_schema", "schema": schema},
    }
    optional_params = {
        "max_tokens": 4096,
        "output_config": caller_output_config,
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-5-20251101-v1:0",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert caller_output_config == {
        "effort": "xhigh",
        "format": {"type": "json_schema", "schema": schema},
    }


def test_bedrock_messages_strips_output_config_with_output_format():
    """
    When both output_config and output_format are present, output_format
    is converted to inline schema and output_config is stripped for
    unsupported models.
    """
    from unittest.mock import patch

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

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=False,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert "output_config" not in result
    assert "output_format" not in result


def test_bedrock_messages_drop_params_strips_output_config_for_pre_4_5():
    """``drop_params=True`` strips ``output_config`` for pre-4.5 Anthropic on /v1/messages."""
    import litellm
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {"effort": "low"},
    }

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert "output_config" not in result


def test_bedrock_messages_drop_params_keeps_output_config_for_4_7():
    """``drop_params=True`` does not strip on opus-4-7 (supports effort)."""
    from unittest.mock import patch

    import litellm
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "output_config": {"effort": "high"},
    }

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        with patch(
            "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
            return_value=True,
        ):
            result = cfg.transform_anthropic_messages_request(
                model="anthropic.claude-opus-4-7",
                messages=messages,
                anthropic_messages_optional_request_params=optional_params,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
    finally:
        litellm.drop_params = original

    assert result.get("output_config") == {"effort": "high"}


@pytest.mark.parametrize(
    "reasoning_effort,expected_effort",
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ],
)
def test_bedrock_messages_maps_reasoning_effort_for_adaptive_model(
    local_model_cost_map, reasoning_effort, expected_effort
):
    """``reasoning_effort`` maps to ``thinking`` + ``output_config.effort`` on /v1/messages."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "reasoning_effort": reasoning_effort,
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert "reasoning_effort" not in result
    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": expected_effort}


def test_bedrock_messages_reasoning_effort_on_non_adaptive_uses_thinking_budget():
    """Non-adaptive models map ``reasoning_effort`` to ``thinking.budget_tokens``."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "reasoning_effort": "medium",
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-5-20251101-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "reasoning_effort" not in result
    assert "output_config" not in result
    thinking = result.get("thinking")
    assert isinstance(thinking, dict)
    assert thinking.get("type") == "enabled"
    assert isinstance(thinking.get("budget_tokens"), int)
    assert thinking["budget_tokens"] >= 1024


def test_bedrock_messages_reasoning_effort_none_clears_thinking():
    """``reasoning_effort='none'`` clears both ``thinking`` and ``output_config``."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "reasoning_effort": "none",
        "output_config": {"effort": "high"},
        "thinking": {"type": "adaptive"},
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-7",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "reasoning_effort" not in result
    assert "thinking" not in result
    assert "output_config" not in result


def test_bedrock_messages_invalid_reasoning_effort_raises_400():
    """Garbage ``reasoning_effort`` raises AnthropicError (400)."""
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    for bad_effort in ("invalid", "disabled", ""):
        with pytest.raises(AnthropicError):
            cfg.transform_anthropic_messages_request(
                model="anthropic.claude-opus-4-7",
                messages=messages,
                anthropic_messages_optional_request_params={
                    "max_tokens": 4096,
                    "reasoning_effort": bad_effort,
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )


def test_bedrock_messages_explicit_output_config_wins_over_reasoning_effort():
    """Explicit ``output_config.effort`` wins over the ``reasoning_effort`` alias."""
    from unittest.mock import patch

    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    optional_params = {
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "output_config": {"effort": "max"},
    }

    with patch(
        "litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ):
        result = cfg.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-7",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert "reasoning_effort" not in result
    assert result.get("output_config") == {"effort": "max"}


def test_bedrock_messages_strips_context_management():
    """
    Ensure context_management is stripped from the request before sending to
    Bedrock Invoke when it carries only LiteLLM-internal edits (e.g.
    clear_thinking_20251015, which is consumed via thinking injection).

    Claude Code sends context_management on every request; leaving such edits
    in the body causes a 400 "context_management: Extra inputs are not
    permitted" from Bedrock.
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

    assert "context_management" not in result, (
        "context_management should be stripped — Bedrock Invoke rejects it"
    )
    assert result.get("max_tokens") == 4096


def test_bedrock_messages_preserves_compact_context_management_and_adds_beta():
    """
    Bedrock InvokeModel supports compaction when paired with the
    ``compact-2026-01-12`` anthropic-beta header, even though the Converse API
    does not. The transformation should:
      1. Keep ``context_management`` with compact_20260112 edits in the body
         (Bedrock rejects unknown top-level fields, but accepts this one with
         the right beta).
      2. Auto-inject ``compact-2026-01-12`` into ``anthropic_beta``.

    Ref: https://github.com/BerriAI/litellm/issues/27532
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {"edits": [{"type": "compact_20260112"}]},
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-6-20250929-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result.get("context_management") == {"edits": [{"type": "compact_20260112"}]}
    assert "compact-2026-01-12" in result.get("anthropic_beta", [])
    assert result["max_tokens"] == 4096


def test_bedrock_messages_filters_unsupported_context_management_edits():
    """
    Mixed edit lists must drop the LiteLLM-internal ``clear_thinking_20251015``
    entries while keeping ``compact_20260112`` and adding the compact beta.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": "all"},
                {"type": "compact_20260112"},
            ]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-6-20250929-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result.get("context_management") == {"edits": [{"type": "compact_20260112"}]}
    assert "compact-2026-01-12" in result.get("anthropic_beta", [])


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
        model="anthropic.claude-opus-4-7",
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
        "context_management",
        "model",
        "stream",
    ):
        assert bad not in result, f"{bad} should be stripped by the allowlist"

    assert result.get("output_config") == {"effort": "low"}
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
    assert "advisor-tool-2026-03-01" not in betas, (
        "user-provided beta not in the Bedrock mapping must be dropped"
    )
    assert "context-1m-2025-08-07" in betas, (
        "user-provided beta that IS in the Bedrock mapping should survive"
    )


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
    assert "advanced-tool-use-2025-11-20" not in betas, (
        "Anthropic-direct spelling should be rewritten, not forwarded verbatim"
    )
    assert "tool-search-tool-2025-10-19" in betas, (
        "user-provided beta should be renamed to the Bedrock-side spelling"
    )


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
        yield {
            "type": "message_stop",
            "usage": {"input_tokens": 10, "output_tokens": 181},
        }

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
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
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
        yield {
            "type": "message_stop",
            "usage": {"input_tokens": 10, "output_tokens": 181},
        }

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


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-opus-4-8",
        "global.anthropic.claude-opus-4-7-v1:0",
        "us.anthropic.claude-fable-5",
        "global.anthropic.claude-fable-5",
    ],
)
def test_bedrock_clear_thinking_injects_adaptive_with_effort_for_adaptive_models(
    local_model_cost_map, model
):
    """clear_thinking_20251015 without a top-level ``thinking`` field must inject
    ``thinking.type=adaptive`` plus ``output_config.effort`` on adaptive-thinking
    models (Opus 4.7/4.8, Fable 5). The legacy ``thinking.type=enabled`` shape is
    rejected by Bedrock for these models (issue #29188). Detection is sourced from
    the cost-map ``supports_adaptive_thinking`` flag; the versioned Bedrock id
    (``-v1:0``) resolves to its suffix-less cost-map entry."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model=model
    )

    assert changed is True
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "low"


def test_bedrock_clear_thinking_converts_legacy_enabled_budget_to_effort():
    """A legacy ``thinking.type=enabled`` with a budget on an adaptive model is
    translated to ``adaptive`` + ``output_config.effort`` derived from the budget
    rather than forwarded as the rejected ``enabled`` shape."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        },
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model="us.anthropic.claude-opus-4-8"
    )

    assert changed is True
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "high"


def test_resolve_clear_thinking_budget_tokens_honors_explicit_zero():
    """A truthiness guard would treat an explicit ``budget_tokens=0`` as missing
    and silently substitute the Bedrock minimum. The resolver must honor ``0``
    and only fall back to the minimum when the caller omits the budget."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    assert cfg._resolve_clear_thinking_budget_tokens(0) == 0
    assert (
        cfg._resolve_clear_thinking_budget_tokens(None)
        == BEDROCK_MIN_THINKING_BUDGET_TOKENS
    )
    assert cfg._resolve_clear_thinking_budget_tokens(12000) == 12000


def test_bedrock_clear_thinking_keeps_enabled_for_non_adaptive_models():
    """Non-adaptive extended-thinking models (e.g. Opus 4.5) still use the legacy
    ``thinking.type=enabled`` + ``budget_tokens`` shape, which they accept."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model="anthropic.claude-opus-4-5-v1:0"
    )

    assert changed is True
    assert request["thinking"] == {
        "type": "enabled",
        "budget_tokens": BEDROCK_MIN_THINKING_BUDGET_TOKENS,
    }
    assert "output_config" not in request


def test_bedrock_invoke_transform_emits_adaptive_thinking_for_opus_4_8():
    """End-to-end: a Claude Code clear_thinking payload (no ``thinking`` field) on
    Opus 4.8 must produce a Bedrock Invoke body with adaptive thinking and an
    effort, never ``thinking.type.enabled``."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    optional_params = {
        "max_tokens": 32000,
        "stream": False,
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="us.anthropic.claude-opus-4-8",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=copy.deepcopy(optional_params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"]["effort"] == "low"


def test_bedrock_invoke_transform_normalizes_system_role_message_into_system():
    """Bedrock Invoke rejects ``role: "system"`` entries in ``messages`` on some
    Claude aliases; they must be moved into the top-level ``system`` field
    (Anthropic Messages carries that content there)."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "hi"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={"max_tokens": 256, "stream": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert all(m.get("role") != "system" for m in result["messages"])
    assert result["messages"] == [{"role": "user", "content": "hi"}]
    assert result["system"] == [
        {"type": "text", "text": "You are a careful assistant."}
    ]


def test_bedrock_invoke_transform_merges_system_role_into_existing_system():
    """A ``role: "system"`` message is appended to any pre-existing top-level
    ``system`` content rather than replacing it."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "system", "content": "Follow the user's formatting."},
        {"role": "user", "content": "hi"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={
            "max_tokens": 256,
            "stream": False,
            "system": "Base system prompt.",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == [{"role": "user", "content": "hi"}]
    assert result["system"] == [
        {"type": "text", "text": "Base system prompt."},
        {"type": "text", "text": "Follow the user's formatting."},
    ]


def test_bedrock_invoke_transform_merges_list_content_system_role_into_system():
    """A ``role: "system"`` message whose content is a list of content blocks is
    merged block-by-block into the top-level ``system`` field, preserving any
    pre-existing system blocks."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Be terse."},
                {"type": "text", "text": "Cite sources."},
            ],
        },
        {"role": "user", "content": "hi"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={
            "max_tokens": 256,
            "stream": False,
            "system": [{"type": "text", "text": "Base."}],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == [{"role": "user", "content": "hi"}]
    assert result["system"] == [
        {"type": "text", "text": "Base."},
        {"type": "text", "text": "Be terse."},
        {"type": "text", "text": "Cite sources."},
    ]


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-opus-4-8",
        "jp.anthropic.claude-opus-4-8",
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-fable-5",
    ],
)
def test_bedrock_invoke_transform_keeps_mid_conversation_system_role_in_place(local_model_cost_map, model):
    """Regression test for the Bedrock prompt-cache collapse: hoisting a
    mid-conversation ``role: "system"`` message (e.g. Claude Code's
    ``mid-conversation-system-2026-04-07`` reminders) into the top-level
    ``system`` field mutates the cache prefix and invalidates the cached message
    history, so on models flagged ``supports_mid_conversation_system`` (Claude
    4.8+, which Invoke accepts the role on) such entries must be forwarded
    in place. Billing-header blocks must still be stripped from the top-level
    ``system`` field even when nothing is hoisted."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "user", "content": "read the file"},
        {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
        {"role": "assistant", "content": "reading"},
        {"role": "user", "content": "continue"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model=model,
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={
            "max_tokens": 256,
            "stream": False,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.205;"},
                {"type": "text", "text": "Base.", "cache_control": {"type": "ephemeral"}},
            ],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == messages
    assert result["system"] == [
        {"type": "text", "text": "Base.", "cache_control": {"type": "ephemeral"}}
    ]


def test_bedrock_invoke_transform_hoists_only_leading_system_run(local_model_cost_map):
    """On models flagged ``supports_mid_conversation_system``, only the leading
    run of ``role: "system"`` messages is hoisted into the top-level ``system``
    field; a later system entry keeps its position in ``messages`` so the
    serialized prefix stays stable across turns."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "system", "content": "Cite sources."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation reminder"},
        {"role": "user", "content": "continue"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-8",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={"max_tokens": 256, "stream": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation reminder"},
        {"role": "user", "content": "continue"},
    ]
    assert result["system"] == [
        {"type": "text", "text": "You are terse."},
        {"type": "text", "text": "Cite sources."},
    ]


def test_bedrock_invoke_transform_hoists_mid_conversation_system_for_older_claude(local_model_cost_map):
    """Regression test for Claude Code 400s on pre-Opus-4.8 Bedrock models:
    Invoke rejects ``role: "system"`` in every position on Opus 4.7, Sonnet 4.6,
    Haiku 4.5, etc. ("role 'system' is not supported on this model"), so on
    models without ``supports_mid_conversation_system`` every system entry must
    be hoisted into the top-level ``system`` field, mid-conversation ones
    included."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "user", "content": "read the file"},
        {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
        {"role": "assistant", "content": "reading"},
        {"role": "user", "content": "continue"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="us.anthropic.claude-opus-4-7",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={
            "max_tokens": 256,
            "stream": False,
            "system": [{"type": "text", "text": "Base."}],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "reading"},
        {"role": "user", "content": "continue"},
    ]
    assert result["system"] == [
        {"type": "text", "text": "Base."},
        {"type": "text", "text": "[Truncated: PARTIAL view of big1.txt]"},
    ]


def test_bedrock_invoke_transform_hoists_all_system_for_unmapped_model(local_model_cost_map):
    """A model with no cost-map entry and no fallback-generalization rule gets
    the hoist-everything behavior: the safe default is a mutated cache prefix,
    never a provider 400 from forwarding a role the model may not accept."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation reminder"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "continue"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="us.anthropic.claude-opus-3-9",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={"max_tokens": 256, "stream": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "continue"},
    ]
    assert result["system"] == [{"type": "text", "text": "mid-conversation reminder"}]


def test_bedrock_invoke_transform_keeps_system_in_place_for_unmapped_future_claude(local_model_cost_map):
    """An unmapped Bedrock Claude at 4.8 or higher resolves through the
    ``claude-mid-conversation-system`` capability rule, so a future model that
    has not landed in the cost map yet keeps the cache-preserving in-place
    behavior instead of falling back to hoist-all."""
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation reminder"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "continue"},
    ]

    result = cfg.transform_anthropic_messages_request(
        model="us.anthropic.claude-opus-4-9",
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params={"max_tokens": 256, "stream": False},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["messages"] == messages
    assert "system" not in result


def test_bedrock_claude_4_8_plus_cost_map_entries_carry_mid_conversation_system_flag():
    """Exact cost-map hits resolve before fallback-generalization rules, so a
    mapped Bedrock Claude 4.8+ entry without ``supports_mid_conversation_system``
    silently loses the cache-preserving in-place handling that the
    ``claude-mid-conversation-system`` capability rule grants unmapped ids.
    Every mapped bedrock entry the rule's own pattern matches must carry the
    flag explicitly."""
    import re

    import litellm

    cost_map_path = os.path.join(os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json")
    with open(cost_map_path) as f:
        cost_map = json.load(f)
    rules = cost_map["fallback_generalizations"]["rules"]
    pattern = re.compile(
        next(r["pattern"] for r in rules if r["name"] == "claude-mid-conversation-system"),
        re.IGNORECASE,
    )
    missing = [
        key
        for key, info in cost_map.items()
        if isinstance(info, dict)
        and str(info.get("litellm_provider", "")).startswith("bedrock")
        and pattern.search(key)
        and info.get("supports_mid_conversation_system") is not True
    ]
    assert missing == []


def test_as_system_content_blocks_handles_each_shape():
    """``_as_system_content_blocks`` normalizes every system shape: ``None`` -> empty,
    a string -> a single text block, a list -> a shallow copy, and any other value
    (e.g. a bare content-block dict) -> wrapped in a single-element list."""
    block = {"type": "text", "text": "x"}
    assert AmazonAnthropicClaudeMessagesConfig._as_system_content_blocks(None) == []
    assert AmazonAnthropicClaudeMessagesConfig._as_system_content_blocks("hello") == [
        {"type": "text", "text": "hello"}
    ]
    blocks = [block]
    out = AmazonAnthropicClaudeMessagesConfig._as_system_content_blocks(blocks)
    assert out == blocks and out is not blocks
    assert AmazonAnthropicClaudeMessagesConfig._as_system_content_blocks(block) == [
        block
    ]


@pytest.mark.parametrize(
    "budget_tokens,expected_effort",
    [
        (0, "low"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET - 1, "low"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET, "medium"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET - 1, "medium"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET, "high"),
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET - 1, "high"),
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET, "xhigh"),
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET * 2, "xhigh"),
    ],
)
def test_effort_from_thinking_budget_tiers(budget_tokens, expected_effort):
    """The budget -> effort mapping pins each tier boundary so a shifted threshold
    is caught."""
    assert (
        AmazonAnthropicClaudeMessagesConfig._effort_from_thinking_budget(budget_tokens)
        == expected_effort
    )


def test_inject_adaptive_thinking_preserves_existing_effort():
    """When the request already carries an ``output_config.effort`` we keep it
    rather than overwriting with the budget-derived tier."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {"output_config": {"effort": "max", "other": "keep"}}

    cfg._inject_adaptive_thinking_for_clear_thinking(
        request, budget_tokens=24000, model="us.anthropic.claude-fable-5"
    )

    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "max", "other": "keep"}


def test_bedrock_clear_thinking_noops_when_thinking_already_adaptive():
    """An incoming ``thinking.type=adaptive`` is already valid for clear_thinking;
    the helper must leave the request untouched and report no change."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "thinking": {"type": "adaptive"},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model="us.anthropic.claude-fable-5"
    )

    assert changed is False
    assert request["thinking"] == {"type": "adaptive"}
    assert "output_config" not in request


def test_bedrock_clear_thinking_replaces_disabled_thinking_on_adaptive_model():
    """A ``thinking.type=disabled`` (or any non-enabled/adaptive shape) is invalid
    for clear_thinking on Bedrock; it must be replaced with a usable config. On an
    adaptive model that means ``thinking.type=adaptive`` + ``output_config.effort``."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "thinking": {"type": "disabled"},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model="us.anthropic.claude-fable-5"
    )

    assert changed is True
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "low"


def test_bedrock_clear_thinking_leaves_enabled_thinking_on_non_adaptive_model():
    """A caller-supplied ``thinking.type=enabled`` on a non-adaptive extended-thinking
    model (Opus 4.5) is already accepted by Bedrock, so the helper leaves it as-is
    and does not convert it to the adaptive shape."""
    cfg = AmazonAnthropicClaudeMessagesConfig()
    request = {
        "max_tokens": 32000,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
    }

    changed = cfg._ensure_thinking_for_clear_thinking_context_management(
        anthropic_messages_request=request, model="anthropic.claude-opus-4-5-v1:0"
    )

    assert changed is False
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "output_config" not in request


@pytest.fixture
def local_beta_headers_config(monkeypatch):
    from litellm.anthropic_beta_headers_manager import reload_beta_headers_config

    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    reload_beta_headers_config()
    yield
    monkeypatch.delenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", raising=False)
    reload_beta_headers_config()


def test_bedrock_messages_preserves_clear_tool_uses_context_management_and_adds_beta(
    local_beta_headers_config,
):
    """
    LIT-3393: Bedrock InvokeModel supports automatic tool-call clearing via
    ``clear_tool_uses_20250919`` under the ``context-management-2025-06-27``
    beta. Before the LIT-3393 fix, the transformation stripped this edit (only
    ``compact_20260112`` survived) AND the beta was filtered out by
    ``filter_and_transform_beta_headers`` for ``bedrock``, producing a Bedrock
    400 ``"context_management: Extra inputs are not permitted"``.

    Post-fix, the edit must reach the body and the beta must reach
    ``anthropic_beta``.

    AWS docs ("Automatic tool call clearing (Beta)"):
    https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-tool-use.md
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {
            "edits": [{"type": "clear_tool_uses_20250919"}]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result.get("context_management") == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }, "clear_tool_uses_20250919 edit must reach Bedrock InvokeModel body"
    assert "context-management-2025-06-27" in result.get("anthropic_beta", []), (
        "context-management-2025-06-27 beta must reach the InvokeModel body so "
        "the tool-call-clearing edit is accepted"
    )


def test_bedrock_messages_preserves_mixed_compact_and_clear_tool_uses_edits(
    local_beta_headers_config,
):
    """
    LIT-3393: a request mixing ``compact_20260112`` and
    ``clear_tool_uses_20250919`` must keep BOTH edits and emit BOTH
    anthropic-beta values (``compact-2026-01-12`` + ``context-management-2025-06-27``).
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {
            "edits": [
                {"type": "compact_20260112"},
                {"type": "clear_tool_uses_20250919"},
            ]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-6-20250929-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    cm = result.get("context_management")
    assert cm is not None
    edit_types = sorted(e.get("type") for e in cm["edits"])
    assert edit_types == ["clear_tool_uses_20250919", "compact_20260112"]

    betas = result.get("anthropic_beta", [])
    assert "compact-2026-01-12" in betas
    assert "context-management-2025-06-27" in betas


def test_bedrock_messages_filters_clear_thinking_keeps_clear_tool_uses(
    local_beta_headers_config,
):
    """
    LIT-3393: ``clear_thinking_20251015`` remains LiteLLM-internal (consumed via
    thinking-injection) and MUST be stripped from the body, while
    ``clear_tool_uses_20250919`` (officially supported on Bedrock InvokeModel)
    survives in the same request.
    """
    from litellm.types.router import GenericLiteLLMParams

    cfg = AmazonAnthropicClaudeMessagesConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    optional_params = {
        "max_tokens": 4096,
        "context_management": {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": "all"},
                {"type": "clear_tool_uses_20250919"},
            ]
        },
    }

    result = cfg.transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-7",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    cm = result.get("context_management")
    assert cm is not None
    assert [e.get("type") for e in cm["edits"]] == [
        "clear_tool_uses_20250919"
    ], "clear_thinking_20251015 must still be stripped (LiteLLM-internal)"

    betas = result.get("anthropic_beta", [])
    assert "context-management-2025-06-27" in betas
    # ``compact-2026-01-12`` was not requested.
    assert "compact-2026-01-12" not in betas


def test_filter_and_transform_beta_headers_passes_context_management_for_bedrock(
    local_beta_headers_config,
):
    """
    LIT-3393: ``anthropic_beta_headers_config.json`` previously mapped
    ``bedrock.context-management-2025-06-27`` to ``null``, so
    ``filter_and_transform_beta_headers`` dropped the header even when the
    transformation tried to set it. This regression guard locks the bundled
    mapping in place.

    Pinned to the bundled local config via ``LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS``
    so the assertion is not subject to whatever the upstream remote currently
    serves or what previous tests left in the module cache.
    """
    from litellm.anthropic_beta_headers_manager import filter_and_transform_beta_headers

    out = filter_and_transform_beta_headers(
        ["context-management-2025-06-27"],
        provider="bedrock",
    )
    assert out == ["context-management-2025-06-27"]

    # Bedrock_converse genuinely lacks it per AWS docs; this guard prevents
    # an accidental flip there.
    out_converse = filter_and_transform_beta_headers(
        ["context-management-2025-06-27"],
        provider="bedrock_converse",
    )
    assert out_converse == []

