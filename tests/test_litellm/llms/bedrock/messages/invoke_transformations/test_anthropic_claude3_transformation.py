import asyncio
import json
import os
import sys
from datetime import datetime

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.bedrock.common_utils import remove_custom_field_from_tools
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
            messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
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
                        "cache_control": {
                            "type": "ephemeral",
                            "ttl": "1h"
                        }
                    }
                ]
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
                        "cache_control": {
                            "type": "ephemeral",
                            "ttl": "1h"
                        }
                    },
                    {
                        "type": "text",
                        "text": "World",
                        "cache_control": {
                            "type": "ephemeral",
                            "ttl": "2h"
                        }
                    }
                ]
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
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    }
                ]
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
