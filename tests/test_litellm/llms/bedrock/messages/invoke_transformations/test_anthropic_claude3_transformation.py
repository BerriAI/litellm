import asyncio
import json
import os
import sys

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaude3MessagesConfig,
    AmazonAnthropicClaudeMessagesStreamDecoder,
)


@pytest.mark.asyncio
async def test_bedrock_sse_wrapper_encodes_dict_chunks():
    """Verify that `bedrock_sse_wrapper` converts dictionary chunks to properly formatted Server-Sent Events and forwards non-dict chunks unchanged."""

    cfg = AmazonAnthropicClaude3MessagesConfig()

    async def _dummy_stream():  # type: ignore[return-type]
        yield {"type": "message_delta", "text": "hello"}
        yield b"raw-bytes"

    # Collect all chunks returned by the wrapper
    collected: list[bytes] = []
    async for chunk in cfg.bedrock_sse_wrapper(_dummy_stream()):
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
