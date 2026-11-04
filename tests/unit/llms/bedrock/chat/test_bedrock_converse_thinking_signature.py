"""
Unit tests verifying that Bedrock Converse does not replay reasoningText.signature
to non-Anthropic models (e.g. GLM 5.3, Nova), and that Bedrock unsupported signature
errors are recognized by is_anthropic_invalid_thinking_block_error.

Closes #49969: [Bug]: Bedrock Converse replays reasoningText.signature to non-Anthropic models (GLM 5.3) causing Bad Request
"""

import json
import pytest

from litellm.llms.anthropic.common_utils import is_anthropic_invalid_thinking_block_error
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


def test_bedrock_converse_preserves_signature_for_claude():
    """Verify that Claude models on Bedrock preserve reasoningText.signature."""
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "pondering the question",
                    "signature": "SIG_FROM_HAIKU",
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    body = AmazonConverseConfig().transform_request(
        model="global.anthropic.claude-haiku-5-5",
        messages=msgs,
        optional_params={},
        litellm_params={},
        headers={},
    )
    dumped = json.dumps(body)
    assert "SIG_FROM_HAIKU" in dumped
    # Verify signature exists in reasoningContent
    messages = body.get("messages", [])
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    assert assistant_msg is not None
    has_sig = any(
        isinstance(part, dict)
        and "reasoningContent" in part
        and "reasoningText" in part["reasoningContent"]
        and part["reasoningContent"]["reasoningText"].get("signature") == "SIG_FROM_HAIKU"
        for part in assistant_msg.get("content", [])
    )
    assert has_sig is True


def test_bedrock_converse_drops_signature_for_non_anthropic_models():
    """Verify that non-Anthropic models on Bedrock (e.g. GLM 5.3) do not receive signature."""
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "pondering the question",
                    "signature": "SIG_FROM_HAIKU",
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    body = AmazonConverseConfig().transform_request(
        model="global.zai.glm-5.3",
        messages=msgs,
        optional_params={},
        litellm_params={},
        headers={},
    )
    dumped = json.dumps(body)
    assert "SIG_FROM_HAIKU" not in dumped

    # Verify thinking text was preserved as regular content text rather than dropped
    messages = body.get("messages", [])
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    assert assistant_msg is not None
    assert any(
        isinstance(part, dict) and part.get("text") == "pondering the question"
        for part in assistant_msg.get("content", [])
    )


def test_is_anthropic_invalid_thinking_block_error_bedrock_unsupported_signature():
    """Verify error matcher detects Bedrock's unsupported signature error string for retry/strip recovery."""
    raw_error = (
        '{"message":"This model doesn\'t support the reasoningContent.reasoningText.signature field. '
        'Remove reasoningContent.reasoningText.signature and try again."}'
    )
    assert is_anthropic_invalid_thinking_block_error(raw_error) is True
