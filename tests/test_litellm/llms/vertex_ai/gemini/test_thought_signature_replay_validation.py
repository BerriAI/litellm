"""
Regression tests for corrupted Gemini thought signatures surviving replay.

When LiteLLM reconstructs a `/v1/responses` conversation from stored response
history, a malformed `thinking_blocks[].signature` or provider field
`thought_signatures` was forwarded verbatim to Vertex AI, which rejected the
request with a TYPE_BYTES Base64 decoding 400. The replay now drops signatures
that are not decodable Base64 before building the Gemini parts.
"""

import pytest

from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
    _is_valid_thought_signature,
)

VALID_SIG = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"
CORRUPTED_SIG = "not!valid!!signature@@"


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        (VALID_SIG, True),
        ("aGVsbG8=", True),
        ("aGVsbG8", True),  # tolerated missing padding
        ("not!valid!!", False),  # invalid characters
        ("!!!!", False),
        ("", False),
        (None, False),
        ("wqk=", True),  # short valid padded signature
    ],
)
def test_is_valid_thought_signature(signature, expected):
    assert _is_valid_thought_signature(signature) is expected


@pytest.mark.parametrize("signature", [CORRUPTED_SIG, "bad sig!", "###"])
def test_replay_drops_invalid_thinking_block_signature(signature):
    """A thinking block whose signature is not Base64 must not reach Vertex as
    a PartType with a thoughtSignature: that part would be rejected."""
    messages = [
        {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [{"type": "thinking", "thinking": "reasoning...", "signature": signature}],
        },
    ]
    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3.7-flash", custom_llm_provider="vertex_ai"
    )

    model_parts = [p for c in contents for p in c.get("parts", []) if c.get("role") == "model"]
    assert all(p.get("thoughtSignature") is None for p in model_parts), (
        f"corrupted signature must not be forwarded, got {model_parts}"
    )
    assert any(p.get("text") == "hello" for p in model_parts), "content part must survive"


def test_replay_keeps_valid_thinking_block_signature():
    """A valid signature must still be replayed byte-for-byte."""
    messages = [
        {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [{"type": "thinking", "thinking": "reasoning...", "signature": VALID_SIG}],
        },
    ]
    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3.7-flash", custom_llm_provider="vertex_ai"
    )

    model_parts = [p for c in contents for p in c.get("parts", []) if c.get("role") == "model"]
    assert any(p.get("thoughtSignature") == VALID_SIG for p in model_parts)


def test_replay_drops_invalid_text_part_thought_signature():
    """A corrupted provider_specific_fields.thought_signatures entry must not be
    attached to the replayed text part."""
    messages = [
        {
            "role": "assistant",
            "content": "answer text",
            "provider_specific_fields": {"thought_signatures": [CORRUPTED_SIG]},
        },
    ]
    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3.7-flash", custom_llm_provider="vertex_ai"
    )

    model_parts = [p for c in contents for p in c.get("parts", []) if c.get("role") == "model"]
    assert all(p.get("thoughtSignature") is None for p in model_parts)
    assert any(p.get("text") == "answer text" for p in model_parts)
