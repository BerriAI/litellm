"""
LIT-2834: tool call / tool result messages must not be passed to the Bedrock
guardrail INPUT payload by default. Without this filter, a RAG tool result
containing PII-shaped data (e.g. an "age" field) trips the pre-guard with a
false positive.

NOTE: this file lives alongside test_bedrock_guardrails.py and is auto-
discovered by pytest. It was split out from the main file purely as a
mechanical artifact of how the patch was delivered (the parent test file
exceeded the size that the upload tool could ship in a single shot). It can
be merged back into test_bedrock_guardrails.py at the maintainer's
discretion with no semantic change.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
)


def _bedrock_input_texts(call):
    """Pull the list of text strings sent in a make_bedrock_api_request call."""
    bg = call.kwargs.get("messages") or []
    out = []
    for m in bg:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    out.append(item["text"])
    return out


@pytest.mark.asyncio
async def test_bedrock_pre_call_hook_drops_tool_messages_by_default():
    """LIT-2834: pre-call guardrail must not receive role=tool results or
    assistant messages with tool_calls when mask_tool_call_messages defaults
    to True.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="lit2834",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook="pre_call",
        default_on=True,
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Look up Alice"},
            {
                "role": "assistant",
                "content": "Sure, let me look.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"name": "Alice"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps({"name": "Alice", "age": 34}),
            },
            {"role": "user", "content": "What team is she on?"},
        ],
    }

    with patch.object(
        guardrail,
        "make_bedrock_api_request",
        new=AsyncMock(
            return_value={"action": "NONE", "output": [], "outputs": [], "assessments": []}
        ),
    ) as mock_make:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    assert mock_make.await_count == 1
    texts = _bedrock_input_texts(mock_make.call_args)
    # tool-result content (with the "age" field) and the assistant tool-call
    # preamble must not appear in the scanned payload.
    assert all('"age"' not in t for t in texts), texts
    assert "Sure, let me look." not in texts, texts
    # user/system messages still flow through.
    assert "Look up Alice" in texts
    assert "What team is she on?" in texts
    assert "sys" in texts


@pytest.mark.asyncio
async def test_bedrock_pre_call_hook_includes_tool_messages_when_opt_out():
    """Legacy behavior must remain available via mask_tool_call_messages=False."""
    guardrail = BedrockGuardrail(
        guardrail_name="lit2834-opt-out",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook="pre_call",
        default_on=True,
        mask_tool_call_messages=False,
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "u"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps({"name": "Alice", "age": 34}),
            },
        ],
    }

    with patch.object(
        guardrail,
        "make_bedrock_api_request",
        new=AsyncMock(
            return_value={"action": "NONE", "output": [], "outputs": [], "assessments": []}
        ),
    ) as mock_make:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    assert mock_make.await_count == 1
    texts = _bedrock_input_texts(mock_make.call_args)
    # Tool result text is still forwarded (legacy behavior).
    assert any('"age"' in t for t in texts), texts


@pytest.mark.asyncio
async def test_bedrock_pre_call_hook_preserves_plain_assistant_text():
    """An assistant message with text but no tool_calls must still be scanned."""
    guardrail = BedrockGuardrail(
        guardrail_name="lit2834-asst-text",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook="pre_call",
        default_on=True,
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "hi how can i help"},
            {"role": "user", "content": "u2"},
        ],
    }

    with patch.object(
        guardrail,
        "make_bedrock_api_request",
        new=AsyncMock(
            return_value={"action": "NONE", "output": [], "outputs": [], "assessments": []}
        ),
    ) as mock_make:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    texts = _bedrock_input_texts(mock_make.call_args)
    assert "hi how can i help" in texts


@pytest.mark.asyncio
async def test_bedrock_moderation_hook_drops_tool_messages_by_default():
    """During-call (async_moderation_hook) goes through the same chokepoint
    and must also drop tool messages by default."""
    guardrail = BedrockGuardrail(
        guardrail_name="lit2834-during",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook="during_call",
        default_on=True,
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "u"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps({"name": "Alice", "age": 34}),
            },
        ],
    }

    with patch.object(
        guardrail,
        "make_bedrock_api_request",
        new=AsyncMock(
            return_value={"action": "NONE", "output": [], "outputs": [], "assessments": []}
        ),
    ) as mock_make:
        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            call_type="completion",
        )

    assert mock_make.await_count == 1
    texts = _bedrock_input_texts(mock_make.call_args)
    assert all('"age"' not in t for t in texts), texts
    assert "u" in texts


def test_is_tool_related_message_classifier():
    """The static helper used by the prepare-payload chokepoint."""
    assert BedrockGuardrail._is_tool_related_message(
        {"role": "tool", "content": "x", "tool_call_id": "c1"}
    )
    assert BedrockGuardrail._is_tool_related_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "n", "arguments": "{}"}}
            ],
        }
    )
    # assistant text alone is NOT a tool message
    assert not BedrockGuardrail._is_tool_related_message(
        {"role": "assistant", "content": "hi"}
    )
    # user is never a tool message
    assert not BedrockGuardrail._is_tool_related_message(
        {"role": "user", "content": "hi"}
    )
    # system is never a tool message
    assert not BedrockGuardrail._is_tool_related_message(
        {"role": "system", "content": "sys"}
    )
