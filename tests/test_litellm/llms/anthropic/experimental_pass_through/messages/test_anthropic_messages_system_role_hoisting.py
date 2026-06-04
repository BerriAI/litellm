"""
Regression tests for #29698 — Anthropic Messages API now accepts
role="system" entries inside the messages[] array, but Bedrock and other
downstream providers still expect system on the top-level field. Verify
that AnthropicMessagesConfig.transform_anthropic_messages_request hoists
any in-messages system entries into the top-level system field before the
adapter forwards downstream.
"""

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _transform(messages, optional_params=None):
    config = AnthropicMessagesConfig()
    optional_params = {**(optional_params or {})}
    optional_params.setdefault("max_tokens", 100)
    return config.transform_anthropic_messages_request(
        model="claude-sonnet-4-8",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )


def test_system_role_in_messages_hoisted_to_top_level_system_field():
    result = _transform(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "you are helpful"},
        ]
    )
    system = result.get("system")
    assert isinstance(system, list)
    assert {"type": "text", "text": "you are helpful"} in system
    assert all(m["role"] != "system" for m in result["messages"])


def test_system_role_merged_after_existing_top_level_string_system():
    result = _transform(
        [
            {"role": "system", "content": "also be concise"},
            {"role": "user", "content": "hi"},
        ],
        optional_params={"system": "you are helpful"},
    )
    system = result["system"]
    assert isinstance(system, list)
    assert system[0] == {"type": "text", "text": "you are helpful"}
    assert system[1] == {"type": "text", "text": "also be concise"}


def test_system_role_merged_after_existing_top_level_list_system():
    result = _transform(
        [
            {"role": "system", "content": "also be concise"},
            {"role": "user", "content": "hi"},
        ],
        optional_params={
            "system": [{"type": "text", "text": "you are helpful"}],
        },
    )
    system = result["system"]
    assert system == [
        {"type": "text", "text": "you are helpful"},
        {"type": "text", "text": "also be concise"},
    ]


def test_system_role_with_list_content_extends_system_blocks():
    result = _transform(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "be terse"},
                    {"type": "text", "text": "use markdown"},
                ],
            },
            {"role": "user", "content": "hi"},
        ]
    )
    system = result["system"]
    assert {"type": "text", "text": "be terse"} in system
    assert {"type": "text", "text": "use markdown"} in system


def test_system_role_empty_or_none_content_dropped_silently():
    result = _transform(
        [
            {"role": "system", "content": None},
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]
    )
    # No new system entries, but the user message survives.
    assert result.get("system") is None
    assert result["messages"] == [{"role": "user", "content": "hi"}]


def test_no_system_role_in_messages_is_a_noop():
    result = _transform(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        optional_params={"system": "you are helpful"},
    )
    # System field is preserved through the existing _filter_billing path,
    # messages are untouched.
    assert "system" in result
    assert len(result["messages"]) == 2
