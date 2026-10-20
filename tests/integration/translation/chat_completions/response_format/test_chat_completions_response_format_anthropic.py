from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_translation

PLACE_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": True,
}
PLACE_PROMPT: Final = "Which city is the capital of France? Answer as JSON with city and country."
PLACE_TOOL_USE: Final = {
    "type": "tool_use",
    "id": "toolu_01Sr1iptyZnsy1eNxabMZejF",
    "name": "json_tool_call",
    "input": {"city": "Paris", "country": "France"},
    "caller": {"type": "direct"},
}

CLAUDE_HAIKU_5_5_JSON_SCHEMA_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="json_schema",
    litellm_request={
        "model": "anthropic/claude-haiku-5-5",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": PLACE_PROMPT}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "place", "schema": PLACE_SCHEMA}},
        "cache": {"no-cache": True},
    },
    expected_provider_request={
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": PLACE_PROMPT}]}],
        "max_tokens": 128,
        "tool_choice": {"name": "json_tool_call", "type": "tool"},
        "tools": [{"name": "json_tool_call", "input_schema": PLACE_SCHEMA}],
    },
    mock_provider_response={
        "model": "claude-haiku-5-5",
        "id": "msg_011CgCoP2UR3vcM5ZWmEBd8D",
        "type": "message",
        "role": "assistant",
        "content": [PLACE_TOOL_USE],
        "container": None,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 528,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 55,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "object": "chat.completion",
        "model": "anthropic/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": '{"city": "Paris", "country": "France"}', "role": "assistant"},
            }
        ],
        "usage": {
            "completion_tokens": 55,
            "prompt_tokens": 528,
            "total_tokens": 583,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 55},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 528,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
            "speed": "standard",
        },
    },
)

CLAUDE_HAIKU_5_5_JSON_SCHEMA_TOOL_NAMED_ENVELOPE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_JSON_SCHEMA_TEST_CASE,
    scenario="json_schema_tool_named_envelope",
    mock_provider_response={
        **CLAUDE_HAIKU_5_5_JSON_SCHEMA_TEST_CASE.mock_provider_response,
        "content": [
            {
                **PLACE_TOOL_USE,
                "input": {"name": "json_tool_call", "arguments": {"input": {"city": "Paris", "country": "France"}}},
            }
        ],
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_JSON_SCHEMA_TEST_CASE, CLAUDE_HAIKU_5_5_JSON_SCHEMA_TOOL_NAMED_ENVELOPE_TEST_CASE],
    ids=lambda case: case.id,
)
def test_chat_completions_response_format_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
