from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.bedrock_chat_completions import GROK_4_7_TEST_CASE
from integration.translation.runner import assert_translation

GROK_4_7_TOOL_SEARCH_TEST_CASE: Final = replace(
    GROK_4_7_TEST_CASE,
    scenario="tool_search",
    litellm_request={
        "model": "bedrock/global.xai.grok-4.7",
        "max_tokens": 1024,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
        "tools": [
            {
                "name": "ToolSearch",
                "description": "Load deferred tools by name.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
            {
                "name": "WebSearch",
                "description": "Search the web.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        ],
        "messages": [
            {"role": "user", "content": "Load the web search tool."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_tool_search_1",
                        "name": "ToolSearch",
                        "input": {"query": "select:WebSearch"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_tool_search_1",
                        "content": [{"type": "tool_reference", "tool_name": "WebSearch"}],
                    }
                ],
            },
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_request={
        "model": "global.xai.grok-4.7",
        "messages": [
            {"role": "user", "content": "Load the web search tool."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_tool_search_1",
                        "type": "function",
                        "function": {"name": "ToolSearch", "arguments": '{"query": "select:WebSearch"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_tool_search_1", "content": ""},
        ],
        "max_completion_tokens": 1024,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ToolSearch",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "description": "Load deferred tools by name.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "WebSearch",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "description": "Search the web.",
                },
            },
        ],
        "reasoning_effort": "low",
    },
    mock_provider_response={
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "annotations": [],
                    "content": "The web search tool is now loaded and available for use.",
                    "refusal": None,
                    "role": "assistant",
                },
            }
        ],
        "created": 1791505405,
        "id": "chatcmpl-xevjeh2pxs6gjvcn2yz76piwdc4pi6fb2sqwb7psnwitm5higvvq",
        "model": "global.xai.grok-4.7",
        "object": "chat.completion",
        "service_tier": "default",
        "usage": {
            "completion_tokens": 76,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 57,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens": 162,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 238,
        },
        "system_fingerprint": None,
    },
    expected_litellm_response={
        **GROK_4_7_TEST_CASE.expected_litellm_response,
        "id": "chatcmpl-xevjeh2pxs6gjvcn2yz76piwdc4pi6fb2sqwb7psnwitm5higvvq",
        "usage": {"input_tokens": 162, "output_tokens": 76},
        "content": [{"type": "text", "text": "The web search tool is now loaded and available for use."}],
    },
)


@pytest.mark.parametrize("case", [GROK_4_7_TOOL_SEARCH_TEST_CASE], ids=lambda case: case.id)
def test_messages_tool_search_bedrock_chat_completions(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
