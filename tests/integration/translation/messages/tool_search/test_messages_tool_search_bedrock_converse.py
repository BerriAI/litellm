from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.runner import assert_translation

GPT_6_1_SOL_TOOL_SEARCH_TEST_CASE: Final = TranslationTestCase(
    scenario="tool_search",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/global.openai.gpt-6.1-sol",
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
    expected_provider_endpoint="/model/global.openai.gpt-6.1-sol/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "inferenceConfig": {"maxTokens": 1024},
        "messages": [
            {"role": "user", "content": [{"text": "Load the web search tool."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "select:WebSearch"},
                            "name": "ToolSearch",
                            "toolUseId": "call_tool_search_1",
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [{"text": "WebSearch"}],
                            "toolUseId": "call_tool_search_1",
                        }
                    }
                ],
            },
        ],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "description": "Load deferred tools by name.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "ToolSearch",
                    }
                },
                {
                    "toolSpec": {
                        "description": "Search the web.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "WebSearch",
                    }
                },
            ]
        },
    },
    mock_provider_response={
        "metrics": {"latencyMs": 1252},
        "output": {
            "message": {
                "content": [{"text": "The web search tool is loaded and ready to use."}],
                "role": "assistant",
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 100,
            "outputTokens": 15,
            "serverToolUsage": {},
            "totalTokens": 115,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/global.openai.gpt-6.1-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 15},
        "content": [{"type": "text", "text": "The web search tool is loaded and ready to use."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


@pytest.mark.parametrize("case", [GPT_6_1_SOL_TOOL_SEARCH_TEST_CASE], ids=lambda case: case.id)
def test_messages_tool_search_bedrock_converse(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
