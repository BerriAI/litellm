from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.runner import assert_translation

GPT_6_1_SOL_TOOL_SEARCH_TEST_CASE: Final = TranslationTestCase(
    scenario="tool_search",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/responses/global.openai.gpt-6.1-sol",
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
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Load the web search tool."}],
            },
            {
                "type": "function_call",
                "call_id": "call_tool_search_1",
                "name": "ToolSearch",
                "arguments": '{"query": "select:WebSearch"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_tool_search_1",
                "output": [{"type": "input_text", "text": "WebSearch"}],
            },
        ],
        "model": "global.openai.gpt-6.1-sol",
        "reasoning": {"effort": "low"},
        "tools": [
            {
                "type": "function",
                "name": "ToolSearch",
                "description": "Load deferred tools by name.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "strict": None,
            },
            {
                "type": "function",
                "name": "WebSearch",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "strict": None,
            },
        ],
    },
    mock_provider_response={
        "id": "resp_0123456789abcdef",
        "object": "response",
        "created_at": 1791510000,
        "status": "completed",
        "model": "global.openai.gpt-6.1-sol",
        "output": [
            {
                "type": "message",
                "id": "msg_0123456789abcdef",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The web search tool is loaded.", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 15,
            "total_tokens": 115,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    },
    expected_litellm_response={
        "id": "resp_0123456789abcdef",
        "type": "message",
        "role": "assistant",
        "model": "bedrock/responses/global.openai.gpt-6.1-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 15},
        "content": [{"type": "text", "text": "The web search tool is loaded."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


@pytest.mark.parametrize("case", [GPT_6_1_SOL_TOOL_SEARCH_TEST_CASE], ids=lambda case: case.id)
def test_messages_tool_search_bedrock_responses(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
