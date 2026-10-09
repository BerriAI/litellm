from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.bedrock_converse import GROK_4_7_TEST_CASE
from integration.translation.runner import assert_translation

GROK_4_7_TOOL_SEARCH_REDACTED_THINKING: Final = "cnNuX2pRenlYWklGSkxscEt5ZmxZa0NmdndJRmVQRjY3NW45cHMrNmE3bUVGVFd3ZjBFRTUzNXZzWlJadTcrUnZCdVpCYkJ6QUY4QUFRQVZZWGR6TFdOeWVYQjBieTF3ZFdKc2FXTXRhMlY1QUVSQmMwVkVNSGQ1VjJJMFVVTkxSSEZqVkdKbWFrWXdaRkF4T0doUWRYbzNUV3QyU1RCMGVqQjFPV2MzVEdWMWNHWm9hVlI2Y3l0RVdtSnJjblowVFVSallVRTlQUUFCQUJGaGQzTXRhMjF6TFdocFpYSmhjbU5vZVFBS1ltVnliUzF6ZEc5eVpRQmNmOG0wT0NOZUU1bjVVTU5RQ0ZYYTVnanoxRUgxOEg2VU5EaGZDOVYwWXJ0MlowNkxpMXpVK1ZFZFFSUW52TWtSc1ArT3hwUDNRK01zR29lU3QrbWNDdjJCc3hGVFlBNnZ6L0xSbDE0ZjRkK3Z1YWhtQ1I0V1ZQNEFhcVlDQUFBUUFKd2NpNVA3MGFYK2J5UDY2dXgzRVdkTzNMNlh4N3NoRmt0amMyaE96ZWhTQXlqaDRESC9kZ3FtWkIxYXNMQUdKLy8vLy84QUFBQUJBQUFBQUFBQUFBQUFBQUFCQUFBQ3hmQ1pzTnZ5Vk4rUlBXY0JmT1B1OEtUcFhoTmQwT2Jna2hzelRRUFBSb25RMUdlNVJnYzl3VVBrSE51c1dFQVdYRE9OSXJ1cVJManNleGJ4RHFub0Nxam1DWmRGbVhoU1l5Mm1wb1JHek5kdkNWYVFlem42dWVtS01YUmhNZ1JBeWc3YWljcVJINlQ5UUxlMm1DOFJkQVB3RjRYeTl6L2hLWUxpcFdPN2lFYm1kMmZRQk1jNm1mZkIvNkszdGlQQU5zQ2dleTFNME12L0h2UHFidGhLdUtxWGNhYm5CVGY3U1hIM0hNMWhNRE40RDR6WUVFdm5kUVdnZm1MM2NhbWllSHhpeWRtdDBObk5QUVYxT3BFanBIem5nMGVET3ZxQjY3OHU1OEM5VnhTNFdSN0VFVDd5NWpiVmYxOTcrdDYyR2VpSnFSMnVvL2tjcjlRdGtHOUdLZFdoUWlOa3puQ01NdzZibDZzd2RwNm5wZGZ5em81WjNlMDgzY2pTU1hid2FFMlRqOVpkbU5lLzdMeXk1YmM0MEEwMUNzSUpXQ0EwSzAvcjYrcy83RzZURE1BTDdzdGxPclZFcjBvY0lLRzFUdk9kNCtUd0NWNlBsaE1SZ2ltbTZXbTVNR1c3YlpmK1dDVnE2RjhSTC94bXJKb3p3aWRBdnpDS0F2NHFHZ3FtN1FpaksvRGNNYTZwa2YxZk5iV0c4SXpGVm5uT0g2UXBHRk5xSUREd2FWS0FDamZOdmUvSHUxbktJaGxPNW1oMWNFaFluTWdUM3hJZmdsWnBGUUYyUjJYcUdYSUpmZmh1bmQ2b3NNMUNVV28wTjlJYlFVaVBEVy82bXJVWU9SS25IbjkxSTlKcEd3NTFQZlFDTzF6OWtjaml4S05iZDNiMDJ3WndKblRqQk9TbWtKbkwzLy80QzRQZmdKdG5pZERVQ3gxRWtOaFExWDIzZXgzMUpHRkk4ZmNmc2NIYzBBUUd4a21TdWdIU0s5UFBhY0djZUMwTkg4blVIVjNrWHV0TnJ3bHYxSUhOQWdxVmdXVEIybjZQUi9xZDJiamVJbmx4bDl2UUlrRDkwb3JuMVRncjc0Ni9XQWNJVkh4cTlRcloyeWswZDducjlZMlhWUG80YXA0cjU5VGhUMG10WHVZd1orNzVOM0Q4Uzh4VDhVMFRwcWozK2FZN01FVVgzdjJyclp4SG15ejIrRFBhaTVXbW9YVElSMysvYkhXSFFtbndzQ3VVTFUwbS9oWFRKRUJJVlY4OUFvY1ZUVDRFcGJRTFNRNldrSi9YQUdjd1pRSXhBS2dMbDB0dFU3LzFiaXgwQ2wyS21mM3ZTSEp4ZXBsM3ZKSTRCRWtIUTZ6Yk1XYnFhdGw0cmY5QUdMNndxTDBmSUFJd2FoYU9TRlNaSlVXRnNNL1lOano5aWN4T0RGQ3dlcVo4RU1lb3h2bVhCWDFYWXpwVXdYeUV1R3VwUy9iTUNsWkk="

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
                "content": [{"toolResult": {"content": [{"text": ""}], "toolUseId": "call_tool_search_1"}}],
            },
        ],
        "inferenceConfig": {"maxTokens": 1024},
        "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "ToolSearch",
                        "description": "Load deferred tools by name.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "WebSearch",
                        "description": "Search the web.",
                    }
                },
            ]
        },
    },
    mock_provider_response={
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"reasoning_tokens": 225}}},
        "metrics": {"latencyMs": 4972},
        "output": {
            "message": {
                "content": [
                    {"reasoningContent": {"redactedContent": GROK_4_7_TOOL_SEARCH_REDACTED_THINKING}},
                    {"text": "The web search tool has been loaded and is now available for use."},
                ],
                "role": "assistant",
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 128,
            "cacheReadInputTokens": 128,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 34,
            "outputTokens": 247,
            "serverToolUsage": {},
            "totalTokens": 409,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/global.xai.grok-4.7",
        "stop_sequence": None,
        "usage": {"input_tokens": 34, "output_tokens": 247, "cache_read_input_tokens": 128},
        "content": [
            {"type": "redacted_thinking", "data": GROK_4_7_TOOL_SEARCH_REDACTED_THINKING},
            {"type": "text", "text": "The web search tool has been loaded and is now available for use."},
        ],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


@pytest.mark.parametrize("case", [GROK_4_7_TOOL_SEARCH_TEST_CASE], ids=lambda case: case.id)
def test_messages_tool_search_bedrock_converse(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
