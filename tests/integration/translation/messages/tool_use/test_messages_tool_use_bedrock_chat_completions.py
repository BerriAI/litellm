import json
import re
from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.bedrock_chat_completions import GROK_4_7_TEST_CASE
from integration.translation.runner import assert_translation

GROK_4_7_TOOL_USE_TEST_CASE: Final = replace(
    GROK_4_7_TEST_CASE,
    scenario="tool_use",
    litellm_request={
        "model": "bedrock/global.xai.grok-4.7",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "Read",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "messages": [{"role": "user", "content": "Read notes.txt."}],
        "cache": {"no-cache": True},
    },
    expected_provider_request={
        "model": "global.xai.grok-4.7",
        "messages": [{"role": "user", "content": "Read notes.txt."}],
        "max_completion_tokens": 1024,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                    "description": "Read a file.",
                },
            }
        ],
    },
    mock_provider_response={
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "annotations": [],
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"file_path":"notes.txt"}'},
                        }
                    ],
                },
            }
        ],
        "created": 1791509000,
        "id": "chatcmpl-toolusegrok47",
        "model": "global.xai.grok-4.7",
        "object": "chat.completion",
        "usage": {"completion_tokens": 20, "prompt_tokens": 40, "total_tokens": 60},
    },
    expected_litellm_response={
        **GROK_4_7_TEST_CASE.expected_litellm_response,
        "id": "chatcmpl-toolusegrok47",
        "usage": {"input_tokens": 40, "output_tokens": 20},
        "content": [{"type": "tool_use", "id": ANY, "name": "Read", "input": {"file_path": "notes.txt"}}],
        "stop_reason": "tool_use",
    },
)


@pytest.mark.parametrize("case", [GROK_4_7_TOOL_USE_TEST_CASE], ids=lambda case: case.id)
def test_messages_tool_use_bedrock_chat_completions(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def test_messages_tool_use_ids_are_unique_across_bedrock_chat_completions_turns(
    gateway: Gateway, provider: SharedProvider
) -> None:
    case: Final = GROK_4_7_TOOL_USE_TEST_CASE
    reply: Final = Reply(body=json.dumps(case.mock_provider_response).encode())
    provider.expect(reply, reply)
    responses: Final = tuple(gateway.request("POST", case.litellm_endpoint, case.litellm_request) for _ in range(2))
    assert len(provider.received()) == 2
    tool_use_ids: Final = tuple(response.json()["content"][0]["id"] for response in responses)
    assert len(set(tool_use_ids)) == 2
    assert all(re.fullmatch(r"call_[0-9a-f]{32}", tool_use_id) for tool_use_id in tool_use_ids)
