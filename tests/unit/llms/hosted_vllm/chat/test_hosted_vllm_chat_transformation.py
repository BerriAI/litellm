import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse


def _chat_response(config: OpenAIGPTConfig, choice_fields: Sequence[Mapping[str, JsonValue]]) -> ModelResponse:
    return config.transform_response(
        model="test-vllm-model",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1735689600,
                "model": "test-vllm-model",
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": f"Answer {index}"},
                        "finish_reason": "stop",
                        **fields,
                    }
                    for index, fields in enumerate(choice_fields)
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ),
        model_response=ModelResponse(),
        logging_obj=Logging(
            model="test-vllm-model",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            litellm_call_id="vllm-choice-fields-test",
            function_id="vllm-choice-fields-test",
        ),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _serialized_choices(response: ModelResponse) -> tuple[Mapping[str, JsonValue], ...]:
    serialized: Final = TypeAdapter(dict[str, JsonValue]).validate_json(response.model_dump_json())
    return TypeAdapter(tuple[Mapping[str, JsonValue], ...]).validate_python(serialized["choices"])


@pytest.mark.parametrize(
    "fields",
    (
        {"stop_reason": "</s>", "token_ids": [42, 100]},
        {"stop_reason": 0, "token_ids": []},
        {"stop_reason": None, "token_ids": None},
        {"stop_reason": "end"},
        {"token_ids": [0]},
        {},
    ),
)
def test_hosted_vllm_chat_preserves_choice_fields(fields: Mapping[str, JsonValue]) -> None:
    response: Final = _chat_response(HostedVLLMChatConfig(), ({**fields, "vendor_marker": "keep"},))
    choice: Final = _serialized_choices(response)[0]

    assert choice == {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "Answer 0",
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "provider_specific_fields": None,
        },
        "provider_specific_fields": {**fields, "vendor_marker": "keep"},
        **fields,
    }


def test_hosted_vllm_chat_keeps_fields_with_their_choice() -> None:
    response: Final = _chat_response(
        HostedVLLMChatConfig(),
        (
            {"stop_reason": "first", "token_ids": [1, 2]},
            {"stop_reason": 0, "token_ids": []},
        ),
    )
    choices: Final = _serialized_choices(response)

    assert len(choices) == 2
    assert choices[0]["stop_reason"] == "first"
    assert choices[0]["token_ids"] == [1, 2]
    assert choices[0]["provider_specific_fields"] == {"stop_reason": "first", "token_ids": [1, 2]}
    assert choices[1]["stop_reason"] == 0
    assert choices[1]["token_ids"] == []
    assert choices[1]["provider_specific_fields"] == {"stop_reason": 0, "token_ids": []}


def test_openai_chat_keeps_vllm_fields_nested() -> None:
    response: Final = _chat_response(OpenAIGPTConfig(), ({"stop_reason": "end", "token_ids": [7]},))
    choice: Final = _serialized_choices(response)[0]

    assert "stop_reason" not in choice
    assert "token_ids" not in choice
    assert choice["provider_specific_fields"] == {"stop_reason": "end", "token_ids": [7]}


def test_hosted_vllm_stream_keeps_openai_chunk_serialization() -> None:
    chunk: Final = json.dumps(
        {
            "id": "chatcmpl-stream-test",
            "created": 1735689600,
            "model": "test-vllm-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Answer"},
                    "finish_reason": "stop",
                    "stop_reason": 0,
                    "token_ids": [7],
                }
            ],
        }
    )
    hosted_stream: Final = HostedVLLMChatConfig().get_model_response_iterator(
        iter((f"data: {chunk}",)), sync_stream=True
    )
    openai_stream: Final = OpenAIGPTConfig().get_model_response_iterator(iter((f"data: {chunk}",)), sync_stream=True)

    assert isinstance(hosted_stream, BaseModelResponseIterator)
    assert isinstance(openai_stream, BaseModelResponseIterator)
    assert next(hosted_stream).model_dump() == next(openai_stream).model_dump()


def test_hosted_vllm_chat_transformation_file_url():
    config = HostedVLLMChatConfig()
    video_url = "https://example.com/video.mp4"
    video_data = f"data:video/mp4;base64,{video_url}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": video_data,
                    },
                }
            ],
        }
    ]
    transformed_response = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_response["messages"] == [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": video_data}}],
        }
    ]


def test_hosted_vllm_supports_reasoning_effort():
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/gpt-oss-120b")
    assert "reasoning_effort" in supported_params
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"


def test_hosted_vllm_supports_thinking():
    """
    Test that hosted_vllm supports the 'thinking' parameter.

    Anthropic-style thinking is converted to OpenAI-style reasoning_effort
    since vLLM is OpenAI-compatible.

    Related issue: https://github.com/BerriAI/litellm/issues/19761
    """
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/GLM-4.6-FP8")
    assert "thinking" in supported_params

    # Test thinking below the low threshold -> "minimal"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET - 1,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert "thinking" not in optional_params  # thinking should NOT be passed
    assert optional_params["reasoning_effort"] == "minimal"

    # Test thinking with high budget_tokens -> "high"
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            }
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"

    # Test that existing reasoning_effort is not overwritten
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            },
            "reasoning_effort": "low",
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "low"


def test_hosted_vllm_reasoning_content_kept_and_thinking_blocks_removed():
    """
    Test that reasoning_content on assistant messages is forwarded to vLLM
    while thinking_blocks are removed and content stays a string.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Here is my answer.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this...",
                    "signature": "abc123",
                }
            ],
            "reasoning_content": "Let me reason about this...",
        },
        {
            "role": "user",
            "content": "Follow up question",
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], str)
    assert assistant_msg["content"] == "Here is my answer."
    assert "thinking_blocks" not in assistant_msg
    assert assistant_msg["reasoning_content"] == "Let me reason about this..."


@pytest.mark.parametrize(
    ("reasoning_content", "expected"),
    [
        ("step one, then step two", "step one, then step two"),
        ("", ""),
        (None, "absent"),
        (42, "absent"),
        (["step one", "step two"], "absent"),
        ({"text": "step one"}, "absent"),
    ],
)
def test_hosted_vllm_forwards_only_string_reasoning_content(reasoning_content, expected):
    config = HostedVLLMChatConfig()
    transformed = config.transform_request(
        model="hosted_vllm/qwen3",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": reasoning_content},
            {"role": "user", "content": "Again"},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][1]
    assert assistant_msg.get("reasoning_content", "absent") == expected
    assert assistant_msg["content"] == "Hi"


def test_hosted_vllm_thinking_blocks_with_list_content():
    """
    Test thinking_blocks are removed and assistant content list is converted
    to a string.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Response text"}],
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Step 1 reasoning",
                    "signature": "sig1",
                },
                {
                    "type": "thinking",
                    "thinking": "Step 2 reasoning",
                    "signature": "sig2",
                },
            ],
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][0]
    assert isinstance(assistant_msg["content"], str)
    assert assistant_msg["content"] == "Response text"
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_assistant_structured_content_is_preserved():
    config = HostedVLLMChatConfig()
    image_block = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png"},
    }
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here is the image"}, image_block],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == [
        {"type": "text", "text": "Here is the image"},
        image_block,
    ]


def test_hosted_vllm_assistant_tool_use_content_becomes_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_assistant_tool_use_does_not_duplicate_existing_tool_calls():
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Boston"},
                }
            ],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Boston"}),
                    },
                }
            ],
        },
    ]

    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assistant_msg = transformed["messages"][0]
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Boston"}),
            },
        }
    ]


def test_hosted_vllm_custom_tools_are_converted_to_function_tools():
    config = HostedVLLMChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "apply_patch",
                        "description": "Apply text patch",
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": "lark", "definition": "start: /.*/"},
                        },
                    },
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "apply_patch"
    assert tools[0]["function"]["description"] == "Apply text patch"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "input" in tools[0]["function"]["parameters"]["properties"]


def test_hosted_vllm_custom_tools_use_top_level_input_schema():
    config = HostedVLLMChatConfig()
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "custom",
                    "name": "search",
                    "description": "Search docs",
                    "input_schema": input_schema,
                }
            ]
        },
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["description"] == "Search docs"
    assert tools[0]["function"]["parameters"] == input_schema
