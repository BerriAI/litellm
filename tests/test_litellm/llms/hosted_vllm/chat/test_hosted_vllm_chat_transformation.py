import json
from copy import deepcopy
from typing import Final

import httpx
import respx

import litellm
from unittest.mock import MagicMock, patch

import pytest

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


@pytest.mark.parametrize("params", [{}, {"forward_reasoning_content": False}, {"forward_reasoning_content": True}])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_forward_reasoning_content_preserves_only_explicit_history(params, is_async):
    config = HostedVLLMChatConfig()
    messages = [
        {"role": "user", "content": "Check the counter"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "  synthetic history\n",
            "thinking_blocks": [{"type": "thinking", "thinking": "Do not convert this", "signature": "sig"}],
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "7"},
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "Never synthesize history", "signature": "sig"}],
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "verify", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "verified"},
    ]
    original = deepcopy(messages)
    arguments = dict(
        model="qwen3.8-flash-next", messages=messages, optional_params={}, litellm_params=params, headers={}
    )
    result = await config.async_transform_request(**arguments) if is_async else config.transform_request(**arguments)
    expected = deepcopy(original)
    expected[1].pop("thinking_blocks")
    expected[3].pop("thinking_blocks")
    if params.get("forward_reasoning_content") is not True:
        expected[1].pop("reasoning_content")
    assert result["messages"] == expected
    assert messages == original
    assert "forward_reasoning_content" not in result


@pytest.mark.asyncio
async def test_forward_reasoning_content_reused_config_and_caller_are_isolated():
    config = HostedVLLMChatConfig()
    messages = [{"role": "assistant", "content": "answer", "reasoning_content": "synthetic history"}]
    original = deepcopy(messages)
    for enabled in (False, True, False, True):
        for transform in (config.transform_request, config.async_transform_request):
            result = transform(
                model="qwen3.8-flash-next",
                messages=messages,
                optional_params={},
                litellm_params={"forward_reasoning_content": enabled},
                headers={},
            )
            if transform == config.async_transform_request:
                result = await result
            assert result["messages"] == (original if enabled else [{"role": "assistant", "content": "answer"}])
            assert messages == original


@pytest.mark.asyncio
async def test_forward_reasoning_content_keeps_async_content_conversion():
    class AsyncContentConfig(HostedVLLMChatConfig):
        async def _async_transform_content_item(self, content_item):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"}}

    config = AsyncContentConfig()
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]},
        {"role": "assistant", "content": "answer", "reasoning_content": "synthetic history"},
    ]
    original = deepcopy(messages)
    result = await config.async_transform_request(
        model="qwen3.8-flash-next",
        messages=messages,
        optional_params={},
        litellm_params={"forward_reasoning_content": True},
        headers={},
    )
    assert result["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,c3ludGhldGlj"
    assert result["messages"][1]["reasoning_content"] == "synthetic history"
    assert messages == original


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


def test_hosted_vllm_chat_transformation_with_audio_url():
    from litellm import completion

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.1-70b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.text = json.dumps(mock_response.json.return_value)
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        try:
            completion(
                model="hosted_vllm/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": "https://example.com/audio.mp3"},
                            },
                        ],
                    },
                ],
                api_base="https://test-vllm.example.com/v1",
            )
        except Exception:
            pass

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        request_data = json.loads(call_kwargs["data"])
        assert request_data["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/audio.mp3"},
                    }
                ],
            }
        ]


def test_hosted_vllm_supports_reasoning_effort():
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(
        model="hosted_vllm/gpt-oss-120b"
    )
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
    supported_params = config.get_supported_openai_params(
        model="hosted_vllm/GLM-4.6-FP8"
    )
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


def test_hosted_vllm_thinking_blocks_prepended_to_assistant_content():
    """
    Test that thinking_blocks on assistant messages are removed and content
    stays a string for vLLM compatibility.
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
    assert "reasoning_content" not in assistant_msg


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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["hosted_vllm", "openai"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("via_router", [False, True])
@pytest.mark.parametrize("forward", [None, False, True])
@pytest.mark.parametrize(
    "history, expected",
    [
        ({"reasoning_content": "source"}, "source"),
        ({"reasoning": "target"}, "target"),
        ({"reasoning_content": "same", "reasoning": "same"}, "same"),
        ({"reasoning_content": "source", "reasoning": "target"}, "target"),
        ({"reasoning_content": "source", "reasoning": None}, "source"),
        ({"reasoning_content": "source", "reasoning": ""}, ""),
        ({"reasoning_content": "", "reasoning": None}, ""),
        ({"reasoning_content": None, "reasoning": None}, None),
        ({}, None),
    ],
)
async def test_reasoning_field_sdk_router_final_wire(
    provider: str,
    is_async: bool,
    via_router: bool,
    forward: bool | None,
    history: dict[str, str | None],
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    messages: Final = [
        {"role": "user", "content": "Check both records"},
        *[
            message
            for index in (1, 2)
            for message in (
                {
                    "role": "assistant",
                    "content": f"Checking {index}",
                    **history,
                    "tool_calls": [
                        {"id": f"call_{index}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": f"call_{index}", "content": f"record {index}"},
            )
        ],
    ]
    original: Final = deepcopy(messages)
    base_params: Final = {
        "model": f"{provider}/reasoning-test",
        "api_base": "https://history-field.invalid/v1",
        "api_key": "test-key",
        **({} if forward is None else {"forward_reasoning_content": forward}),
    }
    router: Final = litellm.Router(
        model_list=[
            {"model_name": alias, "litellm_params": {**base_params, **params}}
            for alias, params in (
                ("legacy", {}),
                ("normalized", {"reasoning_content_field": "reasoning"}),
                ("explicit-default", {"reasoning_content_field": "reasoning_content"}),
            )
        ],
        num_retries=0,
    )
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post("https://history-field.invalid/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-history",
                "object": "chat.completion",
                "created": 1,
                "model": "reasoning-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )
        for alias, field in (("normalized", "reasoning"), ("legacy", None), ("explicit-default", "reasoning_content")):
            kwargs: Final = (
                {"model": alias, "messages": messages}
                if via_router
                else {
                    **base_params,
                    "messages": messages,
                    **({} if field is None else {"reasoning_content_field": field}),
                }
            )
            client: Final = router if via_router else litellm
            response: Final = await client.acompletion(**kwargs) if is_async else client.completion(**kwargs)
            assert response.choices[0].message.content == "Done"
            payload: Final = json.loads(route.calls[-1].request.content)
            forwarded: Final = provider == "openai" or forward is True
            expected_history: Final = (
                ({"reasoning": expected} if expected is not None and forwarded else {})
                if field == "reasoning"
                else {
                    key: value
                    for key, value in history.items()
                    if value is not None and (forwarded or key != "reasoning_content")
                }
            )
            assert payload["messages"] == [
                (
                    {**{key: value for key, value in message.items() if key not in history}, **expected_history}
                    if message["role"] == "assistant"
                    else message
                )
                for message in original
            ]
            assert "reasoning_content_field" not in payload
            assert "forward_reasoning_content" not in payload
            assert messages == original
        assert route.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("provider", ["deepinfra", "together_ai", None])
async def test_reasoning_field_does_not_apply_to_inherited_provider(provider: str | None, is_async: bool):
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    config: Final = OpenAIGPTConfig()
    messages: Final = [{"role": "assistant", "content": "Done", "reasoning_content": "source", "reasoning": "target"}]
    original: Final = deepcopy(messages)
    kwargs: Final = {
        "model": "reasoning-test",
        "messages": messages,
        "optional_params": {},
        "headers": {},
        "litellm_params": {"custom_llm_provider": provider, "reasoning_content_field": "reasoning"},
    }
    result: Final = await config.async_transform_request(**kwargs) if is_async else config.transform_request(**kwargs)
    assert result["messages"] == original
    assert messages == original
