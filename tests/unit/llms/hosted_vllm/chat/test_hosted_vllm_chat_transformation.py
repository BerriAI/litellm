import json
from unittest.mock import MagicMock, patch

import pytest


from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


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


def _vllm_chat_client() -> MagicMock:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    body = {
        "id": "chat-cache-salt",
        "object": "chat.completion",
        "created": 1,
        "model": "Qwen/Qwen3-8B",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.headers = {}
    raw_response.text = json.dumps(body)
    raw_response.json = lambda: body
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = raw_response
    return client


def _sent_chat_body(**call_kwargs: object) -> dict:
    import litellm

    client = _vllm_chat_client()
    litellm.completion(
        model="hosted_vllm/Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "hi"}],
        api_base="http://vllm.internal:8000/v1",
        client=client,
        **call_kwargs,
    )
    return json.loads(client.post.call_args.kwargs["data"])


def _salt_for(key_hash: str, **call_kwargs: object) -> str | None:
    body = _sent_chat_body(hosted_vllm_cache_salt=True, metadata={"user_api_key_hash": key_hash}, **call_kwargs)
    assert "hosted_vllm_cache_salt" not in body
    return body.get("cache_salt")


def test_cache_salt_is_per_key_and_stable():
    alice = _salt_for("hash-alice")
    assert alice is not None and len(alice) == 64
    assert alice == _salt_for("hash-alice")  # same key -> same salt -> prefix reuse kept
    assert alice != _salt_for("hash-bob")  # other key -> no shared prefix-cache hits
    assert alice != "hash-alice"


@pytest.mark.parametrize(
    "caller_salt_kwargs",
    [
        pytest.param({"cache_salt": "end-user-1"}, id="top-level"),
        pytest.param({"extra_body": {"cache_salt": "end-user-1"}}, id="extra-body"),
    ],
)
def test_caller_salt_splits_a_key_but_never_crosses_keys(caller_salt_kwargs: dict):
    alice = _salt_for("hash-alice")
    alice_user = _salt_for("hash-alice", **caller_salt_kwargs)
    assert alice_user not in (alice, "end-user-1")
    assert alice_user == _salt_for("hash-alice", **caller_salt_kwargs)
    assert alice_user != _salt_for("hash-bob", **caller_salt_kwargs)


def test_caller_cannot_submit_another_keys_salt():
    alice = _salt_for("hash-alice")
    assert _salt_for("hash-bob", cache_salt=alice) != alice
    assert _salt_for("hash-bob", extra_body={"cache_salt": alice}) != alice


def test_cache_salt_reads_litellm_metadata():
    body = _sent_chat_body(hosted_vllm_cache_salt=True, litellm_metadata={"user_api_key_hash": "hash-alice"})
    assert body["cache_salt"] == _salt_for("hash-alice")


@pytest.mark.parametrize(
    "call_kwargs, expected_salt",
    [
        pytest.param({"metadata": {"user_api_key_hash": "hash-alice"}}, None, id="not-opted-in-sends-no-salt"),
        pytest.param(
            {"metadata": {"user_api_key_hash": "hash-alice"}, "cache_salt": "caller"},
            "caller",
            id="not-opted-in-keeps-caller-salt",
        ),
        pytest.param({"hosted_vllm_cache_salt": True}, None, id="no-key-hash-sends-no-salt"),
        pytest.param(
            {"hosted_vllm_cache_salt": True, "metadata": {"user_api_key_hash": ""}, "cache_salt": "caller"},
            "caller",
            id="empty-key-hash-keeps-caller-salt",
        ),
        pytest.param(
            {"hosted_vllm_cache_salt": "true", "metadata": {"user_api_key_hash": "hash-alice"}},
            None,
            id="non-bool-flag-is-off",
        ),
    ],
)
def test_cache_salt_untouched_without_opt_in_or_caller_key(call_kwargs: dict, expected_salt: str | None):
    body = _sent_chat_body(**call_kwargs)
    assert body.get("cache_salt") == expected_salt
    assert "hosted_vllm_cache_salt" not in body


def test_cache_salt_set_when_streaming():
    import litellm

    client = _vllm_chat_client()
    client.post.return_value.iter_lines = lambda: iter(())
    litellm.completion(
        model="hosted_vllm/Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "hi"}],
        api_base="http://vllm.internal:8000/v1",
        client=client,
        stream=True,
        hosted_vllm_cache_salt=True,
        metadata={"user_api_key_hash": "hash-alice"},
    )
    body = json.loads(client.post.call_args.kwargs["data"])
    assert body["stream"] is True
    assert body["cache_salt"] == _salt_for("hash-alice")
