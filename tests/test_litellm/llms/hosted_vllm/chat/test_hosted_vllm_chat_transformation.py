import json
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
from litellm.llms.vllm.common_utils import VLLMModelInfo


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


def _model_list_response(*entries: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"data": list(entries)}
    return response


def test_vllm_model_info_maps_context_only_and_authenticates(monkeypatch) -> None:
    request = MagicMock(return_value=_model_list_response({"id": "Qwen/Qwen3-8B", "max_model_len": 262_144}))
    monkeypatch.setattr(litellm.module_level_client, "get", request)

    info = VLLMModelInfo(provider="hosted_vllm").get_model_info(
        model="hosted_vllm/Qwen/Qwen3-8B",
        api_base="https://vllm.example/v1",
        api_key="secret-key",
    )

    assert info is not None
    assert info["max_input_tokens"] == 262_144
    assert info["max_tokens"] is None
    assert info["max_output_tokens"] is None
    request.assert_called_once()
    assert request.call_args.kwargs["url"] == "https://vllm.example/v1/models"
    assert request.call_args.kwargs["headers"] == {"Authorization": "Bearer secret-key"}


def test_vllm_model_info_does_not_discover_without_opt_in(monkeypatch) -> None:
    request = MagicMock()
    monkeypatch.setattr(litellm.module_level_client, "get", request)

    with pytest.raises(Exception, match="isn't mapped yet"):
        litellm.get_model_info(
            "hosted_vllm/not-in-static-map",
            api_base="https://no-discovery.example/v1",
        )

    request.assert_not_called()


@pytest.mark.parametrize("redirect", [False, True])
def test_vllm_discovery_http_transport_bounds_and_credentials(monkeypatch, redirect) -> None:
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if redirect:
            return httpx.Response(302, headers={"Location": "https://other.example/models"})
        return httpx.Response(200, json={"data": [{"id": "shared", "max_model_len": 262144}]})

    with httpx.Client(transport=httpx.MockTransport(serve), follow_redirects=True) as client:
        monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=client))
        provider = VLLMModelInfo(provider="hosted_vllm")
        if redirect:
            with pytest.raises(httpx.HTTPStatusError):
                provider.get_model_info("hosted_vllm/shared", api_base="https://vllm.example/v1", api_key="key")
        else:
            info = provider.get_model_info("hosted_vllm/shared", api_base="https://vllm.example/v1", api_key="key")
            assert info is not None and info["max_input_tokens"] == 262144

    assert len(requests) == 1
    assert str(requests[0].url) == "https://vllm.example/v1/models"
    assert requests[0].headers["Authorization"] == "Bearer key"
    assert set(requests[0].extensions["timeout"].values()) == {5.0}


@pytest.mark.parametrize("value", [None, 0, -1, True, "262144", 262144.0, 1.5])
def test_vllm_model_info_ignores_non_positive_or_non_integer_context(
    monkeypatch,
    value: object,
) -> None:
    monkeypatch.setattr(
        litellm.module_level_client,
        "get",
        MagicMock(return_value=_model_list_response({"id": "model", "max_model_len": value})),
    )

    assert (
        VLLMModelInfo(provider="hosted_vllm").get_model_info(
            model="hosted_vllm/model",
            api_base="https://vllm.example/v1",
        )
        is None
    )


def test_vllm_explicit_base_never_receives_an_ambient_key(monkeypatch) -> None:
    request = MagicMock(return_value=_model_list_response({"id": "model"}))
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "ambient-secret")
    monkeypatch.setattr(litellm.module_level_client, "get", request)

    VLLMModelInfo(provider="hosted_vllm").get_model_info(
        model="hosted_vllm/model",
        api_base="https://operator-supplied.example/v1",
    )

    assert dict(request.call_args.kwargs["headers"]) == {}


def test_hosted_vllm_model_info_uses_provider_environment(monkeypatch) -> None:
    request = MagicMock(return_value=_model_list_response({"id": "env-model", "max_model_len": 32_768}))
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://hosted.example/v1")
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "hosted-secret")
    monkeypatch.setattr(litellm.module_level_client, "get", request)

    info = litellm.get_model_info("hosted_vllm/env-model", discover_model_info=True)

    assert info["litellm_provider"] == "hosted_vllm"
    assert info["max_input_tokens"] == 32_768
    request.assert_called_once()
    assert request.call_args.kwargs["url"] == "https://hosted.example/v1/models"
    assert request.call_args.kwargs["headers"] == {"Authorization": "Bearer hosted-secret"}


def test_bare_vllm_model_keeps_explicit_provider_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        litellm.module_level_client,
        "get",
        MagicMock(return_value=_model_list_response({"id": "shared", "max_model_len": 65_536})),
    )

    info = litellm.get_model_info(
        "shared",
        custom_llm_provider="vllm",
        api_base="https://vllm.example/v1",
        discover_model_info=True,
    )

    assert info["litellm_provider"] == "vllm"
    assert info["max_input_tokens"] == 65_536


def test_vllm_endpoint_scoped_lookup_refreshes_and_removes_without_global_registration(
    monkeypatch,
) -> None:
    responses = [
        _model_list_response({"id": "shared", "max_model_len": 32_768}),
        _model_list_response({"id": "shared", "max_model_len": 65_536}),
        _model_list_response(),
    ]
    request = MagicMock(side_effect=responses)
    monkeypatch.setattr(litellm.module_level_client, "get", request)
    key = "hosted_vllm/shared"
    original_entry = litellm.model_cost.get(key)

    first = litellm.get_model_info(
        key,
        api_base="https://vllm-a.example/v1",
        api_key="key-a",
        discover_model_info=True,
    )
    second = litellm.get_model_info(
        key,
        api_base="https://vllm-a.example/v1",
        api_key="key-a",
        discover_model_info=True,
    )
    with pytest.raises(Exception, match="isn't mapped yet"):
        litellm.get_model_info(
            key,
            api_base="https://vllm-a.example/v1",
            api_key="key-a",
            discover_model_info=True,
        )

    assert first["max_input_tokens"] == 32_768
    assert second["max_input_tokens"] == 65_536
    assert litellm.model_cost.get(key) is original_entry


def test_vllm_same_model_id_is_isolated_by_endpoint(monkeypatch) -> None:
    def get(url: str, headers: dict[str, str], **kwargs) -> MagicMock:
        del headers
        context = 32_768 if "vllm-a" in url else 131_072
        return _model_list_response({"id": "shared", "max_model_len": context})

    monkeypatch.setattr(litellm.module_level_client, "get", get)

    info_a = litellm.get_model_info(
        "hosted_vllm/shared",
        api_base="https://vllm-a.example/v1",
        api_key="key-a",
        discover_model_info=True,
    )
    info_b = litellm.get_model_info(
        "hosted_vllm/shared",
        api_base="https://vllm-b.example/v1",
        api_key="key-b",
        discover_model_info=True,
    )

    assert info_a["max_input_tokens"] == 32_768
    assert info_b["max_input_tokens"] == 131_072


def test_vllm_discovery_failure_does_not_log_the_api_key(monkeypatch, caplog) -> None:
    secret = "do-not-log-this-key"
    monkeypatch.setattr(
        litellm.module_level_client,
        "get",
        MagicMock(return_value=MagicMock(json=lambda: {"error": {"authorization": secret}})),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(Exception, match="isn't mapped yet"):
        litellm.get_model_info(
            "hosted_vllm/unmapped",
            api_base="https://vllm.example/v1",
            api_key=secret,
            discover_model_info=True,
        )

    assert secret not in caplog.text


def test_vllm_endpoint_discovery_survives_price_map_replacement(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(
        litellm.module_level_client,
        "get",
        MagicMock(return_value=_model_list_response({"id": "model", "max_model_len": 98_304})),
    )

    before = litellm.get_model_info("hosted_vllm/model", api_base="https://vllm.example/v1", discover_model_info=True)
    monkeypatch.setattr(litellm, "model_cost", {"unrelated": {"litellm_provider": "openai"}})
    info = litellm.get_model_info(
        "hosted_vllm/model",
        api_base="https://vllm.example/v1",
        api_key="key",
        discover_model_info=True,
    )

    assert info["max_input_tokens"] == 98_304
    assert before["max_input_tokens"] == info["max_input_tokens"]
    assert "hosted_vllm/model" not in litellm.model_cost


def test_vllm_discovery_preserves_static_pricing(monkeypatch) -> None:
    from litellm.utils import _invalidate_model_cost_lowercase_map

    static_info = {
        "litellm_provider": "hosted_vllm",
        "mode": "chat",
        "max_input_tokens": 4_096,
        "max_output_tokens": 1_024,
        "supports_vision": True,
        "input_cost_per_token": 0.25,
        "output_cost_per_token": 0.5,
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(litellm, "model_cost", {"hosted_vllm/priced": static_info})
        scoped.setattr(
            litellm.module_level_client,
            "get",
            MagicMock(return_value=_model_list_response({"id": "priced", "max_model_len": 131_072})),
        )
        _invalidate_model_cost_lowercase_map()

        info = litellm.get_model_info(
            "hosted_vllm/priced",
            api_base="https://vllm.example/v1",
            discover_model_info=True,
        )

        assert info["max_input_tokens"] == 131_072
        assert info["input_cost_per_token"] == 0.25
        assert info["output_cost_per_token"] == 0.5
        assert info["max_output_tokens"] == 1_024
        assert info["supports_vision"] is True
    _invalidate_model_cost_lowercase_map()
