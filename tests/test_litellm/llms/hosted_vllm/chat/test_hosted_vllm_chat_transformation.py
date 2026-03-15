import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
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

    # Test thinking with low budget_tokens -> "minimal" (for < 2000)
    optional_params = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert "thinking" not in optional_params  # thinking should NOT be passed
    assert optional_params["reasoning_effort"] == "minimal"

    # Test thinking with high budget_tokens -> "high"
    optional_params = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 15000}},
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"

    # Test that existing reasoning_effort is not overwritten
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 15000},
            "reasoning_effort": "low",
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "low"


def test_hosted_vllm_thinking_blocks_prepended_to_assistant_content():
    """
    Test that thinking_blocks on assistant messages are converted to content
    blocks prepended before the existing content.
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
    assert isinstance(assistant_msg["content"], list)
    assert assistant_msg["content"][0] == {
        "type": "thinking",
        "thinking": "Let me reason about this...",
    }
    assert assistant_msg["content"][1] == {
        "type": "text",
        "text": "Here is my answer.",
    }
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_thinking_blocks_with_list_content():
    """
    Test thinking_blocks prepended when assistant content is already a list.
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
    assert len(assistant_msg["content"]) == 3
    assert assistant_msg["content"][0] == {
        "type": "thinking",
        "thinking": "Step 1 reasoning",
    }
    assert assistant_msg["content"][1] == {
        "type": "thinking",
        "thinking": "Step 2 reasoning",
    }
    assert assistant_msg["content"][2] == {"type": "text", "text": "Response text"}
    assert "thinking_blocks" not in assistant_msg


# --- End-to-end streaming tests using respx ---
# These tests mock the vLLM HTTP endpoint with real SSE payloads containing
# delta.reasoning (as vLLM/SGLang returns for thinking models) and verify that
# LiteLLM's gateway correctly converts delta.reasoning → delta.reasoning_content
# before the chunks reach the caller.
# See: https://github.com/BerriAI/litellm/issues/20246


def _make_sse_body(chunks: list) -> str:
    """Encode a list of chunk dicts as an SSE body (as vLLM would produce)."""
    lines = []
    for chunk in chunks:
        lines.append("data: " + json.dumps(chunk) + "\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


@pytest.mark.respx()
def test_hosted_vllm_gateway_converts_delta_reasoning_to_reasoning_content(respx_mock):
    """
    End-to-end test: a vLLM backend returns streaming SSE chunks where the delta
    contains a 'reasoning' field (not 'reasoning_content').  LiteLLM must remap
    delta.reasoning → delta.reasoning_content before yielding chunks to callers.

    vLLM/SGLang uses delta.reasoning for thinking models; LiteLLM's public API
    exposes reasoning_content.  The mapping happens inside
    OpenAIChatCompletionStreamingHandler._map_reasoning_to_reasoning_content,
    which HostedVLLMChatConfig inherits from OpenAIGPTConfig.

    See: https://github.com/BerriAI/litellm/issues/20246
    """
    litellm.disable_aiohttp_transport = True

    vllm_sse_body = _make_sse_body([
        {
            "id": "chatcmpl-vllm-01",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "Qwen3-30B-A3B",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning": "Step 1: analyze the problem."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-vllm-01",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "Qwen3-30B-A3B",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": " Step 2: conclude."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-vllm-01",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "Qwen3-30B-A3B",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "The answer is 42."},
                    "finish_reason": "stop",
                }
            ],
        },
    ])

    respx_mock.post("https://test-vllm.example.com/v1/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=vllm_sse_body,
    )

    response = litellm.completion(
        model="hosted_vllm/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "What is 6*7?"}],
        api_base="https://test-vllm.example.com/v1",
        stream=True,
    )

    chunks = list(response)

    # All chunks with reasoning_content set must NOT have a raw 'reasoning' attribute
    reasoning_chunks = [
        c for c in chunks
        if getattr(c.choices[0].delta, "reasoning_content", None) is not None
    ]
    assert len(reasoning_chunks) >= 1, (
        "Expected at least one chunk with reasoning_content, "
        f"got chunks: {[c.choices[0].delta for c in chunks]}"
    )
    for chunk in reasoning_chunks:
        delta = chunk.choices[0].delta
        # The converted field must be named reasoning_content
        assert delta.reasoning_content is not None
        # The raw vLLM field must NOT leak through
        assert not hasattr(delta, "reasoning") or delta.reasoning is None

    # The final content chunk must still be delivered correctly
    content_chunks = [
        c for c in chunks
        if getattr(c.choices[0].delta, "content", None)
    ]
    assert any("42" in c.choices[0].delta.content for c in content_chunks)


@pytest.mark.respx()
def test_hosted_vllm_gateway_preserves_reasoning_content_passthrough(respx_mock):
    """
    End-to-end test: if a vLLM backend already returns delta.reasoning_content
    (rather than delta.reasoning), LiteLLM must pass it through unchanged.
    """
    litellm.disable_aiohttp_transport = True

    vllm_sse_body = _make_sse_body([
        {
            "id": "chatcmpl-vllm-02",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "Qwen3-30B-A3B",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": "Already mapped."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-vllm-02",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "Qwen3-30B-A3B",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Done."},
                    "finish_reason": "stop",
                }
            ],
        },
    ])

    respx_mock.post("https://test-vllm.example.com/v1/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=vllm_sse_body,
    )

    response = litellm.completion(
        model="hosted_vllm/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "Test"}],
        api_base="https://test-vllm.example.com/v1",
        stream=True,
    )

    chunks = list(response)
    reasoning_chunks = [
        c for c in chunks
        if getattr(c.choices[0].delta, "reasoning_content", None) is not None
    ]
    assert len(reasoning_chunks) >= 1
    assert reasoning_chunks[0].choices[0].delta.reasoning_content == "Already mapped."
