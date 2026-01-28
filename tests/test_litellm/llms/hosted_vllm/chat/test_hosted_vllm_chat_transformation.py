import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

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
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = MagicMock()

    with patch.object(
        client.chat.completions.with_raw_response, "create", return_value=MagicMock()
    ) as mock_post:
        try:
            response = completion(
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
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        print(f"mock_post.call_args.kwargs: {mock_post.call_args.kwargs}")
        assert mock_post.call_args.kwargs["messages"] == [
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
