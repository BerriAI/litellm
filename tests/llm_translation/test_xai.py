import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from unittest.mock import patch
from litellm.llms.xai.chat.transformation import XAIChatConfig, XAI_API_BASE
from base_llm_unit_tests import BaseReasoningLLMTests, BaseLLMChatTest


def test_xai_chat_config_get_openai_compatible_provider_info():
    config = XAIChatConfig()

    # Test with default values
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == XAI_API_BASE
    assert api_key == os.environ.get("XAI_API_KEY")

    # Test with custom API key
    custom_api_key = "test_api_key"
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=custom_api_key
    )
    assert api_base == XAI_API_BASE
    assert api_key == custom_api_key

    # Test with custom environment variables for api_base and api_key
    with patch.dict(
        "os.environ",
        {"XAI_API_BASE": "https://env.x.ai/v1", "XAI_API_KEY": "env_api_key"},
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.x.ai/v1"
        assert api_key == "env_api_key"


def test_xai_chat_config_map_openai_params():
    """
    XAI is OpenAI compatible*

    Does not support all OpenAI parameters:
    - max_completion_tokens -> max_tokens

    """
    config = XAIChatConfig()

    # Test mapping of parameters
    non_default_params = {
        "max_completion_tokens": 100,
        "frequency_penalty": 0.5,
        "logit_bias": {"50256": -100},
        "logprobs": 5,
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "xai/grok-beta",
        "n": 2,
        "presence_penalty": 0.2,
        "response_format": {"type": "json_object"},
        "seed": 42,
        "stop": ["END"],
        "stream": True,
        "stream_options": {},
        "temperature": 0.7,
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "top_logprobs": 3,
        "top_p": 0.9,
        "user": "test_user",
        "unsupported_param": "value",
    }
    optional_params = {}
    model = "xai/grok-beta"

    result = config.map_openai_params(non_default_params, optional_params, model)

    # Assert all supported parameters are present in the result
    assert result["max_tokens"] == 100  # max_completion_tokens -> max_tokens
    assert result["frequency_penalty"] == 0.5
    assert result["logit_bias"] == {"50256": -100}
    assert result["logprobs"] == 5
    assert result["n"] == 2
    assert result["presence_penalty"] == 0.2
    assert result["response_format"] == {"type": "json_object"}
    assert result["seed"] == 42
    assert result["stop"] == ["END"]
    assert result["stream"] is True
    assert result["stream_options"] == {}
    assert result["temperature"] == 0.7
    assert result["tool_choice"] == "auto"
    assert result["tools"] == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert result["top_logprobs"] == 3
    assert result["top_p"] == 0.9
    assert result["user"] == "test_user"

    # Assert unsupported parameter is not in the result
    assert "unsupported_param" not in result


@pytest.mark.parametrize("stream", [False, True])
def test_completion_xai(stream):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="xai/grok-3-mini-beta",
            messages=messages,
            stream=stream,
        )
        print(response)

        if stream is True:
            for chunk in response:
                print(chunk)
                assert chunk is not None
                assert isinstance(chunk, litellm.ModelResponseStream)
                assert isinstance(chunk.choices[0], litellm.utils.StreamingChoices)

        else:
            assert response is not None
            assert isinstance(response, litellm.ModelResponse)
            assert response.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_xai_message_name_filtering():
    messages = [
        {
            "role": "system",
            "content": "*I press the green button*",
            "name": "example_user"
        },
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hello", "name": "Jane"},
    ]
    response = completion(
        model="xai/grok-3-mini-beta",
        messages=messages,
    )
    assert response is not None
    assert response.choices[0].message.content is not None


class TestXAIReasoningEffort(BaseReasoningLLMTests):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-3-mini-beta",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    
class TestXAIChat(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-3-mini-beta",
        }
    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass
