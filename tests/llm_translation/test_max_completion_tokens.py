import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from datetime import datetime
from unittest.mock import AsyncMock
from dotenv import load_dotenv

load_dotenv()
import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse

# Adds the parent directory to the system path


def return_mocked_response(model: str):
    if model == "bedrock/mistral.mistral-large-2407-v1:0":
        return {
            "metrics": {"latencyMs": 316},
            "output": {
                "message": {
                    "content": [{"text": "Hello! How are you doing today? How can"}],
                    "role": "assistant",
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
        }


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
    ],
)
@pytest.mark.respx
@pytest.mark.asyncio()
async def test_bedrock_max_completion_tokens(model: str, respx_mock: MockRouter):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    litellm.set_verbose = True

    mock_response = return_mocked_response(model)
    _model = model.split("/")[1]
    print("\n\nmock_response: ", mock_response)
    url = f"https://bedrock-runtime.us-west-2.amazonaws.com/model/{_model}/converse"
    mock_request = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = await litellm.acompletion(
        model=model,
        max_completion_tokens=10,
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)

    print("request_body: ", request_body)

    assert request_body == {
        "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
        "additionalModelRequestFields": {},
        "system": [],
        "inferenceConfig": {"maxTokens": 10},
    }
    print(f"response: {response}")
    assert isinstance(response, ModelResponse)


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-opus-20240229,"],
)
@pytest.mark.respx
@pytest.mark.asyncio()
async def test_anthropic_api_max_completion_tokens(model: str, respx_mock: MockRouter):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to anthropic models
    """
    litellm.set_verbose = True

    mock_response = {
        "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }

    print("\n\nmock_response: ", mock_response)
    url = f"https://api.anthropic.com/v1/messages"
    mock_request = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = await litellm.acompletion(
        model=model,
        max_completion_tokens=10,
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)

    print("request_body: ", request_body)

    assert request_body == {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
        "max_tokens": 10,
        "model": model.split("/")[-1],
    }
    print(f"response: {response}")
    assert isinstance(response, ModelResponse)


def test_all_model_configs():
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.ai21.transformation import (
        VertexAIAi21Config,
    )
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )

    assert (
        "max_completion_tokens" in VertexAILlama3Config().get_supported_openai_params()
    )
    assert VertexAILlama3Config().map_openai_params(
        {"max_completion_tokens": 10}, {}, "llama3", drop_params=False
    ) == {"max_tokens": 10}

    assert "max_completion_tokens" in VertexAIAi21Config().get_supported_openai_params()
    assert VertexAIAi21Config().map_openai_params(
        {"max_completion_tokens": 10}, {}, "llama3", drop_params=False
    ) == {"max_tokens": 10}

    from litellm.llms.fireworks_ai.chat.fireworks_ai_transformation import (
        FireworksAIConfig,
    )

    assert "max_completion_tokens" in FireworksAIConfig().get_supported_openai_params()
    assert FireworksAIConfig().map_openai_params(
        {"max_completion_tokens": 10}, {}, "llama3"
    ) == {"max_tokens": 10}

    from litellm.llms.huggingface_restapi import HuggingfaceConfig

    assert "max_completion_tokens" in HuggingfaceConfig().get_supported_openai_params()
    assert HuggingfaceConfig().map_openai_params({"max_completion_tokens": 10}, {}) == {
        "max_new_tokens": 10
    }

    from litellm.llms.nvidia_nim.chat import NvidiaNimConfig

    assert "max_completion_tokens" in NvidiaNimConfig().get_supported_openai_params(
        model="llama3"
    )
    assert NvidiaNimConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.ollama_chat import OllamaChatConfig

    assert "max_completion_tokens" in OllamaChatConfig().get_supported_openai_params()
    assert OllamaChatConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"num_predict": 10}

    from litellm.llms.predibase import PredibaseConfig

    assert "max_completion_tokens" in PredibaseConfig().get_supported_openai_params()
    assert PredibaseConfig().map_openai_params(
        {"max_completion_tokens": 10},
        {},
    ) == {"max_new_tokens": 10}

    from litellm.llms.text_completion_codestral import MistralTextCompletionConfig

    assert (
        "max_completion_tokens"
        in MistralTextCompletionConfig().get_supported_openai_params()
    )
    assert MistralTextCompletionConfig().map_openai_params(
        {"max_completion_tokens": 10},
        {},
    ) == {"max_tokens": 10}

    from litellm.llms.volcengine import VolcEngineConfig

    assert "max_completion_tokens" in VolcEngineConfig().get_supported_openai_params(
        model="llama3"
    )
    assert VolcEngineConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.AI21.chat import AI21ChatConfig

    assert "max_completion_tokens" in AI21ChatConfig().get_supported_openai_params(
        "jamba-1.5-mini@001"
    )
    assert AI21ChatConfig().map_openai_params(
        model="jamba-1.5-mini@001",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.AzureOpenAI.chat.gpt_transformation import AzureOpenAIConfig

    assert "max_completion_tokens" in AzureOpenAIConfig().get_supported_openai_params()
    assert AzureOpenAIConfig().map_openai_params(
        model="gpt-3.5-turbo",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        api_version="2022-12-01",
        drop_params=False,
    ) == {"max_completion_tokens": 10}

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    assert (
        "max_completion_tokens"
        in AmazonConverseConfig().get_supported_openai_params(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )
    assert AmazonConverseConfig().map_openai_params(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"maxTokens": 10}

    from litellm.llms.text_completion_codestral import MistralTextCompletionConfig

    assert (
        "max_completion_tokens"
        in MistralTextCompletionConfig().get_supported_openai_params()
    )
    assert MistralTextCompletionConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.bedrock.common_utils import (
        AmazonAnthropicClaude3Config,
        AmazonAnthropicConfig,
    )

    assert (
        "max_completion_tokens"
        in AmazonAnthropicClaude3Config().get_supported_openai_params()
    )

    assert AmazonAnthropicClaude3Config().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    assert (
        "max_completion_tokens" in AmazonAnthropicConfig().get_supported_openai_params()
    )

    assert AmazonAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens_to_sample": 10}

    from litellm.llms.databricks.chat import DatabricksConfig

    assert "max_completion_tokens" in DatabricksConfig().get_supported_openai_params()

    assert DatabricksConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    assert (
        "max_completion_tokens"
        in VertexAIAnthropicConfig().get_supported_openai_params()
    )

    assert VertexAIAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
        VertexAIConfig,
        GoogleAIStudioGeminiConfig,
        VertexGeminiConfig,
    )

    assert "max_completion_tokens" in VertexAIConfig().get_supported_openai_params()

    assert VertexAIConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_output_tokens": 10}

    assert (
        "max_completion_tokens"
        in GoogleAIStudioGeminiConfig().get_supported_openai_params()
    )

    assert GoogleAIStudioGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params()

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}
