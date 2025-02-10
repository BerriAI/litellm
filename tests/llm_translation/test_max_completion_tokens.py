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
from unittest.mock import patch, MagicMock, AsyncMock

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
@pytest.mark.asyncio()
async def test_bedrock_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    litellm.set_verbose = True

    client = AsyncHTTPHandler()

    mock_response = return_mocked_response(model)
    _model = model.split("/")[1]
    print("\n\nmock_response: ", mock_response)

    with patch.object(client, "post") as mock_client:
        try:
            response = await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = json.loads(mock_client.call_args.kwargs["data"])

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
            "additionalModelRequestFields": {},
            "system": [],
            "inferenceConfig": {"maxTokens": 10},
        }


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-opus-20240229"],
)
@pytest.mark.asyncio()
async def test_anthropic_api_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to anthropic models
    """
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

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

    client = HTTPHandler()

    print("\n\nmock_response: ", mock_response)

    with patch.object(client, "post") as mock_client:
        try:
            response = await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs["json"]

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "max_tokens": 10,
            "model": model.split("/")[-1],
        }


def test_all_model_configs():
    from litellm.llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import (
        VertexAIAi21Config,
    )
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
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

    from litellm.llms.fireworks_ai.chat.transformation import (
        FireworksAIConfig,
    )

    assert "max_completion_tokens" in FireworksAIConfig().get_supported_openai_params(
        model="llama3"
    )
    assert FireworksAIConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.huggingface.chat.handler import HuggingfaceConfig

    assert "max_completion_tokens" in HuggingfaceConfig().get_supported_openai_params(
        model="llama3"
    )
    assert HuggingfaceConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="llama3",
        drop_params=False,
    ) == {"max_new_tokens": 10}

    from litellm.llms.nvidia_nim.chat import NvidiaNimConfig

    assert "max_completion_tokens" in NvidiaNimConfig().get_supported_openai_params(
        model="llama3"
    )
    assert NvidiaNimConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ollama_chat import OllamaChatConfig

    assert "max_completion_tokens" in OllamaChatConfig().get_supported_openai_params(
        model="llama3"
    )
    assert OllamaChatConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"num_predict": 10}

    from litellm.llms.predibase.chat.transformation import PredibaseConfig

    assert "max_completion_tokens" in PredibaseConfig().get_supported_openai_params(
        model="llama3"
    )
    assert PredibaseConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_new_tokens": 10}

    from litellm.llms.codestral.completion.transformation import (
        CodestralTextCompletionConfig,
    )

    assert (
        "max_completion_tokens"
        in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    )
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.volcengine import VolcEngineConfig

    assert "max_completion_tokens" in VolcEngineConfig().get_supported_openai_params(
        model="llama3"
    )
    assert VolcEngineConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ai21.chat.transformation import AI21ChatConfig

    assert "max_completion_tokens" in AI21ChatConfig().get_supported_openai_params(
        "jamba-1.5-mini@001"
    )
    assert AI21ChatConfig().map_openai_params(
        model="jamba-1.5-mini@001",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

    assert "max_completion_tokens" in AzureOpenAIConfig().get_supported_openai_params(
        model="gpt-3.5-turbo"
    )
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

    from litellm.llms.codestral.completion.transformation import (
        CodestralTextCompletionConfig,
    )

    assert (
        "max_completion_tokens"
        in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    )
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm import (
        AmazonAnthropicClaude3Config,
        AmazonAnthropicConfig,
    )

    assert (
        "max_completion_tokens"
        in AmazonAnthropicClaude3Config().get_supported_openai_params(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )

    assert AmazonAnthropicClaude3Config().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    ) == {"max_tokens": 10}

    assert (
        "max_completion_tokens" in AmazonAnthropicConfig().get_supported_openai_params()
    )

    assert AmazonAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens_to_sample": 10}

    from litellm.llms.databricks.chat.handler import DatabricksConfig

    assert "max_completion_tokens" in DatabricksConfig().get_supported_openai_params()

    assert DatabricksConfig().map_openai_params(
        model="databricks/llama-3-70b-instruct",
        drop_params=False,
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    assert (
        "max_completion_tokens"
        in VertexAIAnthropicConfig().get_supported_openai_params(
            model="claude-3-5-sonnet-20240620"
        )
    )

    assert VertexAIAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="claude-3-5-sonnet-20240620",
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(
        model="gemini-1.0-pro"
    )

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert (
        "max_completion_tokens"
        in GoogleAIStudioGeminiConfig().get_supported_openai_params(
            model="gemini-1.0-pro"
        )
    )

    assert GoogleAIStudioGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(
        model="gemini-1.0-pro"
    )

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}
