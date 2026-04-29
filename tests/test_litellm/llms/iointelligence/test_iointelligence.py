import json
import os
from unittest.mock import patch

import pytest

from litellm import completion, embedding
from litellm.llms.custom_httpx.http_handler import HTTPHandler


@patch.dict(os.environ, {}, clear=True)
def test_completion_io_intelligence():
    """Ensure that the completion function works with IO Intelligence API."""
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            completion(
                model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                client=client,
                max_tokens=5,
                api_key="fake-api-key",
            )

            mock_post.assert_called_once()
            mock_kwargs = mock_post.call_args.kwargs
            assert "api.intelligence.io.solutions/api/v1" in mock_kwargs["url"]
            assert mock_kwargs["headers"]["Authorization"] == "Bearer fake-api-key"
            json_data = json.loads(mock_kwargs["data"])
            assert json_data["max_tokens"] == 5
            assert json_data["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@patch.dict(os.environ, {}, clear=True)
def test_completion_io_intelligence_with_custom_api_base():
    """Ensure custom api_base is respected."""
    messages = [{"role": "user", "content": "Hello"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            completion(
                model="io_intelligence/deepseek-ai/DeepSeek-R1-0528",
                messages=messages,
                client=client,
                max_tokens=10,
                api_key="fake-api-key",
                api_base="https://custom.api.base/v1",
            )

            mock_post.assert_called_once()
            mock_kwargs = mock_post.call_args.kwargs
            assert "custom.api.base" in mock_kwargs["url"]
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@patch.dict(os.environ, {}, clear=True)
def test_completion_io_intelligence_streaming():
    """Ensure streaming parameter is passed correctly."""
    messages = [{"role": "user", "content": "Hello"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            completion(
                model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                client=client,
                max_tokens=5,
                api_key="fake-api-key",
                stream=True,
            )

            mock_post.assert_called_once()
            mock_kwargs = mock_post.call_args.kwargs
            json_data = json.loads(mock_kwargs["data"])
            assert json_data["stream"] is True
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@patch.dict(
    os.environ,
    {"IO_INTELLIGENCE_API_KEY": "test-env-key"},
    clear=True,
)
def test_completion_io_intelligence_env_key():
    """Ensure API key is picked up from environment variable."""
    messages = [{"role": "user", "content": "Hello"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            completion(
                model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                client=client,
                max_tokens=5,
            )

            mock_post.assert_called_once()
            mock_kwargs = mock_post.call_args.kwargs
            assert mock_kwargs["headers"]["Authorization"] == "Bearer test-env-key"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_io_intelligence_with_tools():
    """Ensure function calling params are passed correctly."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            completion(
                model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                client=client,
                max_tokens=5,
                api_key="fake-api-key",
                tools=tools,
                tool_choice="auto",
            )

            mock_post.assert_called_once()
            mock_kwargs = mock_post.call_args.kwargs
            json_data = json.loads(mock_kwargs["data"])
            assert "tools" in json_data
            assert json_data["tool_choice"] == "auto"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_io_intelligence_provider_in_model_prices():
    """Ensure IO Intelligence models are registered in model prices."""
    from litellm import model_cost

    io_models = [k for k in model_cost if k.startswith("io_intelligence/")]
    assert len(io_models) > 0, "No IO Intelligence models found in model_cost"

    # Check a specific model
    assert "io_intelligence/meta-llama/Llama-3.3-70B-Instruct" in model_cost
    model_info = model_cost["io_intelligence/meta-llama/Llama-3.3-70B-Instruct"]
    assert model_info["litellm_provider"] == "io_intelligence"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] > 0


def test_io_intelligence_embedding_model_in_prices():
    """Ensure IO Intelligence embedding model is registered."""
    from litellm import model_cost

    assert "io_intelligence/BAAI/bge-multilingual-gemma2" in model_cost
    model_info = model_cost["io_intelligence/BAAI/bge-multilingual-gemma2"]
    assert model_info["litellm_provider"] == "io_intelligence"
    assert model_info["mode"] == "embedding"


def test_io_intelligence_in_openai_compatible_providers():
    """Ensure io_intelligence is in the openai_compatible_providers list."""
    from litellm.constants import openai_compatible_providers

    assert "io_intelligence" in openai_compatible_providers


def test_io_intelligence_config_supported_params():
    """Test that IOIntelligenceConfig returns expected supported params."""
    from litellm.llms.iointelligence.chat import IOIntelligenceConfig

    config = IOIntelligenceConfig()
    params = config.get_supported_openai_params(
        model="meta-llama/Llama-3.3-70B-Instruct"
    )
    assert "max_tokens" in params
    assert "temperature" in params
    assert "tools" in params
    assert "stream" in params


def test_io_intelligence_config_map_params():
    """Test that IOIntelligenceConfig maps max_completion_tokens to max_tokens."""
    from litellm.llms.iointelligence.chat import IOIntelligenceConfig

    config = IOIntelligenceConfig()
    result = config.map_openai_params(
        non_default_params={"max_completion_tokens": 100},
        optional_params={},
        model="meta-llama/Llama-3.3-70B-Instruct",
        drop_params=False,
    )
    assert result["max_tokens"] == 100


def test_completion_io_intelligence_live():
    """Live integration test - only runs if API key is set."""
    if os.environ.get("IO_INTELLIGENCE_API_KEY") is None:
        pytest.skip("IO_INTELLIGENCE_API_KEY not set")

    messages = [{"role": "user", "content": "Say hello in one word."}]
    response = completion(
        model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
        messages=messages,
        max_tokens=10,
    )
    assert response["object"] == "chat.completion"
    assert len(response["choices"]) == 1
    assert len(response["choices"][0]["message"]["content"]) > 0
