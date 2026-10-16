import json
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config


@pytest.mark.asyncio
async def test_azure_chat_o_series_transformation():
    provider_config = AzureOpenAIO1Config()
    model = "o_series/web-interface-o1-mini"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    optional_params = {}
    litellm_params = {}
    headers = {}

    response = await provider_config.async_transform_request(
        model, messages, optional_params, litellm_params, headers
    )
    print(response)
    assert response["model"] == "web-interface-o1-mini"


def test_azure_o_series_transform_request_flattens_top_level_anyof():
    """Regression test for LIT-6510: the o-series super() chain ends in
    OpenAIGPTConfig, whose flatten gate skips provider 'azure', so
    AzureOpenAIO1Config must flatten tool schema combinators itself."""
    tool = {
        "type": "function",
        "function": {
            "name": "automation_update",
            "description": "Update an automation",
            "parameters": {
                "type": "object",
                "anyOf": [
                    {
                        "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                        "required": ["id", "enabled"],
                    },
                    {
                        "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                        "required": ["id", "schedule"],
                    },
                ],
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }
    optional_params = {"tools": [tool]}

    request = AzureOpenAIO1Config().transform_request(
        model="o3-mini",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    parameters = request["tools"][0]["function"]["parameters"]
    assert "anyOf" not in parameters
    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
    assert parameters["required"] == ["id"]
    assert "anyOf" in tool["function"]["parameters"]
    assert optional_params["tools"][0] is tool


def test_azure_o_series_transform_request_moves_system_messages_first(monkeypatch):
    monkeypatch.setattr(litellm, "openai_system_messages_first", True)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": "dev"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "more"},
    ]

    request = AzureOpenAIO1Config().transform_request(
        model="o3-mini",
        messages=messages,
        optional_params={},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert [m["content"] for m in request["messages"]] == ["dev", "hi", "reply", "more"]
    assert [m["content"] for m in messages] == ["hi", "dev", "reply", "more"]


def test_override_fake_stream():
    """Test that native streaming is not supported for o1."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure/o1-preview",
                "litellm_params": {
                    "model": "azure/o1-preview",
                    "api_key": "my-fake-o1-key",
                    "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
                },
                "model_info": {
                    "supports_native_streaming": True,
                },
            }
        ]
    )

    model_info = litellm.get_model_info(model="azure/o1-preview", custom_llm_provider="azure")
    assert model_info["supports_native_streaming"] is True

    fake_stream = litellm.AzureOpenAIO1Config().should_fake_stream(model="azure/o1-preview", stream=True)
    assert fake_stream is False


def test_azure_o3_streaming():
    """
    Test that o3 models handles fake streaming correctly.
    """
    from openai import AzureOpenAI
    from litellm import completion

    client = AzureOpenAI(
        api_key="my-fake-o1-key",
        base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_create:
        try:
            completion(
                model="azure/o3-mini",
                messages=[{"role": "user", "content": "Hello, world!"}],
                stream=True,
                client=client,
            )
        except Exception as e:
            print(e)
        assert mock_create.call_count == 1
        assert "stream" in mock_create.call_args.kwargs


def test_azure_o_series_routing():
    """
    Allows user to pass model="azure/o_series/<any-deployment-name>" for explicit o_series model routing.
    """
    from openai import AzureOpenAI
    from litellm import completion

    client = AzureOpenAI(
        api_key="my-fake-o1-key",
        base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_create:
        try:
            completion(
                model="azure/o_series/my-random-deployment-name",
                messages=[{"role": "user", "content": "Hello, world!"}],
                stream=True,
                client=client,
            )
        except Exception as e:
            print(e)
        assert mock_create.call_count == 1
        assert "stream" not in mock_create.call_args.kwargs


@patch("litellm.main.azure_o1_chat_completions._get_openai_client")
def test_openai_o_series_max_retries_0(mock_get_openai_client):
    import litellm

    mock_get_openai_client.return_value.chat.completions.with_raw_response.create.return_value.headers = {}
    mock_get_openai_client.return_value.chat.completions.with_raw_response.create.return_value.parse.return_value = (
        ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])
    )
    litellm.set_verbose = True
    response = litellm.completion(
        model="azure/o1-preview",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        api_key="fake-key",
        api_base="https://fake-azure.openai.azure.com",
        api_version="2024-10-21",
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0
    assert response.choices[0].message.content == "Hello"


@pytest.mark.asyncio
async def test_azure_o1_series_response_format_extra_params():
    """
    Tool calling should work for all azure o_series models.
    """
    litellm.turn_on_debug()

    from openai import AsyncAzureOpenAI

    litellm.set_verbose = True

    client = AsyncAzureOpenAI(
        api_key="fake-api-key",
        base_url="https://openai-prod-test.openai.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview",
        api_version="2025-01-01-preview",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response_format = {"type": "json_object"}
    tool_choice = "auto"
    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                client=client,
                model="azure/o_series/<my-deployment-name>",
                api_key="xxxxx",
                api_base="https://openai-prod-test.openai.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview",
                api_version="2024-12-01-preview",
                messages=[{"role": "user", "content": "Hello! return a json object"}],
                tools=tools,
                response_format=response_format,
                tool_choice=tool_choice,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", json.dumps(request_body, indent=4))
        assert request_body["tools"] == tools
        assert request_body["response_format"] == response_format
        assert request_body["tool_choice"] == tool_choice
