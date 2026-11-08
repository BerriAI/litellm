import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.llms.mistral.count_tokens.handler import MistralCountTokensHandler
from litellm.llms.mistral.count_tokens.token_counter import MistralTokenCounter
from litellm.llms.mistral.count_tokens.transformation import MistralCountTokensConfig
from litellm.types.utils import TokenCountResponse

COUNT_URL: Final = "https://api.mistral.ai/v1/responses/input_tokens"


def test_get_mistral_count_tokens_endpoint():
    config: Final = MistralCountTokensConfig()
    assert config.get_mistral_count_tokens_endpoint() == COUNT_URL
    assert (
        config.get_mistral_count_tokens_endpoint("https://custom.mistral.ai")
        == "https://custom.mistral.ai/v1/responses/input_tokens"
    )
    assert (
        config.get_mistral_count_tokens_endpoint("https://custom.mistral.ai/v1")
        == "https://custom.mistral.ai/v1/responses/input_tokens"
    )


def test_get_required_headers():
    config: Final = MistralCountTokensConfig()
    headers: Final = config.get_required_headers("test-key")
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_transform_request_to_count_tokens():
    config: Final = MistralCountTokensConfig()
    transformed: Final = config.transform_request_to_count_tokens(
        model="mistral-large-4",
        input="Hello world",
        instructions="Be concise",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert transformed["model"] == "mistral-large-4"
    assert transformed["input"] == "Hello world"
    assert transformed["instructions"] == "Be concise"
    assert transformed["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup data",
            "parameters": {"type": "object"},
        }
    ]


def test_should_use_token_counting_api():
    counter: Final = MistralTokenCounter()
    assert counter.should_use_token_counting_api("mistral") is True
    assert counter.should_use_token_counting_api("openai") is False
    assert counter.should_use_token_counting_api(None) is False


@pytest.mark.asyncio
async def test_count_tokens_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 42})
        )
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )

    assert result is not None
    assert result.total_tokens == 42
    assert result.tokenizer_type == "mistral_api"
    assert result.model_used == "mistral-large-4"
    assert result.request_model == "mistral/mistral-large-4"
    assert result.error is False


@pytest.mark.asyncio
async def test_count_responses_request_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 55})
        )
        result: Final = await counter.count_responses_request(
            model_to_use="mistral-large-4",
            request={"model": "mistral-large-4", "input": "Count this"},
            policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )

    assert result is not None
    assert result.total_tokens == 55
    assert result.tokenizer_type == "mistral_api"


@pytest.mark.asyncio
async def test_count_tokens_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_AZURE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-4",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="mistral/mistral-large-4",
    )
    assert result is not None
    assert result.error is True
    assert result.status_code == 401
    assert "No Mistral API key is set" in (result.error_message or "")


@pytest.mark.asyncio
async def test_count_tokens_timeout_falls_back_to_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(side_effect=httpx.ConnectTimeout("Connection timed out"))
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_connect_error_falls_back_to_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_upstream_error_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=httpx.Response(400, text="Bad Request"))
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )
    assert result is not None
    assert result.error is True
    assert result.status_code == 400


@pytest.mark.asyncio
async def test_count_tokens_empty_messages():
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-4",
        messages=[],
        contents=None,
        deployment={"litellm_params": {"api_key": "test-key"}},
        request_model="mistral/mistral-large-4",
    )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_openai_format_with_tool_calls_and_tool_results(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": 20}'},
    ]
    tools: Final[list[dict[str, object]]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 58})
        )
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=messages,
            contents=None,
            tools=tools,
            system="Answer concisely.",
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
            messages_format="openai",
        )

    assert result is not None
    assert result.total_tokens == 58
    assert result.error is False
    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["instructions"] == "Answer concisely."
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert sent["input"] == [
        {"role": "user", "content": "What's the weather?"},
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"temp": 20}',
        },
    ]


@pytest.mark.asyncio
async def test_count_tokens_anthropic_format_with_tool_use_and_tool_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [{"type": "text", "text": '{"temp": 20}'}],
                }
            ],
        },
    ]
    tools: Final[list[dict[str, object]]] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 64})
        )
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=messages,
            contents=None,
            tools=tools,
            system="Be precise.",
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
            messages_format="anthropic",
        )

    assert result is not None
    assert result.total_tokens == 64
    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["instructions"] == "Be precise."
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "strict": False,
        }
    ]
    assert {
        "type": "function_call",
        "call_id": "toolu_123",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    } in sent["input"]
    assert {
        "type": "function_call_output",
        "call_id": "toolu_123",
        "output": '{"temp": 20}',
    } in sent["input"]


@pytest.mark.asyncio
async def test_count_tokens_malformed_tool_result_rejected_before_provider_call():
    counter: Final = MistralTokenCounter()
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "fn", "input": {}}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": 123}]}],
        },
    ]
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-4",
        messages=messages,
        contents=None,
        deployment={"litellm_params": {"api_key": "test-key", "drop_params": False}},
        request_model="mistral/mistral-large-4",
        messages_format="anthropic",
    )
    assert result is not None
    assert result.error is True
    assert result.status_code == 400
    assert result.rejected_before_provider_call is True


@pytest.mark.asyncio
async def test_count_tokens_azure_endpoint_falls_back_to_none():
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-4",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={
            "litellm_params": {
                "api_key": "test-key",
                "api_base": "https://my-mistral.inference.ai.azure.com",
            }
        },
        request_model="mistral/mistral-large-4",
    )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_azure_only_credentials_falls_back_to_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_AZURE_API_KEY", "azure-test-key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-4",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="mistral/mistral-large-4",
    )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_mistral_env_key_takes_precedence_over_generic_litellm_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("MISTRAL_API_KEY", "real-mistral-key")
    monkeypatch.setattr(litellm, "api_key", "generic-openai-key")
    counter: Final = MistralTokenCounter()
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 15})
        )
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {}},
            request_model="mistral/mistral-large-4",
        )
    assert result is not None
    assert result.total_tokens == 15
    assert route.calls.last.request.headers["authorization"] == "Bearer real-mistral-key"


def test_mistral_config_get_api_key_and_base(monkeypatch: pytest.MonkeyPatch):
    from litellm.llms.mistral.chat.transformation import MistralConfig

    monkeypatch.setenv("MISTRAL_API_KEY", "env-mistral-key")
    monkeypatch.setenv("MISTRAL_API_BASE", "https://custom.mistral.ai")
    assert MistralConfig.get_api_key() == "env-mistral-key"
    assert MistralConfig.get_api_base() == "https://custom.mistral.ai"


def test_mistral_config_get_models_calls_mistral_endpoint():
    from litellm.llms.mistral.chat.transformation import MistralConfig

    config: Final = MistralConfig()

    def mock_fetch(api_base: str | None, bearer_token: str | None) -> list[str]:
        assert api_base == "https://api.mistral.ai/v1"
        assert bearer_token == "test-key"
        return ["mistral-large-4", "mistral-small-latest"]

    with patch.object(config, "_fetch_model_ids", side_effect=mock_fetch):
        models: Final = config.get_models(api_key="test-key")
        assert models == ["mistral-large-4", "mistral-small-latest"]

