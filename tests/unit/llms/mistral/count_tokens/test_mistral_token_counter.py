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
        model="mistral-large-latest",
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
    assert transformed["model"] == "mistral-large-latest"
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
            model_to_use="mistral-large-latest",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-latest",
        )

    assert result is not None
    assert result.total_tokens == 42
    assert result.tokenizer_type == "mistral_api"
    assert result.model_used == "mistral-large-latest"
    assert result.request_model == "mistral/mistral-large-latest"
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
            model_to_use="mistral-large-latest",
            request={"model": "mistral-large-latest", "input": "Count this"},
            policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-latest",
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
        model_to_use="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={"litellm_params": {}},
        request_model="mistral/mistral-large-latest",
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
            model_to_use="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-latest",
        )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_connect_error_falls_back_to_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-latest",
        )
    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_upstream_error_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=httpx.Response(400, text="Bad Request"))
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-latest",
        )
    assert result is not None
    assert result.error is True
    assert result.status_code == 400


@pytest.mark.asyncio
async def test_count_tokens_empty_messages():
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_tokens(
        model_to_use="mistral-large-latest",
        messages=[],
        contents=None,
        deployment={"litellm_params": {"api_key": "test-key"}},
        request_model="mistral/mistral-large-latest",
    )
    assert result is None
