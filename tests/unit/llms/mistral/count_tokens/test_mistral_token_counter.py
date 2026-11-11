import json
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.core_helpers import decoded_json_object
from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.llms.mistral.count_tokens.token_counter import MistralTokenCounter
from litellm.llms.mistral.count_tokens.transformation import MistralCountTokensConfig

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
    sent: Final = decoded_json_object(json.loads(route.calls.last.request.content))
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
    sent: Final = decoded_json_object(json.loads(route.calls.last.request.content))
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


@pytest.mark.parametrize("mistral_api_key", [None, "env-mistral-key"])
@pytest.mark.asyncio
async def test_count_tokens_never_sends_the_azure_env_key_to_mistral(
    monkeypatch: pytest.MonkeyPatch, mistral_api_key: str | None
):
    monkeypatch.setenv("MISTRAL_AZURE_API_KEY", "azure-test-key")
    if mistral_api_key is None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", mistral_api_key)
    monkeypatch.delenv("MISTRAL_AZURE_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 9})
        )
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {}},
            request_model="mistral/mistral-large-4",
        )
    assert result is None
    assert not route.called


@pytest.mark.asyncio
async def test_count_tokens_mistral_env_key_takes_precedence_over_generic_litellm_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("MISTRAL_API_KEY", "real-mistral-key")
    monkeypatch.delenv("MISTRAL_AZURE_API_KEY", raising=False)
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


_MISTRAL_ENV_VARS: Final = ("MISTRAL_API_KEY", "MISTRAL_API_BASE", "MISTRAL_AZURE_API_KEY", "MISTRAL_AZURE_API_BASE")


def _set_mistral_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name in _MISTRAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("env", "litellm_params"),
    [
        ({"MISTRAL_API_KEY": "env-mistral-key"}, {}),
        ({"MISTRAL_API_KEY": "env-mistral-key", "MISTRAL_API_BASE": "https://egress.example.com"}, {}),
        ({"MISTRAL_API_KEY": "env-mistral-key"}, {"api_key": "deployment-key"}),
        ({}, {"api_key": "deployment-key", "api_base": "https://mistral-gateway.example.com"}),
        ({}, {"api_key": "deployment-key", "api_base": "https://mistral-gateway.example.com/v1"}),
    ],
)
@pytest.mark.asyncio
async def test_count_tokens_sends_the_key_and_base_chat_resolves(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], litellm_params: dict[str, str]
):
    _set_mistral_env(monkeypatch, env)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _, _, chat_key, chat_base = litellm.get_llm_provider(
        model="mistral/mistral-large-4",
        api_base=litellm_params.get("api_base"),
        api_key=litellm_params.get("api_key"),
    )
    with respx.mock:
        route: Final = respx.post(f"{chat_base}/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 7})
        )
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": litellm_params},
            request_model="mistral/mistral-large-4",
        )
    assert result is not None
    assert result.total_tokens == 7
    assert route.calls.last.request.headers["authorization"] == f"Bearer {chat_key}"


@pytest.mark.parametrize(
    "env",
    [
        {"MISTRAL_API_KEY": "mistral-key", "MISTRAL_API_BASE": "https://egress.example.com"},
        {
            "MISTRAL_API_KEY": "mistral-key",
            "MISTRAL_AZURE_API_KEY": "azure-key",
            "MISTRAL_AZURE_API_BASE": "https://mistral-resource.services.ai.azure.com/models",
        },
    ],
)
def test_mistral_model_info_key_and_base_match_chat(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    from litellm.llms.mistral.chat.transformation import MistralConfig
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    _set_mistral_env(monkeypatch, env)
    _, _, chat_key, chat_base = litellm.get_llm_provider(model="mistral/mistral-large-4")
    model_info: Final = ProviderConfigManager.get_provider_model_info(model=None, provider=LlmProviders.MISTRAL)
    assert isinstance(model_info, MistralConfig)
    assert (model_info.get_api_base(), model_info.get_api_key()) == (chat_base, chat_key)


def test_mistral_config_get_models_lists_ids_that_route_to_mistral():
    from litellm.llms.mistral.chat.transformation import MistralConfig

    with respx.mock:
        route: Final = respx.get("https://api.mistral.ai/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "mistral-large-4"}, {"id": "codestral-2508"}]})
        )
        models: Final = MistralConfig().get_models(api_key="test-key")

    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"
    assert models == ["mistral/mistral-large-4", "mistral/codestral-2508"]
    assert {litellm.get_llm_provider(model=model)[1] for model in models} == {"mistral"}


@pytest.mark.asyncio
async def test_native_responses_count_dispatches_to_mistral(monkeypatch: pytest.MonkeyPatch):
    import litellm
    from litellm.llms.mistral.count_tokens.token_counter import MistralTokenCounter
    from litellm.proxy.proxy_server import (
        _native_responses_count,  # pyright: ignore[reportPrivateUsage]  # testing private proxy dispatch helper
    )

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    callback: Final = _native_responses_count(
        provider_counter=counter,
        custom_llm_provider="mistral",
        responses_request={"model": "mistral-large-4", "input": "Hello world"},
        deployment={"litellm_params": {"model": "mistral/mistral-large-4", "api_key": "test-key"}},
        request_model="mistral-large-4",
    )
    assert callback is not None

    with respx.mock:
        route: Final = respx.post("https://api.mistral.ai/v1/responses/input_tokens").mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 5})
        )
        result: Final = await callback("mistral-large-4", None)
        assert result is not None
        assert getattr(result, "total_tokens", None) == 5
        assert route.called
        assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
        assert decoded_json_object(json.loads(route.calls.last.request.content)) == {
            "model": "mistral-large-4",
            "input": "Hello world",
        }


@pytest.mark.asyncio
async def test_mistral_count_tokens_handler_raises_mistral_error_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.mistral.common_utils import MistralError
    from litellm.llms.mistral.count_tokens.handler import MistralCountTokensHandler

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    handler: Final = MistralCountTokensHandler()
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=httpx.Response(400, text="Bad Request"))
        with pytest.raises(MistralError) as exc_info:
            await handler.post_count_request({"model": "mistral-large-4", "input": "hi"}, api_key="test-key")

    assert exc_info.value.status_code == 400
    assert "Bad Request" in exc_info.value.message


@pytest.mark.asyncio
async def test_mistral_count_tokens_handler_raises_mistral_error_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.mistral.common_utils import MistralError
    from litellm.llms.mistral.count_tokens.handler import MistralCountTokensHandler

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    handler: Final = MistralCountTokensHandler()
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=httpx.Response(200, text="not-valid-json"))
        with pytest.raises(MistralError) as exc_info:
            await handler.post_count_request({"model": "mistral-large-4", "input": "hi"}, api_key="test-key")

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_mistral_count_tokens_handler_transform_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.mistral.common_utils import MistralError
    from litellm.llms.mistral.count_tokens.handler import MistralCountTokensHandler

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    handler: Final = MistralCountTokensHandler()
    with patch.object(handler, "transform_request_to_count_tokens", side_effect=ValueError("Invalid request")):
        with pytest.raises(MistralError) as exc_info:
            await handler.handle_count_tokens_request(
                model="mistral-large-4",
                input="hello",
                api_key="test-key",
            )

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_count_tokens_system_blocks_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 12})
        )
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            system=[{"type": "text", "text": "system block"}],
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )

    assert result is not None
    assert result.total_tokens == 12
    assert route.called
    sent_json: Final = decoded_json_object(json.loads(route.calls.last.request.content))
    assert sent_json["instructions"] == "system block"


@pytest.mark.asyncio
async def test_count_tokens_generic_http_error_returns_count_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with patch(
        "litellm.llms.mistral.count_tokens.token_counter.mistral_count_tokens_handler.handle_count_tokens_request",
        side_effect=httpx.HTTPError("generic http error"),
    ):
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )

    assert result is not None
    assert result.error is True
    assert result.status_code == 500


@pytest.mark.asyncio
async def test_count_responses_request_azure_returns_none() -> None:
    counter: Final = MistralTokenCounter()
    result: Final = await counter.count_responses_request(
        model_to_use="mistral-large-4",
        request={"model": "mistral-large-4", "input": "hi"},
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
        deployment={"litellm_params": {"api_base": "https://foo.openai.azure.com", "api_key": "test-key"}},
        request_model="mistral/mistral-large-4",
    )
    assert result is None


def test_get_mistral_count_tokens_endpoint_other_path() -> None:
    config: Final = MistralCountTokensConfig()
    endpoint: Final = config.get_mistral_count_tokens_endpoint("https://custom.mistral.ai/api")
    assert endpoint == "https://custom.mistral.ai/api/v1/responses/input_tokens"
