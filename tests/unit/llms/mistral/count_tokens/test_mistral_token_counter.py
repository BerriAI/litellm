import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import decoded_json_object
from litellm.llms.anthropic.messages.adapters.transformation import AnthropicAdapter
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.count_tokens.token_counter import MistralTokenCounter
from litellm.llms.mistral.count_tokens.transformation import MistralCountTokensConfig
from litellm.types.utils import TokenCountMessagesFormat, TokenCountResponse

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


@pytest.mark.parametrize(
    ("upstream_error", "status_code"),
    [(httpx.ConnectTimeout("Connection timed out"), 408), (httpx.ConnectError("Connection refused"), 500)],
    ids=["timeout", "connect_error"],
)
@pytest.mark.asyncio
async def test_count_tokens_reports_a_failed_connection_as_a_count_error(
    monkeypatch: pytest.MonkeyPatch, upstream_error: httpx.RequestError, status_code: int
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    counter: Final = MistralTokenCounter()
    with respx.mock:
        respx.post(COUNT_URL).mock(side_effect=upstream_error)
        result: Final = await counter.count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "Hello"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
        )
    assert result is not None
    assert (result.error, result.status_code, result.tokenizer_type) == (True, status_code, "mistral_api")


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
async def test_acount_tokens_never_sends_the_azure_env_key_to_mistral(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MISTRAL_API_KEY", "MISTRAL_API_BASE", "MISTRAL_AZURE_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISTRAL_AZURE_API_KEY", "azure-test-key")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 9})
        )
        result: Final = await litellm.acount_tokens(
            model="mistral/mistral-large-4", messages=[{"role": "user", "content": "Hello"}]
        )
    assert (route.called, result.tokenizer_type) == (False, "local_tokenizer")


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
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    _set_mistral_env(monkeypatch, env)
    _, _, chat_key, chat_base = litellm.get_llm_provider(model="mistral/mistral-large-4")
    model_info: Final = ProviderConfigManager.get_provider_model_info(model=None, provider=LlmProviders.MISTRAL)
    assert isinstance(model_info, MistralConfig)
    assert (model_info.get_api_base(), model_info.get_api_key()) == (chat_base, chat_key)


def test_check_provider_endpoint_lists_the_bundled_mistral_models_without_calling_mistral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.utils import get_valid_models

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    with respx.mock:
        listed: Final = get_valid_models(check_provider_endpoint=True, custom_llm_provider="mistral")

    assert listed
    assert listed == get_valid_models(check_provider_endpoint=False, custom_llm_provider="mistral")


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
async def test_mistral_count_tokens_handler_rejects_an_empty_input_without_calling_mistral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.mistral.common_utils import MistralError
    from litellm.llms.mistral.count_tokens.handler import MistralCountTokensHandler

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 1})
        )
        with pytest.raises(MistralError) as exc_info:
            await MistralCountTokensHandler().handle_count_tokens_request(
                model="mistral-large-4", input="", api_key="test-key"
            )

    assert exc_info.value.status_code == 500
    assert not route.called


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
async def test_count_tokens_returns_the_mistral_error_status_instead_of_a_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        respx.post(COUNT_URL).mock(
            return_value=httpx.Response(401, json={"detail": "Unauthorized"}),
        )
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use="mistral-large-4",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "bad-key"}},
            request_model="mistral/mistral-large-4",
        )

    assert result is not None
    assert (result.error, result.status_code, result.total_tokens) == (True, 401, 0)
    assert "Unauthorized" in (result.error_message or "")


async def _sent_count_body(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    chat_params: dict[str, JsonValue] | None = None,
    messages_format: TokenCountMessagesFormat = "openai",
    system: object = None,
    count_options: dict[str, JsonValue] | None = None,
) -> Mapping[str, object]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(
            return_value=httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 3})
        )
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use=model,
            messages=messages,
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model=f"mistral/{model}",
            tools=tools,
            chat_params=chat_params or {},
            messages_format=messages_format,
            system=system,
            count_options=count_options or {},
        )
    assert result is not None and result.total_tokens == 3
    return decoded_json_object(json.loads(route.calls.last.request.content))


async def _sent_count_input(
    monkeypatch: pytest.MonkeyPatch, model: str, messages: list[dict[str, object]]
) -> list[object]:
    sent_input: Final = (await _sent_count_body(monkeypatch, model, messages))["input"]
    assert isinstance(sent_input, list)
    return sent_input


_REASONED_TURN: Final = {"role": "assistant", "content": "Paris.", "reasoning_content": "France's capital is Paris."}
_REASONING_ITEM: Final = {
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "France's capital is Paris."}],
}
_ANSWER_ITEM: Final = {"role": "assistant", "content": "Paris."}
_THINK_CHUNK_TURN: Final = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": [{"type": "text", "text": "France's capital is Paris."}]},
        {"type": "text", "text": "Paris."},
    ],
}


@pytest.mark.parametrize(
    ("model", "assistant_turn", "counted_turn"),
    [
        ("magistral-medium-latest", _REASONED_TURN, [_REASONING_ITEM, _ANSWER_ITEM]),
        (
            "magistral-medium-latest",
            {
                "role": "assistant",
                "content": "Paris.",
                "thinking_blocks": [{"type": "thinking", "thinking": "France's capital is Paris."}],
            },
            [_REASONING_ITEM, _ANSWER_ITEM],
        ),
        ("magistral-medium-latest", {**_REASONED_TURN, "content": None}, []),
        ("magistral-medium-latest", {"role": "assistant", "content": ""}, []),
        ("mistral-no-reasoning-test-model", _REASONED_TURN, [_ANSWER_ITEM]),
        ("magistral-medium-latest", _THINK_CHUNK_TURN, [_REASONING_ITEM, _ANSWER_ITEM]),
        (
            "magistral-medium-latest",
            {**_THINK_CHUNK_TURN, "reasoning_content": "A reasoning text chat skips beside a thinking chunk."},
            [_REASONING_ITEM, _ANSWER_ITEM],
        ),
        ("mistral-no-reasoning-test-model", _THINK_CHUNK_TURN, [_ANSWER_ITEM]),
    ],
    ids=[
        "reasoning_content",
        "thinking_blocks",
        "no_turn_output",
        "empty_text",
        "model_without_reasoning",
        "thinking_chunk",
        "thinking_chunk_beside_reasoning_content",
        "thinking_chunk_model_without_reasoning",
    ],
)
@pytest.mark.asyncio
async def test_count_tokens_counts_reasoning_only_where_mistral_chat_replays_it(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    assistant_turn: dict[str, object],
    counted_turn: list[dict[str, object]],
) -> None:
    sent_input: Final = await _sent_count_input(
        monkeypatch,
        model,
        [
            {"role": "user", "content": "Capital of France?"},
            assistant_turn,
            {"role": "user", "content": "And of Spain?"},
        ],
    )

    assert sent_input == [
        {"role": "user", "content": "Capital of France?"},
        *counted_turn,
        {"role": "user", "content": "And of Spain?"},
    ]


_IMAGE_URL: Final = "https://example.com/cat.png"


@pytest.mark.parametrize(
    ("sibling_blocks", "sibling_parts"),
    [
        ([], []),
        (
            [{"type": "image_url", "image_url": {"url": _IMAGE_URL}}],
            [{"type": "input_image", "image_url": _IMAGE_URL, "detail": "auto"}],
        ),
        (
            [{"type": "file", "file": {"filename": "notes.txt", "file_data": "data:text/plain;base64,aGk="}}],
            [{"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,aGk="}],
        ),
    ],
    ids=["text_only", "with_image", "with_inline_file"],
)
@pytest.mark.asyncio
async def test_count_tokens_counts_files_sent_by_file_id(
    monkeypatch: pytest.MonkeyPatch,
    sibling_blocks: list[dict[str, object]],
    sibling_parts: list[dict[str, object]],
) -> None:
    sent_input: Final = await _sent_count_input(
        monkeypatch,
        "mistral-large-4",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize both"},
                    *sibling_blocks,
                    {"type": "file", "file": {"file_id": "file-openai-shape"}},
                    {"type": "file", "file_id": "file-mistral-shape"},
                ],
            }
        ],
    )

    assert sent_input == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize both"},
                *sibling_parts,
                {"type": "input_file", "file_id": "file-openai-shape"},
                {"type": "input_file", "file_id": "file-mistral-shape"},
            ],
        }
    ]


def test_get_mistral_count_tokens_endpoint_other_path() -> None:
    config: Final = MistralCountTokensConfig()
    endpoint: Final = config.get_mistral_count_tokens_endpoint("https://custom.mistral.ai/api")
    assert endpoint == "https://custom.mistral.ai/api/v1/responses/input_tokens"


async def _count_with(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    deployment_params: dict[str, object],
    count_response: dict[str, object],
    chat_params: dict[str, JsonValue] | None = None,
) -> tuple[TokenCountResponse | None, list[httpx.Request]]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json=count_response))
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use=model,
            messages=[{"role": "user", "content": "Give me a city as JSON"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key", **deployment_params}},
            request_model=f"mistral/{model}",
            chat_params=chat_params or {},
        )
    return result, [call.request for call in route.calls]


_COUNTED: Final = {"object": "response.input_tokens", "input_tokens": 166}


@pytest.mark.asyncio
async def test_count_tokens_sends_the_bridged_response_format_as_text_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema: Final = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    result, requests = await _count_with(
        monkeypatch,
        "mistral-large-4",
        {},
        _COUNTED,
        {"response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}}},
    )

    assert result is not None and result.total_tokens == 166
    (request,) = requests
    assert json.loads(request.content)["text"] == {
        "format": {"type": "json_schema", "name": "city", "schema": schema, "strict": True}
    }


@pytest.mark.parametrize("model", ["mistral-embed", "mistral-ocr-latest"])
@pytest.mark.asyncio
async def test_count_tokens_leaves_models_mistral_cannot_count_to_the_local_estimate(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    result, requests = await _count_with(monkeypatch, model, {}, _COUNTED)

    assert (result, requests) == (None, [])


@pytest.mark.asyncio
async def test_count_tokens_uses_the_deployment_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _, requests = await _count_with(monkeypatch, "mistral-large-4", {"timeout": 7}, _COUNTED)

    (request,) = requests
    assert request.extensions["timeout"]["read"] == 7.0


@pytest.mark.asyncio
async def test_count_tokens_falls_back_when_mistral_answers_without_a_count(monkeypatch: pytest.MonkeyPatch) -> None:
    result, requests = await _count_with(monkeypatch, "mistral-large-4", {}, {"object": "response.input_tokens"})

    assert (result, len(requests)) == (None, 1)


_LOOKUP_TOOL: Final[dict[str, object]] = {
    "type": "function",
    "function": {"name": "lookup", "description": "Look up a city", "parameters": {"type": "object"}},
}
_LOOKUP_ASK: Final[list[dict[str, object]]] = [{"role": "user", "content": "Look up Paris"}]


@pytest.mark.parametrize(
    ("tool_choice", "counted_tool_choice"),
    [
        ("none", "none"),
        ("auto", "auto"),
        ("required", "required"),
        ("any", "required"),
        ({"type": "function", "function": {"name": "lookup"}}, {"type": "function", "name": "lookup"}),
    ],
    ids=["none", "auto", "required", "mistral_any", "named_function"],
)
@pytest.mark.asyncio
async def test_count_tokens_sends_the_tool_choice_chat_sends_in_the_responses_shape(
    monkeypatch: pytest.MonkeyPatch, tool_choice: JsonValue, counted_tool_choice: JsonValue
) -> None:
    sent: Final = await _sent_count_body(
        monkeypatch, "mistral-large-4", _LOOKUP_ASK, tools=[_LOOKUP_TOOL], chat_params={"tool_choice": tool_choice}
    )

    assert sent["tool_choice"] == counted_tool_choice


@pytest.mark.asyncio
async def test_count_tokens_sends_no_tool_choice_without_tools_like_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: Final = await _sent_count_body(monkeypatch, "mistral-large-4", _LOOKUP_ASK, chat_params={"tool_choice": "none"})

    assert "tool_choice" not in sent


def _chat_messages(model: str, messages: list[dict[str, object]], params: dict[str, object]) -> list[object]:
    config: Final = MistralConfig()
    optional_params: Final = config.map_openai_params(
        non_default_params=params, optional_params={}, model=model, drop_params=False
    )
    sent_messages: Final = config.transform_request(
        model=model,
        messages=json.loads(json.dumps(messages)),
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )["messages"]
    assert isinstance(sent_messages, list)
    return sent_messages


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Capital of France?"}],
        [{"role": "system", "content": "Answer in one word."}, {"role": "user", "content": "Capital of France?"}],
        [
            {"role": "system", "content": [{"type": "text", "text": "Answer in one word."}]},
            {"role": "user", "content": "Capital of France?"},
        ],
    ],
    ids=["no_system_message", "system_text", "system_blocks"],
)
@pytest.mark.asyncio
async def test_count_tokens_counts_the_messages_mistral_chat_sends_magistral_with_reasoning(
    monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, object]]
) -> None:
    model: Final = "magistral-medium-latest"
    chat_sent: Final = _chat_messages(model, messages, {"reasoning_effort": "high"})
    counted: Final = await _sent_count_body(monkeypatch, model, messages, chat_params={"reasoning_effort": "high"})
    without_reasoning: Final = await _sent_count_body(monkeypatch, model, messages)

    counted_input: Final = counted["input"]
    assert isinstance(counted_input, list)
    assert len(counted_input) == len(chat_sent)
    assert counted == without_reasoning


@pytest.mark.asyncio
async def test_count_tokens_sends_each_system_message_where_chat_sends_it(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: Final = await _sent_count_body(
        monkeypatch,
        "mistral-large-4",
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": [{"type": "text", "text": "Now answer "}, {"type": "text", "text": "in French."}]},
            {"role": "user", "content": "Say goodbye."},
        ],
    )

    assert (sent["input"], "instructions" in sent) == (
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "Now answer in French."},
            {"role": "user", "content": "Say goodbye."},
        ],
        False,
    )


def _text_of(item: object, keys: tuple[str, ...]) -> object:
    fields: Final = decoded_json_object(item)
    return next((fields[key] for key in keys if fields.get(key)), None)


def _texts(items: list[object], *keys: str) -> list[object]:
    return list(filter(None, (_text_of(item, keys) for item in items)))


@pytest.mark.asyncio
async def test_count_tokens_joins_text_blocks_the_way_mistral_chat_does(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: Final[list[dict[str, object]]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize the plan"}, {"type": "text", "text": " for the release."}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Ship the "}, {"type": "text", "text": "fix."}]},
        {"role": "user", "content": [{"type": "text", "text": "Look it "}, {"type": "text", "text": "up."}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "plan"}'}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "Release "}, {"type": "text", "text": "on Monday."}],
        },
    ]
    chat_sent: Final = _chat_messages("mistral-large-4", messages, {})
    counted: Final = await _sent_count_body(monkeypatch, "mistral-large-4", messages)

    counted_input: Final = counted["input"]
    assert isinstance(counted_input, list)
    assert _texts(counted_input, "content", "output") == _texts(chat_sent, "content") == [
        "Summarize the plan for the release.",
        "Ship the fix.",
        "Look it up.",
        "Release on Monday.",
    ]


@pytest.mark.asyncio
async def test_count_tokens_keeps_file_ids_where_mistral_chat_sends_them(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: Final[list[dict[str, object]]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare"},
                {"type": "file", "file": {"file_id": "file-release-notes"}},
                {"type": "text", "text": "with the plan"},
            ],
        }
    ]
    chat_sent: Final = _chat_messages("mistral-large-4", messages, {})
    sent_input: Final = await _sent_count_input(monkeypatch, "mistral-large-4", messages)

    assert chat_sent == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare"},
                {"type": "file", "file_id": "file-release-notes"},
                {"type": "text", "text": "with the plan"},
            ],
        }
    ]
    assert sent_input == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Compare"},
                {"type": "input_file", "file_id": "file-release-notes"},
                {"type": "input_text", "text": "with the plan"},
            ],
        }
    ]


_CAPITALS_ASKED: Final = (
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Spain?"},
)


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": ""},
            _CAPITALS_ASKED[0],
            {"role": "user", "content": ""},
            _CAPITALS_ASKED[1],
            {"role": "user", "content": "", "name": "reviewer"},
            _CAPITALS_ASKED[2],
        ],
        [
            _CAPITALS_ASKED[0],
            {"role": "tool", "tool_call_id": "call_never_made", "content": ""},
            _CAPITALS_ASKED[1],
            _CAPITALS_ASKED[2],
        ],
    ],
    ids=["empty_system_and_user_turns", "empty_reply_to_no_tool_call"],
)
@pytest.mark.asyncio
async def test_count_tokens_drops_the_empty_turns_mistral_chat_drops(
    monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, object]]
) -> None:
    chat_sent: Final = _chat_messages("mistral-large-4", messages, {})
    sent_input: Final = await _sent_count_input(monkeypatch, "mistral-large-4", messages)

    assert sent_input == chat_sent == list(_CAPITALS_ASKED)


@pytest.mark.asyncio
async def test_count_tokens_joins_anthropic_tool_result_text_the_way_mistral_chat_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "sunny, "}, {"type": "text", "text": "21C"}],
                }
            ],
        },
    ]
    sent: Final = await _sent_count_body(
        monkeypatch,
        "mistral-large-4",
        messages,
        tools=[{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
        messages_format="anthropic",
    )

    sent_input: Final = sent["input"]
    assert isinstance(sent_input, list)
    assert {"type": "function_call_output", "call_id": "toolu_1", "output": "sunny, 21C"} in sent_input


_ANTHROPIC_LOOKUP_TOOL: Final[dict[str, object]] = {
    "name": "lookup",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


_CHAT_MESSAGES: Final = TypeAdapter(list[dict[str, object]])


def _chat_request_for_anthropic(model: str, body: dict[str, object]) -> Mapping[str, object]:
    chat_request: Final = AnthropicAdapter().translate_completion_input_params({**body, "model": model})
    assert chat_request is not None
    return chat_request


@pytest.mark.parametrize(
    ("thinking", "system"),
    [
        ({"type": "enabled", "budget_tokens": 2048}, None),
        ({"type": "enabled", "budget_tokens": 2048}, "Answer in one word."),
        ({"type": "adaptive"}, [{"type": "text", "text": "Answer in one word."}]),
    ],
    ids=["enabled_no_system", "enabled_system_text", "adaptive_system_blocks"],
)
@pytest.mark.asyncio
async def test_anthropic_count_with_thinking_counts_the_system_mistral_chat_sends_magistral(
    monkeypatch: pytest.MonkeyPatch, thinking: dict[str, JsonValue], system: object
) -> None:
    model: Final = "magistral-medium-latest"
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": "Capital of France?"}]
    chat_request: Final = _chat_request_for_anthropic(
        model, {"messages": messages, "thinking": thinking, **({} if system is None else {"system": system})}
    )
    chat_messages: Final = _CHAT_MESSAGES.validate_python(chat_request["messages"])
    chat_params: Final = {name: chat_request[name] for name in ("reasoning_effort", "thinking") if name in chat_request}
    chat_sent: Final = _chat_messages(model, chat_messages, chat_params)
    chat_systems: Final = [m["content"] for m in chat_sent if isinstance(m, dict) and m["role"] == "system"]

    counted: Final = await _sent_count_body(
        monkeypatch, model, messages, messages_format="anthropic", system=system, count_options={"thinking": thinking}
    )
    without_thinking: Final = await _sent_count_body(
        monkeypatch, model, messages, messages_format="anthropic", system=system
    )

    assert counted.get("instructions") == (chat_systems[0] if chat_systems else None)
    assert counted == without_thinking


@pytest.mark.asyncio
async def test_anthropic_count_joins_system_blocks_the_way_mistral_chat_does(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "mistral-large-4"
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": "Capital of France?"}]
    system: Final = [{"type": "text", "text": "Answer in one word."}, {"type": "text", "text": "Be terse."}]
    chat_request: Final = _chat_request_for_anthropic(model, {"messages": messages, "system": system})
    chat_messages: Final = _CHAT_MESSAGES.validate_python(chat_request["messages"])
    chat_system: Final = _chat_messages(model, chat_messages, {})[0]
    assert isinstance(chat_system, dict)

    counted: Final = await _sent_count_body(monkeypatch, model, messages, messages_format="anthropic", system=system)

    assert counted["instructions"] == chat_system["content"]


@pytest.mark.parametrize(
    ("tool_choice", "counted_tool_choice"),
    [
        ({"type": "none"}, "none"),
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "tool", "name": "lookup"}, {"type": "function", "name": "lookup"}),
    ],
    ids=["none", "auto", "any", "named_tool"],
)
@pytest.mark.asyncio
async def test_anthropic_count_sends_the_tool_choice_mistral_chat_sends(
    monkeypatch: pytest.MonkeyPatch, tool_choice: dict[str, JsonValue], counted_tool_choice: JsonValue
) -> None:
    sent: Final = await _sent_count_body(
        monkeypatch,
        "mistral-large-4",
        _LOOKUP_ASK,
        tools=[_ANTHROPIC_LOOKUP_TOOL],
        messages_format="anthropic",
        count_options={"tool_choice": tool_choice},
    )

    assert sent["tool_choice"] == counted_tool_choice


@pytest.mark.asyncio
async def test_anthropic_count_rejects_an_unknown_tool_choice_before_calling_mistral() -> None:
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json=_COUNTED))
        result: Final = await MistralTokenCounter().count_tokens(
            model_to_use="mistral-large-4",
            messages=_LOOKUP_ASK,
            contents=None,
            deployment={"litellm_params": {"api_key": "test-key"}},
            request_model="mistral/mistral-large-4",
            tools=[_ANTHROPIC_LOOKUP_TOOL],
            messages_format="anthropic",
            count_options={"tool_choice": {"type": "sometimes"}},
        )

    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
    assert not route.called
