import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.deepseek.count_tokens.token_counter import DeepSeekTokenCounter
from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import TokenCountMessagesFormat, TokenCountResponse

_DEFAULT_COUNT_ROUTE: Final = "https://api.deepseek.com/anthropic/v1/messages/count_tokens"
_WEATHER_IN_SF: Final[list[dict[str, JsonValue]]] = [{"role": "user", "content": "Weather in SF?"}]
_GET_WEATHER: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


@pytest.fixture(autouse=True)
def _deepseek_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DEEPSEEK_API_BASE", "DEEPSEEK_ANTHROPIC_API_BASE", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


async def _count(
    reply: httpx.Response,
    count_route: str = _DEFAULT_COUNT_ROUTE,
    litellm_params: Mapping[str, JsonValue] = MappingProxyType({"api_key": "sk-deepseek-test"}),
    messages_format: TokenCountMessagesFormat = "anthropic",
    tools: list[dict[str, JsonValue]] | None = None,
    chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    count_options: Mapping[str, JsonValue] = MappingProxyType({}),
    messages: list[dict[str, JsonValue]] = _WEATHER_IN_SF,
) -> tuple[TokenCountResponse | None, respx.Route]:
    with respx.mock:
        route: Final = respx.post(count_route).mock(return_value=reply)
        result: Final = await DeepSeekTokenCounter().count_tokens(
            model_to_use="deepseek-v4-pro",
            messages=messages,
            contents=None,
            deployment={"litellm_params": dict(litellm_params)},
            request_model="deepseek-pro",
            tools=tools,
            messages_format=messages_format,
            chat_params=chat_params,
            count_options=count_options,
        )
    return result, route


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base",
    (
        None,
        "https://api.deepseek.com",
        "https://api.deepseek.com/beta",
        "https://api.deepseek.com/anthropic/v1/messages",
        "https://gateway.example/deepseek",
    ),
)
async def test_count_posts_under_the_base_messages_inference_posts_to(api_base: str | None) -> None:
    messages_url: Final = DeepSeekAnthropicMessagesConfig().get_complete_url(
        api_base=api_base, api_key=None, model="deepseek-v4-pro", optional_params={}, litellm_params={}
    )

    result, route = await _count(
        httpx.Response(200, json={"input_tokens": 91}),
        count_route=f"{messages_url}/count_tokens",
        litellm_params={"api_key": "sk-deepseek-test", **({} if api_base is None else {"api_base": api_base})},
    )

    assert route.calls.last.request.headers["x-api-key"] == "sk-deepseek-test"
    assert result is not None
    assert (result.total_tokens, result.error) == (91, False)


@pytest.mark.asyncio
async def test_count_without_a_key_sends_no_api_key_header() -> None:
    _, route = await _count(httpx.Response(200, json={"input_tokens": 91}), litellm_params={})

    assert "x-api-key" not in route.calls.last.request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("messages_format", ("anthropic", "openai"))
async def test_count_sends_tools_without_the_custom_type_deepseek_refuses(
    messages_format: TokenCountMessagesFormat,
) -> None:
    tool: Final[dict[str, JsonValue]] = (
        {**_GET_WEATHER, "type": "custom"}
        if messages_format == "anthropic"
        else {
            "type": "function",
            "function": {"name": "get_weather", "parameters": _GET_WEATHER["input_schema"]},
        }
    )

    _, route = await _count(
        httpx.Response(200, json={"input_tokens": 91}), messages_format=messages_format, tools=[tool]
    )

    assert json.loads(route.calls.last.request.content)["tools"] == [_GET_WEATHER]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_params", "expected_options"),
    (
        ({}, {}),
        ({"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}}),
        ({"reasoning_effort": "max"}, {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}}),
        ({"reasoning_effort": "unlisted"}, {"thinking": {"type": "enabled"}}),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, {"thinking": {"type": "enabled"}}),
    ),
)
async def test_chat_count_maps_thinking_and_effort_the_way_deepseek_chat_inference_does(
    chat_params: Mapping[str, JsonValue], expected_options: Mapping[str, JsonValue]
) -> None:
    _, route = await _count(
        httpx.Response(200, json={"input_tokens": 91}), messages_format="openai", chat_params=chat_params
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert {name: sent[name] for name in ("thinking", "output_config") if name in sent} == expected_options


@pytest.mark.asyncio
async def test_messages_count_sends_the_thinking_messages_inference_sends() -> None:
    _, route = await _count(
        httpx.Response(200, json={"input_tokens": 91}),
        count_options={"thinking": {"type": "disabled"}, "tool_choice": {"type": "any"}},
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert (sent["thinking"], sent["tool_choice"]) == ({"type": "disabled"}, {"type": "any"})


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (404, 405))
async def test_a_missing_count_route_leaves_the_count_to_the_estimate(status_code: int) -> None:
    result, route = await _count(httpx.Response(status_code, text="no such route"))

    assert route.called
    assert result is None


@pytest.mark.asyncio
async def test_a_rejected_count_answers_with_deepseeks_status() -> None:
    result, _ = await _count(httpx.Response(401, json={"error": {"message": "Authentication Fails"}}))

    assert result is not None
    assert (result.error, result.status_code, result.total_tokens) == (True, 401, 0)
    assert "Authentication Fails" in (result.error_message or "")


@pytest.mark.asyncio
async def test_count_without_messages_leaves_the_count_to_the_estimate() -> None:
    result, route = await _count(httpx.Response(200, json={"input_tokens": 91}), messages=[])

    assert (result, route.called) == (None, False)


@pytest.mark.asyncio
async def test_messages_count_rejects_an_effort_messages_inference_rejects_without_calling_deepseek() -> None:
    options: Final[dict[str, JsonValue]] = {"output_config": {"effort": "unlisted"}}
    with pytest.raises(AnthropicError) as rejected:
        DeepSeekAnthropicMessagesConfig().transform_anthropic_messages_request(
            model="deepseek-v4-pro",
            messages=[dict(message) for message in _WEATHER_IN_SF],
            anthropic_messages_optional_request_params={**options, "max_tokens": 4096},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    result, route = await _count(httpx.Response(200, json={"input_tokens": 91}), count_options=options)

    assert route.called is False
    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
    assert (rejected.value.status_code, str(rejected.value.message)) == (400, result.error_message)


@pytest.mark.asyncio
async def test_chat_count_rejects_a_tool_call_chat_inference_rejects_without_calling_deepseek() -> None:
    messages: Final[list[dict[str, JsonValue]]] = [
        *_WEATHER_IN_SF,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "sunny?"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    result, route = await _count(
        httpx.Response(200, json={"input_tokens": 91}), messages_format="openai", messages=messages
    )

    assert route.called is False
    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
