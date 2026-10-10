"""A request with no model behind a `*` wildcard: a forwarding target (`openai/*`) must answer 400 on the three LLM
endpoints instead of sending the provider a made-up model, and a fixed target (`openai/gpt-4o-mini`) must keep
serving it, for every model shape, streaming and not, and through the OpenAI and Anthropic SDKs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration.compatibility._status_code_audit import (
    UNSET,
    CHAT,
    CHAT_FRAMES,
    EMBEDDING,
    ENDPOINT_BODIES,
    ENDPOINT_PATHS,
    RESPONSE,
    RESPONSES_FRAMES,
    ROUTER_DEFAULTS,
    assembled_text,
    assert_no_provider_call,
    drain_rig_upstream,
    invalid_request,
    json_response,
    one_outbound,
    owned_gateway,
    post,
    register,
    sse_response,
    stream_finished,
    stream_lines,
)
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import RoutedResponse, StoredResponse

pytestmark: Final = pytest.mark.timeout(180)


@pytest.fixture(autouse=True)
def _drained_upstream() -> None:
    drain_rig_upstream()


_MODEL_SHAPES: Final[dict[str, object]] = {"missing": UNSET, "null": None, "empty": ""}
_TARGETS: Final[dict[str, str]] = {
    "forwarding": "openai/*",
    "fixed": "openai/gpt-4o-mini",
    "fixed-chat-stream": "openai/gpt-4o-mini",
    "fixed-responses-stream": "openai/gpt-4o-mini",
}


@dataclass(frozen=True, slots=True)
class _Wildcard:
    gateway: Gateway
    identity: str


def _scripted(flavor: str) -> StoredResponse:
    """The fixed-target stream flavors answer every path with one SSE script: chat frames for chat, Responses
    frames for /v1/responses and for /v1/messages, which reaches an `openai/` target through the Responses bridge."""
    if flavor == "fixed-chat-stream":
        return sse_response(CHAT_FRAMES)
    if flavor == "fixed-responses-stream":
        return sse_response(RESPONSES_FRAMES)
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /chat/completions": json_response(CHAT),
            "POST /responses": json_response(RESPONSE),
            "POST /embeddings": json_response(EMBEDDING),
        },
    )


@pytest.fixture(scope="module")
def wildcard(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Wildcard]:
    """A proxy whose only deployment is the `*` wildcard, one flavor at a time so one proxy runs at once."""
    flavor: Final = str(request.param)
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        identity, handle = register(scenario, f"audit-{flavor}", _scripted(flavor))
        config: Final[dict[str, JsonValue]] = {
            "model_list": [
                {
                    "model_name": "*",
                    "litellm_params": {"model": _TARGETS[flavor], "api_base": handle.api_base(), "api_key": identity},
                }
            ],
            "router_settings": {"num_retries": 0},
        }
        with owned_gateway(tmp_path_factory.mktemp(f"audit-{flavor}"), config) as owned:
            yield _Wildcard(owned, identity)


def _model_field(shape: str) -> dict[str, JsonValue]:
    value: Final = _MODEL_SHAPES[shape]
    return {} if value is UNSET else {"model": value}  # pyright: ignore[reportReturnType]  # None or "" by construction


def _body(endpoint: str, shape: str, *, stream: bool) -> dict[str, JsonValue]:
    return {
        **ENDPOINT_BODIES[endpoint],
        **({"stream": True} if stream else {}),
        **_model_field(shape),
    }


def _shown_model(shape: str) -> str:
    return "None" if shape in {"missing", "null"} else ""


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
@pytest.mark.parametrize("endpoint", tuple(ENDPOINT_PATHS))
def test_forwarding_wildcard_rejects_a_request_without_a_model(
    wildcard: _Wildcard, endpoint: str, shape: str, stream: bool
) -> None:
    response: Final = post(wildcard.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, shape, stream=stream))
    error: Final = invalid_request(response)
    assert f"Invalid model name passed in model={_shown_model(shape)}. " in str(error.get("message")), response.text
    assert_no_provider_call(wildcard.gateway, wildcard.identity)


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
@pytest.mark.parametrize("endpoint", tuple(ENDPOINT_PATHS))
def test_forwarding_wildcard_rejects_a_model_less_request_from_a_key_limited_to_another_model(
    wildcard: _Wildcard, endpoint: str, shape: str
) -> None:
    with wildcard.gateway.scenario() as scenario:
        key: Final = scenario.key(models=["audit-another-model"])
        response: Final = post(
            wildcard.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, shape, stream=False), key=key
        )
        invalid_request(response)
        assert_no_provider_call(wildcard.gateway, wildcard.identity)


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("endpoint", tuple(ENDPOINT_PATHS))
def test_forwarding_wildcard_without_a_model_and_without_a_key_is_401(wildcard: _Wildcard, endpoint: str) -> None:
    response: Final = post(
        wildcard.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, "missing", stream=False), key="sk-audit-bad"
    )
    assert response.status_code == 401, response.text
    assert_no_provider_call(wildcard.gateway, wildcard.identity)


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize(
    ("endpoint", "model"),
    (("chat", "gpt-4o-mini"), ("responses", "gpt-4o-mini"), ("messages", "gpt-4o-mini"), ("chat", "m" * 5000)),
    ids=("chat", "responses", "messages", "chat-5kb-model"),
)
def test_forwarding_wildcard_forwards_a_named_model(wildcard: _Wildcard, endpoint: str, model: str) -> None:
    response: Final = post(wildcard.gateway, ENDPOINT_PATHS[endpoint], {**ENDPOINT_BODIES[endpoint], "model": model})
    assert response.status_code == 200, response.text
    assert "scripted" in response.text, response.text
    outbound: Final = one_outbound(wildcard.gateway, wildcard.identity)
    assert outbound.get("model") == model, outbound


def _openai_call(gateway: Gateway, endpoint: str, *, stream: bool, asynchronous: bool) -> Callable[[], object]:
    base_url: Final = f"{gateway.client.base_url}/v1"

    def sync_call() -> object:
        with openai.OpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            if endpoint == "chat":
                return client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": "Hello"}], stream=stream
                )
            return client.responses.create(model="", input="Hello", store=False, stream=stream)

    async def async_call() -> object:
        async with openai.AsyncOpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            if endpoint == "chat":
                return await client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": "Hello"}], stream=stream
                )
            return await client.responses.create(model="", input="Hello", store=False, stream=stream)

    return (lambda: asyncio.run(async_call())) if asynchronous else sync_call


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize("endpoint", ("chat", "responses"))
def test_forwarding_wildcard_empty_model_through_the_openai_sdk(
    wildcard: _Wildcard, endpoint: str, stream: bool, asynchronous: bool
) -> None:
    call: Final = _openai_call(wildcard.gateway, endpoint, stream=stream, asynchronous=asynchronous)
    with pytest.raises(openai.BadRequestError) as raised:
        call()
    assert raised.value.status_code == 400
    assert "Invalid model name passed in model=. " in str(raised.value.message)
    assert_no_provider_call(wildcard.gateway, wildcard.identity)


def _anthropic_call(gateway: Gateway, *, stream: bool, asynchronous: bool) -> Callable[[], object]:
    base_url: Final = str(gateway.client.base_url)

    def sync_call() -> object:
        with anthropic.Anthropic(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            return client.messages.create(
                model="", max_tokens=16, messages=[{"role": "user", "content": "Hello"}], stream=stream
            )

    async def async_call() -> object:
        async with anthropic.AsyncAnthropic(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            return await client.messages.create(
                model="", max_tokens=16, messages=[{"role": "user", "content": "Hello"}], stream=stream
            )

    return (lambda: asyncio.run(async_call())) if asynchronous else sync_call


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_forwarding_wildcard_empty_model_through_the_anthropic_sdk(
    wildcard: _Wildcard, stream: bool, asynchronous: bool
) -> None:
    call: Final = _anthropic_call(wildcard.gateway, stream=stream, asynchronous=asynchronous)
    with pytest.raises(anthropic.BadRequestError) as raised:
        call()
    assert raised.value.status_code == 400
    assert "Invalid model name passed in model=. " in str(raised.value.message)
    assert_no_provider_call(wildcard.gateway, wildcard.identity)


@pytest.mark.parametrize("wildcard", ("forwarding",), indirect=True)
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
def test_forwarding_wildcard_embeddings_without_a_model(wildcard: _Wildcard, shape: str) -> None:
    pytest.skip(
        "BUG: /v1/embeddings without a model behind a forwarding `*` -> `openai/*` wildcard still reaches the "
        "provider (model 'None/None' or ''), the omission the fix closed for chat/responses/messages; same on base"
    )
    body: Final[dict[str, JsonValue]] = {"input": "Hello", **_model_field(shape)}
    response: Final = post(wildcard.gateway, "/v1/embeddings", body)
    assert response.status_code == 400, response.text
    assert_no_provider_call(wildcard.gateway, wildcard.identity)


@pytest.mark.parametrize("wildcard", ("fixed",), indirect=True)
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
@pytest.mark.parametrize("endpoint", tuple(ENDPOINT_PATHS))
def test_fixed_target_wildcard_serves_a_request_without_a_model(wildcard: _Wildcard, endpoint: str, shape: str) -> None:
    response: Final = post(wildcard.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, shape, stream=False))
    assert response.status_code == 200, response.text
    assert "scripted" in response.text, response.text
    calls: Final = one_outbound(wildcard.gateway, wildcard.identity)
    assert calls.get("model") == "gpt-4o-mini", calls


@pytest.mark.parametrize("wildcard", ("fixed",), indirect=True)
def test_fixed_target_wildcard_without_a_model_through_the_openai_sdk(wildcard: _Wildcard) -> None:
    with openai.OpenAI(
        base_url=f"{wildcard.gateway.client.base_url}/v1", api_key=wildcard.gateway.key, max_retries=0
    ) as client:
        completion: Final = client.chat.completions.create(model="", messages=[{"role": "user", "content": "Hello"}])
    assert completion.choices[0].message.content == "scripted"
    assert one_outbound(wildcard.gateway, wildcard.identity).get("model") == "gpt-4o-mini"


@pytest.mark.parametrize("wildcard", ("fixed",), indirect=True)
def test_fixed_target_wildcard_without_a_model_through_the_anthropic_sdk(wildcard: _Wildcard) -> None:
    async def call() -> anthropic.types.Message:
        async with anthropic.AsyncAnthropic(
            base_url=str(wildcard.gateway.client.base_url), api_key=wildcard.gateway.key, max_retries=0
        ) as client:
            return await client.messages.create(
                model="", max_tokens=16, messages=[{"role": "user", "content": "Hello"}]
            )

    message: Final = asyncio.run(call())
    block: Final = message.content[0]
    assert isinstance(block, anthropic.types.TextBlock), message
    assert block.text == "scripted"
    assert one_outbound(wildcard.gateway, wildcard.identity).get("model") == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("wildcard", "endpoint"),
    (
        ("fixed-chat-stream", "chat"),
        ("fixed-responses-stream", "responses"),
        ("fixed-responses-stream", "messages"),
    ),
    indirect=["wildcard"],
)
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
def test_fixed_target_wildcard_streams_a_request_without_a_model(
    wildcard: _Wildcard, endpoint: str, shape: str
) -> None:
    status, lines = stream_lines(wildcard.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, shape, stream=True))
    assert status == 200, lines
    assert stream_finished(endpoint, lines), lines
    assert assembled_text(lines) == "streamed response", lines
    outbound: Final = one_outbound(wildcard.gateway, wildcard.identity)
    assert outbound.get("model") == "gpt-4o-mini", outbound
    assert outbound.get("stream") is True, outbound


def _prefixed_wildcard_config(
    handle_base: str, identity: str, fallbacks: list[JsonValue] | None
) -> dict[str, JsonValue]:
    return {
        "model_list": [
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_base": handle_base, "api_key": identity},
            },
            {
                "model_name": "audit-fallback",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": handle_base, "api_key": identity},
            },
        ],
        "router_settings": {
            "num_retries": 0,
            "default_litellm_params": ROUTER_DEFAULTS,
            **({} if fallbacks is None else {"fallbacks": fallbacks}),
        },
    }


@pytest.fixture(scope="module")
def fallback_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Wildcard]:
    """A proxy whose only wildcard is provider-prefixed (`anthropic/*`) and whose `*` fallback chain answers every
    model nothing else serves, with router defaults the fallback deployment inherits."""
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        identity, handle = register(scenario, "audit-fallback", _scripted("fixed"))
        config: Final = _prefixed_wildcard_config(handle.api_base(), identity, [{"*": ["audit-fallback"]}])
        with owned_gateway(tmp_path_factory.mktemp("audit-fallback"), config) as owned:
            yield _Wildcard(owned, identity)


@pytest.mark.parametrize("endpoint", ("chat", "responses", "messages"))
@pytest.mark.parametrize("shape", tuple(_MODEL_SHAPES))
def test_missing_model_no_wildcard_matches_is_served_by_the_fallback(
    fallback_gateway: _Wildcard, endpoint: str, shape: str
) -> None:
    response: Final = post(fallback_gateway.gateway, ENDPOINT_PATHS[endpoint], _body(endpoint, shape, stream=False))
    assert response.status_code == 200, response.text
    assert "scripted" in response.text, response.text


def test_unlisted_model_served_by_the_fallback_inherits_the_router_default_max_tokens(
    fallback_gateway: _Wildcard,
) -> None:
    response: Final = post(
        fallback_gateway.gateway,
        "/v1/messages",
        {"model": "audit-unlisted-model", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200, response.text
    assert one_outbound(fallback_gateway.gateway, fallback_gateway.identity).get("max_output_tokens") == 32


@pytest.fixture(scope="module")
def prefixed_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Wildcard]:
    """The same proxy without router fallbacks, so only a request, key or team supplies the fallback chain."""
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        identity, handle = register(scenario, "audit-request-fallback", _scripted("fixed"))
        config: Final = _prefixed_wildcard_config(handle.api_base(), identity, None)
        with owned_gateway(tmp_path_factory.mktemp("audit-request-fallback"), config) as owned:
            yield _Wildcard(owned, identity)


_UNLISTED_MESSAGE: Final[dict[str, JsonValue]] = {
    "model": "audit-unlisted-model",
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.parametrize(
    "fallbacks",
    (
        pytest.param(["audit-fallback"], id="model-names"),
        pytest.param([{"model": "audit-fallback"}], id="client-style"),
        pytest.param([{"model": "audit-fallback", "temperature": 0}], id="client-style-with-params"),
    ),
)
def test_unlisted_model_with_a_request_fallback_inherits_the_router_default_max_tokens(
    prefixed_gateway: _Wildcard, fallbacks: list[JsonValue]
) -> None:
    response: Final = post(prefixed_gateway.gateway, "/v1/messages", {**_UNLISTED_MESSAGE, "fallbacks": fallbacks})
    assert response.status_code == 200, response.text
    assert one_outbound(prefixed_gateway.gateway, prefixed_gateway.identity).get("max_output_tokens") == 32


@pytest.mark.parametrize(
    "fallbacks",
    (
        pytest.param([{"*": ["audit-fallback"]}], id="star-chain"),
        pytest.param([{"model": "audit-fallback"}], id="client-style"),
    ),
)
@pytest.mark.parametrize("owner", ("key", "team"))
def test_unlisted_model_with_a_key_or_team_fallback_inherits_the_router_default_max_tokens(
    prefixed_gateway: _Wildcard, owner: str, fallbacks: list[JsonValue]
) -> None:
    router_settings: Final[dict[str, JsonValue]] = {"fallbacks": fallbacks}
    with prefixed_gateway.gateway.scenario() as scenario:
        key: Final = (
            scenario.key(router_settings=router_settings)
            if owner == "key"
            else scenario.key(team_id=scenario.team(router_settings=router_settings))
        )
        response: Final = post(prefixed_gateway.gateway, "/v1/messages", _UNLISTED_MESSAGE, key=key)
        assert response.status_code == 200, response.text
        assert one_outbound(prefixed_gateway.gateway, prefixed_gateway.identity).get("max_output_tokens") == 32


@pytest.mark.parametrize(
    "fallbacks",
    (
        pytest.param([], id="empty"),
        pytest.param([{"audit-other-model": ["audit-fallback"]}], id="another-model"),
    ),
)
def test_router_star_fallback_serves_an_unlisted_model_whatever_the_request_fallbacks_say(
    fallback_gateway: _Wildcard, fallbacks: list[JsonValue]
) -> None:
    response: Final = post(fallback_gateway.gateway, "/v1/messages", {**_UNLISTED_MESSAGE, "fallbacks": fallbacks})
    assert response.status_code == 200, response.text
    assert one_outbound(fallback_gateway.gateway, fallback_gateway.identity).get("max_output_tokens") == 32
