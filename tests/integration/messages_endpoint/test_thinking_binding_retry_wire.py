import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, count, product
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.anthropic_sse import error_body
from integration._support.anthropic_thinking import (
    JSON_OBJECT,
    SIGNATURE,
    answer,
    message_body,
    prompt,
    standard_events,
    stream_reply,
)
from integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

Provider: TypeAlias = Literal[
    "anthropic", "azure_ai", "bedrock_invoke", "bedrock_mantle", "vertex_ai", "openai_passthrough"
]
Body: TypeAlias = dict[str, JsonValue]

_PROVIDERS: Final[tuple[Provider, ...]] = ("anthropic", "azure_ai", "bedrock_invoke", "bedrock_mantle", "vertex_ai")
_ADAPTIVE_MODELS: Final[Mapping[Provider, str]] = MappingProxyType(
    {
        "anthropic": "anthropic/claude-fable-5-2",
        "azure_ai": "azure_ai/claude-fable-5-2",
        "bedrock_invoke": "bedrock/invoke/anthropic.claude-fable-5-2",
        "bedrock_mantle": "bedrock_mantle/anthropic.claude-fable-5",
        "vertex_ai": "vertex_ai/claude-fable-5-2",
        "openai_passthrough": "openai/claude-fable-5-2",
    }
)
_BUDGET_MODELS: Final[Mapping[Provider, str]] = MappingProxyType(
    {
        "anthropic": "anthropic/claude-sonnet-4-6",
        "azure_ai": "azure_ai/claude-sonnet-4-6",
        "bedrock_invoke": "bedrock/invoke/anthropic.claude-sonnet-4-6",
        "bedrock_mantle": "bedrock_mantle/anthropic.claude-haiku-4-5",
        "vertex_ai": "vertex_ai/claude-sonnet-4-6",
    }
)
_VERTEX_PROJECT: Final = "scripted-project"
_BINDING_BETA: Final = "thinking-binding-controls-2026-08-01"
_CALLER_BETA: Final = "fine-grained-tool-streaming-2025-05-14"
_SIGNATURE_ERROR: Final = "messages.1.content.0: Invalid `signature` in `thinking` block"
_BOUND_ERROR: Final = f"{_SIGNATURE_ERROR}. The block is bound to a different conversation."
_BINDING_ERROR: Final = (
    f'{_BOUND_ERROR} Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block".'
)
_DROP_BLOCK: Final[Body] = {"prefix_mismatch_behavior": "drop_block"}
_ADAPTIVE: Final[Body] = {"type": "adaptive", "display": "summarized"}
_PINNED_ERROR: Final[Body] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}
_BUDGET: Final[Body] = {"type": "enabled", "budget_tokens": 1024}
_THINKING_TYPES: Final = frozenset({"thinking", "redacted_thinking"})
_SIGNED_BLOCKS: Final[tuple[Body, ...]] = (
    {"type": "thinking", "thinking": "weighing the lookup", "signature": SIGNATURE},
    {"type": "redacted_thinking", "data": "scripted-redacted-thinking"},
)
_TOOL_USE: Final[Body] = {"type": "tool_use", "id": "toolu_scripted", "name": "lookup", "input": {"key": "value"}}
_TOOL_RESULT: Final[Body] = {"type": "tool_result", "tool_use_id": "toolu_scripted", "content": "found"}
_TOOLS: Final[list[JsonValue]] = [
    {"name": "lookup", "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}}}
]

_every_route: Final = pytest.mark.parametrize(
    ("provider", "stream"),
    [
        pytest.param(provider, stream, id=f"{provider}-{'stream' if stream else 'non_stream'}")
        for provider, stream in product(_PROVIDERS, (False, True))
    ],
)


@dataclass(frozen=True, slots=True)
class _Exchange:
    response: httpx.Response
    requests: tuple[Request, ...]
    marker: str

    @property
    def bodies(self) -> tuple[Body, ...]:
        return tuple(JSON_OBJECT.validate_json(request.body) for request in self.requests)


def _deployment(scenario: Scenario, provider: Provider, model: str, wire_url: str) -> str:
    match provider:
        case "anthropic" | "azure_ai" | "bedrock_mantle":
            return scenario.model(model=model, api_base=wire_url, api_key=f"scripted-{provider}-key", num_retries=0)
        case "bedrock_invoke":
            return scenario.model(
                model=model,
                api_base=wire_url,
                api_key=None,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                num_retries=0,
            )
        case "vertex_ai":
            return scenario.model(
                model=model,
                api_base=wire_url,
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location="us-east5",
                vertex_credentials=service_account_json(_VERTEX_PROJECT, scenario.gateway.upstream_url.rstrip("/")),
                num_retries=0,
            )
        case "openai_passthrough":
            return scenario.model(
                model=model,
                api_base=f"{wire_url}/v1",
                api_key="scripted-openai-like-key",
                num_retries=0,
                model_info={"supported_endpoints": ["/v1/messages"]},
            )


def _rejection(provider: Provider, message: str) -> Reply:
    if provider == "bedrock_invoke":
        wrapped: Final = {"message": f"The model returned the following errors: {message}"}
        return Reply(status=400, body=json.dumps(wrapped).encode())
    return Reply(status=400, body=error_body(400, message))


def _completion(request: Request, marker: str, stream: bool) -> Reply:
    return stream_reply(request, standard_events(marker)) if stream else Reply(body=message_body(marker))


def _exchange(
    gateway: Gateway,
    provider: Provider,
    stream: bool,
    rejections: Sequence[str | None],
    thinking: Mapping[str, JsonValue] | None,
    models: Mapping[Provider, str] = _ADAPTIVE_MODELS,
) -> _Exchange:
    marker: Final = uuid.uuid4().hex
    attempts: Final = count()

    def respond(request: Request) -> Reply:
        rejection: Final = rejections[min(next(attempts), len(rejections) - 1)]
        return _completion(request, marker, stream) if rejection is None else _rejection(provider, rejection)

    conversation: Final[list[JsonValue]] = [
        {"role": "user", "content": prompt(marker)},
        {"role": "assistant", "content": [*_SIGNED_BLOCKS, _TOOL_USE]},
        {"role": "user", "content": [_TOOL_RESULT]},
    ]
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        deployed: Final = _deployment(scenario, provider, models[provider], wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": deployed,
                "max_tokens": 2048,
                "stream": stream,
                "tools": _TOOLS,
                "messages": conversation,
                **({} if thinking is None else {"thinking": dict(thinking)}),
            },
            headers={"anthropic-beta": _CALLER_BETA},
        )
        return _Exchange(response, wire.drain(), marker)


def _replayed_thinking(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    contents: Final = (object_value(message)["content"] for message in list_value(body["messages"]))
    blocks: Final = chain.from_iterable(content for content in contents if isinstance(content, list))
    return tuple(block for block in blocks if isinstance(block, dict) and block.get("type") in _THINKING_TYPES)


def _betas(provider: Provider, request: Request) -> frozenset[str]:
    if provider == "bedrock_invoke":
        carried: Final = list_value(JSON_OBJECT.validate_json(request.body).get("anthropic_beta", []))
        return frozenset(string_value(beta) for beta in carried)
    return frozenset(beta.strip() for beta in request.headers.get("anthropic-beta", "").split(","))


@_every_route
@pytest.mark.parametrize(
    ("caller_thinking", "retried_thinking"),
    [
        (_ADAPTIVE, {**_ADAPTIVE, "block_binding": _DROP_BLOCK}),
        (None, {"type": "adaptive", "block_binding": _DROP_BLOCK}),
    ],
    ids=["adaptive_with_display", "no_thinking"],
)
def test_regression_binding_mismatch_retries_with_drop_block_and_keeps_thinking(
    gateway: Gateway,
    provider: Provider,
    stream: bool,
    caller_thinking: Body | None,
    retried_thinking: Body,
) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (_BINDING_ERROR, None), caller_thinking)

    assert exchange.response.status_code == 200, exchange.response.text
    assert answer(exchange.marker) in exchange.response.text
    assert len(exchange.requests) == 2, [request.target for request in exchange.requests]
    first, second = exchange.bodies
    assert first.get("thinking") == caller_thinking, first
    assert _replayed_thinking(first) == _SIGNED_BLOCKS, first
    assert _CALLER_BETA in _betas(provider, exchange.requests[0])
    assert second["messages"] == first["messages"], second
    assert second["thinking"] == retried_thinking, second
    assert {_BINDING_BETA, _CALLER_BETA} <= _betas(provider, exchange.requests[1])


@_every_route
def test_regression_binding_mismatch_on_both_attempts_returns_the_provider_error(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (_BINDING_ERROR, _BINDING_ERROR), _ADAPTIVE)

    assert exchange.response.status_code == 400, exchange.response.text
    assert _BOUND_ERROR in exchange.response.text
    assert len(exchange.requests) == 2, [request.target for request in exchange.requests]
    assert _replayed_thinking(exchange.bodies[1]) == _SIGNED_BLOCKS, exchange.bodies[1]


@_every_route
def test_regression_invalid_signature_after_binding_retry_recovers_without_thinking(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (_BINDING_ERROR, _SIGNATURE_ERROR, None), _ADAPTIVE)

    assert exchange.response.status_code == 200, exchange.response.text
    assert answer(exchange.marker) in exchange.response.text
    assert len(exchange.requests) == 3, [request.target for request in exchange.requests]
    _, dropping, stripped = exchange.bodies
    assert dropping["thinking"] == {**_ADAPTIVE, "block_binding": _DROP_BLOCK}, dropping
    assert _replayed_thinking(dropping) == _SIGNED_BLOCKS, dropping
    assert _replayed_thinking(stripped) == (), stripped
    assert "thinking" not in stripped, stripped


@_every_route
def test_regression_invalid_signature_after_binding_retry_stops_at_the_third_attempt(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (_BINDING_ERROR, _SIGNATURE_ERROR), _ADAPTIVE)

    assert exchange.response.status_code == 400, exchange.response.text
    assert _SIGNATURE_ERROR in exchange.response.text
    assert len(exchange.requests) == 3, [request.target for request in exchange.requests]


@_every_route
def test_guard_caller_pinned_binding_error_is_not_retried(gateway: Gateway, provider: Provider, stream: bool) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (_BINDING_ERROR, None), _PINNED_ERROR)

    assert exchange.response.status_code == 400, exchange.response.text
    assert _BOUND_ERROR in exchange.response.text
    assert len(exchange.requests) == 1, [request.target for request in exchange.requests]
    assert exchange.bodies[0]["thinking"] == _PINNED_ERROR, exchange.bodies[0]


@_every_route
@pytest.mark.parametrize(
    ("rejection", "caller_thinking", "models"),
    [
        (_SIGNATURE_ERROR, _ADAPTIVE, _ADAPTIVE_MODELS),
        (_BINDING_ERROR, _BUDGET, _BUDGET_MODELS),
    ],
    ids=["signature_error_without_binding_sentence", "binding_error_with_budget_thinking"],
)
def test_guard_other_thinking_rejections_retry_without_thinking(
    gateway: Gateway,
    provider: Provider,
    stream: bool,
    rejection: str,
    caller_thinking: Body,
    models: Mapping[Provider, str],
) -> None:
    exchange: Final = _exchange(gateway, provider, stream, (rejection, None), caller_thinking, models)

    assert exchange.response.status_code == 200, exchange.response.text
    assert answer(exchange.marker) in exchange.response.text
    assert len(exchange.requests) == 2, [request.target for request in exchange.requests]
    first, second = exchange.bodies
    assert first["thinking"] == caller_thinking, first
    assert _replayed_thinking(first) == _SIGNED_BLOCKS, first
    assert _replayed_thinking(second) == (), second
    assert "thinking" not in second, second


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("caller_thinking", [_ADAPTIVE, None], ids=["adaptive_with_display", "no_thinking"])
def test_guard_openai_compatible_passthrough_retries_binding_mismatch_without_thinking(
    gateway: Gateway, stream: bool, caller_thinking: Body | None
) -> None:
    exchange: Final = _exchange(gateway, "openai_passthrough", stream, (_BINDING_ERROR, None), caller_thinking)

    assert exchange.response.status_code == 200, exchange.response.text
    assert answer(exchange.marker) in exchange.response.text
    assert len(exchange.requests) == 2, [request.target for request in exchange.requests]
    first, second = exchange.bodies
    assert first.get("thinking") == caller_thinking, first
    assert _replayed_thinking(first) == _SIGNED_BLOCKS, first
    assert _replayed_thinking(second) == (), second
    assert "thinking" not in second, second
    assert _BINDING_BETA not in _betas("openai_passthrough", exchange.requests[1])
