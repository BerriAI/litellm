import threading
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from integration._support.anthropic_sse import SseEvent, event_types, parse_sse
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.openai_wire import (
    answering_model_discovery,
    chat_stream,
    openai_error,
    posted_targets,
    responses_stream,
)
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_cooldown import cooldown_config, cooled, router_settings
from integration.routing.test_payment_required_mapping import deployment
from pydantic import JsonValue

_BACKEND: Final = "glm-5-fp8"
_HOSTED: Final = f"hosted_vllm/{_BACKEND}"
_BRIDGED: Final = "openai/gpt-5.3-codex"
_PROVIDER_KEY: Final = "synthetic-hosted-vllm-key"
_ANSWER: Final = "scripted answer before the upstream stalls"
_FALLBACK_ANSWER: Final = "scripted answer from the fallback deployment"
_PROMPT: Final = "Keep streaming while the upstream thinks."
_CHAT_TARGET: Final = "/v1/chat/completions"
_RESPONSES_TARGET: Final = "/v1/responses"
_STREAM_TIMEOUT_SECONDS: Final = 1
_DEPLOYMENT_TIMEOUT_SECONDS: Final = 30
_PAUSE_SECONDS: Final = 3.0
_ROW_SECONDS: Final = 60
_HOLD_SECONDS: Final = 2.0 * _ROW_SECONDS
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_ROWS: Final = 'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
_EITHER_GROUP_ROWS: Final = 'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE model_group IN (%s, %s)'
_OWNED_PROXY: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

Endpoint = Literal["chat", "responses", "messages"]
Shape = Literal["prompt", "pause_after_content", "pause_after_role", "stall_before_first", "hold_headers", "fail"]


def _stream_parts(target: str, answer: str) -> tuple[bytes, bytes, bytes]:
    identity: Final = f"stall-{uuid.uuid4().hex}"
    if target == _RESPONSES_TARGET:
        return responses_stream(identity, _BACKEND, answer)
    return chat_stream(identity, _BACKEND, answer)


def _sse(chunks: tuple[bytes, ...], *, pause: float = 0, gate: threading.Event | None = None) -> Reply:
    return Reply(content_type="text/event-stream", chunks=chunks, pause_between_chunks=pause, gate_before_first=gate)


@dataclass(frozen=True, slots=True)
class _Peer:
    shape: Shape
    answer: str
    release: threading.Event

    def __call__(self, request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target in (_CHAT_TARGET, _RESPONSES_TARGET), request
        opened, delta, closed = _stream_parts(request.target, self.answer)
        match self.shape:
            case "fail":
                return openai_error(500)
            case "hold_headers":
                assert self.release.wait(_HOLD_SECONDS), request
                return _sse((opened, delta, closed))
            case "pause_after_content":
                return _sse((opened + delta, closed), pause=_PAUSE_SECONDS)
            case "pause_after_role":
                return _sse((opened, delta + closed), pause=_PAUSE_SECONDS)
            case "stall_before_first":
                return _sse((opened, delta, closed), gate=self.release)
            case "prompt":
                return _sse((opened, delta, closed))


@contextmanager
def _peer(shape: Shape, answer: str = _ANSWER) -> Generator[Wire]:
    release: Final = threading.Event()
    with wire_server(answering_model_discovery(_Peer(shape, answer, release))) as wire:
        try:
            yield wire
        finally:
            release.set()


def _params(wire: Wire, backend: str) -> dict[str, JsonValue]:
    return {
        "model": backend,
        "api_base": f"{wire.url}/v1",
        "api_key": _PROVIDER_KEY,
        "timeout": _DEPLOYMENT_TIMEOUT_SECONDS,
        "stream_timeout": _STREAM_TIMEOUT_SECONDS,
    }


def _deployment(scenario: Scenario, wire: Wire, backend: str = _HOSTED) -> tuple[str, str]:
    group: Final = f"integration-stall-{uuid.uuid4().hex[:12]}"
    return group, deployment(scenario, group, _params(wire, backend))


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"
        case "messages":
            return "/v1/messages"


def _body(endpoint: Endpoint, model: str, extra: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "stream": True, **extra}
        case "responses":
            return {"model": model, "input": _PROMPT, "stream": True, **extra}
        case "messages":
            return {
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 16,
                "stream": True,
                **extra,
            }


def _streamed(gateway: Gateway, endpoint: Endpoint, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", _path(endpoint), _body(endpoint, model, {**_EXTRA, **extra}))


def _upstream_targets(wire: Wire) -> tuple[str, ...]:
    eventually(wire.received.qsize, lambda size: size >= 1, 10)
    return posted_targets(wire)


def _chat_errors(events: tuple[SseEvent, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(event.data["error"]) for event in events if "error" in event.data)


def _error_class(message: JsonValue) -> str:
    return string_value(message).split(":", 1)[0]


def _failure_row(gateway: Gateway, query: str, groups: tuple[str, ...]) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: gateway.rows(query, groups), lambda found: len(found) >= 1, seconds=_ROW_SECONDS)
    assert [row["status"] for row in rows] == ["failure"], rows
    return object_value(object_value(rows[0]["metadata"])["error_information"])


def _logged_failure(gateway: Gateway, model: str) -> tuple[JsonValue, JsonValue]:
    error: Final = _failure_row(gateway, _ROWS, (model,))
    return error["error_code"], error["error_class"]


@pytest.mark.parametrize(
    ("backend", "target"),
    (
        pytest.param(_HOSTED, _CHAT_TARGET, id="hosted-vllm"),
        pytest.param(_BRIDGED, _RESPONSES_TARGET, id="bridged-to-responses"),
    ),
)
def test_a_chat_stream_that_stalls_past_its_stream_timeout_ends_with_a_408_timeout_and_logs_it(
    gateway: Gateway, backend: str, target: str
) -> None:
    with _peer("pause_after_content") as wire, gateway.scenario() as scenario:
        model, _ = _deployment(scenario, wire, backend)
        response: Final = _streamed(gateway, "chat", model)
        assert response.status_code == 200, response.text
        assert _ANSWER in response.text, response.text
        events: Final = parse_sse(response.text)
        assert _chat_errors(events[:-1]) == () and "error" in events[-1].data, response.text
        error: Final = object_value(events[-1].data["error"])
        observed: Final = (error["code"], _error_class(error["message"]), *_logged_failure(gateway, model))
        assert observed == ("408", "litellm.Timeout", "408", "Timeout"), (error, observed)
        assert _upstream_targets(wire) == (target,)


def test_a_native_responses_stream_that_stalls_past_its_stream_timeout_fails_with_request_timeout_and_logs_it(
    gateway: Gateway,
) -> None:
    with _peer("pause_after_content") as wire, gateway.scenario() as scenario:
        model, _ = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "responses", model)
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        types: Final = event_types(events)
        assert "response.output_text.delta" in types, response.text
        assert types[-1] == "response.failed" and "response.completed" not in types, response.text
        failed: Final = object_value(object_value(events[-1].data["response"])["error"])
        observed: Final = (failed["code"], *_logged_failure(gateway, model))
        assert observed == ("request_timeout", "408", "Timeout"), (failed, observed)
        assert _upstream_targets(wire) == (_RESPONSES_TARGET,)


def test_a_messages_stream_that_stalls_past_its_stream_timeout_ends_with_a_timeout_error_event_and_logs_it(
    gateway: Gateway,
) -> None:
    with _peer("pause_after_content") as wire, gateway.scenario() as scenario:
        model, _ = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "messages", model)
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        types: Final = event_types(events)
        assert "content_block_delta" in types and "message_stop" not in types, response.text
        assert types[-1] == "error", response.text
        error: Final = object_value(events[-1].data["error"])
        observed: Final = (_error_class(error["message"]), *_logged_failure(gateway, model))
        assert observed == ("litellm.Timeout", "408", "Timeout"), (error, observed)
        assert _upstream_targets(wire) == (_CHAT_TARGET,)


def test_a_chat_stream_that_never_pauses_is_answered_in_full_under_the_same_stream_timeout(gateway: Gateway) -> None:
    with _peer("prompt") as wire, gateway.scenario() as scenario:
        model, _ = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "chat", model)
        assert response.status_code == 200, response.text
        assert _ANSWER in response.text, response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        events: Final = parse_sse(response.text)
        assert _chat_errors(events) == (), response.text
        assert _finish_reasons(events) == ("stop",), response.text
        assert _upstream_targets(wire) == (_CHAT_TARGET,)
        rows: Final = eventually(lambda: gateway.rows(_ROWS, (model,)), lambda found: len(found) >= 1, _ROW_SECONDS)
        assert [row["status"] for row in rows] == ["success"], rows


@pytest.mark.parametrize(
    ("endpoint", "shape"),
    (
        pytest.param("chat", "pause_after_role", id="chat-after-the-role-chunk"),
        pytest.param("messages", "stall_before_first", id="messages-before-the-first-chunk"),
    ),
)
def test_a_stream_that_stalls_before_any_content_falls_back_to_the_next_deployment(
    gateway: Gateway, endpoint: Endpoint, shape: Shape
) -> None:
    with _peer(shape) as stalled, _peer("prompt", _FALLBACK_ANSWER) as healthy, gateway.scenario() as scenario:
        primary, _ = _deployment(scenario, stalled)
        fallback, _ = _deployment(scenario, healthy)
        response: Final = _streamed(gateway, endpoint, primary, fallbacks=[fallback])
        assert response.status_code == 200, response.text
        assert _FALLBACK_ANSWER in response.text and _ANSWER not in response.text, response.text
        events: Final = parse_sse(response.text)
        assert _chat_errors(events) == () and _completed(endpoint, events), response.text
        assert (_upstream_targets(stalled), _upstream_targets(healthy)) == ((_CHAT_TARGET,), (_CHAT_TARGET,))


@dataclass(frozen=True, slots=True)
class _CooldownRig:
    gateway: Gateway
    cache: OwnedRedis


@pytest.fixture(scope="module")
def cooldown_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CooldownRig]:
    directory: Final = tmp_path_factory.mktemp("stream-stall-cooldown")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                directory,
                {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
                config=cooldown_config(directory, router_settings(cache, allowed_fails=0), prometheus=False),
            )
        )
        yield _CooldownRig(owned.gateway, cache)


@_OWNED_PROXY
@pytest.mark.parametrize(
    ("endpoint", "backend", "shape"),
    (
        pytest.param("chat", _HOSTED, "pause_after_content", id="chat"),
        pytest.param("chat", _HOSTED, "stall_before_first", id="chat-before-the-first-chunk"),
        pytest.param("responses", _HOSTED, "pause_after_content", id="responses"),
        pytest.param("chat", _BRIDGED, "pause_after_content", id="chat-bridged-to-responses"),
    ),
)
def test_a_deployment_whose_stream_stalls_stays_out_of_cooldown(
    cooldown_rig: _CooldownRig, endpoint: Endpoint, backend: str, shape: Shape
) -> None:
    with _peer(shape) as wire, cooldown_rig.gateway.scenario() as scenario:
        group, identity = _deployment(scenario, wire, backend)
        response: Final = _streamed(cooldown_rig.gateway, endpoint, group)
        error: Final = _failure_row(cooldown_rig.gateway, _ROWS, (group,))
        assert not cooled(cooldown_rig.cache, identity), (error, response.text)


@_OWNED_PROXY
def test_a_deployment_whose_stream_times_out_before_it_opens_is_cooled_down(cooldown_rig: _CooldownRig) -> None:
    with _peer("hold_headers") as wire, cooldown_rig.gateway.scenario() as scenario:
        group, identity = _deployment(scenario, wire)
        response: Final = _streamed(cooldown_rig.gateway, "chat", group)
        assert response.status_code == 408, response.text
        assert eventually(lambda: cooled(cooldown_rig.cache, identity), bool, _ROW_SECONDS)


def _fallback_cooled(cooldown_rig: _CooldownRig, fallback_shape: Shape) -> tuple[bool, dict[str, JsonValue], str]:
    with (
        _peer("fail") as failing,
        _peer(fallback_shape) as fallback_wire,
        cooldown_rig.gateway.scenario() as scenario,
    ):
        primary, _ = _deployment(scenario, failing)
        fallback, identity = _deployment(scenario, fallback_wire)
        response: Final = _streamed(cooldown_rig.gateway, "chat", primary, fallbacks=[fallback])
        error: Final = _failure_row(cooldown_rig.gateway, _EITHER_GROUP_ROWS, (primary, fallback))
        assert _upstream_targets(fallback_wire) == (_CHAT_TARGET,), response.text
        return cooled(cooldown_rig.cache, identity), error, response.text


@_OWNED_PROXY
def test_a_fallback_deployment_whose_stream_stalls_stays_out_of_cooldown(cooldown_rig: _CooldownRig) -> None:
    is_cooled, error, text = _fallback_cooled(cooldown_rig, "stall_before_first")
    assert not is_cooled, (error, text)


@_OWNED_PROXY
def test_a_fallback_deployment_whose_stream_times_out_before_it_opens_is_cooled_down(
    cooldown_rig: _CooldownRig,
) -> None:
    is_cooled, error, text = _fallback_cooled(cooldown_rig, "hold_headers")
    assert is_cooled, (error, text)


def _completed(endpoint: Endpoint, events: tuple[SseEvent, ...]) -> bool:
    match endpoint:
        case "chat":
            return _finish_reasons(events) == ("stop",)
        case "messages":
            return event_types(events)[-1:] == ("message_stop",)
        case "responses":
            return event_types(events)[-1:] == ("response.completed",)


def _finish_reasons(events: tuple[SseEvent, ...]) -> tuple[JsonValue, ...]:
    choices: Final = tuple(_first_choice(event.data) for event in events if event.data.get("choices"))
    return tuple(choice["finish_reason"] for choice in choices if choice.get("finish_reason"))


def _first_choice(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = data["choices"]
    assert isinstance(choices, list), data
    return object_value(choices[0])
