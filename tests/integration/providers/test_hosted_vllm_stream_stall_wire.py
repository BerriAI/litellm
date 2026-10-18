import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal

import httpx
from integration._support.anthropic_sse import SseEvent, event_types, parse_sse
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_stream, posted_targets, responses_stream
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_BACKEND: Final = "glm-5-fp8"
_PROVIDER_KEY: Final = "synthetic-hosted-vllm-key"
_ANSWER: Final = "scripted answer before the upstream stalls"
_PROMPT: Final = "Keep streaming while the upstream thinks."
_CHAT_TARGET: Final = "/v1/chat/completions"
_RESPONSES_TARGET: Final = "/v1/responses"
_STREAM_TIMEOUT_SECONDS: Final = 1
_DEPLOYMENT_TIMEOUT_SECONDS: Final = 30
_PAUSE_SECONDS: Final = 3.0
_ROW_SECONDS: Final = 60
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_ROWS: Final = 'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s'

Endpoint = Literal["chat", "responses", "messages"]


def _stream_parts(target: str) -> tuple[bytes, bytes, bytes]:
    identity: Final = f"stall-{uuid.uuid4().hex}"
    if target == _RESPONSES_TARGET:
        return responses_stream(identity, _BACKEND, _ANSWER)
    return chat_stream(identity, _BACKEND, _ANSWER)


@dataclass(frozen=True, slots=True)
class _Peer:
    pauses: bool

    def __call__(self, request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target in (_CHAT_TARGET, _RESPONSES_TARGET), request
        opened, delta, closed = _stream_parts(request.target)
        if not self.pauses:
            return Reply(content_type="text/event-stream", chunks=(opened, delta, closed))
        return Reply(
            content_type="text/event-stream", chunks=(opened + delta, closed), pause_between_chunks=_PAUSE_SECONDS
        )


@contextmanager
def _peer(*, pauses: bool) -> Generator[Wire]:
    with wire_server(answering_model_discovery(_Peer(pauses))) as wire:
        yield wire


def _deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"hosted_vllm/{_BACKEND}",
        api_base=f"{wire.url}/v1",
        api_key=_PROVIDER_KEY,
        timeout=_DEPLOYMENT_TIMEOUT_SECONDS,
        stream_timeout=_STREAM_TIMEOUT_SECONDS,
    )


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"
        case "messages":
            return "/v1/messages"


def _body(endpoint: Endpoint, model: str) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "stream": True, **_EXTRA}
        case "responses":
            return {"model": model, "input": _PROMPT, "stream": True, **_EXTRA}
        case "messages":
            return {
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 16,
                "stream": True,
                **_EXTRA,
            }


def _streamed(gateway: Gateway, endpoint: Endpoint, model: str) -> httpx.Response:
    return gateway.request("POST", _path(endpoint), _body(endpoint, model))


def _upstream_targets(wire: Wire) -> tuple[str, ...]:
    eventually(wire.received.qsize, lambda size: size >= 1, 10)
    return posted_targets(wire)


def _chat_errors(events: tuple[SseEvent, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(event.data["error"]) for event in events if "error" in event.data)


def _error_class(message: JsonValue) -> str:
    return string_value(message).split(":", 1)[0]


def _logged_failure(model: str) -> tuple[JsonValue, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_ROWS, (model,)), lambda found: len(found) >= 1, seconds=_ROW_SECONDS)
    assert [row["status"] for row in rows] == ["failure"], rows
    error: Final = object_value(object_value(rows[0]["metadata"])["error_information"])
    return error["error_code"], error["error_class"]


def test_a_chat_stream_that_stalls_past_its_stream_timeout_ends_with_a_408_timeout_and_logs_it(
    gateway: Gateway,
) -> None:
    with _peer(pauses=True) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "chat", model)
        assert response.status_code == 200, response.text
        assert _ANSWER in response.text, response.text
        events: Final = parse_sse(response.text)
        assert _chat_errors(events[:-1]) == () and "error" in events[-1].data, response.text
        error: Final = object_value(events[-1].data["error"])
        observed: Final = (error["code"], _error_class(error["message"]), *_logged_failure(model))
        assert observed == ("408", "litellm.Timeout", "408", "Timeout"), (error, observed)
        assert _upstream_targets(wire) == (_CHAT_TARGET,)


def test_a_native_responses_stream_that_stalls_past_its_stream_timeout_fails_with_request_timeout_and_logs_it(
    gateway: Gateway,
) -> None:
    with _peer(pauses=True) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "responses", model)
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        types: Final = event_types(events)
        assert "response.output_text.delta" in types, response.text
        assert types[-1] == "response.failed" and "response.completed" not in types, response.text
        failed: Final = object_value(object_value(events[-1].data["response"])["error"])
        observed: Final = (failed["code"], *_logged_failure(model))
        assert observed == ("request_timeout", "408", "Timeout"), (failed, observed)
        assert _upstream_targets(wire) == (_RESPONSES_TARGET,)


def test_a_messages_stream_that_stalls_past_its_stream_timeout_ends_with_a_timeout_error_event_and_logs_it(
    gateway: Gateway,
) -> None:
    with _peer(pauses=True) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "messages", model)
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        types: Final = event_types(events)
        assert "content_block_delta" in types and "message_stop" not in types, response.text
        assert types[-1] == "error", response.text
        error: Final = object_value(events[-1].data["error"])
        observed: Final = (_error_class(error["message"]), *_logged_failure(model))
        assert observed == ("litellm.Timeout", "408", "Timeout"), (error, observed)
        assert _upstream_targets(wire) == (_CHAT_TARGET,)


def test_a_chat_stream_that_never_pauses_is_answered_in_full_under_the_same_stream_timeout(gateway: Gateway) -> None:
    with _peer(pauses=False) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _streamed(gateway, "chat", model)
        assert response.status_code == 200, response.text
        assert _ANSWER in response.text, response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        events: Final = parse_sse(response.text)
        assert _chat_errors(events) == (), response.text
        assert _finish_reasons(events) == ("stop",), response.text
        assert _upstream_targets(wire) == (_CHAT_TARGET,)
        rows: Final = eventually(lambda: read_rows(_ROWS, (model,)), lambda found: len(found) >= 1, _ROW_SECONDS)
        assert [row["status"] for row in rows] == ["success"], rows


def _finish_reasons(events: tuple[SseEvent, ...]) -> tuple[JsonValue, ...]:
    choices: Final = tuple(_first_choice(event.data) for event in events if event.data.get("choices"))
    return tuple(choice["finish_reason"] for choice in choices if choice.get("finish_reason"))


def _first_choice(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = data["choices"]
    assert isinstance(choices, list), data
    return object_value(choices[0])
