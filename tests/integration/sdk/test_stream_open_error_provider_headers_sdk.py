import asyncio
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import pytest
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

Route: TypeAlias = Literal["gemini", "vertex_ai", "codestral", "predibase", "openai", "azure", "anthropic"]
Call: TypeAlias = Literal["completion", "acompletion", "responses", "aresponses", "text_completion", "atext_completion"]

_GEMINI_MODEL: Final = "gemini-3.8-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_TOKEN_TARGET: Final = "/_oauth/token"
_TOKEN_REPLY: Final = Reply(
    body=json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()
)
_SCRIPTED: Final = ("retry-after", "x-request-id")
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")

_STREAM_OPEN_CASES: Final[tuple[tuple[Route, Call], ...]] = (
    ("gemini", "completion"),
    ("gemini", "acompletion"),
    ("gemini", "responses"),
    ("gemini", "aresponses"),
    ("vertex_ai", "completion"),
    ("vertex_ai", "acompletion"),
    ("vertex_ai", "responses"),
    ("vertex_ai", "aresponses"),
    ("codestral", "atext_completion"),
    ("predibase", "acompletion"),
)
_EAGER_CASES: Final[tuple[tuple[Route, Call], ...]] = (
    ("openai", "acompletion"),
    ("azure", "acompletion"),
    ("anthropic", "acompletion"),
    ("codestral", "text_completion"),
    ("predibase", "completion"),
)
_STREAM_TARGETS: Final[Mapping[Route, str]] = MappingProxyType(
    {
        "gemini": ":streamGenerateContent",
        "vertex_ai": ":streamGenerateContent",
        "codestral": "/v1/fim/completions",
        "predibase": "/scripted-tenant/deployments/v2/llms/llama-3-8b-instruct/generate_stream",
        "openai": "/chat/completions",
        "azure": "/chat/completions",
        "anthropic": "/v1/messages",
    }
)
_SYNC_TWINS: Final[Mapping[Call, Call]] = MappingProxyType(
    {
        "completion": "completion",
        "acompletion": "completion",
        "responses": "responses",
        "aresponses": "responses",
        "text_completion": "text_completion",
        "atext_completion": "text_completion",
    }
)


def _headers(marker: str, retry_after: str = "7") -> Mapping[str, str]:
    return MappingProxyType({"retry-after": retry_after, "x-request-id": f"req-{marker}"})


def _gemini_error(status: int) -> bytes:
    return json.dumps(
        {"error": {"code": status, "message": f"scripted provider error {status}", "status": "X"}}
    ).encode()


def _error_body(route: Route, status: int) -> bytes:
    match route:
        case "anthropic":
            return json.dumps(
                {"type": "error", "error": {"type": "rate_limit_error", "message": "scripted provider error"}}
            ).encode()
        case "openai" | "azure" | "codestral" | "predibase":
            return json.dumps(
                {"error": {"message": "scripted provider error", "type": "requests", "code": "rate_limit_exceeded"}}
            ).encode()
        case "gemini" | "vertex_ai":
            return _gemini_error(status)


def _failing(route: Route, status: int, headers: Mapping[str, str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == _TOKEN_TARGET:
            return _TOKEN_REPLY
        assert request.method == "POST", request.method
        return Reply(status=status, body=_error_body(route, status), headers=headers)

    return respond


def _deployment(route: Route, wire: Wire) -> dict[str, str]:
    match route:
        case "gemini":
            return {"model": f"gemini/{_GEMINI_MODEL}", "api_base": wire.url, "api_key": "scripted-key"}
        case "vertex_ai":
            return {
                "model": f"vertex_ai/{_GEMINI_MODEL}",
                "api_base": wire.url,
                "vertex_project": _PROJECT,
                "vertex_location": _LOCATION,
                "vertex_credentials": service_account_json(_PROJECT, wire.url),
            }
        case "codestral":
            return {
                "model": "text-completion-codestral/codestral-latest",
                "api_base": f"{wire.url}/v1/fim/completions",
                "api_key": "scripted-key",
            }
        case "openai":
            return {"model": "openai/gpt-6.1-sol", "api_base": wire.url, "api_key": "scripted-key"}
        case "azure":
            return {
                "model": "azure/gpt-6.1-sol",
                "api_base": wire.url,
                "api_key": "scripted-key",
                "api_version": "2025-04-01-preview",
            }
        case "anthropic":
            return {"model": "anthropic/claude-opus-5-5", "api_base": wire.url, "api_key": "scripted-key"}
        case "predibase":
            return {"model": "predibase/llama-3-8b-instruct", "api_base": wire.url, "api_key": "scripted-key"}


def _messages(marker: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"stream open probe marker-{marker}"}]


def _drain_sync(response: object) -> None:
    assert not isinstance(response, (str, bytes)) and hasattr(response, "__iter__"), type(response)
    for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # narrowed by the hasattr assert above
        pass


async def _drain_async(response: object) -> None:
    assert hasattr(response, "__aiter__"), type(response)
    async for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # narrowed by the hasattr assert above
        pass


async def _invoke_async(call: Call, deployment: dict[str, str], marker: str, stream: bool) -> None:
    match call:
        case "acompletion":
            completion: Final = await litellm.acompletion(
                **deployment, messages=_messages(marker), stream=stream, num_retries=0
            )
            if stream:
                await _drain_async(completion)
        case "aresponses":
            responses: Final = await litellm.aresponses(
                **deployment, input=f"stream open probe marker-{marker}", stream=stream, num_retries=0
            )
            if stream:
                await _drain_async(responses)
        case "atext_completion":
            text: Final = await litellm.atext_completion(
                **deployment, prompt=f"stream open probe marker-{marker}", stream=stream, num_retries=0
            )
            if stream:
                await _drain_async(text)
        case "completion" | "responses" | "text_completion":
            raise AssertionError(call)


def _invoke(call: Call, deployment: dict[str, str], marker: str, stream: bool) -> None:
    match call:
        case "completion":
            completion: Final = litellm.completion(
                **deployment, messages=_messages(marker), stream=stream, num_retries=0
            )
            if stream:
                _drain_sync(completion)
        case "responses":
            responses: Final = litellm.responses(
                **deployment, input=f"stream open probe marker-{marker}", stream=stream, num_retries=0
            )
            if stream:
                _drain_sync(responses)
        case "text_completion":
            text: Final = litellm.text_completion(
                **deployment, prompt=f"stream open probe marker-{marker}", stream=stream, num_retries=0
            )
            if stream:
                _drain_sync(text)
        case "acompletion" | "aresponses" | "atext_completion":
            asyncio.run(_invoke_async(call, deployment, marker, stream))


def _raised(invoke: Callable[[], None]) -> Exception:
    try:
        invoke()
    except Exception as error:
        return error
    pytest.fail("the provider error never reached the caller")


def _carried(error: Exception) -> Mapping[str, str]:
    headers: Final = getattr(error, "litellm_response_headers", None) or {}
    return MappingProxyType({str(name).lower(): str(value) for name, value in dict(headers).items()})


def _scripted(error: Exception) -> Mapping[str, str | None]:
    carried: Final = _carried(error)
    return MappingProxyType({name: carried.get(name) for name in _SCRIPTED})


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.target != _TOKEN_TARGET)


def _assert_one_stream_attempt(route: Route, wire: Wire, marker: str) -> None:
    calls: Final = _provider_calls(wire)
    assert len(calls) == 1, [request.target for request in calls]
    assert marker.encode() in calls[0].body and _STREAM_TARGETS[route] in calls[0].target, calls[0]


def _twin_and_stream_errors(
    route: Route, call: Call, status: int, headers: Mapping[str, str], marker: str
) -> tuple[Exception, Exception]:
    with wire_server(_failing(route, status, headers)) as wire:
        twin: Final = _raised(lambda: _invoke(_SYNC_TWINS[call], _deployment(route, wire), marker, stream=False))
        assert len(_provider_calls(wire)) == 1
        raised: Final = _raised(lambda: _invoke(call, _deployment(route, wire), marker, stream=True))
        _assert_one_stream_attempt(route, wire, marker)
    return twin, raised


@pytest.fixture(autouse=True)
def _predibase_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDIBASE_TENANT_ID", "scripted-tenant")


@pytest.mark.parametrize(("route", "call"), _STREAM_OPEN_CASES)
def test_stream_open_rate_limit_carries_the_provider_headers_of_its_non_stream_twin(route: Route, call: Call) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = _headers(marker)
    twin, raised = _twin_and_stream_errors(route, call, 429, headers, marker)
    assert isinstance(twin, litellm.RateLimitError), repr(twin)
    assert dict(_scripted(twin)) == dict(headers), dict(_carried(twin))
    assert isinstance(raised, MidStreamFallbackError), repr(raised)
    assert raised.status_code == 429 and raised.is_pre_first_chunk and raised.generated_content == "", repr(raised)
    assert isinstance(raised.original_exception, litellm.RateLimitError), repr(raised.original_exception)
    assert dict(_scripted(raised)) == dict(headers), dict(_carried(raised))
    assert sorted(_carried(raised)) == sorted(_carried(twin)), (dict(_carried(raised)), dict(_carried(twin)))


@pytest.mark.parametrize("status", (500, 502, 503, 504))
def test_stream_open_server_error_carries_the_provider_headers_of_its_non_stream_twin(status: int) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = _headers(marker)
    twin, raised = _twin_and_stream_errors("gemini", "acompletion", status, headers, marker)
    assert dict(_scripted(twin)) == dict(headers), dict(_carried(twin))
    assert isinstance(raised, MidStreamFallbackError), repr(raised)
    assert type(raised.original_exception) is type(twin), (repr(raised.original_exception), repr(twin))
    assert raised.status_code == getattr(twin, "status_code"), (raised.status_code, getattr(twin, "status_code"))
    assert dict(_scripted(raised)) == dict(headers), dict(_carried(raised))


@pytest.mark.parametrize("status", (400, 401, 403, 404, 408))
def test_stream_open_client_error_raises_its_non_stream_twin_with_the_provider_headers(status: int) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = _headers(marker)
    twin, raised = _twin_and_stream_errors("gemini", "acompletion", status, headers, marker)
    assert not isinstance(raised, MidStreamFallbackError), repr(raised)
    assert type(raised) is type(twin) and getattr(raised, "status_code") == status, (repr(raised), repr(twin))
    assert dict(_scripted(raised)) == dict(_scripted(twin)) == dict(headers), dict(_carried(raised))


@pytest.mark.parametrize(
    ("retry_after", "request_id"),
    (
        ("", ""),
        ("abc", "scripted-request"),
        ("-5", "scripted-request"),
        ("Wed, 21 Oct 2015 07:28:00 GMT", "scripted-request"),
        ("1", "r" * 5000),
    ),
    ids=("empty", "not_a_number", "negative", "http_date", "five_kb_request_id"),
)
def test_stream_open_rate_limit_keeps_odd_provider_header_values_as_the_non_stream_twin_does(
    retry_after: str, request_id: str
) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = MappingProxyType({"retry-after": retry_after, "x-request-id": request_id})
    twin, raised = _twin_and_stream_errors("gemini", "acompletion", 429, headers, marker)
    assert isinstance(raised, MidStreamFallbackError), repr(raised)
    assert dict(_scripted(twin)) == dict(headers), dict(_carried(twin))
    assert dict(_scripted(raised)) == dict(_scripted(twin)), (dict(_scripted(raised)), dict(_scripted(twin)))


@pytest.mark.parametrize(("route", "call"), _EAGER_CASES)
def test_eager_stream_opener_raises_the_same_error_as_its_non_stream_twin(route: Route, call: Call) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = _headers(marker)
    twin, raised = _twin_and_stream_errors(route, call, 429, headers, marker)
    assert not isinstance(raised, MidStreamFallbackError), repr(raised)
    assert type(raised) is type(twin) and isinstance(raised, litellm.RateLimitError), (repr(raised), repr(twin))
    assert dict(_scripted(raised)) == dict(_scripted(twin)) == dict(headers), dict(_carried(raised))


def _gemini_frame(text: str) -> bytes:
    candidate: Final = {"content": {"role": "model", "parts": [{"text": text}]}}
    usage: Final = {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4}
    return b"data: " + json.dumps({"candidates": [candidate], "usageMetadata": usage}).encode() + b"\n\n"


def _gemini_error_frame() -> bytes:
    return (
        b"data: "
        + json.dumps({"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}).encode()
        + (b"\n\n")
    )


def _openai_frame(text: str) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-scripted",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-6.1-sol",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def _openai_error_frame() -> bytes:
    error: Final = {"error": {"message": "server overloaded", "type": "server_error", "code": "server_error"}}
    return b"data: " + json.dumps(error).encode() + b"\n\n"


def _anthropic_frame(event: str, data: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


_ANTHROPIC_START: Final = _anthropic_frame(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    },
)
_ANTHROPIC_TEXT: Final = (
    _anthropic_frame(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    _anthropic_frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial "}},
    ),
)
_ANTHROPIC_OVERLOADED: Final = _anthropic_frame(
    "error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
)

InBand: TypeAlias = Literal[
    "gemini_error_before_content",
    "gemini_error_after_content",
    "gemini_drop_after_content",
    "anthropic_overloaded_before_content",
    "anthropic_overloaded_after_content",
    "openai_error_after_content",
]


def _in_band(kind: InBand, headers: Mapping[str, str]) -> tuple[Route, Reply]:
    match kind:
        case "gemini_error_before_content":
            return "gemini", Reply(content_type="text/event-stream", chunks=(_gemini_error_frame(),), headers=headers)
        case "gemini_error_after_content":
            chunks: Final = (_gemini_frame("partial "), _gemini_error_frame())
            return "gemini", Reply(content_type="text/event-stream", chunks=chunks, headers=headers)
        case "gemini_drop_after_content":
            dropped: Final = (_gemini_frame("partial "), _gemini_frame("never sent"))
            return "gemini", Reply(content_type="text/event-stream", chunks=dropped, abort_after=1, headers=headers)
        case "anthropic_overloaded_before_content":
            early: Final = (_ANTHROPIC_START, _ANTHROPIC_OVERLOADED)
            return "anthropic", Reply(content_type="text/event-stream", chunks=early, headers=headers)
        case "anthropic_overloaded_after_content":
            late: Final = (_ANTHROPIC_START, *_ANTHROPIC_TEXT, _ANTHROPIC_OVERLOADED)
            return "anthropic", Reply(content_type="text/event-stream", chunks=late, headers=headers)
        case "openai_error_after_content":
            openai_chunks: Final = (_openai_frame("partial "), _openai_error_frame())
            return "openai", Reply(content_type="text/event-stream", chunks=openai_chunks, headers=headers)


@pytest.mark.parametrize(
    ("kind", "before_content"),
    (
        ("gemini_error_before_content", True),
        ("gemini_error_after_content", False),
        ("gemini_drop_after_content", False),
        ("anthropic_overloaded_before_content", True),
        ("anthropic_overloaded_after_content", False),
        ("openai_error_after_content", False),
    ),
)
def test_in_band_stream_error_wraps_without_another_requests_provider_headers(
    kind: InBand, before_content: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    route, reply = _in_band(kind, _headers(marker))
    with wire_server(lambda _request: reply) as wire:
        raised: Final = _raised(lambda: _invoke("acompletion", _deployment(route, wire), marker, stream=True))
        calls: Final = _provider_calls(wire)
    assert isinstance(raised, MidStreamFallbackError), repr(raised)
    assert raised.is_pre_first_chunk is before_content, repr(raised)
    assert raised.generated_content == ("" if before_content else "partial "), raised.generated_content
    assert _scripted(raised)["x-request-id"] in (None, f"req-{marker}"), dict(_carried(raised))
    assert len(calls) == 1 and marker.encode() in calls[0].body, calls


async def test_messages_bridge_stream_open_rate_limit_answers_an_in_band_rate_limit_error_frame() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_failing("gemini", 429, _headers(marker))) as wire:
        stream: Final = await litellm.anthropic.messages.acreate(
            **_deployment("gemini", wire), max_tokens=16, messages=_messages(marker), stream=True
        )
        assert hasattr(stream, "__aiter__"), type(stream)
        frames: Final = [frame async for frame in stream]  # pyright: ignore[reportGeneralTypeIssues]  # narrowed above
        calls: Final = _provider_calls(wire)
    text: Final = b"".join(frame if isinstance(frame, bytes) else str(frame).encode() for frame in frames).decode()
    events: Final = tuple(line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: "))
    assert events[0] == "message_start" and events[-1] == "error", text
    error_data: Final = json.loads(text.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert error_data["error"]["type"] == "rate_limit_error", error_data
    assert len(calls) == 1 and marker.encode() in calls[0].body, calls


def _router(primary: Wire, fallback: Wire | None, num_retries: int) -> Router:
    healthy: Final = (
        ({"model_name": "healthy", "litellm_params": _deployment("gemini", fallback)},) if fallback is not None else ()
    )
    return Router(
        model_list=[{"model_name": "limited", "litellm_params": _deployment("gemini", primary)}, *healthy],
        num_retries=num_retries,
        retry_after=0,
        fallbacks=[{"limited": ["healthy"]}] if fallback is not None else [],
    )


def _healthy_stream(_request: Request) -> Reply:
    candidate: Final = {"content": {"role": "model", "parts": [{"text": "answer"}]}, "finishReason": "STOP"}
    usage: Final = {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4}
    frame: Final = b"data: " + json.dumps({"candidates": [candidate], "usageMetadata": usage}).encode() + b"\n\n"
    return Reply(content_type="text/event-stream", chunks=(_gemini_frame("healthy "), frame))


def _text_of(chunk: object) -> str:
    choices: Final = getattr(chunk, "choices", None) or ()
    return "".join(str(choice.delta.content or "") for choice in choices)


async def _router_stream_error(router: Router, marker: str) -> Exception:
    try:
        await _drain_async(await router.acompletion(model="limited", messages=_messages(marker), stream=True))
    except Exception as error:
        return error
    raise AssertionError(f"marker-{marker} streamed without the scripted 429")


async def test_router_async_stream_open_rate_limit_retries_then_surfaces_the_429() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_failing("gemini", 429, _headers(marker, retry_after="1"))) as wire:
        raised: Final = await _router_stream_error(_router(wire, None, num_retries=1), marker)
        calls: Final = _provider_calls(wire)
    assert getattr(raised, "status_code", None) == 429, repr(raised)
    assert not isinstance(raised, MidStreamFallbackError), repr(raised)
    assert len(calls) == 2 and all(marker.encode() in request.body for request in calls), calls


async def _fallback_text(router: Router, marker: str) -> str:
    response: Final = await router.acompletion(model="limited", messages=_messages(marker), stream=True)
    assert isinstance(response, CustomStreamWrapper), type(response)
    return "".join([_text_of(chunk) async for chunk in response])


async def test_router_async_stream_open_rate_limit_falls_back_on_every_call() -> None:
    markers: Final = (uuid.uuid4().hex, uuid.uuid4().hex)
    with wire_server(_failing("gemini", 429, _headers(markers[0]))) as limited, wire_server(_healthy_stream) as healthy:
        router: Final = _router(limited, healthy, num_retries=0)
        first: Final = await _fallback_text(router, markers[0])
        first_calls: Final = (_provider_calls(limited), _provider_calls(healthy))
        second: Final = await _fallback_text(router, markers[1])
        second_calls: Final = (_provider_calls(limited), _provider_calls(healthy))
    assert first == second == "healthy answer", (first, second)
    for marker, (limited_calls, healthy_calls) in zip(markers, (first_calls, second_calls), strict=True):
        assert [marker.encode() in request.body for request in limited_calls] == [True], limited_calls
        assert [marker.encode() in request.body for request in healthy_calls] == [True], healthy_calls


def test_router_sync_stream_open_rate_limit_falls_back_to_the_healthy_deployment() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_failing("gemini", 429, _headers(marker))) as limited, wire_server(_healthy_stream) as healthy:
        response: Final = _router(limited, healthy, num_retries=0).completion(
            model="limited", messages=_messages(marker), stream=True
        )
        assert isinstance(response, CustomStreamWrapper), type(response)
        text: Final = "".join(_text_of(chunk) for chunk in response)
        limited_calls: Final = _provider_calls(limited)
        healthy_calls: Final = _provider_calls(healthy)
    assert text == "healthy answer", text
    assert len(limited_calls) == 1 and marker.encode() in limited_calls[0].body, limited_calls
    assert len(healthy_calls) == 1 and marker.encode() in healthy_calls[0].body, healthy_calls


def _retry_gap_seconds(retry_after: str, stream: bool) -> float:
    marker: Final = uuid.uuid4().hex
    arrivals: Final = SimpleQueue[float]()
    failing: Final = _failing("gemini", 429, _headers(marker, retry_after=retry_after))

    def respond(request: Request) -> Reply:
        arrivals.put(time.monotonic())
        return failing(request)

    with wire_server(respond) as wire:
        router: Final = _router(wire, None, num_retries=1)

        def complete() -> None:
            router.completion(model="limited", messages=_messages(marker), stream=stream, complete_response=stream)

        raised: Final = _raised(complete)
        calls: Final = _provider_calls(wire)
    assert getattr(raised, "status_code", None) == 429, repr(raised)
    assert [marker.encode() in request.body for request in calls] == [True, True], calls
    first, second = arrivals.get_nowait(), arrivals.get_nowait()
    return second - first


def test_router_sync_complete_response_stream_waits_the_provider_retry_after_like_its_non_stream_twin() -> None:
    retry_after: Final = 3
    gaps: Final = (
        _retry_gap_seconds(str(retry_after), stream=False),
        _retry_gap_seconds(str(retry_after), stream=True),
    )
    assert min(gaps) >= retry_after, gaps


def _per_request_rate_limit(request: Request) -> Reply:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return Reply(status=429, body=_gemini_error(429), headers=_headers(found.group(1)))


async def _stream_error(deployment: dict[str, str], marker: str) -> Exception:
    try:
        await _invoke_async("acompletion", deployment, marker, stream=True)
    except Exception as error:
        return error
    raise AssertionError(f"marker-{marker} streamed without the scripted 429")


async def test_concurrent_stream_open_rate_limits_each_carry_their_own_provider_request_id() -> None:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(40))
    with wire_server(_per_request_rate_limit) as wire:
        deployment: Final = _deployment("gemini", wire)
        errors: Final = await asyncio.gather(*(_stream_error(deployment, marker) for marker in markers))
        calls: Final = _provider_calls(wire)
    for marker, error in zip(markers, errors, strict=True):
        assert isinstance(error, MidStreamFallbackError), repr(error)
        assert dict(_scripted(error)) == dict(_headers(marker)), (marker, dict(_carried(error)))
    forwarded: Final = sorted(
        found.group(1) for found in (_MARKER.search(call.body.decode()) for call in calls) if found
    )
    assert forwarded == sorted(markers), forwarded
