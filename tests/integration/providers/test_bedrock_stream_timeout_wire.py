import base64
import json
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_CONVERSE_MODEL_ID: Final = "global.moonshotai.kimi-k3"
_CONVERSE_MODEL: Final = f"bedrock/converse/{_CONVERSE_MODEL_ID}"
_INVOKE_MODEL_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_INVOKE_MODEL: Final = f"bedrock/invoke/{_INVOKE_MODEL_ID}"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_ANSWER_HEAD: Final = "scripted answer part one "
_ANSWER_TAIL: Final = "scripted answer part two"
_ANSWER: Final = _ANSWER_HEAD + _ANSWER_TAIL
_PROMPT: Final = "How long does the gateway wait for this stream?"
_USER_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": _PROMPT}
_TIMEOUT_SECONDS: Final = 1
_PAUSE_SECONDS: Final = 3.0
_HOLD_SECONDS: Final = 40.0
_CLIENT_WINDOW: Final = 10.0
_RETRY_WINDOW: Final = 25.0
_SHORT_WINDOW: Final = 3.0
_AWS: Final[dict[str, JsonValue]] = {
    "api_key": None,
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_MODELS: Final = TypeAdapter(list[dict[str, JsonValue]])
_TIMEOUT_PASSED: Final = re.compile(r"Timeout passed=(?:Timeout\(timeout=)?(-?\d+\.\d+)")
_USAGE: Final[dict[str, JsonValue]] = {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}
_CONVERSE_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
        "stopReason": "end_turn",
        **_USAGE,
        "metrics": {"latencyMs": 1},
    }
).encode()
_INVOKE_RESPONSE: Final = json.dumps(
    {
        "id": "msg_invoke",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": _ANSWER}],
        "model": _INVOKE_MODEL_ID,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
).encode()

Endpoint = Literal["chat", "messages", "responses"]
Mode = Literal["fast", "stall", "pause", "cut"]


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    return _frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


_CONVERSE_FRAMES: Final = (
    _frame("messageStart", {"role": "assistant"}),
    _frame("contentBlockDelta", {"delta": {"text": _ANSWER_HEAD}, "contentBlockIndex": 0}),
    _frame("contentBlockDelta", {"delta": {"text": _ANSWER_TAIL}, "contentBlockIndex": 0}),
    _frame("contentBlockStop", {"contentBlockIndex": 0}),
    _frame("messageStop", {"stopReason": "end_turn"}),
    _frame("metadata", _USAGE),
)
_INVOKE_FRAMES: Final = (
    _invoke_chunk(
        {
            "type": "message_start",
            "message": {
                "id": "msg_invoke",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": _INVOKE_MODEL_ID,
                "stop_reason": None,
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
        }
    ),
    _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _ANSWER_HEAD}}),
    _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _ANSWER_TAIL}}),
    _invoke_chunk({"type": "content_block_stop", "index": 0}),
    _invoke_chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
    _invoke_chunk({"type": "message_stop"}),
)


def _is_stream(target: str) -> bool:
    return target.endswith("converse-stream") or target.endswith("invoke-with-response-stream")


def _frames_for(target: str) -> tuple[bytes, ...]:
    return _INVOKE_FRAMES if "/invoke" in target else _CONVERSE_FRAMES


def _frames_before_the_pause(target: str, mode: Mode) -> int:
    if mode == "pause":
        return 1
    return 3 if "/invoke" in target else 2


def _json_for(target: str) -> bytes:
    return _INVOKE_RESPONSE if "/invoke" in target else _CONVERSE_RESPONSE


@dataclass(frozen=True, slots=True)
class _Peer:
    mode: Mode
    release: threading.Event

    def __call__(self, request: Request) -> Reply:
        target: Final = unquote(request.target)
        if self.mode == "stall":
            self.release.wait(timeout=_HOLD_SECONDS)
        if not _is_stream(target):
            return Reply(body=_json_for(target))
        frames: Final = _frames_for(target)
        if self.mode in ("pause", "cut"):
            sent: Final = _frames_before_the_pause(target, self.mode)
            return Reply(
                chunks=(b"".join(frames[:sent]), b"".join(frames[sent:])),
                content_type=_EVENT_STREAM,
                pause_between_chunks=_PAUSE_SECONDS,
            )
        return Reply(body=b"".join(frames), content_type=_EVENT_STREAM)


@contextmanager
def _peer(mode: Mode) -> Iterator[Wire]:
    release: Final = threading.Event()
    with wire_server(_Peer(mode, release)) as wire:
        try:
            yield wire
        finally:
            release.set()


def _converse(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(model=_CONVERSE_MODEL, api_base=wire.url, **_AWS, **extra)


def _invoke(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(
        model=_INVOKE_MODEL, api_base=wire.url, aws_bedrock_runtime_endpoint=wire.url, **_AWS, **extra
    )


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _auth(gateway: Gateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {gateway.key}"}


def _openai(gateway: Gateway, window: float = _CLIENT_WINDOW) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0, timeout=window)


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0, timeout=_CLIENT_WINDOW
    )


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0, timeout=_CLIENT_WINDOW)


def _async_anthropic(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0, timeout=_CLIENT_WINDOW
    )


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(endpoint: Endpoint, model: str, *, stream: bool = True, **extra: JsonValue) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"model": model, "messages": [_USER_TURN], "stream": stream, **_EXTRA, **extra}
        case "messages":
            return {"model": model, "messages": [_USER_TURN], "max_tokens": 16, "stream": stream, **_EXTRA, **extra}
        case "responses":
            return {"model": model, "input": _PROMPT, "stream": stream, **_EXTRA, **extra}


@dataclass(frozen=True, slots=True)
class _Streamed:
    status: int
    headers: Mapping[str, str]
    text: str


def _streamed(
    gateway: Gateway,
    endpoint: Endpoint,
    body: Mapping[str, JsonValue] | None = None,
    *,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    key: str | None = None,
    window: float = _CLIENT_WINDOW,
) -> _Streamed:
    with httpx.Client(base_url=_proxy_url(gateway), timeout=window, trust_env=False) as client:
        return _streamed_on(client, gateway, endpoint, body, content=content, headers=headers, key=key)


def _streamed_on(
    client: httpx.Client,
    gateway: Gateway,
    endpoint: Endpoint,
    body: Mapping[str, JsonValue] | None = None,
    *,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    key: str | None = None,
) -> _Streamed:
    request_headers: Final = {
        "Authorization": f"Bearer {key or gateway.key}",
        **(headers or {}),
        **({"Content-Type": "application/json"} if content is not None else {}),
    }
    with client.stream("POST", _path(endpoint), json=body, content=content, headers=request_headers) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
        return _Streamed(response.status_code, dict(response.headers), "\n".join(lines))


def _timeout_passed(text: str) -> str:
    found: Final = _TIMEOUT_PASSED.search(text)
    assert found is not None, text
    return found.group(1)


_STREAMS_FROM_THE_FIRST_FRAME: Final = frozenset({"messages", "responses"})


def _assert_no_answer(text: str) -> None:
    assert _ANSWER_HEAD not in text, text
    assert _ANSWER_TAIL not in text, text


def _assert_timed_out(status: int, text: str, seconds: float = _TIMEOUT_SECONDS) -> None:
    assert status == 408, text
    assert _timeout_passed(text) == f"{seconds:.1f}", text
    _assert_no_answer(text)


def _assert_mid_stream_timeout(served: _Streamed, endpoint: Endpoint) -> None:
    if endpoint in _STREAMS_FROM_THE_FIRST_FRAME:
        assert served.status == 200, served.text
        assert "error" in served.text, served.text
    else:
        assert served.status >= 400, served.text
    assert "Timeout" in served.text, served.text
    _assert_no_answer(served.text)


def _assert_cut_off_mid_answer(served: _Streamed) -> None:
    assert served.status == 200, served.text
    assert _ANSWER_HEAD in served.text, served.text
    assert _ANSWER_TAIL not in served.text, served.text
    assert "Timeout" in served.text, served.text


def _assert_answered(served: _Streamed) -> None:
    assert served.status == 200, served.text
    assert _ANSWER_HEAD in served.text, served.text
    assert _ANSWER_TAIL in served.text, served.text
    assert "Timeout" not in served.text, served.text


def _failure_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert [row["status"] for row in rows] == ["failure"] * expected, rows
    assert len({row["request_id"] for row in rows}) == expected, rows
    return rows


def _received(wire: Wire, expected: int) -> None:
    eventually(wire.received.qsize, lambda size: size == expected, 10)
    assert len(wire.drain()) == expected


def _model_identity(gateway: Gateway, model: str) -> str:
    entries: Final = _MODELS.validate_python(gateway.get("/model/info")["data"])
    (entry,) = tuple(item for item in entries if item["model_name"] == model)
    return string_value(object_value(entry["model_info"])["id"])


def _model_timeout(gateway: Gateway, model: str) -> JsonValue:
    entries: Final = _MODELS.validate_python(gateway.get("/model/info")["data"])
    (entry,) = tuple(item for item in entries if item["model_name"] == model)
    return object_value(entry["litellm_params"]).get("timeout")


def test_h1_chat_stream_through_the_openai_sdk_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            _openai(gateway).chat.completions.create(model=model, messages=[_USER_TURN], stream=True, extra_body=_EXTRA)
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)
        _failure_rows(model, 1)


async def test_h2_chat_stream_through_the_async_openai_sdk_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            await _async_openai(gateway).chat.completions.create(
                model=model, messages=[_USER_TURN], stream=True, extra_body=_EXTRA
            )
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)


def test_h3_messages_stream_through_the_anthropic_sdk_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(anthropic.APIStatusError) as caught:
            _anthropic(gateway).messages.create(
                model=model, max_tokens=16, messages=[_USER_TURN], stream=True, extra_body=_EXTRA
            )
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)
        _failure_rows(model, 1)


async def test_h4_messages_stream_through_the_async_anthropic_sdk_fails_at_the_deployment_timeout(
    gateway: Gateway,
) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(anthropic.APIStatusError) as caught:
            await _async_anthropic(gateway).messages.create(
                model=model, max_tokens=16, messages=[_USER_TURN], stream=True, extra_body=_EXTRA
            )
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)


def test_h5_responses_stream_through_the_openai_sdk_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            _openai(gateway).responses.create(model=model, input=_PROMPT, stream=True, extra_body=_EXTRA)
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)
        _failure_rows(model, 1)


async def test_h6_responses_stream_through_raw_httpx_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=_CLIENT_WINDOW, trust_env=False) as client:
            response: Final = await client.post("/v1/responses", json=_body("responses", model), headers=_auth(gateway))
        _assert_timed_out(response.status_code, response.text)
        _received(wire, 1)


async def test_h7_invoke_stream_through_the_async_openai_sdk_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _invoke(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            await _async_openai(gateway).chat.completions.create(
                model=model, messages=[_USER_TURN], stream=True, extra_body=_EXTRA
            )
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)
        _failure_rows(model, 1)


@pytest.mark.parametrize("endpoint", ("chat", "messages", "responses"))
def test_h8_to_h10_a_converse_stream_that_pauses_past_the_timeout_ends_with_a_timeout_error(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    with _peer("pause") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_mid_stream_timeout(_streamed(gateway, endpoint, _body(endpoint, model)), endpoint)
        _received(wire, 1)


def test_h11_an_invoke_stream_that_pauses_past_the_timeout_ends_with_a_timeout_error(gateway: Gateway) -> None:
    with _peer("pause") as wire, gateway.scenario() as scenario:
        model: Final = _invoke(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_mid_stream_timeout(_streamed(gateway, "chat", _body("chat", model)), "chat")
        _received(wire, 1)


@pytest.mark.parametrize("endpoint", ("chat", "messages", "responses"))
def test_h12_a_converse_stream_cut_off_mid_answer_reports_the_timeout_in_band(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    with _peer("cut") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_cut_off_mid_answer(_streamed(gateway, endpoint, _body(endpoint, model)))
        _received(wire, 1)


def test_h13_an_invoke_stream_cut_off_mid_answer_reports_the_timeout_in_band(gateway: Gateway) -> None:
    with _peer("cut") as wire, gateway.scenario() as scenario:
        model: Final = _invoke(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_cut_off_mid_answer(_streamed(gateway, "chat", _body("chat", model)))
        _received(wire, 1)


def test_c1_chat_without_streaming_already_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            _openai(gateway).chat.completions.create(model=model, messages=[_USER_TURN], extra_body=_EXTRA)
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)
        _failure_rows(model, 1)


def test_c2_invoke_without_streaming_already_fails_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _invoke(scenario, wire, timeout=_TIMEOUT_SECONDS)
        with pytest.raises(openai.APIStatusError) as caught:
            _openai(gateway).chat.completions.create(model=model, messages=[_USER_TURN], extra_body=_EXTRA)
        _assert_timed_out(caught.value.status_code, caught.value.response.text)
        _received(wire, 1)


@pytest.mark.parametrize("endpoint", ("chat", "messages", "responses"))
def test_c3_to_c5_a_prompt_converse_stream_under_the_timeout_is_answered(gateway: Gateway, endpoint: Endpoint) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_answered(_streamed(gateway, endpoint, _body(endpoint, model)))
        _received(wire, 1)


def test_c6_a_prompt_invoke_stream_under_the_timeout_is_answered(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _invoke(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_answered(_streamed(gateway, "chat", _body("chat", model)))
        _received(wire, 1)


_PASSTHROUGH_BODY: Final[dict[str, JsonValue]] = {"messages": [{"role": "user", "content": [{"text": _PROMPT}]}]}


def test_c7_the_bedrock_passthrough_stream_is_relayed_verbatim(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        response: Final = gateway.request("POST", f"/bedrock/model/{model}/converse-stream", _PASSTHROUGH_BODY)
        assert response.status_code == 200, response.text
        assert response.headers.get("content-type") == _EVENT_STREAM, dict(response.headers)
        assert response.content == b"".join(_CONVERSE_FRAMES), response.text
        _received(wire, 1)


def test_c8_the_bedrock_passthrough_stream_already_retries_at_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS, num_retries=2)
        with httpx.Client(base_url=_proxy_url(gateway), timeout=_RETRY_WINDOW, trust_env=False) as client:
            response: Final = client.post(
                f"/bedrock/model/{model}/converse-stream", json=_PASSTHROUGH_BODY, headers=_auth(gateway)
            )
        assert response.status_code >= 400, response.text
        assert "Timeout" in response.text, response.text
        assert response.headers["x-litellm-timeout"] == f"{_TIMEOUT_SECONDS:.1f}", response.headers
        _assert_no_answer(response.text)
        _received(wire, 3)


def test_c9_model_info_reports_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        assert _model_timeout(gateway, model) == _TIMEOUT_SECONDS


def test_c10_a_deployment_without_any_timeout_keeps_waiting_on_a_stalled_stream(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        with pytest.raises(httpx.ReadTimeout):
            _streamed(gateway, "chat", _body("chat", model), window=_SHORT_WINDOW)
        _received(wire, 1)


_SOURCES: Final = (
    pytest.param({"timeout": _TIMEOUT_SECONDS}, {}, id="p1-body-timeout"),
    pytest.param({"request_timeout": _TIMEOUT_SECONDS}, {}, id="p2-body-request-timeout"),
    pytest.param({}, {"x-litellm-timeout": str(_TIMEOUT_SECONDS)}, id="p3-header-timeout"),
    pytest.param({"stream_timeout": _TIMEOUT_SECONDS}, {}, id="p4-body-stream-timeout"),
    pytest.param({}, {"x-litellm-stream-timeout": str(_TIMEOUT_SECONDS)}, id="p5-header-stream-timeout"),
)


@pytest.mark.parametrize(("extra", "headers"), _SOURCES)
def test_p1_to_p5_every_request_level_timeout_source_bounds_the_stream(
    gateway: Gateway, extra: Mapping[str, JsonValue], headers: Mapping[str, str]
) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        served: Final = _streamed(gateway, "chat", _body("chat", model, **extra), headers=headers)
        _assert_timed_out(served.status, served.text)
        _received(wire, 1)


def test_p6_the_request_timeout_beats_the_deployment_timeout(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=30)
        served: Final = _streamed(gateway, "chat", _body("chat", model, timeout=_TIMEOUT_SECONDS))
        _assert_timed_out(served.status, served.text)
        _received(wire, 1)


def test_r1_retries_each_wait_the_timeout_and_the_last_one_answers(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        served: Final = _streamed(gateway, "chat", _body("chat", model, num_retries=2), window=_RETRY_WINDOW)
        _assert_timed_out(served.status, served.text)
        _received(wire, 3)


def test_r2_a_stream_timeout_falls_back_to_the_next_deployment(gateway: Gateway) -> None:
    with _peer("stall") as stalled, _peer("fast") as prompt, gateway.scenario() as scenario:
        slow: Final = _converse(scenario, stalled, timeout=_TIMEOUT_SECONDS)
        fast: Final = _converse(scenario, prompt)
        with httpx.Client(base_url=_proxy_url(gateway), timeout=_RETRY_WINDOW, trust_env=False) as same_worker:
            _assert_answered(_streamed_on(same_worker, gateway, "chat", _body("chat", fast)))
            served: Final = _streamed_on(same_worker, gateway, "chat", _body("chat", slow, fallbacks=[fast]))
        _assert_answered(served)
        assert served.headers.get("x-litellm-attempted-fallbacks") == "1", served.headers
        _received(stalled, 1)
        _received(prompt, 2)


_BAD_VALUES: Final = (
    pytest.param("abc", id="s1-string"),
    pytest.param([1], id="s2-list"),
    pytest.param("x" * 5120, id="s4-five-kilobytes"),
)


@pytest.mark.parametrize("value", _BAD_VALUES)
def test_s1_s2_s4_an_unusable_timeout_value_is_an_error_the_caller_sees(gateway: Gateway, value: JsonValue) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        served: Final = _streamed(gateway, "chat", _body("chat", model, timeout=value))
        assert served.status >= 400, served.text
        assert "error" in served.text, served.text
        assert wire.received.qsize() == 0, wire.drain()
        _assert_answered(_streamed(gateway, "chat", _body("chat", model)))
        _received(wire, 1)


def test_s3_an_empty_timeout_string_is_ignored(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        _assert_answered(_streamed(gateway, "chat", _body("chat", model, timeout="")))
        _received(wire, 1)


def test_s5_the_last_duplicated_timeout_key_wins(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        duplicated: Final = (
            json.dumps({**_body("chat", model), "timeout": 30})[:-1] + f', "timeout": {_TIMEOUT_SECONDS}}}'
        )
        served: Final = _streamed(gateway, "chat", content=duplicated.encode())
        _assert_timed_out(served.status, served.text)
        _received(wire, 1)


def test_s6_a_zero_timeout_leaves_the_deployment_timeout_in_force(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        served: Final = _streamed(gateway, "chat", _body("chat", model, timeout=0))
        _assert_timed_out(served.status, served.text)
        _received(wire, 1)


def test_s7_a_negative_timeout_has_already_expired_for_streams_and_non_streams_alike(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        streamed: Final = _streamed(gateway, "chat", _body("chat", model, timeout=-1))
        _assert_timed_out(streamed.status, streamed.text, -1)
        plain: Final = _streamed(gateway, "chat", _body("chat", model, stream=False, timeout=-1))
        _assert_timed_out(plain.status, plain.text, -1)


def test_s8_a_malformed_deployment_timeout_fails_only_its_own_deployment(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        broken: Final = _converse(scenario, wire, timeout="fast")
        healthy: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        served: Final = _streamed(gateway, "chat", _body("chat", broken))
        assert served.status >= 400, served.text
        assert "error" in served.text, served.text
        assert wire.received.qsize() == 0, wire.drain()
        _assert_answered(_streamed(gateway, "chat", _body("chat", healthy)))
        _received(wire, 1)


def test_s9_an_unauthenticated_stream_never_reaches_the_deployment(gateway: Gateway) -> None:
    with _peer("fast") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire)
        served: Final = _streamed(gateway, "chat", _body("chat", model, timeout=_TIMEOUT_SECONDS), key="sk-not-a-key")
        assert served.status == 401, served.text
        assert wire.received.qsize() == 0, wire.drain()


def test_e1_a_null_timeout_reads_as_missing(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        served: Final = _streamed(gateway, "chat", _body("chat", model, timeout=None))
        _assert_timed_out(served.status, served.text)
        _received(wire, 1)


def test_e2_a_timeout_updated_while_traffic_flows_applies_to_the_next_stream(gateway: Gateway) -> None:
    with _peer("stall") as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire, timeout=_TIMEOUT_SECONDS)
        _assert_timed_out(*_status_and_text(_streamed(gateway, "chat", _body("chat", model))))
        gateway.post(
            "/model/update",
            {
                "model_name": model,
                "litellm_params": {"timeout": 3},
                "model_info": {"id": _model_identity(gateway, model)},
            },
        )
        eventually(lambda: _model_timeout(gateway, model), lambda value: value == 3, 30)
        eventually(
            lambda: _timeout_passed(_streamed(gateway, "chat", _body("chat", model)).text),
            lambda passed: passed == "3.0",
            45,
        )
        received: Final = wire.drain()
        assert len(received) >= 2, received


def _status_and_text(served: _Streamed) -> tuple[int, str]:
    return served.status, served.text
