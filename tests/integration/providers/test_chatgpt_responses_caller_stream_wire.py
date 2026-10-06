import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support import agentic_probe as ap
from integration._support import codex_vendor as cv
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Wire, wire_server
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseCompletedEvent,
    ResponseInputItemParam,
    ResponseTextDeltaEvent,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

_MODEL: Final = "chatgpt/gpt-5.5"
_NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
_INCOMPLETE_GATE: Final = threading.Event()


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy
    probe: Path

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    @property
    def base_url(self) -> str:
        return str(self.proxy.gateway.client.base_url).rstrip("/")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("chatgpt-caller-stream-rig")
    probe: Final = directory / "agentic-probe.jsonl"
    probe.touch()
    vendor: Final = cv.CodexVendor(incomplete_gate=_INCOMPLETE_GATE)
    with gateway_from_environment() as gateway, wire_server(vendor.respond) as wire:
        overrides: Final = {
            "CHATGPT_TOKEN_DIR": str(cv.login(directory)),
            "CHATGPT_API_BASE": wire.url,
            ap.OUT_ENVIRONMENT: str(probe),
        }
        config: Final = cv.proxy_config(directory, probe=True)
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as owned:
            yield _Rig(wire, owned, probe)


@pytest.fixture
def model(rig: _Rig) -> Iterator[str]:
    rig.wire.drain()
    with rig.gateway.scenario() as scenario:
        yield scenario.model(model=_MODEL, api_base=rig.wire.url, api_key=None)


def _prompt(marker: str, *directives: str) -> str:
    return " ".join((f"Say marker-{marker}", *directives))


def _sdk_input(marker: str) -> list[ResponseInputItemParam]:  # mutable-ok: the OpenAI SDK input parameter is a list
    return [EasyInputMessageParam(role="user", content=_prompt(marker))]


def _responses_body(
    model: str,
    marker: str,
    *directives: str,
    stream: bool | None = None,
    extra_body: Mapping[str, JsonValue] | None = None,
    cache_bust: bool = True,
) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "input": [{"role": "user", "content": _prompt(marker, *directives)}],
        **(_NO_CACHE if cache_bust else {}),
        **({} if stream is None else {"stream": stream}),
        **({} if extra_body is None else {"extra_body": dict(extra_body)}),
    }


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.gateway.request("POST", path, body)


def _only_forwarded(rig: _Rig, marker: str, *, stream: bool = True) -> Mapping[str, JsonValue]:
    (request,) = rig.wire.drain()
    return cv.forwarded(request, marker, stream=stream)


def _spend_rows(model: str, expected: int) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, litellm_call_id, status, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE model_group = %s',
            (model,),
        ),
        lambda rows: len(rows) >= expected,
        seconds=70,
    )


def _landed(model: str, response_id: str, *, expected: int = 1) -> Sequence[Mapping[str, JsonValue]]:
    rows: Final = _spend_rows(model, expected)
    (row,) = [row for row in rows if rv.same_response(string_value(row["request_id"]), response_id)]
    assert row["status"] == "success", rows
    return rows


def _landed_by_call_id(model: str, call_id: str) -> None:
    rows: Final = _spend_rows(model, 1)
    (row,) = [row for row in rows if row["litellm_call_id"] == call_id]
    assert row["status"] == "success", rows


def _output_text(body: Mapping[str, JsonValue]) -> str:
    (item,) = rv.ITEMS.validate_python(body["output"])
    (part,) = rv.ITEMS.validate_python(item["content"])
    return string_value(part["text"])


def _assert_json_answer(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    assert "event:" not in response.text, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.content)
    assert _output_text(body) == rv.answer(marker), body
    assert cv.totals(rv.JSON_OBJECT.validate_python(body["usage"])) == cv.totals(cv.USAGE), body
    return string_value(body["id"])


def _sse_frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {"))


def _assert_sse_answer(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    frames: Final = _sse_frames(response.text)
    (completed,) = [frame for frame in frames if frame.get("type") == "response.completed"]
    deltas: Final = "".join(
        string_value(frame["delta"]) for frame in frames if frame.get("type") == "response.output_text.delta"
    )
    assert deltas == rv.answer(marker), response.text
    return string_value(rv.JSON_OBJECT.validate_python(completed["response"])["id"])


def _assert_error(response: httpx.Response, *, status: int | None, message: str) -> None:
    assert response.status_code >= 400, response.text
    assert status is None or response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    assert "event:" not in response.text, response.text
    assert message in response.text, response.text


def _openai(rig: _Rig) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{rig.base_url}/v1", api_key=rig.gateway.key, max_retries=0)


def _async_openai(rig: _Rig) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=f"{rig.base_url}/v1", api_key=rig.gateway.key, max_retries=0)


def _anthropic(rig: _Rig) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=rig.base_url, api_key=rig.gateway.key, max_retries=0)


def test_openai_sdk_request_without_a_stream_flag_gets_the_aggregated_json_response(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    raw: Final = _openai(rig).responses.with_raw_response.create(
        model=model, input=_sdk_input(marker), extra_body=_NO_CACHE
    )
    assert raw.headers["content-type"].startswith("application/json"), raw.headers
    response: Final = raw.parse()
    assert response.output_text == rv.answer(marker), raw.text
    assert response.usage is not None and response.usage.total_tokens == cv.USAGE["total_tokens"], raw.text
    _only_forwarded(rig, marker)
    _landed(model, response.id)


async def test_async_openai_sdk_request_with_stream_false_gets_the_aggregated_json_response(
    rig: _Rig, model: str
) -> None:
    marker: Final = uuid.uuid4().hex
    raw: Final = await _async_openai(rig).responses.with_raw_response.create(
        model=model, input=_sdk_input(marker), stream=False, extra_body=_NO_CACHE
    )
    assert raw.headers["content-type"].startswith("application/json"), raw.headers
    response: Final = raw.parse()
    assert response.output_text == rv.answer(marker), raw.text
    _only_forwarded(rig, marker)
    _landed(model, response.id)


def test_raw_request_without_a_stream_key_gets_json_not_sse(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker))
    identity: Final = _assert_json_answer(response, marker)
    _only_forwarded(rig, marker)
    _landed(model, identity)


def test_openai_sdk_stream_request_still_streams(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    events: Final = list(
        _openai(rig).responses.create(model=model, input=_sdk_input(marker), stream=True, extra_body=_NO_CACHE)
    )
    deltas: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
    assert deltas == rv.answer(marker), events
    (completed,) = [event for event in events if isinstance(event, ResponseCompletedEvent)]
    _only_forwarded(rig, marker)
    _landed(model, completed.response.id)


async def test_async_openai_sdk_stream_request_still_streams(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = await _async_openai(rig).responses.create(
        model=model, input=_sdk_input(marker), stream=True, extra_body=_NO_CACHE
    )
    events: Final = [event async for event in stream]
    deltas: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
    assert deltas == rv.answer(marker), events
    (completed,) = [event for event in events if isinstance(event, ResponseCompletedEvent)]
    _only_forwarded(rig, marker)
    _landed(model, completed.response.id)


def test_openai_sdk_chat_completion_is_bridged_to_a_json_answer(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    completion: Final = _openai(rig).chat.completions.create(
        model=model, messages=[{"role": "user", "content": _prompt(marker)}], extra_body=_NO_CACHE
    )
    assert completion.choices[0].message.content == rv.answer(marker), completion
    _only_forwarded(rig, marker)
    _landed(model, completion.id)


async def test_async_openai_sdk_chat_completion_is_bridged_to_a_json_answer(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    completion: Final = await _async_openai(rig).chat.completions.create(
        model=model, messages=[{"role": "user", "content": _prompt(marker)}], extra_body=_NO_CACHE
    )
    assert completion.choices[0].message.content == rv.answer(marker), completion
    _only_forwarded(rig, marker)
    _landed(model, completion.id)


def test_openai_sdk_chat_completion_stream_still_streams(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    chunks: Final = list(
        _openai(rig).chat.completions.create(
            model=model, messages=[{"role": "user", "content": _prompt(marker)}], stream=True, extra_body=_NO_CACHE
        )
    )
    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == rv.answer(marker), chunks
    _only_forwarded(rig, marker)
    _landed(model, chunks[0].id)


def test_anthropic_sdk_message_is_bridged_to_a_json_answer(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    message: Final = _anthropic(rig).messages.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": _prompt(marker)}], extra_body=_NO_CACHE
    )
    (block,) = message.content
    assert block.type == "text" and block.text == rv.answer(marker), message
    _only_forwarded(rig, marker)
    _landed(model, message.id)


def test_anthropic_sdk_message_stream_still_streams(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    with _anthropic(rig).messages.stream(
        model=model, max_tokens=64, messages=[{"role": "user", "content": _prompt(marker)}], extra_body=_NO_CACHE
    ) as stream:
        text: Final = "".join(stream.text_stream)
        message: Final = stream.get_final_message()
    assert text == rv.answer(marker), message
    _only_forwarded(rig, marker)
    _landed_by_call_id(model, stream.response.headers["x-litellm-call-id"])


def test_identical_request_is_served_from_the_response_cache_as_json(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = _responses_body(model, marker, cache_bust=False)
    first: Final = _post(rig, "/v1/responses", body)
    identity: Final = _assert_json_answer(first, marker)
    assert "x-litellm-cache-key" not in first.headers, first.headers
    _only_forwarded(rig, marker)
    _landed(model, identity)
    second: Final = eventually(
        lambda: _post(rig, "/v1/responses", body), lambda found: "x-litellm-cache-key" in found.headers, seconds=20
    )
    assert rv.same_response(_assert_json_answer(second, marker), identity), second.text
    assert rig.wire.drain() == (), "the cached answer reached the vendor"
    rows: Final = _landed(model, identity, expected=2)
    (cached,) = [row for row in rows if "_cache_hit" in string_value(row["request_id"])]
    assert rv.same_response(string_value(cached["request_id"]).split("_cache_hit")[0], identity), rows
    assert (cached["status"], cached["cache_hit"]) == ("success", "True"), rows
    assert cached["spend"] == 0, rows


def test_agentic_hook_sees_the_aggregated_response_once(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker))
    identity: Final = _assert_json_answer(response, marker)
    _landed(model, identity)
    (line,) = eventually(lambda: ap.lines(rig.probe, marker), lambda found: len(found) >= 1)
    assert (line["surface"], line["response_type"], line["stream"], line["provider"]) == (
        "responses",
        "ResponsesAPIResponse",
        False,
        "chatgpt",
    ), line
    assert ap.lines(rig.probe, marker) == (line,), ap.lines(rig.probe, marker)


def test_agentic_hook_stays_out_of_a_stream_request(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker, stream=True))
    identity: Final = _assert_sse_answer(response, marker)
    _landed(model, identity)
    assert ap.lines(rig.probe, marker) == (), ap.lines(rig.probe, marker)


@dataclass(frozen=True, slots=True)
class _Junk:
    label: str
    fragment: str
    streams: bool


_JUNK: Final = (
    _Junk("null", '"stream": null', False),
    _Junk("empty-string", '"stream": ""', False),
    _Junk("empty-list", '"stream": []', False),
    _Junk("zero", '"stream": 0', False),
    _Junk("false-twice", '"stream": false, "stream": false', False),
    _Junk("true-then-false", '"stream": true, "stream": false', False),
    _Junk("one", '"stream": 1', True),
    _Junk("string-false", '"stream": "false"', True),
    _Junk("five-kb-string", f'"stream": "{"x" * 5000}"', True),
    _Junk("true-twice", '"stream": true, "stream": true', True),
)


@pytest.mark.parametrize("junk", _JUNK, ids=[junk.label for junk in _JUNK])
def test_odd_stream_values_decide_the_shape_by_their_truth(rig: _Rig, model: str, junk: _Junk) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = json.dumps(_responses_body(model, marker))[:-1] + f", {junk.fragment}}}"
    response: Final = rig.gateway.client.post(
        "/v1/responses",
        content=body.encode(),
        headers={"Authorization": f"Bearer {rig.gateway.key}", "content-type": "application/json"},
    )
    identity: Final = _assert_sse_answer(response, marker) if junk.streams else _assert_json_answer(response, marker)
    _only_forwarded(rig, marker)
    _landed(model, identity)


def test_extra_body_stream_true_streams(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker, extra_body={"stream": True}))
    identity: Final = _assert_sse_answer(response, marker)
    _only_forwarded(rig, marker)
    _landed(model, identity)


def test_extra_body_stream_false_gets_json(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker, extra_body={"stream": False}))
    identity: Final = _assert_json_answer(response, marker)
    _only_forwarded(rig, marker, stream=False)
    _landed(model, identity)


def test_string_input_is_refused_by_the_vendor_as_a_400(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", {"model": model, "input": _prompt(marker), **_NO_CACHE})
    _assert_error(response, status=400, message=cv.INPUT_MUST_BE_A_LIST)
    (request,) = rig.wire.drain()
    assert rv.JSON_OBJECT.validate_json(request.body)["input"] == _prompt(marker), request.body


def test_vendor_401_reaches_the_caller_as_401(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker, cv.UNAUTHORIZED_DIRECTIVE))
    _assert_error(response, status=401, message=cv.UNAUTHORIZED)
    _only_forwarded(rig, marker)


def test_vendor_response_failed_event_reaches_the_caller_as_a_json_error(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(rig, "/v1/responses", _responses_body(model, marker, cv.FAILED_DIRECTIVE))
    _assert_error(response, status=None, message=cv.failure_message(marker))
    _only_forwarded(rig, marker)


def test_vendor_stream_dying_mid_transfer_answers_a_json_error_and_leaves_the_proxy_serving(
    rig: _Rig, model: str
) -> None:
    marker: Final = uuid.uuid4().hex
    _INCOMPLETE_GATE.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: Final = pool.submit(
            _post, rig, "/v1/responses", _responses_body(model, marker, cv.INCOMPLETE_DIRECTIVE)
        )
        eventually(rig.wire.received.qsize, lambda size: size >= 1)
        liveliness: Final = rig.gateway.request("GET", "/health/liveliness", None)
        assert liveliness.status_code == 200, liveliness.text
        assert not pending.done(), pending.result().text
        _INCOMPLETE_GATE.set()
        response: Final = pending.result(timeout=60)
    assert response.status_code >= 400, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    assert "event:" not in response.text, response.text
    assert "error" in rv.JSON_OBJECT.validate_json(response.content), response.text
    _only_forwarded(rig, marker)
    follow_up: Final = uuid.uuid4().hex
    identity: Final = _assert_json_answer(_post(rig, "/v1/responses", _responses_body(model, follow_up)), follow_up)
    _only_forwarded(rig, follow_up)
    _landed(model, identity, expected=2)


def test_repeated_identical_cache_busted_requests_each_land_once(rig: _Rig, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = _responses_body(model, marker)
    identities: Final = tuple(_assert_json_answer(_post(rig, "/v1/responses", body), marker) for _ in range(5))
    assert len(set(identities)) == 5, identities
    received: Final = rig.wire.drain()
    assert len(received) == 5, [request.target for request in received]
    for request in received:
        cv.forwarded(request, marker)
    rows: Final = _spend_rows(model, 5)
    assert len(rows) == 5, rows
    for identity in identities:
        (row,) = [row for row in rows if rv.same_response(string_value(row["request_id"]), identity)]
        assert row["status"] == "success", rows
