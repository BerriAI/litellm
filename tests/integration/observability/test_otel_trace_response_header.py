import itertools
import json
import re
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

MARKER: Final = re.compile(rb"traceresp-[0-9a-f]{32}")
TRACERESPONSE: Final = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-01")
SERVER: Final = 2
CALL_ID: Final = "litellm.call_id"
ANSWER: Final = "trace ok"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
UPSTREAM_CALL_ID: Final = "upstream-chosen-call-id"
HOSTILE_UPSTREAM_VALUES: Final = MappingProxyType(
    {
        "empty": "",
        "numeric": "12345",
        "big": "00-" + "a" * 5000,
        "garbage": "not-a-traceparent",
        "joined": f"00-{'1' * 32}-{'2' * 16}-01, 00-{'3' * 32}-{'4' * 16}-01",
    }
)
RELAY_CASES: Final = ("none", "continued", "restarted", "callid", *HOSTILE_UPSTREAM_VALUES)


def _marker() -> str:
    return "traceresp-" + uuid.uuid4().hex


def _trace_id() -> str:
    return uuid.uuid4().hex


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def _chat_reply(identity: str, stream: bool) -> Reply:
    message: Final = {"role": "assistant", "content": ANSWER}
    usage: Final = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": message}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": ANSWER,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _provider(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    stream: Final = JSON_OBJECT.validate_json(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


def _query(request: Request) -> Mapping[str, str]:
    return MappingProxyType({name: values[0] for name, values in parse_qs(urlsplit(request.target).query).items()})


def _continued_traceresponse(request: Request) -> Reply:
    incoming: Final = TRACERESPONSE.fullmatch(request.headers.get("traceparent", ""))
    if incoming is None:
        return Reply(status=400, body=b'{"error":"no sampled traceparent reached the upstream"}')
    return Reply(body=b'{"case":"continued"}', headers={"traceresponse": f"00-{incoming.group(1)}-{_span_id()}-01"})


def _relay(request: Request) -> Reply:
    case: Final = urlsplit(request.target).path.rsplit("/", 1)[-1]
    query: Final = _query(request)
    body: Final = json.dumps({"case": case}).encode()
    if case == "continued":
        return _continued_traceresponse(request)
    if case == "restarted":
        return Reply(body=body, headers={"traceresponse": f"00-{query['trace']}-{query['span']}-01"})
    if case == "callid":
        return Reply(status=int(query.get("status", "200")), body=body, headers={"x-litellm-call-id": UPSTREAM_CALL_ID})
    if case in HOSTILE_UPSTREAM_VALUES:
        return Reply(body=body, headers={"traceresponse": HOSTILE_UPSTREAM_VALUES[case]})
    return Reply(body=body)


@dataclass(frozen=True, slots=True)
class ExportedSpan:
    trace_id: str
    span_id: str
    parent_span_id: str
    kind: int
    attributes: Mapping[str, JsonValue]
    links: tuple[tuple[str, str], ...]


def _attribute(value: JsonValue) -> JsonValue:
    typed: Final = object_value(value)
    return next((typed[key] for key in ("stringValue", "intValue", "boolValue", "doubleValue") if key in typed), None)


def _link(value: JsonValue) -> tuple[str, str]:
    link: Final = object_value(value)
    return str(link["traceId"]), str(link["spanId"])


def _exported(value: JsonValue) -> ExportedSpan:
    span: Final = object_value(value)
    attributes: Final = tuple(object_value(attribute) for attribute in list_value(span.get("attributes", [])))
    return ExportedSpan(
        trace_id=str(span["traceId"]),
        span_id=str(span["spanId"]),
        parent_span_id=str(span.get("parentSpanId", "")),
        kind=int(str(span.get("kind", 0))),
        attributes=MappingProxyType({str(item["key"]): _attribute(item["value"]) for item in attributes}),
        links=tuple(_link(link) for link in list_value(span.get("links", []))),
    )


def _scope_spans(resource: JsonValue) -> Iterator[JsonValue]:
    return itertools.chain.from_iterable(
        list_value(object_value(scope).get("spans", [])) for scope in list_value(object_value(resource)["scopeSpans"])
    )


def _batch_spans(body: bytes) -> tuple[ExportedSpan, ...]:
    resources: Final = list_value(JSON_OBJECT.validate_json(body).get("resourceSpans", []))
    return tuple(_exported(span) for span in itertools.chain.from_iterable(map(_scope_spans, resources)))


@dataclass(frozen=True, slots=True)
class Collector:
    wire: Wire
    outage: threading.Event
    accepted: Sequence[Request]
    guard: threading.Lock

    def spans(self) -> tuple[ExportedSpan, ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        return tuple(itertools.chain.from_iterable(_batch_spans(batch.body) for batch in batches))

    def with_span_id(self, span_id: str) -> tuple[ExportedSpan, ...]:
        return tuple(span for span in self.spans() if span.span_id == span_id)

    def span(self, span_id: str) -> ExportedSpan:
        eventually(lambda: self.with_span_id(span_id), lambda found: len(found) >= 1, seconds=30)
        found: Final = self.with_span_id(span_id)
        assert len(found) == 1, found
        return found[0]

    def server_spans_for_call(self, call_id: str) -> tuple[ExportedSpan, ...]:
        return tuple(span for span in self.spans() if span.kind == SERVER and span.attributes.get(CALL_ID) == call_id)

    def server_span_for_call(self, call_id: str) -> ExportedSpan:
        eventually(lambda: self.server_spans_for_call(call_id), lambda found: len(found) >= 1, seconds=30)
        found: Final = self.server_spans_for_call(call_id)
        assert len(found) == 1, found
        return found[0]


@pytest.fixture(scope="module")
def collector() -> Iterator[Collector]:
    outage: Final = threading.Event()
    accepted: Final[deque[Request]] = deque()  # mutable-ok: the sink thread appends each accepted export batch
    guard: Final = threading.Lock()

    def sink(request: Request) -> Reply:
        if outage.is_set():
            return Reply(status=503, body=b'{"error":"sink down"}')
        with guard:
            accepted.append(request)
        return Reply()

    with wire_server(sink) as wire:
        yield Collector(wire, outage, accepted, guard)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_provider) as wire:
        yield wire


@pytest.fixture(scope="module")
def relay() -> Iterator[Wire]:
    with wire_server(_relay) as wire:
        yield wire


@dataclass(frozen=True, slots=True)
class Answer:
    headers: httpx.Headers
    text: str


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    process: OwnedProxy
    model: str
    upstream: Wire
    sink: Collector

    def base_url(self) -> str:
        return str(self.proxy.client.base_url)

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=self.base_url() + "/v1", api_key=self.proxy.key, max_retries=0)

    def anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=self.base_url(), api_key=self.proxy.key, max_retries=0)

    def post(self, path: str, body: Mapping[str, JsonValue], headers: Sequence[tuple[str, str]] = ()) -> httpx.Response:
        return self.proxy.client.post(
            path, json=body, headers=[("Authorization", f"Bearer {self.proxy.key}"), *headers]
        )

    def chat(self, marker: str, *, headers: Sequence[tuple[str, str]] = ()) -> httpx.Response:
        return self.post(
            "/v1/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            headers,
        )

    def upstream_count(self, marker: str) -> int:
        return sum(1 for request in self.upstream.drain() if marker.encode() in request.body)


def _config(provider: Wire, relay: Wire, sink: Collector, model: str) -> dict[str, JsonValue]:
    config: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **config,
        "model_list": [
            *list_value(config.get("model_list") or []),
            {
                "model_name": model,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": provider.url + "/v1",
                },
            },
        ],
        "litellm_settings": {**object_value(config["litellm_settings"]), "callbacks": ["otel"]},
        "callback_settings": {"otel": {"exporter": "http/json", "endpoint": sink.wire.url}},
        "general_settings": {
            **object_value(config["general_settings"]),
            "pass_through_endpoints": [
                {"path": f"/relay/{case}", "target": f"{relay.url}/relay/{case}"} for case in RELAY_CASES
            ],
        },
    }


@dataclass(frozen=True, slots=True)
class RigFactory:
    provider: Wire
    relay: Wire
    sink: Collector
    directory: Path
    otel_v2: bool

    def start(self) -> Iterator[Rig]:
        model: Final = "traceresp-model-" + uuid.uuid4().hex[:12]
        path: Final = self.directory / f"traceresponse-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(_config(self.provider, self.relay, self.sink, model)))
        gate: Final = {"LITELLM_OTEL_V2": "1"} if self.otel_v2 else {}
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                self.directory,
                {**gate, "OTEL_BSP_SCHEDULE_DELAY": "300"},
                config=path,
                remove_environment=() if self.otel_v2 else ("LITELLM_OTEL_V2",),
                workers=2,
            ) as owned,
        ):
            yield Rig(owned.gateway, owned, model, self.provider, self.sink)


@pytest.fixture(scope="module")
def factory(
    provider: Wire, relay: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[bool], RigFactory]:
    return lambda otel_v2: RigFactory(provider, relay, collector, tmp_path_factory.mktemp("traceresponse"), otel_v2)


@pytest.fixture(scope="module")
def rig(factory: Callable[[bool], RigFactory]) -> Iterator[Rig]:
    yield from factory(True).start()


@pytest.fixture(scope="module")
def control(factory: Callable[[bool], RigFactory]) -> Iterator[Rig]:
    yield from factory(False).start()


def _named(headers: httpx.Headers) -> tuple[str, str]:
    values: Final = headers.get_list("traceresponse")
    assert len(values) == 1, values
    match: Final = TRACERESPONSE.fullmatch(values[0])
    assert match is not None, values
    return match.group(1), match.group(2)


def _assert_names_its_server_span(rig: Rig, headers: httpx.Headers) -> ExportedSpan:
    trace_id, span_id = _named(headers)
    span: Final = rig.sink.span(span_id)
    assert (span.kind, span.trace_id) == (SERVER, trace_id), span
    call_ids: Final = headers.get_list("x-litellm-call-id")
    assert len(call_ids) == 1 and span.attributes.get(CALL_ID) == call_ids[0], (call_ids, span)
    return span


def _chat_sdk(rig: Rig, marker: str) -> Answer:
    completion: Final = rig.openai_client().chat.completions.with_raw_response.create(
        model=rig.model, messages=[{"role": "user", "content": marker}], extra_body={"cache": {"no-cache": True}}
    )
    return Answer(completion.headers, completion.parse().choices[0].message.content or "")


def _chat_stream_sdk(rig: Rig, marker: str) -> Answer:
    with rig.openai_client().chat.completions.with_streaming_response.create(
        model=rig.model,
        messages=[{"role": "user", "content": marker}],
        stream=True,
        extra_body={"cache": {"no-cache": True}},
    ) as chunks:
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks.parse() if chunk.choices)
        return Answer(chunks.headers, text)


def _messages_sdk(rig: Rig, marker: str) -> Answer:
    reply: Final = rig.anthropic_client().messages.with_raw_response.create(
        model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    )
    block: Final = reply.parse().content[0]
    assert isinstance(block, anthropic.types.TextBlock), block
    return Answer(reply.headers, block.text)


def _messages_stream_sdk(rig: Rig, marker: str) -> Answer:
    with rig.anthropic_client().messages.with_streaming_response.create(
        model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
    ) as streamed:
        text: Final = "".join(
            event.delta.text
            for event in streamed.parse()
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        )
        return Answer(streamed.headers, text)


def _responses_sdk(rig: Rig, marker: str) -> Answer:
    created: Final = rig.openai_client().responses.with_raw_response.create(model=rig.model, input=marker)
    return Answer(created.headers, created.parse().output_text)


def _responses_stream_sdk(rig: Rig, marker: str) -> Answer:
    with rig.openai_client().responses.with_streaming_response.create(
        model=rig.model, input=marker, stream=True
    ) as events:
        text: Final = "".join(event.delta for event in events.parse() if event.type == "response.output_text.delta")
        return Answer(events.headers, text)


SDK_ROUTES: Final[Mapping[str, Callable[[Rig, str], Answer]]] = MappingProxyType(
    {
        "chat": _chat_sdk,
        "chat_stream": _chat_stream_sdk,
        "messages": _messages_sdk,
        "messages_stream": _messages_stream_sdk,
        "responses": _responses_sdk,
        "responses_stream": _responses_stream_sdk,
    }
)


@pytest.mark.parametrize("route", tuple(SDK_ROUTES))
def test_sdk_response_names_the_server_span_that_carries_its_call_id(rig: Rig, route: str) -> None:
    marker: Final = _marker()
    answer: Final = SDK_ROUTES[route](rig, marker)
    assert answer.text == ANSWER, answer
    _assert_names_its_server_span(rig, answer.headers)
    assert rig.upstream_count(marker) == 1


def _spend_rows(response_id: str) -> int:
    return len(read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)))


@pytest.mark.parametrize("stream", (False, True), ids=("plain", "stream"))
def test_a_response_cache_hit_names_its_own_server_span(rig: Rig, stream: bool) -> None:
    marker: Final = _marker()
    body: Final[dict[str, JsonValue]] = {
        "model": rig.model,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
    }
    first: Final = rig.post("/v1/chat/completions", body)
    assert first.status_code == 200, first.text
    first_span: Final = _assert_names_its_server_span(rig, first.headers)
    eventually(lambda: _spend_rows(f"chatcmpl-{marker}"), lambda rows: rows == 1, seconds=70)
    second: Final = rig.post("/v1/chat/completions", body)
    assert second.status_code == 200, second.text
    assert f"chatcmpl-{marker}" in second.text, second.text
    second_span: Final = _assert_names_its_server_span(rig, second.headers)
    assert second_span.span_id != first_span.span_id
    assert second.headers["x-litellm-call-id"] != first.headers["x-litellm-call-id"]
    assert rig.upstream_count(marker) == 1


def test_a_sampled_caller_traceparent_is_continued_in_the_header(rig: Rig) -> None:
    trace_id: Final = _trace_id()
    parent: Final = _span_id()
    response: Final = rig.chat(_marker(), headers=(("traceparent", f"00-{trace_id}-{parent}-01"),))
    assert response.status_code == 200, response.text
    span: Final = _assert_names_its_server_span(rig, response.headers)
    assert (span.trace_id, span.parent_span_id) == (trace_id, parent), span


def test_an_unsampled_caller_traceparent_gets_no_header(rig: Rig) -> None:
    response: Final = rig.chat(_marker(), headers=(("traceparent", f"00-{_trace_id()}-{_span_id()}-00"),))
    assert response.status_code == 200, response.text
    assert response.headers.get_list("traceresponse") == []


@pytest.mark.parametrize(
    "traceparent",
    ("", "12345", "00-" + "f" * 5000, "not-a-traceparent", f"00-{'5' * 32}-{'6' * 16}-01, 00-{'7' * 32}-{'8' * 16}-01"),
    ids=("empty", "numeric", "five_kilobytes", "garbage", "list"),
)
def test_a_malformed_caller_traceparent_starts_a_fresh_trace_in_the_header(rig: Rig, traceparent: str) -> None:
    response: Final = rig.chat(_marker(), headers=(("traceparent", traceparent),))
    assert response.status_code == 200, response.text
    span: Final = _assert_names_its_server_span(rig, response.headers)
    assert span.parent_span_id == "", span
    assert span.trace_id not in traceparent, span


def test_a_traceparent_sent_twice_is_continued_once(rig: Rig) -> None:
    trace_id: Final = _trace_id()
    parent: Final = _span_id()
    value: Final = f"00-{trace_id}-{parent}-01"
    response: Final = rig.chat(_marker(), headers=(("traceparent", value), ("traceparent", value)))
    assert response.status_code == 200, response.text
    span: Final = _assert_names_its_server_span(rig, response.headers)
    assert (span.trace_id, span.parent_span_id) == (trace_id, parent), span


def _error_bad_key(rig: Rig) -> httpx.Response:
    return rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.model, "messages": [{"role": "user", "content": _marker()}]},
        headers={"Authorization": "Bearer sk-not-a-real-key"},
    )


def _error_unknown_model(rig: Rig) -> httpx.Response:
    return rig.post(
        "/v1/chat/completions",
        {"model": "no-such-model-" + uuid.uuid4().hex, "messages": [{"role": "user", "content": "x"}]},
    )


def _error_invalid_body(rig: Rig) -> httpx.Response:
    return rig.post("/key/generate", {"models": "not-a-list"})


def _error_unhandled(rig: Rig) -> httpx.Response:
    return rig.proxy.client.post(
        "/v1/chat/completions",
        content=b"[1]",
        headers={"Authorization": f"Bearer {rig.proxy.key}", "content-type": "application/json"},
    )


ERRORS: Final[Mapping[str, tuple[int, Callable[[Rig], httpx.Response]]]] = MappingProxyType(
    {
        "bad_key": (401, _error_bad_key),
        "unknown_model": (400, _error_unknown_model),
        "invalid_body": (422, _error_invalid_body),
        "unhandled": (500, _error_unhandled),
    }
)


@pytest.mark.parametrize("error", tuple(ERRORS))
def test_an_error_response_names_its_own_server_span(rig: Rig, error: str) -> None:
    status, send = ERRORS[error]
    response: Final = send(rig)
    assert response.status_code == status, response.text
    trace_id, span_id = _named(response.headers)
    span: Final = rig.sink.span(span_id)
    assert (span.kind, span.trace_id) == (SERVER, trace_id), span


def test_an_excluded_health_route_gets_no_header(rig: Rig) -> None:
    response: Final = rig.proxy.client.get("/health/liveliness")
    assert response.status_code == 200, response.text
    assert response.headers.get_list("traceresponse") == []


def test_a_pass_through_without_an_upstream_header_gets_the_gateway_header(rig: Rig) -> None:
    response: Final = rig.post("/relay/none", {"marker": _marker()})
    assert response.status_code == 200, response.text
    assert response.json() == {"case": "none"}
    span: Final = _assert_names_its_server_span(rig, response.headers)
    assert span.links == (), span


def test_a_pass_through_upstream_header_in_the_same_trace_is_relayed_unchanged_without_a_link(rig: Rig) -> None:
    trace_id: Final = _trace_id()
    response: Final = rig.post(
        "/relay/continued", {"marker": _marker()}, (("traceparent", f"00-{trace_id}-{_span_id()}-01"),)
    )
    assert response.status_code == 200, response.text
    relayed: Final = response.headers.get_list("traceresponse")
    assert len(relayed) == 1 and relayed[0].startswith(f"00-{trace_id}-"), relayed
    span: Final = rig.sink.server_span_for_call(response.headers["x-litellm-call-id"])
    assert span.trace_id == trace_id, span
    assert span.span_id not in relayed[0], (span, relayed)
    assert span.links == (), span


def test_a_pass_through_upstream_that_restarted_the_trace_is_relayed_and_linked(rig: Rig) -> None:
    upstream_trace: Final = _trace_id()
    upstream_span: Final = _span_id()
    response: Final = rig.proxy.client.post(
        "/relay/restarted",
        params={"trace": upstream_trace, "span": upstream_span},
        json={"marker": _marker()},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get_list("traceresponse") == [f"00-{upstream_trace}-{upstream_span}-01"]
    span: Final = rig.sink.server_span_for_call(response.headers["x-litellm-call-id"])
    assert span.trace_id != upstream_trace, span
    assert span.links == ((upstream_trace, upstream_span),), span


@pytest.mark.parametrize("case", tuple(HOSTILE_UPSTREAM_VALUES))
def test_a_hostile_pass_through_upstream_header_is_relayed_once_unchanged_and_never_linked(rig: Rig, case: str) -> None:
    response: Final = rig.post(f"/relay/{case}", {"marker": _marker()})
    assert response.status_code == 200, response.text
    assert response.headers.get_list("traceresponse") == [HOSTILE_UPSTREAM_VALUES[case]]
    span: Final = rig.sink.server_span_for_call(response.headers["x-litellm-call-id"])
    assert span.links == (), span


@pytest.mark.parametrize("status", (200, 404, 500))
def test_a_pass_through_upstream_call_id_never_replaces_the_gateway_call_id(rig: Rig, status: int) -> None:
    response: Final = rig.proxy.client.post(
        "/relay/callid",
        params={"status": str(status)},
        json={"marker": _marker()},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == status, response.text
    call_ids: Final = response.headers.get_list("x-litellm-call-id")
    assert len(call_ids) == 1 and call_ids[0] != UPSTREAM_CALL_ID, call_ids
    _assert_names_its_server_span(rig, response.headers)
    assert rig.sink.server_spans_for_call(UPSTREAM_CALL_ID) == ()


def _exposed(response: httpx.Response) -> frozenset[str]:
    listed: Final = response.headers.get("access-control-expose-headers", "")
    return frozenset(name.strip().lower() for name in listed.split(",") if name.strip())


def test_a_browser_origin_can_read_the_header(rig: Rig) -> None:
    response: Final = rig.proxy.request("GET", "/v1/models", headers={"Origin": "http://dashboard.example"})
    assert response.status_code == 200, response.text
    assert "traceresponse" in _exposed(response), response.headers
    _named(response.headers)


def test_with_the_v2_gate_off_no_response_carries_or_exposes_the_header(control: Rig) -> None:
    marker: Final = _marker()
    chat: Final = control.chat(marker)
    assert chat.status_code == 200, chat.text
    assert chat.headers.get_list("traceresponse") == []
    assert control.upstream_count(marker) == 1
    models: Final = control.proxy.request("GET", "/v1/models", headers={"Origin": "http://dashboard.example"})
    assert models.status_code == 200, models.text
    assert _exposed(models), models.headers
    assert "traceresponse" not in _exposed(models), models.headers
    assert models.headers.get_list("traceresponse") == []


def _raw_send(rig: Rig, index: int) -> httpx.Response:
    marker: Final = _marker()
    stream: Final = index % 2 == 0
    route: Final = index % 3
    if route == 0:
        return rig.post(
            "/v1/chat/completions",
            {"model": rig.model, "messages": [{"role": "user", "content": marker}], "stream": stream},
        )
    if route == 1:
        return rig.post("/v1/responses", {"model": rig.model, "input": marker, "stream": stream})
    return rig.post(
        "/v1/messages",
        {"model": rig.model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], "stream": stream},
    )


def _burst(rig: Rig, count: int) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(lambda index: _raw_send(rig, index), range(count)))


def _assert_each_names_its_own_span(rig: Rig, responses: Sequence[httpx.Response]) -> None:
    assert [response.status_code for response in responses] == [200] * len(responses), [
        response.text for response in responses
    ]
    spans: Final = tuple(_assert_names_its_server_span(rig, response.headers) for response in responses)
    assert len({span.span_id for span in spans}) == len(responses), spans
    assert len({span.trace_id for span in spans}) == len(responses), spans


def test_concurrent_mixed_requests_each_name_their_own_server_span(rig: Rig) -> None:
    _assert_each_names_its_own_span(rig, _burst(rig, 24))


def test_a_sink_outage_keeps_every_header_and_the_spans_land_after_recovery(rig: Rig) -> None:
    rig.sink.outage.set()
    try:
        responses: Final = _burst(rig, 20)
        assert all(len(TRACERESPONSE.findall(response.headers.get("traceresponse", ""))) == 1 for response in responses)
        eventually(lambda: any(True for _ in rig.sink.wire.drain()), lambda seen: seen, seconds=30)
    finally:
        rig.sink.outage.clear()
    _assert_each_names_its_own_span(rig, responses)


def _fresh_chat(rig: Rig) -> httpx.Response:
    with httpx.Client(base_url=rig.base_url(), timeout=30, trust_env=False) as client:
        return client.post(
            "/v1/chat/completions",
            json={
                "model": rig.model,
                "messages": [{"role": "user", "content": _marker()}],
                "cache": {"no-cache": True},
            },
            headers={"Authorization": f"Bearer {rig.proxy.key}"},
        )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_replacement_workers_still_name_their_server_spans(factory: Callable[[bool], RigFactory]) -> None:
    started: Final = factory(True).start()
    fresh: Final = next(started)
    try:
        originals: Final = eventually(fresh.process.worker_pids, lambda found: len(found) == 2, seconds=30)
        replacements: Final = frozenset(fresh.process.kill_worker(pid) for pid in sorted(originals))
        assert not replacements & originals, (originals, replacements)
        assert fresh.process.process.poll() is None, "Proxy root exited after its workers were replaced"
        _assert_each_names_its_own_span(fresh, tuple(_fresh_chat(fresh) for _ in range(8)))
    finally:
        started.close()
