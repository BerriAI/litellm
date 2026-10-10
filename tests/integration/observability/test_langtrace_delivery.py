import asyncio
import json
import re
import signal
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import psutil
import pytest
import yaml
from anthropic import Anthropic
from integration._support.client import Gateway, eventually
from integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status
from pydantic import TypeAdapter

TRACE_PATH: Final = "/api/trace"
STOCK_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
_PROXY_CONFIG: Final = TypeAdapter(dict[str, object])
_SETTINGS: Final = TypeAdapter(dict[str, object])
_MARKER: Final = re.compile(rb"lt[0-9a-f]{32}")
_USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


def _marker() -> str:
    return "lt" + uuid.uuid4().hex


def _sse(events: Sequence[object]) -> tuple[bytes, ...]:
    return tuple(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (b"data: [DONE]\n\n",)


def _chat_reply(marker: str, stream: bool) -> Reply:
    identity: Final = "chatcmpl-" + marker
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "echo " + marker},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _USAGE,
                }
            ).encode()
        )
    head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            (
                {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "echo "}}]},
                {**head, "choices": [{"index": 0, "delta": {"content": marker}}]},
                {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**head, "choices": [], "usage": _USAGE},
            )
        ),
    )


def _responses_reply(marker: str, stream: bool) -> Reply:
    completed: Final = {
        "id": "resp_" + marker,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + marker,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "echo " + marker, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_" + marker,
            "output_index": 0,
            "content_index": 0,
            "delta": "echo " + marker,
        },
        {"type": "response.completed", "response": completed},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    match: Final = _MARKER.search(request.body)
    assert match is not None, request.body[:300]
    marker: Final = match.group().decode()
    if b'"fail"' in request.body:
        return Reply(status=401, body=json.dumps({"error": {"message": "bad provider key " + marker}}).encode())
    stream: Final = json.loads(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(marker, stream)
    return _chat_reply(marker, stream)


def _config(tmp_path: Path, **litellm_settings: object) -> Path:
    config: Final = _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))
    settings: Final = {**_SETTINGS.validate_python(config["litellm_settings"]), **litellm_settings}
    path: Final = tmp_path / "langtrace.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": settings}))
    return path


def _spans(batches: Sequence[Request]) -> tuple[Span, ...]:
    return tuple(
        span
        for batch in batches
        for resource_spans in ExportTraceServiceRequest.FromString(batch.body).resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    )


def _prompt_events(span: Span) -> tuple[str, ...]:
    return tuple(
        attribute.value.string_value
        for event in span.events
        if event.name == "gen_ai.content.prompt"
        for attribute in event.attributes
        if attribute.key == "gen_ai.prompt"
    )


def _assert_prompted_with(span: Span, marker: str) -> Span:
    prompts: Final = _prompt_events(span)
    assert any(marker in prompt for prompt in prompts), (span.name, prompts)
    return span


def _spans_carrying(batches: Sequence[Request], marker: str, name: str | None = "litellm_request") -> tuple[Span, ...]:
    return tuple(
        span for span in _spans(batches) if name in (None, span.name) and marker.encode() in span.SerializeToString()
    )


def _streamed_text(sse: str, key: str) -> str:
    def strings(node: object) -> Iterator[str]:
        if isinstance(node, dict):
            for field_name, value in node.items():
                if field_name == key and isinstance(value, str):
                    yield value
                else:
                    yield from strings(value)
        if isinstance(node, list):
            for item in node:
                yield from strings(item)

    events: Final = tuple(
        json.loads(line.removeprefix("data: "))
        for line in sse.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    return "".join(text for event in events for text in strings(event))


def _accepted(request: Request) -> Reply:
    return Reply(body=b'{"message":"Traces added successfully"}')


@dataclass(frozen=True, slots=True)
class _Sink:
    wire: Wire
    api_key: str
    # mutable-ok: drain() consumes, so batches accumulate across polls
    received: list[Request] = field(default_factory=list)

    def collect(self) -> tuple[Request, ...]:
        self.received.extend(self.wire.drain())
        return tuple(self.received)

    def spans_for(self, marker: str) -> tuple[Span, ...]:
        return _spans_carrying(self.collect(), marker)

    def assert_wire_contract(self, batches: Sequence[Request], target: str = TRACE_PATH) -> None:
        for request in batches:
            assert (request.method, request.target) == ("POST", target), (request.method, request.target)
            assert request.headers.get("x-api-key") == self.api_key, request.headers
            assert "api_key" not in request.headers, request.headers
            assert request.headers.get("content-type") == "application/x-protobuf", request.headers
        assert self.api_key.encode() not in b"".join(batch.body for batch in batches)

    def delivered_once(self, marker: str, seconds: float = 20) -> Span:
        batches: Final = eventually(
            self.collect, lambda value: len(_spans_carrying(value, marker)) >= 1, seconds=seconds
        )
        self.assert_wire_contract(batches)
        settled: Final = eventually(
            self.collect, lambda value: len(_spans_carrying(value, marker)) >= 2, seconds=1, return_last_on_timeout=True
        )
        spans: Final = _spans_carrying(settled, marker)
        assert len(spans) == 1, [span.span_id for span in spans]
        return _assert_prompted_with(spans[0], marker)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    model: str
    provider: Wire
    sink: _Sink

    def provider_hits(self, marker: str) -> int:
        return sum(marker.encode() in request.body for request in self.provider.drain())


@contextmanager
def _langtrace_rig(
    gateway: Gateway,
    tmp_path: Path,
    *,
    mode: str = "callbacks",
    host: Callable[[str], str] = lambda url: url,
    api_key: str | None = None,
    workers: int = 1,
    respond: Callable[[Request], Reply] = _accepted,
    sink_port: int = 0,
) -> Iterator[_Rig]:
    key: Final = "synthetic-langtrace-key-" + uuid.uuid4().hex if api_key is None else api_key
    with wire_server(_upstream) as provider, wire_server(respond, port=sink_port) as sink:
        overrides: Final = {
            "LANGTRACE_API_KEY": key,
            "LANGTRACE_API_HOST": host(sink.url),
            "OTEL_BSP_SCHEDULE_DELAY": "300",
        }
        config: Final = _config(tmp_path, **{mode: ["langtrace"]})
        with (
            owned_proxy(gateway, tmp_path, overrides, config=config, workers=workers) as proxy,
            proxy.scenario() as scenario,
        ):
            yield _Rig(proxy, scenario.model(api_base=provider.url + "/v1"), provider, _Sink(sink, key))


def _chat_httpx(rig: _Rig, marker: str, stream: bool) -> str:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.model, "messages": [{"role": "user", "content": marker}], "stream": stream},
    )
    assert response.status_code == 200, response.text
    return _streamed_text(response.text, "content") if stream else response.text


def _chat_openai_sync_stream(rig: _Rig, marker: str, stream: bool) -> str:
    with OpenAI(base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0) as client:
        chunks: Final = client.chat.completions.create(
            model=rig.model, messages=[{"role": "user", "content": marker}], stream=True
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)


def _chat_openai_async(rig: _Rig, marker: str, stream: bool) -> str:
    async def call() -> str:
        async with AsyncOpenAI(base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0) as client:
            completion: Final = await client.chat.completions.create(
                model=rig.model, messages=[{"role": "user", "content": marker}]
            )
            return completion.model_dump_json()

    return asyncio.run(call())


def _messages_anthropic(rig: _Rig, marker: str, stream: bool) -> str:
    with Anthropic(base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0) as client:
        message: Final = client.messages.create(
            model=rig.model, max_tokens=64, messages=[{"role": "user", "content": marker}]
        )
        return message.model_dump_json()


def _messages_httpx(rig: _Rig, marker: str, stream: bool) -> str:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        {"model": rig.model, "max_tokens": 64, "messages": [{"role": "user", "content": marker}], "stream": stream},
    )
    assert response.status_code == 200, response.text
    return _streamed_text(response.text, "text") if stream else response.text


def _responses_openai(rig: _Rig, marker: str, stream: bool) -> str:
    with OpenAI(base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0) as client:
        return client.responses.create(model=rig.model, input=marker).model_dump_json()


def _responses_httpx(rig: _Rig, marker: str, stream: bool) -> str:
    response: Final = rig.proxy.request(
        "POST", "/v1/responses", {"model": rig.model, "input": marker, "stream": stream}
    )
    assert response.status_code == 200, response.text
    return _streamed_text(response.text, "delta") if stream else response.text


@dataclass(frozen=True, slots=True)
class _Surface:
    call: Callable[[_Rig, str, bool], str]
    stream: bool


_SURFACES: Final = (
    pytest.param(_Surface(_chat_httpx, False), id="chat-httpx"),
    pytest.param(_Surface(_chat_openai_sync_stream, True), id="chat-openai-sync-stream"),
    pytest.param(_Surface(_chat_openai_async, False), id="chat-openai-async"),
    pytest.param(_Surface(_messages_anthropic, False), id="messages-anthropic"),
    pytest.param(_Surface(_messages_httpx, True), id="messages-httpx-stream"),
    pytest.param(_Surface(_responses_openai, False), id="responses-openai"),
    pytest.param(_Surface(_responses_httpx, True), id="responses-httpx-stream"),
)


def _assert_delivered(rig: _Rig, surface: _Surface, marker: str) -> Span:
    text: Final = surface.call(rig, marker, surface.stream)
    assert "echo " + marker in text, text
    assert rig.provider_hits(marker) == 1
    return rig.sink.delivered_once(marker)


@pytest.mark.parametrize("surface", _SURFACES)
def test_langtrace_span_reaches_api_trace_with_x_api_key(gateway: Gateway, tmp_path: Path, surface: _Surface) -> None:
    with _langtrace_rig(gateway, tmp_path) as rig:
        _assert_delivered(rig, surface, _marker())


def test_langtrace_exports_cache_hit_twin_as_its_own_span(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with _langtrace_rig(gateway, tmp_path) as rig:
        first: Final = rig.proxy.request(
            "POST", "/v1/chat/completions", {"model": rig.model, "messages": [{"role": "user", "content": marker}]}
        )
        assert first.status_code == 200, first.text
        rig.sink.delivered_once(marker)
        second: Final = rig.proxy.request(
            "POST", "/v1/chat/completions", {"model": rig.model, "messages": [{"role": "user", "content": marker}]}
        )
        assert second.status_code == 200 and second.headers.get("x-litellm-cache-key"), second.headers
        assert second.json()["id"] == first.json()["id"], second.text
        assert rig.provider_hits(marker) == 1
        batches: Final = eventually(
            rig.sink.collect, lambda value: len(_spans_carrying(value, marker)) >= 2, seconds=20
        )
        rig.sink.assert_wire_contract(batches)
        assert len(_spans_carrying(batches, marker)) == 2


def test_langtrace_success_callback_mode_delivers(gateway: Gateway, tmp_path: Path) -> None:
    with _langtrace_rig(gateway, tmp_path, mode="success_callback") as rig:
        _assert_delivered(rig, _Surface(_chat_httpx, False), _marker())


def test_langtrace_failure_callback_mode_exports_provider_error_span(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with _langtrace_rig(gateway, tmp_path, mode="failure_callback") as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": rig.model, "messages": [{"role": "user", "content": marker + " fail"}], "user": "fail"},
        )
        assert response.status_code == 401, response.text
        assert "bad provider key " + marker in response.text, response.text
        assert rig.provider_hits(marker) == 1
        span: Final = rig.sink.delivered_once(marker)
        assert span.status.code == Status.STATUS_CODE_ERROR, span.status


@pytest.mark.parametrize("status", (403, 404), ids=("forbidden", "not-found"))
def test_langtrace_rejecting_sink_leaves_callers_and_later_exports_intact(
    gateway: Gateway, tmp_path: Path, status: int
) -> None:
    scripted: Final[SimpleQueue[int]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        return Reply(status=scripted.get_nowait()) if not scripted.empty() else _accepted(request)

    rejected: Final = _marker()
    accepted: Final = _marker()
    with _langtrace_rig(gateway, tmp_path, respond=respond) as rig:
        scripted.put(status)
        assert "echo " + rejected in _chat_httpx(rig, rejected, False)
        batches: Final = eventually(
            rig.sink.collect, lambda value: len(_spans_carrying(value, rejected)) >= 1, seconds=20
        )
        rig.sink.assert_wire_contract(batches)
        assert scripted.empty()
        assert "echo " + accepted in _chat_httpx(rig, accepted, False)
        rig.sink.delivered_once(accepted)
        assert rig.proxy.request("GET", "/health/liveliness").status_code == 200


def test_langtrace_missing_api_key_logs_startup_error_and_exports_nothing(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with wire_server(_upstream) as provider, wire_server(_accepted) as sink:
        overrides: Final = {"LANGTRACE_API_HOST": sink.url, "OTEL_BSP_SCHEDULE_DELAY": "300"}
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                overrides,
                config=_config(tmp_path, callbacks=["langtrace"]),
                remove_environment=("LANGTRACE_API_KEY",),
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            assert "LANGTRACE_API_KEY not found in environment variables" in owned.log.read_text()
            rig: Final = _Rig(owned.gateway, scenario.model(api_base=provider.url + "/v1"), provider, _Sink(sink, ""))
            assert "echo " + marker in _chat_httpx(rig, marker, False)
            assert rig.provider_hits(marker) == 1
            batches: Final = eventually(
                rig.sink.collect, lambda value: len(value) >= 1, seconds=2, return_last_on_timeout=True
            )
            assert batches == (), batches


def test_langtrace_empty_api_key_still_posts_to_api_trace(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with _langtrace_rig(gateway, tmp_path, api_key="") as rig:
        assert "echo " + marker in _chat_httpx(rig, marker, False)
        batches: Final = eventually(
            rig.sink.collect, lambda value: len(_spans_carrying(value, marker)) >= 1, seconds=20
        )
        for request in batches:
            assert (request.method, request.target) == ("POST", TRACE_PATH), (request.method, request.target)
            assert "api_key" not in request.headers, request.headers
            assert request.headers.get("x-api-key", "") == "", request.headers


@pytest.mark.parametrize(
    "host",
    (lambda url: url + "/", lambda url: url + TRACE_PATH, lambda url: url + TRACE_PATH + "/"),
    ids=("trailing-slash", "already-suffixed", "suffixed-trailing-slash"),
)
def test_langtrace_api_host_variants_append_api_trace_exactly_once(
    gateway: Gateway, tmp_path: Path, host: Callable[[str], str]
) -> None:
    with _langtrace_rig(gateway, tmp_path, host=host) as rig:
        _assert_delivered(rig, _Surface(_chat_httpx, False), _marker())


def test_langtrace_logs_repeated_identical_requests_once_each(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with _langtrace_rig(gateway, tmp_path) as rig:
        body: Final = {
            "model": rig.model,
            "messages": [{"role": "user", "content": marker}],
            "cache": {"no-cache": True},
        }
        responses: Final = tuple(rig.proxy.request("POST", "/v1/chat/completions", body) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        assert rig.provider_hits(marker) == 2
        batches: Final = eventually(
            rig.sink.collect, lambda value: len(_spans_carrying(value, marker)) >= 2, seconds=20
        )
        rig.sink.assert_wire_contract(batches)
        settled: Final = eventually(
            rig.sink.collect,
            lambda value: len(_spans_carrying(value, marker)) >= 3,
            seconds=1,
            return_last_on_timeout=True,
        )
        assert len(_spans_carrying(settled, marker)) == 2


def test_generic_otel_callback_keeps_v1_traces_suffix_on_api_trace_endpoint(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with wire_server(_upstream) as provider, wire_server(_accepted) as sink:
        overrides: Final = {
            "OTEL_EXPORTER": "otlp_http",
            "OTEL_ENDPOINT": sink.url + TRACE_PATH,
            "OTEL_HEADERS": "x-api-key=generic-otel-key",
            "OTEL_BSP_SCHEDULE_DELAY": "300",
        }
        with (
            owned_proxy(gateway, tmp_path, overrides, config=_config(tmp_path, callbacks=["otel"])) as proxy,
            proxy.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1")
            response: Final = proxy.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
            )
            assert response.status_code == 200, response.text
            collector: Final = _Sink(sink, "generic-otel-key")
            batches: Final = eventually(
                collector.collect, lambda value: len(_spans_carrying(value, marker)) >= 1, seconds=20
            )
            collector.assert_wire_contract(batches, target=TRACE_PATH + "/v1/traces")


def test_langtrace_otel_v2_route_still_targets_collector_v1_traces(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = _marker()
    with wire_server(_upstream) as provider, wire_server(_accepted) as collector:
        overrides: Final = {
            "LITELLM_OTEL_V2": "true",
            "LANGTRACE_API_KEY": "unused-by-the-collector-route",
            "OTEL_EXPORTER_OTLP_ENDPOINT": collector.url,
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=collector-key",
            "OTEL_BSP_SCHEDULE_DELAY": "300",
        }
        with (
            owned_proxy(gateway, tmp_path, overrides, config=_config(tmp_path, callbacks=["langtrace"])) as proxy,
            proxy.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1")
            response: Final = proxy.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
            )
            assert response.status_code == 200, response.text
            sink: Final = _Sink(collector, "collector-key")
            batches: Final = eventually(
                sink.collect, lambda value: len(_spans_carrying(value, marker, name=None)) >= 1, seconds=20
            )
            sink.assert_wire_contract(batches, target="/v1/traces")


_BURST: Final = (
    (_chat_httpx, False),
    (_chat_httpx, True),
    (_messages_httpx, False),
    (_messages_httpx, True),
    (_responses_httpx, False),
    (_responses_httpx, True),
)


def _burst_call(rig: _Rig, index: int, marker: str) -> str:
    call, stream = _BURST[index % len(_BURST)]
    return call(rig, marker, stream)


def _burst(rig: _Rig, size: int) -> tuple[str, ...]:
    markers: Final = tuple(_marker() for _ in range(size))
    with ThreadPoolExecutor(max_workers=size) as pool:
        texts: Final = tuple(pool.map(_burst_call, repeat(rig), range(size), markers))
    for marker, text in zip(markers, texts, strict=True):
        assert "echo " + marker in text, text
    return markers


def _assert_each_once(sink: _Sink, markers: Sequence[str], seconds: float = 30) -> None:
    batches: Final = eventually(
        sink.collect, lambda value: all(_spans_carrying(value, marker) for marker in markers), seconds=seconds
    )
    sink.assert_wire_contract(batches)
    settled: Final = eventually(
        sink.collect,
        lambda value: any(len(_spans_carrying(value, marker)) > 1 for marker in markers),
        seconds=1,
        return_last_on_timeout=True,
    )
    counts: Final = {marker: len(_spans_carrying(settled, marker)) for marker in markers}
    assert all(count == 1 for count in counts.values()), counts
    for marker in markers:
        _assert_prompted_with(_spans_carrying(settled, marker)[0], marker)


def test_langtrace_two_workers_deliver_every_burst_span_exactly_once(gateway: Gateway, tmp_path: Path) -> None:
    with _langtrace_rig(gateway, tmp_path, workers=2) as rig:
        markers: Final = _burst(rig, 24)
        _assert_each_once(rig.sink, markers)


def test_langtrace_sink_outage_mid_burst_recovers_on_the_same_port(gateway: Gateway, tmp_path: Path) -> None:
    key: Final = "synthetic-langtrace-key-" + uuid.uuid4().hex
    with wire_server(_accepted) as probe:
        port: Final = int(probe.url.rsplit(":", 1)[1])
    host: Final = f"http://127.0.0.1:{port}"
    with wire_server(_upstream) as provider:
        overrides: Final = {"LANGTRACE_API_KEY": key, "LANGTRACE_API_HOST": host, "OTEL_BSP_SCHEDULE_DELAY": "300"}
        with (
            owned_proxy_process(
                gateway, tmp_path, overrides, config=_config(tmp_path, callbacks=["langtrace"]), workers=2
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1")
            with wire_server(_accepted, port=port) as sink:
                rig: Final = _Rig(owned.gateway, model, provider, _Sink(sink, key))
                _assert_each_once(rig.sink, _burst(rig, 6))
            log: Final = owned.log
            failures_before: Final = log.read_text().count("Exception while exporting Span batch")
            outage: Final = _burst(rig, 12)
            eventually(
                lambda: log.read_text().count("Exception while exporting Span batch"),
                lambda value: value > failures_before,
                seconds=20,
            )
            assert owned.gateway.request("GET", "/health/liveliness").status_code == 200
            with wire_server(_accepted, port=port) as revived:
                recovered: Final = _Rig(owned.gateway, model, provider, _Sink(revived, key))
                _assert_each_once(recovered.sink, _burst(recovered, 6))
                counts: Final = {marker: len(recovered.sink.spans_for(marker)) for marker in outage}
                assert all(count <= 1 for count in counts.values()), counts


def test_langtrace_slow_sink_does_not_delay_callers_or_duplicate_spans(gateway: Gateway, tmp_path: Path) -> None:
    def slow(request: Request) -> Reply:
        time.sleep(1)
        return _accepted(request)

    with _langtrace_rig(gateway, tmp_path, respond=slow) as rig:
        started: Final = time.monotonic()
        markers: Final = _burst(rig, 6)
        assert time.monotonic() - started < 5
        _assert_each_once(rig.sink, markers, seconds=40)


def test_langtrace_survives_a_killed_worker(gateway: Gateway, tmp_path: Path) -> None:
    key: Final = "synthetic-langtrace-key-" + uuid.uuid4().hex
    with wire_server(_upstream) as provider, wire_server(_accepted) as sink:
        overrides: Final = {"LANGTRACE_API_KEY": key, "LANGTRACE_API_HOST": sink.url, "OTEL_BSP_SCHEDULE_DELAY": "300"}
        with (
            owned_proxy_process(
                gateway, tmp_path, overrides, config=_config(tmp_path, callbacks=["langtrace"]), workers=2
            ) as owned,
            httpx.Client(
                base_url=owned.gateway.client.base_url,
                timeout=15,
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as fresh_connections,
        ):
            proxy: Final = Gateway(fresh_connections, owned.gateway.key, owned.gateway.upstream_url)
            with proxy.scenario() as scenario:
                rig: Final = _Rig(proxy, scenario.model(api_base=provider.url + "/v1"), provider, _Sink(sink, key))
                _assert_kill_and_recovery(owned, rig)


def _cmdline(process: psutil.Process) -> str:
    try:
        return " ".join(process.cmdline())
    except psutil.Error:
        return ""


def _assert_kill_and_recovery(owned: OwnedProxy, rig: _Rig) -> None:
    _assert_delivered(rig, _Surface(_chat_httpx, False), _marker())

    def uvicorn_workers() -> tuple[psutil.Process, ...]:
        return tuple(child for child in psutil.Process(owned.process.pid).children() if "spawn_main" in _cmdline(child))

    workers: Final = uvicorn_workers()
    assert len(workers) == 2, workers
    workers[0].send_signal(signal.SIGKILL)
    eventually(
        uvicorn_workers,
        lambda value: len(value) == 2 and workers[0].pid not in {child.pid for child in value},
        seconds=30,
    )
    _assert_each_once(rig.sink, _burst(rig, 6))
