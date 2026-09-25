import asyncio
import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import group_members, signal_group, stop_root_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

MARKER: Final = re.compile(rb"traceaudit-[0-9a-f]{32}")
TRACE_ID: Final = re.compile(r"[0-9a-f]{32}")
SERVER_KINDS: Final = frozenset({"2", "span_kind_server", "server"})
TRACE_HEADER: Final = "x-litellm-trace-id"
CALL_ID_HEADER: Final = "x-litellm-call-id"
CALL_ID_ATTRIBUTE: Final = "litellm.call_id"


def _marker() -> str:
    return "traceaudit-" + uuid.uuid4().hex


def _chat_reply(identity: str, stream: bool) -> Reply:
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
                            "message": {"role": "assistant", "content": "trace header ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            b"data: "
            + json.dumps(
                {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "trace"}}]}
            ).encode()
            + b"\n\n",
            b"data: "
            + json.dumps(
                {
                    **chunk,
                    "choices": [{"index": 0, "delta": {"content": " header ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
            + b"\n\n",
            b"data: [DONE]\n\n",
        ),
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
                "content": [{"type": "output_text", "text": "trace header ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": "trace header ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _decoded_responses_id(identity: str) -> str:
    try:
        return base64.b64decode(identity.removeprefix("resp_").encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return identity


def _canonical_id(identity: str) -> str:
    return _decoded_responses_id(identity).rpartition("response_id:")[2]


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    stream: Final = json.loads(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"
    )


@dataclass(frozen=True, slots=True)
class OtlpSpan:
    trace_id: str
    span_id: str
    parent_span_id: str
    kind: str
    name: str
    attributes: Mapping[str, JsonValue]


def _attribute_value(attribute: Mapping[str, JsonValue]) -> JsonValue:
    value: Final = attribute.get("value")
    if isinstance(value, Mapping):
        return next(iter(value.values()), None)
    return value


@dataclass(frozen=True, slots=True)
class Collector:
    wire: Wire
    outage: threading.Event
    accepted: Sequence[Request]
    guard: threading.Lock

    def spans(self) -> tuple[OtlpSpan, ...]:
        with self.guard:
            batches: Final = tuple(self.accepted)
        return tuple(
            OtlpSpan(
                trace_id=str(span.get("traceId", "")),
                span_id=str(span.get("spanId", "")),
                parent_span_id=str(span.get("parentSpanId", "")),
                kind=str(span.get("kind", "")),
                name=str(span.get("name", "")),
                attributes={
                    str(attribute["key"]): _attribute_value(attribute) for attribute in span.get("attributes", ())
                },
            )
            for batch in batches
            for resource in json.loads(batch.body)["resourceSpans"]
            for scope in resource["scopeSpans"]
            for span in scope["spans"]
        )

    def server_spans(self, trace_id: str) -> tuple[OtlpSpan, ...]:
        return tuple(span for span in self.spans() if span.trace_id == trace_id and span.kind.lower() in SERVER_KINDS)

    def llm_spans(self, trace_id: str) -> tuple[OtlpSpan, ...]:
        return tuple(
            span
            for span in self.spans()
            if span.trace_id == trace_id and any(key.startswith("gen_ai.") for key in span.attributes)
        )

    def llm_span_ids(self) -> frozenset[str]:
        return frozenset(
            _canonical_id(str(span.attributes["gen_ai.response.id"]))
            for span in self.spans()
            if isinstance(span.attributes.get("gen_ai.response.id"), str)
        )

    def call_ids(self) -> tuple[str, ...]:
        return tuple(
            str(span.attributes[CALL_ID_ATTRIBUTE])
            for span in self.spans()
            if span.kind.lower() in SERVER_KINDS and CALL_ID_ATTRIBUTE in span.attributes
        )


@contextmanager
def _collector() -> Iterator[Collector]:
    outage: Final = threading.Event()
    accepted: Final[deque[Request]] = deque()  # mutable-ok: sink thread appends each accepted batch
    guard: Final = threading.Lock()

    def sink(request: Request) -> Reply:
        if outage.is_set():
            return Reply(status=503, body=b'{"error":"sink down"}')
        with guard:
            accepted.append(request)
        return Reply()

    with wire_server(sink) as wire:
        yield Collector(wire, outage, accepted, guard)


@pytest.fixture(scope="session")
def collector() -> Iterator[Collector]:
    with _collector() as sink:
        yield sink


@pytest.fixture(scope="session")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@dataclass(frozen=True, slots=True)
class OwnedProxy:
    gateway: Gateway
    process: subprocess.Popen[bytes]
    log: Path


@contextmanager
def _owned_proxy_process(
    key: str,
    upstream_url: str,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    config: Path,
    remove_environment: tuple[str, ...] = (),
    workers: int = 1,
) -> Iterator[OwnedProxy]:
    """Same launch contract as _support.process.owned_proxy_process, plus a workers count."""
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    root: Final = Path(__file__).resolve().parents[3]
    pythonpath: Final = os.pathsep.join(
        entry
        for entry in (str(root), str(root / "tests"), str(root / "tests/e2e"), os.environ.get("PYTHONPATH", ""))
        if entry
    )
    environment: Final = {
        **{name: value for name, value in os.environ.items() if name not in remove_environment},
        "LITELLM_MASTER_KEY": key,
        "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
        "STORE_MODEL_IN_DB": "True",
        "PYTHONPATH": pythonpath,
        **overrides,
    }
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    log_path: Final = output / f"owned-proxy-{uuid.uuid4().hex}.log"
    with log_path.open("w") as log:
        process: Final = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "integration._support.proxy",
                "--config",
                str(config),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--num_workers",
                str(workers),
                "--telemetry",
                "False",
                "--use_prisma_db_push",
                "--enforce_prisma_migration_check",
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
                deadline: Final = time.monotonic() + 120
                while True:
                    assert process.poll() is None, "Owned proxy exited before readiness"
                    try:
                        if client.get("/health/readiness", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "Owned proxy readiness deadline exceeded"
                    time.sleep(0.1)
                yield OwnedProxy(Gateway(client, key, upstream_url), process, log_path)
        finally:
            root_stopped: Final = stop_root_process(process)
            residual: Final = group_members(process.pid)
            if residual:
                signal_group(process.pid, signal.SIGTERM)
                psutil.wait_procs(residual, timeout=5)
            remaining: Final = group_members(process.pid)
            if remaining:
                signal_group(process.pid, signal.SIGKILL)
                psutil.wait_procs(remaining, timeout=3)
            process.wait(timeout=3)
            survivors: Final = group_members(process.pid)
            assert not survivors, "Owned proxy child survived cleanup"
            assert root_stopped and not remaining, "Owned proxy required forced cleanup"


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    process: OwnedProxy
    model: str
    upstream: Wire
    sink: Collector

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0)

    def async_openai_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=str(self.proxy.client.base_url) + "/v1", api_key=self.proxy.key, max_retries=0
        )

    def anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def async_anthropic_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(base_url=str(self.proxy.client.base_url), api_key=self.proxy.key, max_retries=0)

    def post(
        self,
        path: str,
        body: Mapping[str, JsonValue],
        *,
        key: str | None = None,
        headers: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    ) -> httpx.Response:
        sent: Final = [("Authorization", f"Bearer {self.proxy.key if key is None else key}")]
        if headers is not None:
            sent.extend(headers.items() if isinstance(headers, Mapping) else headers)
        return self.proxy.client.post(path, json=body, headers=sent)

    def chat(
        self,
        marker: str,
        *,
        key: str | None = None,
        headers: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        model: str | None = None,
        **extra: JsonValue,
    ) -> httpx.Response:
        return self.post(
            "/v1/chat/completions",
            {
                "model": model or self.model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
                **extra,
            },
            key=key,
            headers=headers,
        )

    def upstream_bodies(self, marker: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(json.loads(request.body) for request in self.upstream.drain() if marker.encode() in request.body)

    def spend_session(self, response_id: str) -> str | None:
        rows: Final = eventually(
            lambda: read_rows('SELECT session_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        value: Final = rows[0]["session_id"]
        assert value is None or isinstance(value, str), rows
        return value


@dataclass(frozen=True, slots=True)
class RigFactory:
    provider: Wire
    sink: Collector
    directory: Path
    overrides: Mapping[str, str]
    remove_environment: tuple[str, ...]
    workers: int

    def start(self) -> Iterator[Rig]:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update({"callbacks": ["otel"]})
        config["general_settings"].update({"disable_model_info_refresh": True})
        config["callback_settings"] = {
            "otel": {"exporter": "http/json", "endpoint": self.sink.wire.url, "mapper_names": ["genai"]},
        }
        path: Final = self.directory / f"otel-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(config))
        key: Final = "sk-" + uuid.uuid4().hex
        overrides: Final = {
            "REDIS_HOST": os.environ["REDIS_HOST"],
            "REDIS_PORT": os.environ["REDIS_PORT"],
            **self.overrides,
        }
        with (
            _owned_proxy_process(
                key,
                self.provider.url,
                self.directory,
                overrides,
                config=path,
                remove_environment=self.remove_environment,
                workers=self.workers,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=self.provider.url + "/v1")
            yield Rig(owned.gateway, owned, model, self.provider, self.sink)


@pytest.fixture(scope="session")
def rig(provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    overrides: Final = {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300"}
    yield from RigFactory(provider, collector, tmp_path_factory.mktemp("otel-trace"), overrides, (), 2).start()


@pytest.fixture(scope="session")
def otel_off_rig(provider: Wire, collector: Collector, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    overrides: Final = {"OTEL_BSP_SCHEDULE_DELAY": "300"}
    yield from RigFactory(
        provider, collector, tmp_path_factory.mktemp("otel-off"), overrides, ("LITELLM_OTEL_V2",), 1
    ).start()


@pytest.fixture(scope="session")
def sampler_collector() -> Iterator[Collector]:
    with _collector() as sink:
        yield sink


@pytest.fixture(scope="session")
def sampler_off_rig(
    provider: Wire, sampler_collector: Collector, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Rig]:
    overrides: Final = {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300", "OTEL_TRACES_SAMPLER": "always_off"}
    yield from RigFactory(
        provider, sampler_collector, tmp_path_factory.mktemp("otel-sampler"), overrides, (), 1
    ).start()


def assert_trace_header(response: httpx.Response) -> str:
    value: Final = response.headers.get(TRACE_HEADER)
    assert value is not None, f"missing {TRACE_HEADER} on {response.status_code}: {dict(response.headers)}"
    assert TRACE_ID.fullmatch(value), f"malformed {TRACE_HEADER} {value!r}: {dict(response.headers)}"
    return value


def _assert_header_absent(response: httpx.Response) -> None:
    assert TRACE_HEADER not in response.headers, (
        f"unexpected {TRACE_HEADER}={response.headers.get(TRACE_HEADER)!r} on {response.status_code}: "
        f"{dict(response.headers)}"
    )


def _server_span(sink: Collector, trace_id: str, call_id: str, seconds: float = 40) -> OtlpSpan:
    found: Final = eventually(
        lambda: sink.server_spans(trace_id),
        lambda spans: any(span.attributes.get(CALL_ID_ATTRIBUTE) == call_id for span in spans),
        seconds=seconds,
    )
    matching: Final = tuple(span for span in found if span.attributes.get(CALL_ID_ATTRIBUTE) == call_id)
    assert matching, found
    return matching[0]


def _assert_llm_span(sink: Collector, trace_id: str, seconds: float = 40) -> None:
    assert eventually(lambda: sink.llm_spans(trace_id), lambda spans: len(spans) >= 1, seconds=seconds)


@pytest.mark.covers("other.observability.otel.trace_header.chat_completions_nonstream_sdk")
def test_h1_chat_completions_nonstream_openai_sdk(rig: Rig) -> None:
    marker: Final = _marker()
    raw: Final = rig.openai_client().chat.completions.with_raw_response.create(
        model=rig.model, messages=[{"role": "user", "content": marker}], extra_body={"cache": {"no-cache": True}}
    )
    completion: Final = raw.parse()
    assert completion.id == f"chatcmpl-{marker}", completion
    assert completion.model and completion.choices[0].message.content == "trace header ok", completion
    trace_id: Final = assert_trace_header(raw.http_response)
    call_id: Final = raw.http_response.headers[CALL_ID_HEADER]
    assert _server_span(rig.sink, trace_id, call_id).trace_id == trace_id
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.chat_completions_stream_async_sdk")
def test_h2_chat_completions_stream_openai_async_sdk(rig: Rig) -> None:
    marker: Final = _marker()

    async def consume() -> httpx.Response:
        raw: Final = await rig.async_openai_client().chat.completions.with_raw_response.create(
            model=rig.model,
            messages=[{"role": "user", "content": marker}],
            stream=True,
            extra_body={"cache": {"no-cache": True}},
        )
        chunks: Final = [chunk async for chunk in raw.parse()]
        assert chunks[0].id == f"chatcmpl-{marker}", chunks
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "trace header ok"
        return raw.http_response

    response: Final = asyncio.run(consume())
    trace_id: Final = assert_trace_header(response)
    call_id: Final = response.headers[CALL_ID_HEADER]
    _server_span(rig.sink, trace_id, call_id)
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.messages_nonstream_anthropic_sdk")
def test_h3_messages_nonstream_anthropic_sdk(rig: Rig) -> None:
    marker: Final = _marker()
    raw: Final = rig.anthropic_client().messages.with_raw_response.create(
        model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    )
    message: Final = raw.parse()
    assert message.type == "message" and message.id, message
    assert message.content[0].type == "text" and message.content[0].text == "trace header ok", message
    trace_id: Final = assert_trace_header(raw.http_response)
    call_id: Final = raw.http_response.headers[CALL_ID_HEADER]
    _server_span(rig.sink, trace_id, call_id)
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.messages_stream_anthropic_async_sdk")
def test_h4_messages_stream_anthropic_async_sdk(rig: Rig) -> None:
    marker: Final = _marker()

    async def consume() -> httpx.Response:
        raw: Final = await rig.async_anthropic_client().messages.with_raw_response.create(
            model=rig.model, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
        )
        text: Final = "".join(
            [
                event.delta.text
                async for event in raw.parse()
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            ]
        )
        assert text == "trace header ok", text
        return raw.http_response

    response: Final = asyncio.run(consume())
    trace_id: Final = assert_trace_header(response)
    call_id: Final = response.headers[CALL_ID_HEADER]
    _server_span(rig.sink, trace_id, call_id)
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.responses_nonstream_sdk")
def test_h5_responses_nonstream_openai_sdk(rig: Rig) -> None:
    marker: Final = _marker()
    raw: Final = rig.openai_client().responses.with_raw_response.create(model=rig.model, input=marker)
    response: Final = raw.parse()
    assert response.output_text == "trace header ok", response
    assert response.output[0].id == f"msg_resp_{marker}", response
    trace_id: Final = assert_trace_header(raw.http_response)
    call_id: Final = raw.http_response.headers[CALL_ID_HEADER]
    _server_span(rig.sink, trace_id, call_id)
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.responses_stream_raw_sse")
def test_h6_responses_stream_raw_sse(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.client.stream(
        "POST",
        "/v1/responses",
        json={"model": rig.model, "input": marker, "stream": True},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    ) as response:
        body: Final = response.read().decode()
        assert response.status_code == 200, body
        trace_id: Final = assert_trace_header(response)
        call_id: Final = response.headers[CALL_ID_HEADER]
    completed: Final = tuple(event for event in _sse_events(body) if event.get("type") == "response.completed")
    assert len(completed) == 1, body
    assert str(completed[0]["response"]["id"]).startswith("resp_"), completed
    _server_span(rig.sink, trace_id, call_id)
    _assert_llm_span(rig.sink, trace_id)
    assert len(rig.upstream_bodies(marker)) == 1


@pytest.mark.covers("other.observability.otel.trace_header.error_400_unknown_model")
def test_h7_unknown_model_400_response_still_carries_trace_header(rig: Rig) -> None:
    response: Final = rig.chat(_marker(), model="does-not-exist-" + uuid.uuid4().hex)
    assert response.status_code == 400, response.text
    trace_id: Final = assert_trace_header(response)
    assert eventually(lambda: rig.sink.server_spans(trace_id), lambda spans: len(spans) >= 1, seconds=40)


@pytest.mark.covers("other.observability.otel.trace_header.error_401_bad_key")
def test_h8_bad_key_401_response_still_carries_trace_header(rig: Rig) -> None:
    response: Final = rig.chat(_marker(), key="sk-wrong")
    assert response.status_code == 401, response.text
    trace_id: Final = assert_trace_header(response)
    assert eventually(lambda: rig.sink.server_spans(trace_id), lambda spans: len(spans) >= 1, seconds=40)


@pytest.mark.covers("other.observability.otel.trace_header.traceparent_propagation")
def test_h9_traceparent_trace_id_is_returned_and_parents_the_server_span(rig: Rig) -> None:
    trace: Final = uuid.uuid4().hex
    parent: Final = uuid.uuid4().hex[:16]
    response: Final = rig.chat(_marker(), headers={"traceparent": f"00-{trace}-{parent}-01"})
    assert response.status_code == 200, response.text
    assert assert_trace_header(response) == trace
    server: Final = eventually(lambda: rig.sink.server_spans(trace), lambda spans: len(spans) == 1, seconds=40)[0]
    assert server.parent_span_id == parent, server


@pytest.mark.covers("other.observability.otel.trace_header.concurrent_burst_unique_traces")
def test_h10_twenty_four_concurrent_mixed_requests_get_distinct_matching_traces(rig: Rig) -> None:
    markers: Final = tuple(_marker() for _ in range(24))

    async def one(client: httpx.AsyncClient, index: int) -> tuple[int, httpx.Response]:
        marker: Final = markers[index]
        body: Final = {"model": rig.model, "stream": index % 2 == 0}
        route: Final = index % 3
        if route == 0:
            payload: Final = {**body, "messages": [{"role": "user", "content": marker}]}
            response: Final = await client.post("/v1/chat/completions", json=payload)
        elif route == 1:
            response = await client.post("/v1/responses", json={**body, "input": marker})
        else:
            response = await client.post(
                "/v1/messages", json={**body, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]}
            )
        await response.aread()
        return index, response

    async def burst() -> tuple[tuple[int, httpx.Response], ...]:
        async with httpx.AsyncClient(
            base_url=str(rig.proxy.client.base_url),
            headers={"Authorization": f"Bearer {rig.proxy.key}"},
            timeout=30,
            trust_env=False,
        ) as client:
            return await asyncio.gather(*(one(client, index) for index in range(24)))

    results: Final = asyncio.run(burst())
    assert all(response.status_code == 200 for _, response in results), [
        response.text for _, response in results if response.status_code != 200
    ]
    pairs: Final = tuple((assert_trace_header(response), response.headers[CALL_ID_HEADER]) for _, response in results)
    assert len({trace for trace, _ in pairs}) == 24, pairs
    for trace_id, call_id in pairs:
        found: Final = eventually(
            lambda trace_id=trace_id: rig.sink.server_spans(trace_id),  # noqa: B023  # bound per iteration
            lambda spans: len(spans) >= 1,
            seconds=40,
        )
        assert len(found) == 1, found
        assert found[0].attributes.get(CALL_ID_ATTRIBUTE) == call_id, found


@pytest.mark.covers("other.observability.otel.trace_header.excluded_route_no_header")
def test_e1_health_liveliness_never_carries_trace_header(rig: Rig) -> None:
    response: Final = rig.proxy.client.get("/health/liveliness")
    assert response.status_code == 200, response.text
    _assert_header_absent(response)


@pytest.mark.covers("other.observability.otel.trace_header.otel_v2_off_no_header")
def test_e2_otel_v2_disabled_serves_call_id_but_no_trace_header(otel_off_rig: Rig) -> None:
    response: Final = otel_off_rig.chat(_marker())
    assert response.status_code == 200, response.text
    _assert_header_absent(response)
    assert response.headers.get(CALL_ID_HEADER), dict(response.headers)


@pytest.mark.covers("other.observability.otel.trace_header.always_off_sampler_no_header_no_spans")
def test_e3_always_off_sampler_suppresses_header_and_exports(sampler_off_rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = sampler_off_rig.chat(marker)
    assert response.status_code == 200, response.text
    _assert_header_absent(response)
    assert len(sampler_off_rig.upstream_bodies(marker)) == 1
    sampler_off_rig.spend_session(str(response.json()["id"]))
    assert not sampler_off_rig.sink.wire.drain(), "OTLP export arrived under always_off sampler"


@pytest.mark.covers("other.observability.otel.trace_header.request_header_stays_session_id")
def test_s1_x_litellm_trace_id_request_header_still_lands_as_spend_session(rig: Rig) -> None:
    marker: Final = _marker()
    session: Final = "sess-" + uuid.uuid4().hex
    response: Final = rig.chat(marker, headers={TRACE_HEADER: session})
    assert response.status_code == 200, response.text
    identity: Final = response.json()["id"]
    assert identity == f"chatcmpl-{marker}", response.text
    returned: Final = assert_trace_header(response)
    assert returned != session, "response header echoed the caller session id"
    assert rig.spend_session(identity) == session
    call_id: Final = response.headers[CALL_ID_HEADER]
    assert _server_span(rig.sink, returned, call_id).trace_id == returned


def _s2_chat(
    rig: Rig, marker: str, headers: Mapping[str, str] | Sequence[tuple[str, str]] | None, *, key: str | None = None
) -> tuple[httpx.Response, str]:
    response: Final = rig.chat(marker, headers=headers, key=key)
    return response, assert_trace_header(response)


@pytest.mark.covers("other.observability.otel.trace_header.hostile_header_five_kb")
def test_s2_five_kilobyte_trace_header_value(rig: Rig) -> None:
    marker: Final = _marker()
    sent: Final = ("t" * 5000) + uuid.uuid4().hex
    response, _trace_id = _s2_chat(rig, marker, {TRACE_HEADER: sent})
    assert response.status_code == 200, response.text
    assert rig.spend_session(response.json()["id"]) == sent


@pytest.mark.covers("other.observability.otel.trace_header.hostile_header_numeric")
def test_s2_numeric_trace_header_value(rig: Rig) -> None:
    response, _trace_id = _s2_chat(rig, _marker(), {TRACE_HEADER: "123"})
    assert response.status_code == 200, response.text
    assert rig.spend_session(response.json()["id"]) == "123"


@pytest.mark.covers("other.observability.otel.trace_header.hostile_header_duplicate")
def test_s2_duplicate_trace_header_values(rig: Rig) -> None:
    first: Final = "dup-first-" + uuid.uuid4().hex
    second: Final = "dup-second-" + uuid.uuid4().hex
    response, _trace_id = _s2_chat(rig, _marker(), [(TRACE_HEADER, first), (TRACE_HEADER, second)])
    assert response.status_code == 200, response.text
    assert rig.spend_session(response.json()["id"]) == second


@pytest.mark.covers("other.observability.otel.trace_header.hostile_header_empty")
def test_s2_empty_trace_header_value(rig: Rig) -> None:
    response, _trace_id = _s2_chat(rig, _marker(), {TRACE_HEADER: ""})
    assert response.status_code == 200, response.text
    observed: Final = rig.spend_session(response.json()["id"])
    assert observed and observed != "", observed


@pytest.mark.covers("other.observability.otel.trace_header.hostile_header_unauthenticated")
def test_s2_unauthenticated_trace_header(rig: Rig) -> None:
    marker: Final = _marker()
    response, _trace_id = _s2_chat(rig, marker, {TRACE_HEADER: "sess-" + uuid.uuid4().hex}, key="sk-wrong")
    assert response.status_code == 401, response.text
    assert rig.upstream_bodies(marker) == ()


@pytest.mark.covers("other.observability.otel.trace_header.cors_expose_headers")
def test_s3_cors_expose_headers_lists_trace_id(rig: Rig) -> None:
    liveliness: Final = rig.proxy.client.get("/health/liveliness", headers={"Origin": "http://localhost:4000"})
    assert liveliness.status_code == 200, liveliness.text
    exposed: Final = {
        value.strip().lower()
        for value in liveliness.headers.get("access-control-expose-headers", "").split(",")
        if value.strip()
    }
    assert TRACE_HEADER in exposed, dict(liveliness.headers)
    response: Final = rig.chat(_marker(), headers={"Origin": "http://localhost:4000"})
    assert response.status_code == 200, response.text
    assert TRACE_HEADER in {
        value.strip().lower()
        for value in response.headers.get("access-control-expose-headers", "").split(",")
        if value.strip()
    }, dict(response.headers)


def _chat_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return str(response.json()["id"])
    identities: Final = frozenset(str(event["id"]) for event in _sse_events(response.text))
    assert len(identities) == 1, response.text
    return next(iter(identities))


def _responses_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return str(response.json()["id"])
    completed: Final = tuple(
        event["response"]["id"] for event in _sse_events(response.text) if event.get("type") == "response.completed"
    )
    assert len(completed) == 1, response.text
    return str(completed[0])


def _message_id(response: httpx.Response) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return str(response.json()["id"])
    starts: Final = tuple(
        event["message"]["id"] for event in _sse_events(response.text) if event.get("type") == "message_start"
    )
    assert len(starts) == 1, response.text
    return starts[0]


def _burst(rig: Rig, count: int) -> tuple[tuple[int, httpx.Response | None, str | None], ...]:
    markers: Final = tuple(_marker() for _ in range(count))

    def one(index: int) -> tuple[int, httpx.Response | None, str | None]:
        marker: Final = markers[index]
        route: Final = index % 3
        stream: Final = index % 2 == 0
        try:
            if route == 0:
                response: Final = rig.post(
                    "/v1/chat/completions",
                    {"model": rig.model, "messages": [{"role": "user", "content": marker}], "stream": stream},
                )
            elif route == 1:
                response = rig.post("/v1/responses", {"model": rig.model, "input": marker, "stream": stream})
            else:
                response = rig.post(
                    "/v1/messages",
                    {
                        "model": rig.model,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": marker}],
                        "stream": stream,
                    },
                )
            response.read()
            return index, response, None
        except httpx.HTTPError as error:
            return index, None, repr(error)

    with ThreadPoolExecutor(max_workers=10) as pool:
        return tuple(pool.map(one, range(count)))


def _landed_trace_ids(sink: Collector) -> tuple[str, ...]:
    return tuple(span.trace_id for span in sink.spans() if span.kind.lower() in SERVER_KINDS)


@pytest.mark.covers("other.observability.otel.trace_header.sink_outage_burst")
def test_c1_sink_outage_during_burst_never_duplicates_a_server_span(rig: Rig) -> None:
    rig.sink.outage.set()
    try:
        results: Final = _burst(rig, 30)
        assert all(error is None for _, _, error in results), [error for _, _, error in results if error]
        responses: Final = tuple(response for _, response, _ in results if response is not None)
        assert all(response.status_code == 200 for response in responses), [
            response.text for response in responses if response.status_code != 200
        ]
        eventually(lambda: any(True for _ in rig.sink.wire.drain()), lambda seen: seen, seconds=30)
    finally:
        rig.sink.outage.clear()
    probe: Final = rig.chat(_marker())
    assert probe.status_code == 200, probe.text
    probe_id: Final = str(probe.json()["id"])
    assert eventually(lambda: rig.sink.llm_span_ids(), lambda ids: probe_id in ids, seconds=60), (
        "exporter never recovered after the outage cleared"
    )
    identities: Final = tuple(
        _canonical_id(
            _chat_id(response)
            if index % 3 == 0
            else _responses_id(response)
            if index % 3 == 1
            else _message_id(response)
        )
        for index, response, _ in results
        if response is not None
    )
    landed_ids: Final = rig.sink.llm_span_ids()
    landed_traces: Final = _landed_trace_ids(rig.sink)
    landed_calls: Final = rig.sink.call_ids()
    call_ids: Final = tuple(str(response.headers.get(CALL_ID_HEADER, "")) for response in responses)
    trace_ids: Final = tuple(
        header for header in (response.headers.get(TRACE_HEADER) for response in responses) if header is not None
    )
    record: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", ".")) / "c1-landed.txt"
    record.write_text(
        json.dumps(
            {
                "sent": len(responses),
                "landed_by_call_id": sum(1 for call_id in call_ids if call_id in landed_calls),
                "landed_by_trace_id": sum(1 for trace in trace_ids if trace in landed_traces),
                "landed_by_response_id": sum(1 for identity in identities if identity in landed_ids),
                "duplicates_by_call_id": sum(
                    1 for call_id in set(call_ids) if call_id and landed_calls.count(call_id) > 1
                ),
            }
        )
        + "\n"
    )
    assert not [call_id for call_id in set(call_ids) if call_id and landed_calls.count(call_id) > 1], call_ids
    for response in responses:
        assert_trace_header(response)


@pytest.mark.covers("other.observability.otel.trace_header.worker_kill_burst")
def test_c2_killing_one_worker_mid_burst_keeps_serving_with_headers(rig: Rig) -> None:
    root: Final = psutil.Process(rig.process.process.pid)
    workers: Final = eventually(
        lambda: tuple(child for child in root.children() if "resource_tracker" not in " ".join(child.cmdline())),
        lambda found: len(found) == 2,
        seconds=30,
    )
    markers: Final = tuple(_marker() for _ in range(24))

    def one(index: int) -> tuple[int, httpx.Response | None, str | None]:
        if index == 8:
            os.kill(workers[0].pid, signal.SIGKILL)
        try:
            response: Final = rig.chat(markers[index])
            return index, response, None
        except httpx.HTTPError as error:
            return index, None, repr(error)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = tuple(pool.map(one, range(24)))
    assert rig.process.process.poll() is None, "Proxy root exited after a worker was killed"
    after: Final = rig.chat(_marker())
    assert after.status_code == 200, after.text
    assert_trace_header(after)
    failures: Final = tuple(error for _, _, error in results if error)
    assert all(error.startswith(("ReadError(", "RemoteProtocolError(", "ConnectError(")) for error in failures), (
        failures
    )
    served: Final = tuple(response for _, response, error in results if response is not None and not error)
    assert len(served) >= 18, results
    assert all(response.status_code == 200 for response in served), [response.text for response in served]
    for response in served:
        assert_trace_header(response)
