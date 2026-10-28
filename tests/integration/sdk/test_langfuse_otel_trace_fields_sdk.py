import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import Reply, Request, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

CALLS: Final = {
    "sync": "completion",
    "sync_stream": "completion",
    "async": "acompletion",
    "async_stream": "acompletion",
}
SDK_CALLS: Final = """
import asyncio
import sys

import litellm

litellm.callbacks = ["langfuse_otel"]
api_base, marker = sys.argv[1], sys.argv[2]


def request(call):
    return {
        "model": "openai/gpt-4o-mini",
        "api_base": api_base,
        "api_key": "synthetic-provider-key",
        "messages": [{"role": "user", "content": f"{marker}-{call}"}],
        "stream": call.endswith("_stream"),
    }


async def async_calls():
    await litellm.acompletion(**request("async"))
    async for _ in await litellm.acompletion(**request("async_stream")):
        pass


litellm.completion(**request("sync"))
for _ in litellm.completion(**request("sync_stream")):
    pass
asyncio.run(async_calls())
print("calls done", flush=True)
sys.stdin.read()
"""


APP_TRACE_CALLS: Final = """
import asyncio
import json
import sys
import threading
import time

import litellm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

litellm.callbacks = ["langfuse_otel"]
api_base, marker, sink, writes_onto_app_span = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
app = TracerProvider()
app.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=sink + "/api/public/otel/v1/traces")))
tracer = app.get_tracer("checkout-app")


def request(call, **extra):
    return {
        "model": "openai/gpt-4o-mini",
        "api_base": api_base,
        "api_key": "synthetic-provider-key",
        "messages": [{"role": "user", "content": f"{marker}-{call}"}],
        **extra,
    }


def name_trace(root, call):
    root.set_attribute("app.call", f"{marker}-{call}")
    root.set_attribute("langfuse.trace.name", f"checkout-{marker}-{call}")
    root.set_attribute("langfuse.trace.input", json.dumps({"cart": f"{marker}-{call}"}))


def close_trace(root, call):
    root.set_attribute("langfuse.trace.output", json.dumps({"order": f"{marker}-{call}"}))
    root.end()


def litellm_wrote_onto(root):
    return not writes_onto_app_span or "langfuse.observation.type" in root.attributes


def ambient_sync():
    with tracer.start_as_current_span("checkout", end_on_exit=False) as root:
        name_trace(root, "ambient_sync")
        litellm.completion(**request("ambient_sync"))
        deadline = time.monotonic() + 30
        while not litellm_wrote_onto(root) and time.monotonic() < deadline:
            threading.Event().wait(0.05)
        close_trace(root, "ambient_sync")


async def ambient_async():
    with tracer.start_as_current_span("checkout", end_on_exit=False) as root:
        name_trace(root, "ambient_async")
        await litellm.acompletion(**request("ambient_async"))
        deadline = time.monotonic() + 30
        while not litellm_wrote_onto(root) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        close_trace(root, "ambient_async")


def caller_span_sync():
    root = tracer.start_span("checkout")
    name_trace(root, "caller_span_sync")
    litellm.completion(**request("caller_span_sync", metadata={"litellm_parent_otel_span": root}))
    close_trace(root, "caller_span_sync")


async def caller_span_async():
    root = tracer.start_span("checkout")
    name_trace(root, "caller_span_async")
    await litellm.acompletion(**request("caller_span_async", metadata={"litellm_parent_otel_span": root}))
    close_trace(root, "caller_span_async")


async def async_calls():
    await ambient_async()
    await caller_span_async()


ambient_sync()
caller_span_sync()
asyncio.run(async_calls())
print("calls done", flush=True)
sys.stdin.read()
"""
APP_CALLS: Final = ("ambient_sync", "ambient_async", "caller_span_sync", "caller_span_async")


def _exported_spans(body: bytes) -> Iterator[Span]:
    for resource in ExportTraceServiceRequest.FromString(body).resource_spans:
        for scope in resource.scope_spans:
            yield from scope.spans


def _attributes(span: Span) -> Mapping[str, str]:
    return {pair.key: pair.value.string_value for pair in span.attributes}


@dataclass(frozen=True, slots=True)
class TraceSpan:
    trace_id: str
    parent_span_id: str
    attributes: Mapping[str, str]


class TraceSink:
    def __init__(self) -> None:
        self.lock: Final = threading.Lock()
        self.spans: Final[list[TraceSpan]] = []  # mutable-ok: spans accumulate as both exporters deliver

    def respond(self, request: Request) -> Reply:
        if not request.target.endswith("/v1/traces"):
            return Reply(body=b"")
        spans: Final = tuple(
            TraceSpan(span.trace_id.hex(), span.parent_span_id.hex(), _attributes(span))
            for span in _exported_spans(request.body)
        )
        with self.lock:
            self.spans.extend(spans)
        return Reply(body=b"")

    def traces(self, marker: str, calls: tuple[str, ...]) -> Mapping[str, tuple[TraceSpan, ...]]:
        with self.lock:
            snapshot: Final = tuple(self.spans)
        roots: Final = {
            call: next(
                (
                    span
                    for span in snapshot
                    if not span.parent_span_id and span.attributes.get("app.call") == f"{marker}-{call}"
                ),
                None,
            )
            for call in calls
        }
        return {
            call: tuple(span for span in snapshot if span.trace_id == root.trace_id)
            for call, root in roots.items()
            if root is not None
        }


def _trace_values(spans: tuple[TraceSpan, ...], keys: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    for span in spans:
        yield from ((key, span.attributes[key]) for key in keys if key in span.attributes)


def _logged(spans: tuple[TraceSpan, ...]) -> bool:
    return any(span.attributes.get("langfuse.observation.type") == "generation" for span in spans)


def _chat_reply(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    content: Final = body["messages"][0]["content"]
    if body.get("stream"):
        delta: Final = {"id": content, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
        return Reply(
            content_type="text/event-stream",
            chunks=(
                f"data: {json.dumps({**delta, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'ok'}}]})}\n\n".encode(),
                f"data: {json.dumps({**delta, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
                b"data: [DONE]\n\n",
            ),
        )
    return Reply(
        body=json.dumps(
            {
                "id": content,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ).encode()
    )


class GenerationSink:
    def __init__(self) -> None:
        self.lock: Final = threading.Lock()
        self.generations: Final[list[Mapping[str, str]]] = []  # mutable-ok: spans accumulate as the exporter delivers

    def respond(self, request: Request) -> Reply:
        if not request.target.endswith("/v1/traces"):
            return Reply(body=b"")
        spans: Final = tuple(_attributes(span) for span in _exported_spans(request.body))
        with self.lock:
            self.generations.extend(span for span in spans if span.get("langfuse.observation.type") == "generation")
        return Reply(body=b"")

    def by_call(self, marker: str) -> Mapping[str, tuple[Mapping[str, str], ...]]:
        with self.lock:
            snapshot: Final = tuple(self.generations)
        return {
            call: tuple(span for span in snapshot if f'"{marker}-{call}"' in span.get("langfuse.observation.input", ""))
            for call in CALLS
        }


def test_sdk_sync_and_async_calls_derive_trace_name_and_io_from_their_call_type() -> None:
    marker: Final = "lit8281-sdk-" + uuid.uuid4().hex
    sink: Final = GenerationSink()
    with wire_server(sink.respond) as destination, wire_server(_chat_reply) as provider:
        child: Final = subprocess.Popen(
            [sys.executable, "-P", "-c", SDK_CALLS, provider.url + "/v1", marker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "LANGFUSE_PUBLIC_KEY": "pk-lf-sdk",
                "LANGFUSE_SECRET_KEY": "sk-lf-sdk",
                "LANGFUSE_HOST": destination.url,
                "OTEL_BSP_SCHEDULE_DELAY": "200",
            },
        )
        try:
            spans: Final = eventually(
                lambda: sink.by_call(marker),
                lambda found: all(len(records) == 1 for records in found.values()) or child.poll() is not None,
                seconds=70,
            )
        finally:
            output: Final = child.communicate(input=b"", timeout=60)[0].decode()
    assert {call: len(records) for call, records in spans.items()} == dict.fromkeys(CALLS, 1), output
    for call, call_type in CALLS.items():
        attributes = spans[call][0]
        assert attributes["langfuse.trace.name"] == f"litellm-{call_type}", (call, attributes)
        assert attributes["langfuse.trace.input"] == attributes["langfuse.observation.input"], (call, attributes)
        assert attributes["langfuse.trace.output"] == attributes["langfuse.observation.output"], (call, attributes)
        assert json.loads(attributes["langfuse.trace.output"]) == {"role": "assistant", "content": "ok"}, call


@pytest.mark.parametrize(
    ("environment", "writes_onto_app_span", "calls"),
    [
        pytest.param({}, True, APP_CALLS, id="litellm_writes_onto_the_app_span"),
        pytest.param({"USE_OTEL_LITELLM_REQUEST_SPAN": "true"}, False, APP_CALLS, id="litellm_request_child_span"),
        pytest.param(
            {"LITELLM_OTEL_V2": "1", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only"},
            False,
            ("ambient_async",),
            id="otel_v2_llm_call_child_span",
        ),
    ],
)
def test_sdk_call_inside_an_app_trace_leaves_the_app_trace_name_and_io_alone(
    environment: Mapping[str, str], writes_onto_app_span: bool, calls: tuple[str, ...]
) -> None:
    marker: Final = "lit8281-app-" + uuid.uuid4().hex
    sink: Final = TraceSink()
    with wire_server(sink.respond) as destination, wire_server(_chat_reply) as provider:
        child: Final = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-c",
                APP_TRACE_CALLS,
                provider.url + "/v1",
                marker,
                destination.url,
                "1" if writes_onto_app_span else "0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "LANGFUSE_PUBLIC_KEY": "pk-lf-sdk",
                "LANGFUSE_SECRET_KEY": "sk-lf-sdk",
                "LANGFUSE_HOST": destination.url,
                "OTEL_BSP_SCHEDULE_DELAY": "200",
                **environment,
            },
        )
        try:
            traces: Final = eventually(
                lambda: sink.traces(marker, calls),
                lambda found: (
                    (len(found) == len(calls) and all(map(_logged, found.values()))) or child.poll() is not None
                ),
                seconds=70,
            )
        finally:
            output: Final = child.communicate(input=b"", timeout=60)[0].decode()
    assert sorted(call for call, spans in traces.items() if _logged(spans)) == sorted(calls), output
    for call, spans in traces.items():
        _assert_the_app_trace_fields_stand(marker, call, spans)


def _assert_the_app_trace_fields_stand(marker: str, call: str, spans: tuple[TraceSpan, ...]) -> None:
    app: Final = {
        "langfuse.trace.name": f"checkout-{marker}-{call}",
        "langfuse.trace.input": json.dumps({"cart": f"{marker}-{call}"}),
        "langfuse.trace.output": json.dumps({"order": f"{marker}-{call}"}),
    }
    root: Final = next(span for span in spans if not span.parent_span_id)
    assert {key: root.attributes.get(key) for key in app} == app, (call, root.attributes)
    written: Final = set(_trace_values(spans, (*app, "langfuse.trace.tags")))
    assert written == set(app.items()), (call, written)
