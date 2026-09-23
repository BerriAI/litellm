import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

REPLY_TEXT: Final = "langfuse-otel scripted reply"


@dataclass(frozen=True, slots=True)
class SpanRecord:
    trace_id: str
    parent_span_id: str
    attributes: dict[str, object]


def _attribute_value(value: object) -> object:
    kind: Final = value.WhichOneof("value")
    return getattr(value, kind) if kind is not None else None


def _span_records(body: bytes) -> tuple[SpanRecord, ...]:
    export: Final = ExportTraceServiceRequest.FromString(body)
    return tuple(
        SpanRecord(
            trace_id=span.trace_id.hex(),
            parent_span_id=span.parent_span_id.hex(),
            attributes={pair.key: _attribute_value(pair.value) for pair in span.attributes},
        )
        for resource in export.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


def _otlp_sink(request: Request) -> Reply:
    if request.target.endswith("/v1/traces"):
        assert request.headers.get("x-langfuse-ingestion-version") == "4", request.headers
    return Reply(body=b"")


def _chat_completion(marker: str) -> bytes:
    return json.dumps(
        {
            "id": marker,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": REPLY_TEXT},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }
    ).encode()


def _responses_result(marker: str) -> bytes:
    return json.dumps(
        {
            "id": marker,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": "msg-lit8281",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        }
    ).encode()


def _chat_stream(marker: str) -> tuple[bytes, ...]:
    def frame(delta: dict[str, object], finish: str | None = None) -> bytes:
        return (
            b"data: "
            + json.dumps(
                {
                    "id": marker,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            ).encode()
            + b"\n\n"
        )

    return (
        frame({"role": "assistant", "content": "langfuse-otel "}),
        frame({"content": "scripted reply"}),
        frame({}, finish="stop"),
        b"data: [DONE]\n\n",
    )


def _generation_span(collector: Wire, marker: str) -> SpanRecord:
    batches: list[Request] = []  # mutable-ok: accumulated across eventually() polls

    def spans() -> tuple[SpanRecord, ...]:
        batches.extend(collector.drain())
        return tuple(
            record
            for request in batches
            if request.target.endswith("/v1/traces")
            for record in _span_records(request.body)
            if record.attributes.get("langfuse.observation.type") == "generation"
            and marker in str(tuple(record.attributes.values()))
        )

    found: Final = eventually(spans, lambda values: len(values) == 1, seconds=70)
    return found[0]


def _assert_derived_trace_fields(span: SpanRecord, call_type: str) -> None:
    attributes: Final = span.attributes
    assert attributes.get("langfuse.observation.input") is not None, attributes
    assert attributes.get("langfuse.observation.output") is not None, attributes
    assert attributes.get("langfuse.trace.name") == f"litellm-{call_type}", attributes
    assert attributes.get("langfuse.trace.input") == attributes.get("langfuse.observation.input"), attributes
    assert attributes.get("langfuse.trace.output") == attributes.get("langfuse.observation.output"), attributes


def _langfuse_config(directory: Path, callbacks: tuple[str, ...]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = list(callbacks)
    path: Final = directory / "langfuse.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def langfuse_sink() -> Iterator[Wire]:
    with wire_server(_otlp_sink) as wire:
        yield wire


@pytest.fixture(scope="module")
def langfuse_v1_proxy(langfuse_sink: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("langfuse-v1")
    with gateway_from_environment() as gateway:
        with owned_proxy(
            gateway,
            directory,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
                "LANGFUSE_HOST": langfuse_sink.url,
            },
            config=_langfuse_config(directory, ("langfuse_otel",)),
        ) as proxy:
            yield proxy


@pytest.fixture(scope="module")
def langfuse_v2_proxy(langfuse_sink: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("langfuse-v2")
    with gateway_from_environment() as gateway:
        with owned_proxy(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
                "LANGFUSE_HOST": langfuse_sink.url,
            },
            config=_langfuse_config(directory, ("langfuse_otel",)),
        ) as proxy:
            yield proxy


@pytest.mark.covers("other.observability.langfuse_otel.chat_derives_trace_fields")
def test_chat_completion_derives_trace_name_input_and_output(langfuse_v1_proxy: Gateway, langfuse_sink: Wire) -> None:
    marker: Final = "lit8281-chat-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(body=_chat_completion(marker))

    with wire_server(upstream) as provider, langfuse_v1_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v1_proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
        )
        assert response.status_code == 200, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        assert span.attributes["llm.response.id"] == marker, span.attributes
        assert span.attributes["langfuse.observation.type"] == "generation", span.attributes
        _assert_derived_trace_fields(span, "acompletion")
        assert marker in str(span.attributes["langfuse.observation.input"]), span.attributes
        assert REPLY_TEXT in str(span.attributes["langfuse.observation.output"]), span.attributes


@pytest.mark.covers("other.observability.langfuse_otel.parented_chat_derives_trace_fields_and_keeps_parent")
def test_parented_chat_completion_derives_trace_fields_inside_inbound_trace(
    langfuse_v1_proxy: Gateway, langfuse_sink: Wire
) -> None:
    marker: Final = "lit8281-parent-" + uuid.uuid4().hex
    trace_id: Final = uuid.uuid4().hex
    parent_id: Final = uuid.uuid4().hex[:16]

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(body=_chat_completion(marker))

    with wire_server(upstream) as provider, langfuse_v1_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v1_proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
            headers={"traceparent": f"00-{trace_id}-{parent_id}-01"},
        )
        assert response.status_code == 200, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        assert span.trace_id == trace_id, (span.trace_id, trace_id)
        assert span.parent_span_id == parent_id, (span.parent_span_id, parent_id)
        _assert_derived_trace_fields(span, "acompletion")


@pytest.mark.covers("other.observability.langfuse_otel.streaming_chat_derives_trace_fields")
def test_streaming_chat_completion_derives_trace_fields(langfuse_v1_proxy: Gateway, langfuse_sink: Wire) -> None:
    marker: Final = "lit8281-stream-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(marker))

    with wire_server(upstream) as provider, langfuse_v1_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v1_proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        assert "scripted reply" in response.text, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        _assert_derived_trace_fields(span, "acompletion")
        assert REPLY_TEXT in str(span.attributes["langfuse.observation.output"]), span.attributes


@pytest.mark.covers("other.observability.langfuse_otel.caller_trace_name_and_tags_win")
def test_caller_supplied_trace_name_and_tags_are_emitted(langfuse_v1_proxy: Gateway, langfuse_sink: Wire) -> None:
    marker: Final = "lit8281-caller-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(body=_chat_completion(marker))

    with wire_server(upstream) as provider, langfuse_v1_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v1_proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"trace_name": "caller-trace", "tags": ["caller-tag"]},
            },
        )
        assert response.status_code == 200, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        assert span.attributes.get("langfuse.trace.name") == "caller-trace", span.attributes
        tags: Final = json.loads(str(span.attributes["langfuse.trace.tags"]))
        assert tags[0] == "caller-tag", span.attributes


@pytest.mark.covers("other.observability.langfuse_otel.responses_derives_trace_fields")
def test_responses_call_derives_trace_fields(langfuse_v1_proxy: Gateway, langfuse_sink: Wire) -> None:
    marker: Final = "lit8281-resp-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/responses"), request.target
        return Reply(body=_responses_result(marker))

    with wire_server(upstream) as provider, langfuse_v1_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v1_proxy.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": marker},
        )
        assert response.status_code == 200, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        _assert_derived_trace_fields(span, "aresponses")


@pytest.mark.covers("other.observability.langfuse_otel.v2_derives_trace_fields")
def test_otel_v2_derives_trace_name_input_and_output(langfuse_v2_proxy: Gateway, langfuse_sink: Wire) -> None:
    marker: Final = "lit8281-v2-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(body=_chat_completion(marker))

    with wire_server(upstream) as provider, langfuse_v2_proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = langfuse_v2_proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
        )
        assert response.status_code == 200, response.text
        span: Final = _generation_span(langfuse_sink, marker)
        _assert_derived_trace_fields(span, "acompletion")
