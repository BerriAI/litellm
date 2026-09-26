import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.common.v1.common_pb2 import AnyValue

MARKER_JSON_KEYS: Final = ("llm.response.id", "gen_ai.response.id")


def _attribute_value(value: AnyValue) -> object:
    kind: Final = value.WhichOneof("value")
    return getattr(value, kind) if kind is not None else None


def _spans(body: bytes) -> tuple[tuple[str, dict[str, object]], ...]:
    export: Final = trace_service_pb2.ExportTraceServiceRequest()
    export.ParseFromString(body)
    return tuple(
        (
            span.trace_id.hex(),
            {attribute.key: _attribute_value(attribute.value) for attribute in span.attributes},
        )
        for resource in export.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


def _sse_frame(value: Mapping[str, object]) -> bytes:
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def _upstream_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": marker,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"reply {marker}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _upstream_stream_reply(marker: str) -> Reply:
    chunk: Final = {
        "id": marker,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"reply {marker}"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    final: Final = {
        "id": marker,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(_sse_frame(chunk), _sse_frame(final), b"data: [DONE]\n\n"),
    )


def _responses_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": marker,
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"reply {marker}"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _responses_stream_reply(marker: str) -> Reply:
    response: Final = {
        "id": marker,
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o-mini",
        "status": "in_progress",
        "output": [],
        "usage": None,
    }
    completed: Final = {
        "id": marker,
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "item-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"reply {marker}"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    events: Final = (
        {"type": "response.created", "response": response},
        {"type": "response.output_item.added", "output_index": 0, "item": completed["output"][0]},
        {
            "type": "response.output_text.delta",
            "item_id": "item-1",
            "output_index": 0,
            "content_index": 0,
            "delta": f"reply {marker}",
        },
        {
            "type": "response.output_text.done",
            "item_id": "item-1",
            "output_index": 0,
            "content_index": 0,
            "text": f"reply {marker}",
        },
        {"type": "response.output_item.done", "output_index": 0, "item": completed["output"][0]},
        {"type": "response.completed", "response": completed},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(_sse_frame(event) for event in events) + (b"data: [DONE]\n\n",),
    )


def _upstream_reply_for(request: Request, marker: str) -> Reply:
    body: Final = json.loads(request.body) if request.body else {}
    if request.target.endswith("/responses"):
        if body.get("stream") is True:
            return _responses_stream_reply(marker)
        return _responses_reply(marker)
    if body.get("stream") is True:
        return _upstream_stream_reply(marker)
    return _upstream_reply(marker)


def _marker_from_body(request: Request) -> str:
    body: Final = json.loads(request.body) if request.body else {}
    if body.get("input") is not None:
        value: Final = body["input"]
        if isinstance(value, str):
            return value
        return str(value)
    messages: Final = body.get("messages")
    if not messages:
        return ""
    return str(messages[0]["content"])


def _sink(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _drained_spans(sink: Wire, batches: list[bytes]) -> tuple[tuple[str, dict[str, object]], ...]:
    batches.extend(request.body for request in sink.drain())
    return tuple(span for body in batches for span in _spans(body))


def _generation_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if attributes.get("llm.response.id") == marker or attributes.get("gen_ai.response.id") == marker
    )


def _span_attributes_containing_marker(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if any(isinstance(value, str) and marker in value for value in attributes.values())
    )


def _generation_marker_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if attributes.get("langfuse.observation.type") == "generation"
        and any(isinstance(value, str) and marker in value for value in attributes.values())
    )


def _dedupe_spans(spans: tuple[tuple[str, dict[str, object]], ...]) -> tuple[tuple[str, dict[str, object]], ...]:
    seen: Final = set()
    unique: Final = []
    for trace_id, attributes in spans:
        fingerprint: Final = (trace_id, frozenset(attributes.items()))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append((trace_id, attributes))
    return tuple(unique)


def _span_containing_marker(
    spans: tuple[tuple[str, dict[str, object]], ...], marker: str
) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(
        (trace_id, attributes)
        for trace_id, attributes in spans
        if any(isinstance(value, str) and marker in value for value in attributes.values())
    )


def _arize_generation_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    return tuple(
        attributes
        for _trace_id, attributes in _drained_spans(sink, batches)
        if attributes.get("llm.response.id") == marker and "session.id" in attributes
    )


def _trace_user_span_attributes(sink: Wire, batches: list[bytes], marker: str) -> tuple[dict[str, object], ...]:
    spans: Final = _drained_spans(sink, batches)
    generation_trace: Final = next(
        (
            trace_id
            for trace_id, attributes in spans
            if attributes.get("llm.response.id") == marker or attributes.get("gen_ai.response.id") == marker
        ),
        None,
    )
    if generation_trace is None:
        return ()
    return tuple(
        attributes for trace_id, attributes in spans if trace_id == generation_trace and "user.id" in attributes
    )


def _user_id_session_id(attributes: Mapping[str, object]) -> dict[str, object]:
    return {key: attributes.get(key) for key in ("user.id", "session.id")}


def _proxy_config(directory: Path, name: str, callbacks: tuple[str, ...]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": list(callbacks)})
    path: Final = directory / name
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _observability_proxy(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    callbacks: tuple[str, ...] = ("langfuse_otel",),
    config_name: str = "langfuse_otel.yaml",
) -> Iterator[Gateway]:
    path: Final = _proxy_config(directory, config_name, callbacks)
    with owned_proxy(
        gateway,
        directory,
        {"OTEL_BSP_SCHEDULE_DELAY": "100", **overrides},
        config=path,
        workers=2,
    ) as candidate:
        yield candidate


@contextmanager
def _langfuse_proxy(
    gateway: Gateway, directory: Path, collector_url: str, overrides: Mapping[str, str] | None = None
) -> Iterator[Gateway]:
    with _observability_proxy(
        gateway,
        directory,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-integration",
            "LANGFUSE_SECRET_KEY": "sk-integration",
            "LANGFUSE_HOST": collector_url,
            **(overrides or {}),
        },
    ) as candidate:
        yield candidate
