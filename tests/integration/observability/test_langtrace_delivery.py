import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from pydantic import TypeAdapter

TRACE_PATH: Final = "/api/trace"
STOCK_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
_PROXY_CONFIG: Final = TypeAdapter(dict[str, object])
_SETTINGS: Final = TypeAdapter(dict[str, object])


def _completion(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + text,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _langtrace_config(tmp_path: Path) -> Path:
    config: Final = _PROXY_CONFIG.validate_python(yaml.safe_load(STOCK_CONFIG.read_text()))
    settings: Final = {**_SETTINGS.validate_python(config["litellm_settings"]), "callbacks": ["langtrace"]}
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


def _spans_prompted_with(batches: Sequence[Request], marker: str) -> tuple[Span, ...]:
    return tuple(span for span in _spans(batches) if any(marker in prompt for prompt in _prompt_events(span)))


def test_langtrace_callback_posts_protobuf_spans_to_api_trace_with_x_api_key(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "langtrace-" + uuid.uuid4().hex
    api_key: Final = "synthetic-langtrace-key-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        return _completion(json.loads(request.body)["messages"][0]["content"])

    def langtrace(request: Request) -> Reply:
        return Reply(body=b'{"message":"Traces added successfully"}')

    with wire_server(upstream) as provider, wire_server(langtrace) as sink:
        received: Final[list[Request]] = []  # mutable-ok: drain() consumes, so batches accumulate across polls

        def collect() -> tuple[Request, ...]:
            received.extend(sink.drain())
            return tuple(received)

        overrides: Final = {
            "LANGTRACE_API_KEY": api_key,
            "LANGTRACE_API_HOST": sink.url,
            "OTEL_BSP_SCHEDULE_DELAY": "300",
        }
        with (
            owned_proxy(gateway, tmp_path, overrides, config=_langtrace_config(tmp_path)) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1")
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            )
            assert response.status_code == 200, response.text
            assert response.json()["id"] == "chatcmpl-" + marker, response.text
            requests: Final = eventually(
                collect, lambda batches: len(_spans_prompted_with(batches, marker)) >= 1, seconds=20
            )

    for request in requests:
        assert (request.method, request.target) == ("POST", TRACE_PATH), requests
        assert request.headers.get("x-api-key") == api_key, request.headers
        assert "api_key" not in request.headers, request.headers
        assert request.headers.get("content-type") == "application/x-protobuf", request.headers
    spans: Final = _spans_prompted_with(requests, marker)
    assert len(spans) == 1, requests
    assert spans[0].name == "litellm_request", spans[0]
    assert api_key.encode() not in b"".join(batch.body for batch in requests)
