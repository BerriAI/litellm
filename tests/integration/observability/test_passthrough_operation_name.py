import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.timeout(180)

_EXPORTERS: Final = {
    "otel-v1-semconv": {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"},
    "otel-v2": {"LITELLM_OTEL_V2": "1"},
}
_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_passthrough_operation",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_STREAM_REQUESTS: Final = {"stream-requested": {"query": "q", "stream": True}, "stream-detected": {"query": "q"}}


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    sink: str


def _upstream(request: Request) -> Reply:
    if request.target.endswith("/v1/messages"):
        return Reply(body=json.dumps(_ANTHROPIC_MESSAGE).encode())
    _, _, reported = request.target.partition("/report/")
    headers: Final = {"x-litellm-operation-name": reported} if reported else {}
    if request.target.startswith("/app/stream/cut"):
        return Reply(
            content_type="text/event-stream",
            chunks=(b'data: {"chunk": 1}\n\n', b'data: {"chunk": 2}\n\n'),
            abort_after=1,
        )
    if request.target.startswith("/app/stream"):
        return Reply(content_type="text/event-stream", chunks=(b'data: {"chunk": 1}\n\n',), headers=headers)
    return Reply(body=b'{"chunks": []}', headers=headers)


@pytest.fixture(scope="module", params=sorted(_EXPORTERS))
def rig(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks
) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-operation")
    exporter: Final = _EXPORTERS[request.param]
    with gateway_from_environment() as base, wire_server(_upstream) as upstream:
        shared: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config: Final = {
            **shared,
            "litellm_settings": {**shared["litellm_settings"], "callbacks": ["otel"]},
            "callback_settings": {
                "otel": {"exporter": "http/json", "endpoint": audit_sinks.operator, "use_simple_processor": True}
            },
            "general_settings": {
                **shared["general_settings"],
                "pass_through_endpoints": [
                    {"path": "/doc-app", "target": upstream.url + "/app", "include_subpath": True}
                ],
            },
            "environment_variables": {
                **shared["environment_variables"],
                "ANTHROPIC_API_BASE": upstream.url,
                "ANTHROPIC_API_KEY": "scripted",
            },
        }
        path: Final = directory / "passthrough-operation.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": audit_sinks.operator, **exporter}
        unset: Final = tuple(
            name for name in ("LITELLM_OTEL_V2", "OTEL_SEMCONV_STABILITY_OPT_IN") if name not in exporter
        )
        with owned_proxy(base, directory, overrides, config=path, remove_environment=unset) as proxy:
            yield _Rig(proxy, audit_sinks.operator)


def _span_of(rig: _Rig, since: int, call_id: str) -> Span:
    spans: Final = eventually(
        lambda: tuple(
            span
            for span in recorded_spans(rig.sink, since)[1]
            if span["attributes"].get("litellm.call_id") == call_id and "gen_ai.operation.name" in span["attributes"]
        ),
        lambda found: len(found) == 1,
        seconds=60,
        return_last_on_timeout=True,
    )
    assert len(spans) == 1, spans
    return spans[0]


def _operation_span(rig: _Rig, method: str, path: str, body: dict[str, object] | None = None) -> Span:
    since, _ = recorded_spans(rig.sink)
    response: Final = rig.proxy.request(method, path, body)
    assert response.status_code == 200, response.text
    return _span_of(rig, since, response.headers["x-litellm-call-id"])


def _cut_stream_span(rig: _Rig, path: str) -> Span:
    since, _ = recorded_spans(rig.sink)
    headers: Final = {"Authorization": f"Bearer {rig.proxy.key}"}
    with rig.proxy.client.stream("POST", path, json={"query": "q", "stream": True}, headers=headers) as response:
        assert response.status_code == 200
        with pytest.raises(httpx.RemoteProtocolError):
            response.read()
    return _span_of(rig, since, response.headers["x-litellm-call-id"])


def _assert_operation(span: Span, operation: str) -> None:
    assert span["attributes"]["gen_ai.operation.name"] == operation, span
    assert span["name"].split(" ")[0] == operation, span


def test_configured_route_without_the_header_is_labelled_passthrough(rig: _Rig) -> None:
    _assert_operation(_operation_span(rig, "POST", "/doc-app/search", {"query": "q"}), "litellm.passthrough")


def test_operation_reported_by_the_target_labels_the_span(rig: _Rig) -> None:
    _assert_operation(_operation_span(rig, "POST", "/doc-app/report/retrieval", {"query": "q"}), "retrieval")


def test_unknown_operation_reported_by_the_target_is_ignored(rig: _Rig) -> None:
    _assert_operation(_operation_span(rig, "POST", "/doc-app/report/summarize", {"query": "q"}), "litellm.passthrough")


@pytest.mark.parametrize("body", _STREAM_REQUESTS.values(), ids=_STREAM_REQUESTS.keys())
def test_streamed_configured_route_without_the_header_is_labelled_passthrough(
    rig: _Rig, body: dict[str, object]
) -> None:
    _assert_operation(_operation_span(rig, "POST", "/doc-app/stream", body), "litellm.passthrough")


@pytest.mark.parametrize("body", _STREAM_REQUESTS.values(), ids=_STREAM_REQUESTS.keys())
def test_operation_reported_by_a_streamed_target_labels_the_span(rig: _Rig, body: dict[str, object]) -> None:
    _assert_operation(_operation_span(rig, "POST", "/doc-app/stream/report/retrieval", body), "retrieval")


def test_configured_route_stream_cut_by_the_target_is_labelled_passthrough(rig: _Rig) -> None:
    _assert_operation(_cut_stream_span(rig, "/doc-app/stream/cut"), "litellm.passthrough")


def test_provider_route_stays_chat(rig: _Rig) -> None:
    body: Final = {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    _assert_operation(_operation_span(rig, "POST", "/anthropic/v1/messages", body), "chat")


def test_provider_route_litellm_does_not_parse_stays_chat(rig: _Rig) -> None:
    _assert_operation(_operation_span(rig, "GET", "/anthropic/v1/models"), "chat")
