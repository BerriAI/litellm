import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans, spans_for_trace
from integration._support.process import owned_proxy
from integration._support.wire import wire_server
from integration.management._batch_file_caps import (
    UPLOADS,
    batch_file,
    caps_config,
    marker,
    provider,
    provider_environment,
    uploads_seen,
)
from pydantic import JsonValue

COUNTER_SPAN: Final = "redis.incr rate_limits"
SERVER: Final = 2
CAP: Final = 3


def _config(directory: Path, otel: Path, provider_url: str) -> Path:
    tracing: Final = yaml.safe_load(otel.read_text())
    caps: Final = yaml.safe_load(caps_config(directory, provider_url, {UPLOADS: CAP}).read_text())
    path: Final = directory / "otel-file-usage-caps.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **tracing,
                "model_list": caps["model_list"],
                "files_settings": caps["files_settings"],
                "general_settings": {**tracing["general_settings"], UPLOADS: CAP},
            }
        )
    )
    return path


def _names(spans: tuple[Span, ...]) -> list[str]:
    return sorted(span["name"] for span in spans)


def _traced_upload(candidate: Gateway, content: bytes, trace_id: str) -> int:
    response: Final = candidate.client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("batch.jsonl", content, "application/jsonl")},
        headers={
            "Authorization": f"Bearer {candidate.key}",
            "traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01",
        },
    )
    assert response.status_code == 200, response.text
    return response.status_code


def test_a_counted_upload_exports_its_counter_increment_as_a_rate_limits_span(
    gateway: Gateway,
    tmp_path: Path,
    audit_sinks: SpanSinks,
    otel_audit_config: Callable[[Path, Mapping[str, JsonValue]], Path],
) -> None:
    with wire_server(provider) as files:
        config: Final = _config(tmp_path, otel_audit_config(tmp_path, {}), files.url)
        environment: Final = {**provider_environment(files.url), "LITELLM_OTEL_V2": "1"}
        with owned_proxy(gateway, tmp_path, environment, config=config) as candidate:
            since, _ = recorded_spans(audit_sinks.operator)
            mark: Final = marker()
            trace_id: Final = uuid.uuid4().hex
            assert _traced_upload(candidate, batch_file(mark, 1), trace_id) == 200
            trace: Final = eventually(
                lambda: spans_for_trace(recorded_spans(audit_sinks.operator, since)[1], trace_id),
                lambda spans: COUNTER_SPAN in _names(spans),
                seconds=40,
            )
            assert _names(trace).count(COUNTER_SPAN) == 1, _names(trace)
            assert sum(1 for span in trace if span["kind"] == SERVER) == 1, _names(trace)
            (counter,) = (span for span in trace if span["name"] == COUNTER_SPAN)
            assert counter["attributes"].get("litellm.service.target") == "rate_limits", counter["attributes"]
            assert len(uploads_seen(files.drain(), mark)) == 1
