import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]


@pytest.fixture(scope="module")
def gateway(audit_sinks: SpanSinks) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        yield base


def _traces(spans: tuple[Span, ...]) -> dict[str, frozenset[str]]:
    trace_ids: Final = {span["trace_id"] for span in spans}
    return {trace: frozenset(span["name"] for span in spans if span["trace_id"] == trace) for trace in trace_ids}


def test_default_otel_logger_keeps_spend_flush_outside_the_request_trace(
    gateway: Gateway, audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    config: Final = otel_audit_config(tmp_path, {})
    overrides: Final = {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": audit_sinks.operator}
    with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        start, _ = recorded_spans(audit_sinks.operator)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"otel v1 {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        expected: Final = frozenset({"postgres", "redis", "raw_gen_ai_request"})
        traces: Final = eventually(
            lambda: _traces(recorded_spans(audit_sinks.operator, start)[1]),
            lambda grouped: any(expected <= names for names in grouped.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        assert any(expected <= names for names in traces.values()), {
            trace: sorted(names) for trace, names in traces.items()
        }
        request_trace: Final = next(trace for trace, names in traces.items() if expected <= names)
        key_info: Final = eventually(
            lambda: candidate.request("GET", "/key/info", key=key, params={"key": key}),
            lambda response: response.status_code == 200
            and float(str(object_value(response.json()["info"])["spend"])) > 0,
            seconds=60,
        )
        assert key_info.status_code == 200, key_info.text
        assert float(str(object_value(key_info.json()["info"])["spend"])) > 0
        request_spans: Final = tuple(
            span for span in recorded_spans(audit_sinks.operator, start)[1] if span["trace_id"] == request_trace
        )
        assert not any(span["name"] == "batch_write_to_db" for span in request_spans), request_spans
        assert not any(
            span["name"] == "postgres"
            and span["attributes"].get("call_type") == "commit_spend_updates"
            and span["attributes"].get("table_name") == "LiteLLM_VerificationToken"
            for span in request_spans
        ), request_spans
