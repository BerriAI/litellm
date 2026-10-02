import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
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


def test_default_otel_logger_puts_datastore_model_and_spend_writer_spans_in_the_request_trace(
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
        expected: Final = frozenset({"postgres", "redis", "raw_gen_ai_request", "batch_write_to_db"})
        traces: Final = eventually(
            lambda: _traces(recorded_spans(audit_sinks.operator, start)[1]),
            lambda grouped: any(expected <= names for names in grouped.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        assert any(expected <= names for names in traces.values()), {
            trace: sorted(names) for trace, names in traces.items()
        }
