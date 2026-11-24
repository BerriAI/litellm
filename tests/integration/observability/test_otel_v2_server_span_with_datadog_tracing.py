import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans, spans_for_trace
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]
SERVER: Final = 2


@pytest.fixture(scope="module")
def gateway(audit_sinks: SpanSinks) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        yield base


def _datadog_agent(request: Request) -> Reply:
    return Reply(body=b'{"rate_by_service": {}}')


def _started_by_ddtrace_run(agent_url: str) -> Mapping[str, str]:
    package: Final = find_spec("ddtrace")
    assert package is not None and package.origin is not None, "ddtrace is not installed"
    bootstrap: Final = Path(package.origin).parent / "bootstrap"
    assert (bootstrap / "sitecustomize.py").is_file(), bootstrap
    return {
        "PYTHONPATH": os.pathsep.join(path for path in (str(bootstrap), os.environ.get("PYTHONPATH", "")) if path),
        "USE_DDTRACE": "true",
        "DD_TRACE_ENABLED": "true",
        "DD_TRACE_AGENT_URL": agent_url,
    }


def _is_datadog_trace_of(request: Request, path: str) -> bool:
    return request.target.endswith("/traces") and path.encode() in request.body


def _server_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(span for span in spans if span["kind"] == SERVER)


def test_a_request_keeps_its_server_span_and_team_identity_when_datadog_tracing_is_on(
    gateway: Gateway, audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    config: Final = otel_audit_config(tmp_path, {})
    with (
        wire_server(_datadog_agent) as agent,
        owned_proxy(
            gateway, tmp_path, {"LITELLM_OTEL_V2": "1", **_started_by_ddtrace_run(agent.url)}, config=config
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(models=[model], team_id=team)
        start, _ = recorded_spans(audit_sinks.operator)
        trace_id: Final = uuid.uuid4().hex
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"datadog and otel {uuid.uuid4().hex}"}]},
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text
        trace: Final = eventually(
            lambda: spans_for_trace(recorded_spans(audit_sinks.operator, start)[1], trace_id),
            lambda spans: len(_server_spans(spans)) > 0,
            seconds=60,
            return_last_on_timeout=True,
        )
        names: Final = sorted(span["name"] for span in trace)
        assert len(_server_spans(trace)) == 1, names
        (server,) = _server_spans(trace)
        assert server["name"] == "POST /v1/chat/completions", names
        assert server["attributes"].get("litellm.team.id") == team, server["attributes"]
        datadog_traces: Final = eventually(
            lambda: tuple(sent for sent in agent.drain() if _is_datadog_trace_of(sent, "/v1/chat/completions")),
            lambda sent: len(sent) > 0,
            seconds=60,
            return_last_on_timeout=True,
        )
        assert datadog_traces, "Datadog received no trace for the request"
