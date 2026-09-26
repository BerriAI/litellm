from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    Gateway,
    eventually,
    gateway_from_environment,
)
from integration._support.otlp_sink import (
    Span,
    SpanSinks,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import owned_proxy, owned_proxy_process
from pydantic import JsonValue

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]

DB_SYSTEM_KEYS: Final = frozenset({"db.system.name", "db.system"})


@pytest.fixture(scope="module")
def gateway(audit_sinks: SpanSinks) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        yield base


def _config_with(
    directory: Path,
    otel_audit_config: AuditConfigWriter,
    *,
    otel: Mapping[str, JsonValue] = MappingProxyType({}),
    extra: Callable[[dict[str, JsonValue]], None] | None = None,
) -> Path:
    config: Final = yaml.safe_load(otel_audit_config(directory, {}).read_text())
    config["callback_settings"]["otel"].update(dict(otel))
    if extra is not None:
        extra(config)
    path: Final = directory / f"otel-excl-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _add_callback(gateway: Gateway, team_id: str, callback_vars: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        f"/team/{team_id}/callback",
        {"callback_name": "langfuse_otel", "callback_vars": dict(callback_vars)},
    )


def _drive(candidate: Gateway, langfuse_vars: Mapping[str, JsonValue]) -> httpx.Response:
    with candidate.scenario() as scenario:
        model: Final = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
        team_id: Final = scenario.team()
        callback: Final = _add_callback(candidate, team_id, langfuse_vars)
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(team_id=team_id)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"otel-excl-{uuid.uuid4().hex}"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        return response


def _trace_id(sink_url: str, response: httpx.Response, seconds: float = 40) -> str:
    call_id: Final = response.headers.get("x-litellm-call-id")
    response_id: Final = response.json().get("id")

    def look() -> str | None:
        _, spans = recorded_spans(sink_url)
        return next(
            (
                str(span["trace_id"])
                for span in spans
                if (call_id is not None and span["attributes"].get("litellm.call_id") == call_id)
                or (response_id is not None and span["attributes"].get("gen_ai.response.id") == response_id)
            ),
            None,
        )

    found: Final = eventually(look, lambda value: value is not None, seconds=seconds)
    assert found is not None
    return found


def _trace_spans(sink_url: str, trace_id: str, seconds: float = 30) -> tuple[Span, ...]:
    """The trace's spans once the post-call tail has landed.

    The spend-writer and other post-response spans flush after the request
    answers, so absence assertions poll for the whole window instead of
    settling at the first glimpse of the root span.
    """
    deadline: Final = time.monotonic() + seconds
    group: tuple[Span, ...] = ()  # rebind-ok: drains samples until the post-call tail lands
    while time.monotonic() < deadline:
        _, spans = recorded_spans(sink_url)
        group = spans_for_trace(spans, trace_id)
        time.sleep(0.5)
    assert group, f"trace {trace_id} never reached {sink_url}"
    return group


def _await_db_span(sink_url: str, trace_id: str | None, needle: str, seconds: float = 40, since: int = 0) -> None:
    def seen() -> bool:
        _, spans = recorded_spans(sink_url, since)
        group: Final = spans if trace_id is None else spans_for_trace(spans, trace_id)
        return any(
            needle in str(span["name"]) or needle in {str(span["attributes"].get(k)) for k in DB_SYSTEM_KEYS}
            for span in group
        )

    landed: Final = eventually(seen, bool, seconds=seconds)
    assert landed, f"{needle} span never landed at {sink_url}"


def _db_systems(spans: tuple[Span, ...]) -> set[str]:
    return {str(span["attributes"][key]) for span in spans for key in DB_SYSTEM_KEYS if key in span["attributes"]}


def _assert_core_spans_present(spans: tuple[Span, ...]) -> None:
    attributes_by_span: Final = tuple(span["attributes"] for span in spans)
    assert any(span["kind"] == 2 for span in spans), "request root span missing"
    assert any("gen_ai.operation.name" in attrs for attrs in attributes_by_span), "model span missing"
    assert any("litellm.guardrail.name" in attrs for attrs in attributes_by_span), "guardrail span missing"
    names: Final = sorted(str(span["name"]) for span in spans)
    assert any(name.startswith("auth") for name in names), f"auth span missing in {names}"


def _guardrail_block(config: dict) -> None:
    config["guardrails"] = [
        {
            "guardrail_name": f"excl-filter-{uuid.uuid4().hex[:8]}",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "default_on": True,
                "patterns": [
                    {
                        "pattern_type": "regex",
                        "pattern_name": "excl_secret",
                        "pattern": "TOPSECRET\\d{9}",
                        "action": "BLOCK",
                    }
                ],
            },
        }
    ]


@pytest.mark.timeout(180)
def test_excluded_services_drops_db_spans_at_tenant_only(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(
        tmp_path, otel_audit_config, otel={"excluded_services": ["redis", "postgres"]}, extra=_guardrail_block
    )
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2) as candidate:
        ten_start, _ = recorded_spans(audit_sinks.tenant)
        op_start, _ = recorded_spans(audit_sinks.operator)
        traffic: Final = _drive(candidate, langfuse_vars)
        _await_db_span(audit_sinks.operator, None, "postgresql", seconds=60, since=op_start)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace)
        _assert_core_spans_present(tenant_spans)
        assert _db_systems(tenant_spans) == set(), (
            f"db spans reached tenant: {sorted(str(s['name']) for s in tenant_spans)}"
        )
        operator_trace: Final = _trace_id(audit_sinks.operator, traffic)
        assert operator_trace == tenant_trace
        trace_systems: Final = _db_systems(_trace_spans(audit_sinks.operator, operator_trace))
        assert "redis" in trace_systems, f"operator trace lost redis spans: {trace_systems}"
        _, all_operator = recorded_spans(audit_sinks.operator, op_start)
        operator_systems: Final = _db_systems(all_operator)
        assert "postgresql" in operator_systems, f"operator lost aux db spans: {operator_systems}"
        _, all_tenant = recorded_spans(audit_sinks.tenant, ten_start)
        names: Final = sorted(str(span["name"]) for span in all_tenant)
        assert _db_systems(all_tenant) == set(), f"aux db spans reached tenant: {names}"
        assert not any("batch_write_to_db" in name for name in names), f"spend writer reached tenant: {names}"


def test_env_excluded_services_drops_only_redis(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config)
    with owned_proxy(
        gateway, tmp_path, {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "redis"}, config=config, workers=2
    ) as candidate:
        start, _ = recorded_spans(audit_sinks.tenant)
        _drive(candidate, langfuse_vars)
        _await_db_span(audit_sinks.tenant, None, "postgresql", seconds=60, since=start)
        _, tenant_spans = recorded_spans(audit_sinks.tenant, start)
        systems: Final = _db_systems(tenant_spans)
        assert "postgresql" in systems, f"postgresql spans missing at tenant: {systems}"
        assert "redis" not in systems, f"redis spans reached tenant: {sorted(str(s['name']) for s in tenant_spans)}"


def test_config_excluded_services_wins_over_env(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    def with_langfuse_otel(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["otel", "langfuse_otel"]

    config: Final = _config_with(
        tmp_path, otel_audit_config, otel={"excluded_services": ["postgres"]}, extra=with_langfuse_otel
    )
    with owned_proxy(
        gateway, tmp_path, {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "redis"}, config=config, workers=2
    ) as candidate:
        start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
        tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _, all_tenant = recorded_spans(audit_sinks.tenant, start)
        systems: Final = _db_systems(tenant_spans)
        assert "redis" in systems, f"redis spans missing at tenant: {systems}"
        assert "postgresql" not in _db_systems(all_tenant), (
            f"postgresql spans reached tenant: {_db_systems(all_tenant)}"
        )


def test_excluded_services_applies_with_preset_ordered_first(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    def preset_first(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["langfuse_otel", "otel"]

    config: Final = _config_with(
        tmp_path, otel_audit_config, otel={"excluded_services": ["postgres"]}, extra=preset_first
    )
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2) as candidate:
        start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
        tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _, all_tenant = recorded_spans(audit_sinks.tenant, start)
        systems: Final = _db_systems(tenant_spans)
        assert "redis" in systems, f"redis spans missing at tenant: {systems}"
        assert "postgresql" not in _db_systems(all_tenant), (
            f"postgresql spans reached tenant: {_db_systems(all_tenant)}"
        )


def test_bogus_excluded_service_fails_proxy_start(
    gateway: Gateway, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"excluded_services": ["auth"]})
    log_dir: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path)))
    before: Final = frozenset(log_dir.glob("owned-proxy-*.log"))
    with pytest.raises(AssertionError, match="readiness"):
        with owned_proxy_process(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2):
            pass
    logs: Final = [path.read_text() for path in frozenset(log_dir.glob("owned-proxy-*.log")) - before]
    assert logs, "no owned proxy log written"
    text: Final = "\n".join(logs)
    assert "'auth' is not a datastore service" in text, text[-3000:]
    assert "postgres, redis" in text, text[-3000:]


def test_valid_config_excluded_services_tolerates_bogus_env(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"excluded_services": ["postgres"]})
    with owned_proxy(
        gateway, tmp_path, {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "auth"}, config=config, workers=2
    ) as candidate:
        start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
        tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _, all_tenant = recorded_spans(audit_sinks.tenant, start)
        systems: Final = _db_systems(tenant_spans)
        assert "redis" in systems, f"redis spans missing at tenant: {systems}"
        assert "postgresql" not in _db_systems(all_tenant), (
            f"postgresql spans reached tenant: {_db_systems(all_tenant)}"
        )


def test_bogus_excluded_services_env_fails_proxy_start_with_preset_alongside_otel(
    gateway: Gateway, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    def with_langfuse_otel(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["otel", "langfuse_otel"]

    config: Final = _config_with(
        tmp_path, otel_audit_config, otel={"excluded_services": ["postgres"]}, extra=with_langfuse_otel
    )
    log_dir: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path)))
    before: Final = frozenset(log_dir.glob("owned-proxy-*.log"))
    with pytest.raises(AssertionError, match="readiness"):
        with owned_proxy_process(
            gateway, tmp_path, {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "auth"}, config=config, workers=2
        ):
            pass
    logs: Final = [path.read_text() for path in frozenset(log_dir.glob("owned-proxy-*.log")) - before]
    assert logs, "no owned proxy log written"
    text: Final = "\n".join(logs)
    assert "'auth' is not a datastore service" in text, text[-3000:]


def test_bogus_excluded_services_env_fails_proxy_start_without_otel_callback(
    gateway: Gateway, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    def presets_only(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["langfuse_otel"]

    config: Final = _config_with(tmp_path, otel_audit_config, extra=presets_only)
    log_dir: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path)))
    before: Final = frozenset(log_dir.glob("owned-proxy-*.log"))
    with pytest.raises(AssertionError, match="readiness"):
        with owned_proxy_process(
            gateway, tmp_path, {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "auth"}, config=config, workers=2
        ):
            pass
    logs: Final = [path.read_text() for path in frozenset(log_dir.glob("owned-proxy-*.log")) - before]
    assert logs, "no owned proxy log written"
    text: Final = "\n".join(logs)
    assert "'auth' is not a datastore service" in text, text[-3000:]


def test_postgres_exclusion_covers_batch_write_to_db(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"excluded_services": ["postgres"]})
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2) as candidate:
        op_start, _ = recorded_spans(audit_sinks.operator)
        ten_start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        _await_db_span(audit_sinks.operator, None, "batch_write_to_db", seconds=60, since=op_start)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
        tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _, all_tenant = recorded_spans(audit_sinks.tenant, ten_start)
        names: Final = sorted(str(span["name"]) for span in all_tenant)
        assert "redis" in _db_systems(tenant_spans), f"redis spans missing at tenant: {names}"
        assert not any("batch_write_to_db" in name for name in names), f"spend writer reached tenant: {names}"
