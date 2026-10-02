from __future__ import annotations

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


def _operator_langfuse(audit_sinks: SpanSinks) -> dict[str, str]:
    return {
        "LANGFUSE_HOST": audit_sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        "OTEL_EXPORTER": "http/json",
        "OTEL_ENDPOINT": audit_sinks.operator,
    }


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


def _trace_spans_when(
    sink_url: str,
    trace_id: str,
    ready: Callable[[tuple[Span, ...]], bool],
    seconds: float = 30,
) -> tuple[Span, ...]:
    spans: Final = eventually(
        lambda: spans_for_trace(recorded_spans(sink_url)[1], trace_id),
        ready,
        seconds=seconds,
    )
    return spans


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


def _assert_tenant_keeps_redis_without_postgres(
    candidate: Gateway, audit_sinks: SpanSinks, langfuse_vars: Mapping[str, JsonValue]
) -> None:
    tenant_start, _ = recorded_spans(audit_sinks.tenant)
    operator_start, _ = recorded_spans(audit_sinks.operator)
    traffic: Final = _drive(candidate, langfuse_vars)
    _await_db_span(audit_sinks.operator, None, "batch_write_to_db", seconds=60, since=operator_start)
    tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
    _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
    systems: Final = _db_systems(_trace_spans(audit_sinks.tenant, tenant_trace, seconds=15))
    assert "redis" in systems, f"redis spans missing at tenant: {systems}"
    _, all_tenant = recorded_spans(audit_sinks.tenant, tenant_start)
    assert "postgresql" not in _db_systems(all_tenant), f"postgresql spans reached tenant: {_db_systems(all_tenant)}"


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


@pytest.mark.timeout(180)
def test_without_excluded_services_the_tenant_still_gets_redis_and_postgres_spans(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, extra=_guardrail_block)
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2) as candidate:
        tenant_start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        _await_db_span(audit_sinks.tenant, None, "batch_write_to_db", seconds=60, since=tenant_start)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis", seconds=60)
        _assert_core_spans_present(_trace_spans(audit_sinks.tenant, tenant_trace, seconds=15))
        _, all_tenant = recorded_spans(audit_sinks.tenant, tenant_start)
        systems: Final = _db_systems(all_tenant)
        assert {"redis", "postgresql"} <= systems, f"datastore spans missing at tenant: {systems}"


@pytest.mark.parametrize("otel", [None, True, "on", "", []], ids=["null", "true", "on", "empty_string", "empty_list"])
@pytest.mark.timeout(180)
def test_a_non_mapping_otel_block_still_publishes_the_tenant_fan_out(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
    otel: JsonValue,
) -> None:
    def with_callback_settings(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["langfuse_otel"]
        config["callback_settings"]["otel"] = otel

    config: Final = _config_with(tmp_path, otel_audit_config, extra=with_callback_settings)
    overrides: Final = {"LITELLM_OTEL_V2": "1", **_operator_langfuse(audit_sinks)}
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis")
        tenant_spans: Final = _trace_spans_when(
            audit_sinks.tenant,
            tenant_trace,
            lambda spans: any(span["kind"] == 2 for span in spans) and "redis" in _db_systems(spans),
            seconds=15,
        )
        assert any(span["kind"] == 2 for span in tenant_spans), "tenant SERVER root span missing"
        assert "redis" in _db_systems(tenant_spans), f"tenant redis span missing: {_db_systems(tenant_spans)}"
        operator_trace: Final = _trace_id(audit_sinks.operator, traffic)
        operator_spans: Final = _trace_spans_when(
            audit_sinks.operator,
            operator_trace,
            lambda spans: any(span["kind"] == 2 for span in spans),
            seconds=15,
        )
        assert any(span["kind"] == 2 for span in operator_spans), "operator SERVER root span missing"


@pytest.mark.parametrize("name", ["EXCLUDED_SERVICES", "excluded_services"])
@pytest.mark.timeout(180)
def test_a_bare_excluded_services_env_var_is_ignored(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
    name: str,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config)
    overrides: Final = {"LITELLM_OTEL_V2": "1", name: "redis,postgres"}
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        tenant_start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _await_db_span(audit_sinks.tenant, tenant_trace, "redis")
        tenant_spans: Final = _trace_spans_when(
            audit_sinks.tenant,
            tenant_trace,
            lambda spans: "redis" in _db_systems(spans),
            seconds=15,
        )
        assert "redis" in _db_systems(tenant_spans), f"redis span missing at tenant: {_db_systems(tenant_spans)}"
        _await_db_span(audit_sinks.tenant, None, "postgresql", since=tenant_start)
        _, all_tenant = recorded_spans(audit_sinks.tenant, tenant_start)
        systems: Final = _db_systems(all_tenant)
        assert {"redis", "postgresql"} <= systems, f"datastore spans missing at tenant: {systems}"


@pytest.mark.timeout(180)
def test_the_documented_env_var_wins_over_a_bare_excluded_services(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config)
    overrides: Final = {
        "LITELLM_OTEL_V2": "1",
        "LITELLM_OTEL_EXCLUDED_SERVICES": "redis",
        "EXCLUDED_SERVICES": "postgres",
    }
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        tenant_start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _await_db_span(audit_sinks.tenant, None, "postgresql", since=tenant_start)
        _, tenant_spans = recorded_spans(audit_sinks.tenant, tenant_start)
        systems: Final = _db_systems(tenant_spans)
        assert "postgresql" in systems, f"postgresql spans missing at tenant: {systems}"
        assert "redis" not in systems, f"redis spans reached tenant: {systems}"


@pytest.mark.parametrize(
    ("env_name", "redis_reaches_tenant"),
    [
        pytest.param("LITELLM_OTEL_EXCLUDED_SERVICES", False, id="exact-case"),
        pytest.param("litellm_otel_excluded_services", True, id="wrong-case"),
    ],
)
@pytest.mark.timeout(180)
def test_case_sensitive_otel_settings_read_only_the_exact_env_name(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: Mapping[str, JsonValue],
    tmp_path: Path,
    env_name: str,
    redis_reaches_tenant: bool,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"_case_sensitive": True})
    overrides: Final = {"LITELLM_OTEL_V2": "1", env_name: "redis"}
    with owned_proxy(
        gateway,
        tmp_path,
        overrides,
        config=config,
        remove_environment=("LITELLM_OTEL_EXCLUDED_SERVICES", "litellm_otel_excluded_services"),
        workers=2,
    ) as candidate:
        tenant_start, _ = recorded_spans(audit_sinks.tenant)
        traffic: Final = _drive(candidate, langfuse_vars)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, traffic)
        if redis_reaches_tenant:
            _await_db_span(audit_sinks.tenant, tenant_trace, "redis")
            _trace_spans_when(
                audit_sinks.tenant,
                tenant_trace,
                lambda spans: "redis" in _db_systems(spans),
                seconds=15,
            )
        else:
            _trace_spans(audit_sinks.tenant, tenant_trace, seconds=15)
        _await_db_span(audit_sinks.tenant, None, "postgresql", seconds=60, since=tenant_start)
        _, tenant_spans = recorded_spans(audit_sinks.tenant, tenant_start)
        systems: Final = _db_systems(tenant_spans)
        assert "postgresql" in systems, f"postgresql spans missing at tenant: {systems}"
        assert ("redis" in systems) is redis_reaches_tenant, (
            f"tenant redis presence={('redis' in systems)}; expected={redis_reaches_tenant}; systems={systems}"
        )


@pytest.mark.timeout(180)
def test_env_ignore_empty_keeps_the_default_service_name(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: Mapping[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"_env_ignore_empty": True})
    overrides: Final = {"LITELLM_OTEL_V2": "1", "OTEL_SERVICE_NAME": ""}
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        traffic: Final = _drive(candidate, langfuse_vars)
        operator_trace: Final = _trace_id(audit_sinks.operator, traffic)
        operator_spans: Final = _trace_spans_when(
            audit_sinks.operator,
            operator_trace,
            lambda spans: any(span["kind"] == 2 for span in spans),
            seconds=15,
        )
        service_names: Final = tuple(span["resource"].get("service.name") for span in operator_spans)
        assert service_names and all(name == "litellm" for name in service_names), (
            f"operator service.name values={service_names}"
        )


@pytest.mark.timeout(180)
def test_env_parse_none_str_reads_a_null_traces_endpoint_as_unset(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: Mapping[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"_env_parse_none_str": "null"})
    overrides: Final = {"LITELLM_OTEL_V2": "1", "OTEL_TRACES_ENDPOINT": "null"}
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        traffic: Final = _drive(candidate, langfuse_vars)
        operator_trace: Final = _trace_id(audit_sinks.operator, traffic)
        operator_spans: Final = _trace_spans_when(
            audit_sinks.operator,
            operator_trace,
            lambda spans: any(span["kind"] == 2 for span in spans),
            seconds=15,
        )
        assert any(span["kind"] == 2 for span in operator_spans), "operator SERVER root span missing"


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


@pytest.mark.timeout(180)
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
        _assert_tenant_keeps_redis_without_postgres(candidate, audit_sinks, langfuse_vars)


@pytest.mark.timeout(180)
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
    overrides: Final = {"LITELLM_OTEL_V2": "1", **_operator_langfuse(audit_sinks)}
    with owned_proxy(gateway, tmp_path, overrides, config=config, workers=2) as candidate:
        _assert_tenant_keeps_redis_without_postgres(candidate, audit_sinks, langfuse_vars)


@pytest.mark.timeout(180)
def test_bogus_excluded_service_logs_error_and_drops_at_proxy_start(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    config: Final = _config_with(tmp_path, otel_audit_config, otel={"excluded_services": ["auth", "postgres"]})
    with owned_proxy_process(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=config, workers=2) as owned:
        assert "'auth' is not a datastore service; ignored" in owned.log.read_text(), owned.log.read_text()[-3000:]
        _assert_tenant_keeps_redis_without_postgres(owned.gateway, audit_sinks, langfuse_vars)


@pytest.mark.timeout(180)
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
        _assert_tenant_keeps_redis_without_postgres(candidate, audit_sinks, langfuse_vars)


@pytest.mark.timeout(180)
def test_bogus_excluded_services_env_logs_and_drops_with_preset_alongside_otel(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    def with_langfuse_otel(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["otel", "langfuse_otel"]

    config: Final = _config_with(tmp_path, otel_audit_config, extra=with_langfuse_otel)
    overrides: Final = {"LITELLM_OTEL_V2": "1", "LITELLM_OTEL_EXCLUDED_SERVICES": "auth,postgres"}
    with owned_proxy_process(gateway, tmp_path, overrides, config=config, workers=2) as owned:
        assert "'auth' is not a datastore service; ignored" in owned.log.read_text(), owned.log.read_text()[-3000:]
        _assert_tenant_keeps_redis_without_postgres(owned.gateway, audit_sinks, langfuse_vars)


@pytest.mark.timeout(180)
def test_bogus_excluded_services_env_logs_and_drops_without_otel_callback(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    langfuse_vars: dict[str, JsonValue],
    tmp_path: Path,
) -> None:
    def presets_only(config: dict) -> None:
        config["litellm_settings"]["callbacks"] = ["langfuse_otel"]

    config: Final = _config_with(tmp_path, otel_audit_config, extra=presets_only)
    overrides: Final = {
        "LITELLM_OTEL_V2": "1",
        "LITELLM_OTEL_EXCLUDED_SERVICES": "auth,postgres",
        **_operator_langfuse(audit_sinks),
    }
    with owned_proxy_process(gateway, tmp_path, overrides, config=config, workers=2) as owned:
        assert "'auth' is not a datastore service; ignored" in owned.log.read_text(), owned.log.read_text()[-3000:]
        _assert_tenant_keeps_redis_without_postgres(owned.gateway, audit_sinks, langfuse_vars)


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
