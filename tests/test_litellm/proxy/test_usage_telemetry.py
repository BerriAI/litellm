"""
Tests for the opt-in OSS usage telemetry recorder and its wiring helpers.

These exercise the real OpenTelemetry SDK through an InMemoryMetricReader, so
what they assert is the actual exported data: counter names, attribute sets,
and aggregated values. Nothing the proxy sends is mocked.
"""

import json
import uuid

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm.proxy import usage_telemetry as ut
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in (ut.ENABLED_ENV, ut.ENDPOINT_ENV, ut.EXPORT_INTERVAL_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    ut.shutdown_usage_telemetry_recorder()


def _data_points(reader: InMemoryMetricReader, metric_name: str):
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    return list(metric.data.data_points)
    return []


def _all_data_points(reader: InMemoryMetricReader):
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
        for point in metric.data.data_points
    ]


def _recorder_with_reader() -> tuple[ut.UsageTelemetryRecorder, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    return ut.UsageTelemetryRecorder(MeterProvider(metric_readers=[reader])), reader


# ── Middleware sink: litellm.usage.requests ────────────────────────────────────


def test_record_aggregates_by_route_and_status_class():
    recorder, reader = _recorder_with_reader()

    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=500)

    points = _data_points(reader, "litellm.usage.requests")
    by_class = {point.attributes["http.response.status_class"]: point.value for point in points}
    assert by_class == {"2xx": 2, "5xx": 1}
    for point in points:
        assert point.attributes["http.route"] == "/chat/completions"
        assert point.attributes["litellm.endpoint.category"] == "llm"


# ── Success callback: llm_requests / tokens / spend ────────────────────────────

_SUCCESS_PAYLOAD = {
    "model": "gpt-4.1",
    "custom_llm_provider": "openai",
    "call_type": "acompletion",
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "response_cost": 0.0123,
    # sensitive/identifying fields that must never appear as attributes
    "model_group": "my-secret-alias",
    "api_base": "https://customer-proxy.example.com",
    "metadata": {
        "user_api_key": "sk-secret-key-value",
        "user_api_key_team_id": "team-secret",
        "user_api_key_user_id": "user-secret",
    },
}


async def test_success_event_exports_only_anonymous_attributes():
    recorder, reader = _recorder_with_reader()

    await recorder.async_log_success_event(
        kwargs={"standard_logging_object": dict(_SUCCESS_PAYLOAD)},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    llm_points = _data_points(reader, "litellm.usage.llm_requests")
    assert len(llm_points) == 1
    assert llm_points[0].value == 1
    assert dict(llm_points[0].attributes) == {
        "litellm.model": "gpt-4.1",
        "litellm.provider": "openai",
        "litellm.call_type": "acompletion",
    }

    token_points = _data_points(reader, "litellm.usage.tokens")
    by_kind = {point.attributes["litellm.token.kind"]: point.value for point in token_points}
    assert by_kind == {"prompt": 10, "completion": 5}

    spend_points = _data_points(reader, "litellm.usage.spend_usd")
    assert len(spend_points) == 1
    assert spend_points[0].value == pytest.approx(0.0123)

    leaked = [
        value
        for point in _all_data_points(reader)
        for value in point.attributes.values()
        if value
        in ("sk-secret-key-value", "team-secret", "user-secret", "my-secret-alias", "https://customer-proxy.example.com")
    ]
    assert leaked == []


async def test_success_event_with_missing_payload_records_nothing():
    recorder, reader = _recorder_with_reader()

    await recorder.async_log_success_event(kwargs={}, response_obj=None, start_time=None, end_time=None)
    await recorder.async_log_success_event(
        kwargs={"standard_logging_object": None}, response_obj=None, start_time=None, end_time=None
    )
    await recorder.async_log_success_event(
        kwargs={"standard_logging_object": "not-a-dict"}, response_obj=None, start_time=None, end_time=None
    )
    await recorder.async_log_success_event(
        kwargs={"standard_logging_object": {"model": "gpt-4.1"}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert _all_data_points(reader) == []


# ── Env flag and factory ───────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["true", "TRUE", "1"])
def test_usage_telemetry_enabled_true_values(monkeypatch, value):
    monkeypatch.setenv(ut.ENABLED_ENV, value)
    assert ut.usage_telemetry_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "yes", ""])
def test_usage_telemetry_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv(ut.ENABLED_ENV, value)
    assert ut.usage_telemetry_enabled() is False


def test_usage_telemetry_enabled_unset_is_false():
    assert ut.usage_telemetry_enabled() is False


def test_build_recorder_returns_none_when_disabled_and_builds_no_provider():
    builds = []
    assert (
        ut.build_usage_telemetry_recorder(
            litellm_version="1.0", instance_id="i-1", provider_factory=lambda config: builds.append(config)
        )
        is None
    )
    assert builds == []


def test_build_recorder_uses_injected_provider_factory(monkeypatch):
    monkeypatch.setenv(ut.ENABLED_ENV, "true")
    built: dict[str, ut.UsageTelemetryConfig] = {}

    reader = InMemoryMetricReader()

    def _factory(config: ut.UsageTelemetryConfig) -> MeterProvider:
        built["config"] = config
        return MeterProvider(metric_readers=[reader])

    recorder = ut.build_usage_telemetry_recorder(
        litellm_version="9.9", instance_id="inst-1", provider_factory=_factory
    )
    assert isinstance(recorder, ut.UsageTelemetryRecorder)
    assert built["config"].litellm_version == "9.9"
    assert built["config"].instance_id == "inst-1"
    ut.shutdown_usage_telemetry_recorder()


def test_build_recorder_returns_none_when_provider_factory_raises(monkeypatch):
    monkeypatch.setenv(ut.ENABLED_ENV, "true")

    def _explode(config: ut.UsageTelemetryConfig) -> MeterProvider:
        raise OSError("no network")

    assert (
        ut.build_usage_telemetry_recorder(litellm_version="1.0", instance_id="i-1", provider_factory=_explode)
        is None
    )


def test_export_interval_invalid_falls_back(monkeypatch):
    monkeypatch.setenv(ut.ENABLED_ENV, "true")
    monkeypatch.setenv(ut.EXPORT_INTERVAL_ENV, "not-a-number")
    config = ut.load_usage_telemetry_config(litellm_version="1.0", instance_id="i-1")
    assert config.export_interval_ms == ut.DEFAULT_EXPORT_INTERVAL_MS


def test_metrics_endpoint_appends_signal_path():
    assert ut._metrics_endpoint("http://127.0.0.1:4318") == "http://127.0.0.1:4318/v1/metrics"
    assert ut._metrics_endpoint("http://127.0.0.1:4318/v1/metrics") == "http://127.0.0.1:4318/v1/metrics"


# ── Sink composition ──────────────────────────────────────────────────────────


class _SpySink:
    def __init__(self, raises: bool = False) -> None:
        self.calls = []
        self._raises = raises

    def record(self, *, category, route, status_code) -> None:
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append((category, route, status_code))


def test_compose_sinks_empty_returns_none():
    assert ut.compose_gateway_request_sinks(None, None) is None


def test_compose_sinks_single_returns_it():
    sink = _SpySink()
    assert ut.compose_gateway_request_sinks(None, sink) is sink


def test_compose_sinks_fans_out_and_isolates_failures():
    raising = _SpySink(raises=True)
    spy = _SpySink()
    sink = ut.compose_gateway_request_sinks(raising, spy)
    assert sink is not None
    sink.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    assert spy.calls == [(BillableCategory.LLM, "/chat/completions", 200)]


# ── Instance id resolution ────────────────────────────────────────────────────


class _FakeConfigTable:
    def __init__(self, row=None) -> None:
        self._row = row
        self.created = []

    async def find_unique(self, *, where):
        return self._row

    async def create(self, *, data):
        self.created.append(data)
        self._row = type("Row", (), {"param_value": data["param_value"]})()


class _FakePrisma:
    def __init__(self, row=None) -> None:
        self.db = type("DB", (), {"litellm_config": _FakeConfigTable(row)})()


class _Row:
    def __init__(self, param_value) -> None:
        self.param_value = param_value


async def test_resolve_instance_id_without_db_returns_uuid():
    resolved = await ut.resolve_instance_id(None)
    assert str(uuid.UUID(resolved)) == resolved


async def test_resolve_instance_id_returns_existing():
    prisma = _FakePrisma(row=_Row({"instance_id": "existing-id"}))
    assert await ut.resolve_instance_id(prisma) == "existing-id"


async def test_resolve_instance_id_persists_new_id():
    prisma = _FakePrisma(row=None)
    resolved = await ut.resolve_instance_id(prisma)
    assert str(uuid.UUID(resolved)) == resolved
    (created,) = prisma.db.litellm_config.created
    assert created["param_name"] == ut.INSTANCE_ID_CONFIG_KEY
    assert json.loads(created["param_value"]) == {"instance_id": resolved}


async def test_resolve_instance_id_falls_back_on_db_error():
    class _ExplodingTable:
        async def find_unique(self, *, where):
            raise RuntimeError("db down")

    prisma = type("P", (), {"db": type("DB", (), {"litellm_config": _ExplodingTable()})()})()
    resolved = await ut.resolve_instance_id(prisma)
    assert str(uuid.UUID(resolved)) == resolved
