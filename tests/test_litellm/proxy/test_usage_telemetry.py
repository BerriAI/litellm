"""
Tests for the opt-in OSS usage telemetry recorder and its wiring helpers.

These exercise the real OpenTelemetry SDK through an InMemoryMetricReader, so
what they assert is the actual exported data: counter names, attribute sets,
and aggregated values. Nothing the proxy sends is mocked.
"""

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Metric

import litellm
from litellm.proxy import usage_telemetry as ut
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ut.ENABLED_ENV, ut.ENDPOINT_ENV, ut.EXPORT_INTERVAL_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    ut.shutdown_usage_telemetry_recorder()


def _metric(reader: InMemoryMetricReader, metric_name: str) -> Metric | None:
    data = reader.get_metrics_data()
    if data is None:
        return None
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    return metric
    return None


def _data_points(reader: InMemoryMetricReader, metric_name: str) -> list:
    metric: Final = _metric(reader, metric_name)
    return [] if metric is None else list(metric.data.data_points)


def _all_data_points(reader: InMemoryMetricReader) -> list:
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
    reader: Final = InMemoryMetricReader()
    return ut.UsageTelemetryRecorder(MeterProvider(metric_readers=[reader])), reader


def _config(**overrides: object) -> ut.UsageTelemetryConfig:
    base: Final[dict[str, object]] = {
        "endpoint": "https://telemetry.litellm.ai/v1/metrics",
        "export_interval_ms": 60_000,
        "litellm_version": "1.2.3",
        "instance_id": "inst-1",
    }
    return ut.UsageTelemetryConfig(**{**base, **overrides})


def test_record_aggregates_by_route_and_status_class() -> None:
    recorder, reader = _recorder_with_reader()

    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=500)

    points: Final = _data_points(reader, "litellm.usage.requests")
    by_class: Final = {point.attributes["http.response.status_class"]: point.value for point in points}
    assert by_class == {"2xx": 2, "5xx": 1}
    for point in points:
        assert point.attributes["http.route"] == "/chat/completions"
        assert point.attributes["litellm.endpoint.category"] == "llm"


_PUBLIC_MODEL: Final = next(key for key in ut._PUBLIC_MODELS if "/" not in key)

_SUCCESS_PAYLOAD: Final[dict] = {
    "model": _PUBLIC_MODEL,
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


async def test_success_event_exports_only_anonymous_attributes() -> None:
    recorder, reader = _recorder_with_reader()

    await recorder.async_log_success_event(
        kwargs={"standard_logging_object": dict(_SUCCESS_PAYLOAD)},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    llm_points: Final = _data_points(reader, "litellm.usage.llm_requests")
    assert len(llm_points) == 1
    assert llm_points[0].value == 1
    assert dict(llm_points[0].attributes) == {
        "litellm.model": _PUBLIC_MODEL,
        "litellm.provider": "openai",
        "litellm.call_type": "acompletion",
    }

    token_points: Final = _data_points(reader, "litellm.usage.tokens")
    by_kind: Final = {point.attributes["litellm.token.kind"]: point.value for point in token_points}
    assert by_kind == {"prompt": 10, "completion": 5}

    spend_points: Final = _data_points(reader, "litellm.usage.spend_usd")
    assert len(spend_points) == 1
    assert spend_points[0].value == pytest.approx(0.0123)

    leaked: Final = [
        value
        for point in _all_data_points(reader)
        for value in point.attributes.values()
        if value
        in (
            "sk-secret-key-value",
            "team-secret",
            "user-secret",
            "my-secret-alias",
            "https://customer-proxy.example.com",
        )
    ]
    assert leaked == []


async def test_private_model_name_collapses_to_other() -> None:
    """A model name absent from the shipped pricing map could be an internal
    identifier (a fine-tune name, a private deployment alias), so it must not
    be exported; the label falls back to "other"."""
    recorder, reader = _recorder_with_reader()

    await recorder.async_log_success_event(
        kwargs={
            "standard_logging_object": {
                **_SUCCESS_PAYLOAD,
                "model": "my-internal-finetune-v7",
                "custom_llm_provider": "openai",
            }
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    llm_points: Final = _data_points(reader, "litellm.usage.llm_requests")
    assert len(llm_points) == 1
    assert llm_points[0].attributes["litellm.model"] == "other"


def test_model_registered_at_runtime_still_labels_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can add private names to litellm.model_cost for billing
    (register_model, custom pricing config). Those names are private
    identifiers and must still label as "other": only the pricing map
    shipped with the package counts as public."""
    monkeypatch.setitem(litellm.model_cost, "my-private-finetune", {"input_cost_per_token": 0.0})
    assert ut._public_model_label("my-private-finetune", "openai") == "other"
    assert ut._public_model_label("openai/my-private-finetune", "openai") == "other"


def test_shipped_provider_prefixed_model_labels_as_itself() -> None:
    shipped: Final = min(
        key for key in ut._PUBLIC_MODELS if "/" in key and key.partition("/")[2] not in ut._PUBLIC_MODELS
    )
    provider, _, model = shipped.partition("/")
    assert ut._public_model_label(model, provider) == shipped


def test_provider_prefixed_bare_shipped_model_labels_as_bare_name() -> None:
    bare: Final = min(key for key in ut._PUBLIC_MODELS if "/" not in key and f"openai/{key}" not in ut._PUBLIC_MODELS)
    assert ut._public_model_label(f"openai/{bare}", "openai") == bare
    assert ut._public_model_label(f"openai/{bare}", None) == "other"


async def test_success_event_with_missing_payload_records_nothing() -> None:
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


@pytest.mark.parametrize("value", ["true", "TRUE", "1"])
def test_usage_telemetry_enabled_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ut.ENABLED_ENV, value)
    assert ut.usage_telemetry_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "yes", ""])
def test_usage_telemetry_enabled_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ut.ENABLED_ENV, value)
    assert ut.usage_telemetry_enabled() is False


def test_usage_telemetry_enabled_unset_is_false() -> None:
    assert ut.usage_telemetry_enabled() is False


def test_build_recorder_returns_none_when_disabled_and_builds_no_provider() -> None:
    builds: Final[list] = []
    assert (
        ut.build_usage_telemetry_recorder(
            litellm_version="1.0", instance_id="i-1", provider_factory=lambda config: builds.append(config)
        )
        is None
    )
    assert builds == []


def test_build_recorder_uses_injected_provider_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ut.ENABLED_ENV, "true")
    built: Final[dict[str, ut.UsageTelemetryConfig]] = {}

    reader: Final = InMemoryMetricReader()

    def _factory(config: ut.UsageTelemetryConfig) -> MeterProvider:
        built["config"] = config
        return MeterProvider(metric_readers=[reader])

    recorder: Final = ut.build_usage_telemetry_recorder(
        litellm_version="9.9", instance_id="inst-1", provider_factory=_factory
    )
    assert isinstance(recorder, ut.UsageTelemetryRecorder)
    assert built["config"].litellm_version == "9.9"
    assert built["config"].instance_id == "inst-1"
    ut.shutdown_usage_telemetry_recorder()


def test_build_recorder_returns_none_when_provider_factory_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ut.ENABLED_ENV, "true")

    def _explode(config: ut.UsageTelemetryConfig) -> MeterProvider:
        raise OSError("no network")

    assert (
        ut.build_usage_telemetry_recorder(litellm_version="1.0", instance_id="i-1", provider_factory=_explode) is None
    )


def test_export_interval_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ut.ENABLED_ENV, "true")
    monkeypatch.setenv(ut.EXPORT_INTERVAL_ENV, "not-a-number")
    config: Final = ut.load_usage_telemetry_config(litellm_version="1.0", instance_id="i-1")
    assert config.export_interval_ms == ut.DEFAULT_EXPORT_INTERVAL_MS


def test_metrics_endpoint_appends_signal_path() -> None:
    assert ut._metrics_endpoint("http://127.0.0.1:4318") == "http://127.0.0.1:4318/v1/metrics"
    assert ut._metrics_endpoint("http://127.0.0.1:4318/v1/metrics") == "http://127.0.0.1:4318/v1/metrics"


def test_exporter_ignores_otlp_header_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a falsy headers value OTLPMetricExporter falls back to the
    OTEL_EXPORTER_OTLP_*_HEADERS env vars, which would forward a deployment's
    own collector credentials to telemetry.litellm.ai."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer secret-token")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_HEADERS", "x-api-key=secret-key")

    exporter: Final = ut._build_exporter(_config())
    headers: Final[Mapping[str, str]] = exporter._session.headers  # pyright: ignore[reportPrivateUsage]  # asserts the session's real header set

    assert "authorization" not in {key.lower() for key in headers}
    assert "x-api-key" not in {key.lower() for key in headers}
    assert headers["User-Agent"] == "litellm-proxy/1.2.3"


def test_meter_provider_ignores_otel_resource_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource.create merges OTEL_RESOURCE_ATTRIBUTES and OTEL_SERVICE_NAME,
    which would stamp a deployment's host name and service name onto the
    telemetry resource. The plain Resource constructor reads no env."""
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "host.name=secret-host,k8s.pod.name=p1")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "customer-svc")

    provider: Final = ut.build_usage_meter_provider(
        _config(endpoint="http://127.0.0.1:1", litellm_version="9.9", instance_id="i-9")
    )
    attrs: Final = provider._sdk_config.resource.attributes  # pyright: ignore[reportPrivateUsage]  # asserts the resource actually exported

    assert dict(attrs) == {
        "service.name": "litellm-proxy",
        "litellm.version": "9.9",
        "litellm.instance.id": "i-9",
    }


class _SpySink:
    def __init__(self, raises: bool = False) -> None:
        self.calls: list[tuple[BillableCategory, str, int]] = []
        self._raises: Final = raises

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append((category, route, status_code))


def test_compose_sinks_empty_returns_none() -> None:
    assert ut.compose_gateway_request_sinks(None, None) is None


def test_compose_sinks_single_returns_it() -> None:
    sink: Final = _SpySink()
    assert ut.compose_gateway_request_sinks(None, sink) is sink


def test_compose_sinks_fans_out_and_isolates_failures() -> None:
    raising: Final = _SpySink(raises=True)
    spy: Final = _SpySink()
    sink: Final = ut.compose_gateway_request_sinks(raising, spy)
    assert sink is not None
    sink.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200)
    assert spy.calls == [(BillableCategory.LLM, "/chat/completions", 200)]


@dataclass
class _FakeRow:
    param_value: object


@dataclass
class _FakeConfigTable:
    rows: list[_FakeRow] = field(default_factory=list)
    find_calls: int = 0
    created: list[dict] = field(default_factory=list)
    create_raises: bool = False

    async def find_unique(self, *, where: dict) -> _FakeRow | None:
        self.find_calls += 1
        return self.rows[0] if self.rows else None

    async def create(self, *, data: dict) -> None:
        if self.create_raises:
            raise RuntimeError("duplicate key")
        self.created.append(data)
        self.rows.append(_FakeRow(data["param_value"]))


@dataclass
class _FakeDb:
    litellm_config: object


class _ReplicaTable:
    """Stands in for the read replica: any access through it is a bug, because
    a row just written to the primary may not be visible on the replica yet."""

    async def find_unique(self, *, where: dict) -> None:
        raise AssertionError("read went through the replica (db) instead of writer_db")

    async def create(self, *, data: dict) -> None:
        raise AssertionError("write went through the replica (db) instead of writer_db")


@dataclass
class _FakePrisma:
    writer_db: _FakeDb
    db: _FakeDb = field(default_factory=lambda: _FakeDb(_ReplicaTable()))


async def test_resolve_instance_id_without_db_returns_uuid() -> None:
    resolved: Final = await ut.resolve_instance_id(None)
    assert str(uuid.UUID(resolved)) == resolved


async def test_resolve_instance_id_returns_existing() -> None:
    prisma: Final = _FakePrisma(_FakeDb(_FakeConfigTable(rows=[_FakeRow({"instance_id": "existing-id"})])))
    assert await ut.resolve_instance_id(prisma) == "existing-id"


async def test_resolve_instance_id_persists_new_id() -> None:
    prisma: Final = _FakePrisma(_FakeDb(_FakeConfigTable()))
    resolved: Final = await ut.resolve_instance_id(prisma)
    assert str(uuid.UUID(resolved)) == resolved
    (created,) = prisma.writer_db.litellm_config.created
    assert created["param_name"] == ut.INSTANCE_ID_CONFIG_KEY
    assert json.loads(created["param_value"]) == {"instance_id": resolved}


async def test_resolve_instance_id_insert_race_returns_winner_row() -> None:
    """When two workers miss the row and race the create, the loser must adopt
    the winner's persisted id rather than falling back to a per-process uuid.
    Every read goes through writer_db so the winner's row is visible despite
    replica lag."""
    winner: Final = _FakeRow({"instance_id": "winner-id"})

    class _RaceTable:
        def __init__(self) -> None:
            self.find_calls: Final[list] = []

        async def find_unique(self, *, where: dict) -> _FakeRow | None:
            self.find_calls.append(where)
            return None if len(self.find_calls) == 1 else winner

        async def create(self, *, data: dict) -> None:
            raise RuntimeError("duplicate key")

    race_table: Final = _RaceTable()
    prisma: Final = _FakePrisma(_FakeDb(race_table))
    assert await ut.resolve_instance_id(prisma) == "winner-id"
    assert len(race_table.find_calls) == 2


async def test_resolve_instance_id_falls_back_on_db_error() -> None:
    class _ExplodingTable:
        async def find_unique(self, *, where: dict) -> None:
            raise RuntimeError("db down")

    prisma: Final = _FakePrisma(_FakeDb(_ExplodingTable()))
    resolved: Final = await ut.resolve_instance_id(prisma)
    assert str(uuid.UUID(resolved)) == resolved
