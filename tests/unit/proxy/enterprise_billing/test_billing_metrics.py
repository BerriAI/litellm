from collections.abc import Generator, Mapping
from typing import Final

import pytest
import requests
import requests_mock
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Sum
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy._types import EnterpriseLicenseData
from litellm.proxy.enterprise_billing import billing_metrics as bm
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory


class _RecordedRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    url: str | None
    body: str | bytes | None
    headers: dict[str, str]


def _request_history(http_mock: requests_mock.Mocker) -> tuple[_RecordedRequest, ...]:
    return TypeAdapter(tuple[_RecordedRequest, ...]).validate_python(http_mock.request_history)


def _metric_category(attributes: Mapping[str, object] | None) -> str:
    if attributes is None:
        return ""
    return str(attributes.get("litellm.endpoint.category"))


_ENV_VARS: Final = (
    bm.ENDPOINT_ENV,
    bm.CLIENT_ID_ENV,
    bm.CLIENT_SECRET_ENV,
    bm.TOKEN_URL_ENV,
    bm.EXPORT_INTERVAL_ENV,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    bm.shutdown_billing_metrics_recorder()


def _set_full_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4318")
    monkeypatch.setenv(bm.CLIENT_ID_ENV, "client-id")
    monkeypatch.setenv(bm.CLIENT_SECRET_ENV, "client-secret")
    monkeypatch.setenv(bm.TOKEN_URL_ENV, "https://auth.example/oauth2/token")


def _config(license_id: str | None = "org-1") -> bm.BillingMetricsConfig:
    return bm.BillingMetricsConfig(
        endpoint="https://collector.example:4318",
        token_url="https://auth.example/oauth2/token",
        client_id="client-id",
        client_secret="client-secret",
        export_interval_ms=60_000,
        litellm_version="1.2.3",
        license_id=license_id,
    )


@pytest.mark.parametrize("missing_env", (bm.ENDPOINT_ENV, bm.CLIENT_ID_ENV, bm.CLIENT_SECRET_ENV, bm.TOKEN_URL_ENV))
def test_missing_required_env_disables_metrics_without_logging_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing_env: str,
) -> None:
    _set_full_env(monkeypatch)
    monkeypatch.setenv(bm.CLIENT_SECRET_ENV, "secret-for-test")
    monkeypatch.delenv(missing_env)

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None
    assert f"licensed deployment missing config ({missing_env})" in caplog.text
    assert "secret-for-test" not in caplog.text


def test_empty_required_env_disables_metrics(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _set_full_env(monkeypatch)
    monkeypatch.setenv(bm.ENDPOINT_ENV, "")

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None
    assert bm.ENDPOINT_ENV in caplog.text


@pytest.mark.parametrize(
    ("env_name", "url"),
    (
        (bm.ENDPOINT_ENV, "http://collector.example:4318"),
        (bm.TOKEN_URL_ENV, "http://auth.example/oauth2/token"),
    ),
)
def test_http_urls_disable_metrics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_name: str,
    url: str,
) -> None:
    _set_full_env(monkeypatch)
    monkeypatch.setenv(env_name, url)

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None
    assert f"{env_name} must use HTTPS" in caplog.text


def test_valid_config_contains_bearer_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_full_env(monkeypatch)

    license_data: Final[EnterpriseLicenseData] = {"user_id": "org-1"}
    config: Final = bm.load_billing_metrics_config(license_data=license_data, litellm_version="1.0")
    assert config is not None

    assert config == bm.BillingMetricsConfig(
        endpoint="https://collector.example:4318",
        token_url="https://auth.example/oauth2/token",
        client_id="client-id",
        client_secret="client-secret",
        export_interval_ms=bm.DEFAULT_EXPORT_INTERVAL_MS,
        litellm_version="1.0",
        license_id="org-1",
    )
    assert "client-secret" not in repr(config)
    assert "license-secret" not in repr(config)


def test_not_premium_returns_none_without_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _set_full_env(monkeypatch)

    assert bm.build_billing_metrics_recorder(premium=False, license_data=None, litellm_version="1.0") is None
    assert "Enterprise billing metrics disabled" not in caplog.text


def test_premium_without_config_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert bm.build_billing_metrics_recorder(premium=True, license_data={"user_id": "x"}, litellm_version="1.0") is None


def test_premium_builds_recorder_and_logs_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _set_full_env(monkeypatch)
    caplog.set_level("INFO", logger=bm.verbose_proxy_logger.name)
    token_url: Final = "https://auth.example/oauth2/token"
    export_url: Final = "https://collector.example:4318/v1/metrics"
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(token_url, json={"access_token": "access-token", "expires_in": 3600})
        http_mock.post(export_url, status_code=200)
        recorder: Final = bm.build_billing_metrics_recorder(
            premium=True, license_data={"user_id": "org-1"}, litellm_version="1.0"
        )
        assert recorder is not None
        assert "Enterprise billing metrics enabled: exporting to https://collector.example:4318 every 60000 ms" in caplog.text
        bm.shutdown_billing_metrics_recorder()


class _ShutdownCountingMeterProvider(MeterProvider):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls = 0

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        self.shutdown_calls += 1
        super().shutdown(timeout_millis=timeout_millis)


def test_shutdown_flushes_active_recorder_once(monkeypatch: pytest.MonkeyPatch) -> None:
    provider: Final = _ShutdownCountingMeterProvider()
    _set_full_env(monkeypatch)

    def create_provider(config: bm.BillingMetricsConfig) -> MeterProvider:
        return provider

    monkeypatch.setattr(bm, "build_bearer_meter_provider", create_provider)
    recorder: Final = bm.build_billing_metrics_recorder(premium=True, license_data=None, litellm_version="1.0")
    assert recorder is not None

    bm.shutdown_billing_metrics_recorder()
    bm.shutdown_billing_metrics_recorder()

    assert provider.shutdown_calls == 1


class _FakeClock:
    def __init__(self, now: float) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def set(self, now: float) -> None:
        self._now = now


def test_bearer_auth_fetches_and_caches_client_credentials_token() -> None:
    token_url: Final = "https://auth.example/oauth2/token"
    export_url: Final = "https://collector.example/v1/metrics"
    auth: Final = bm.BillingMetricsBearerAuth(
        client_id="client-id",
        client_secret="client-secret",
        token_url=token_url,
        clock=lambda: 100.0,
    )
    session: Final = requests.Session()
    session.auth = auth
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(token_url, json={"access_token": "access-token", "expires_in": 3600})
        http_mock.post(export_url, status_code=200)
        session.post(export_url, data=b"metrics")
        session.post(export_url, data=b"metrics")

    request_records: Final = _request_history(http_mock)
    token_requests: Final = tuple(request for request in request_records if request.url == token_url)
    export_requests: Final = tuple(request for request in request_records if request.url == export_url)
    assert len(token_requests) == 1
    assert token_requests[0].headers["Authorization"].startswith("Basic ")
    assert token_requests[0].body == "grant_type=client_credentials&scope=telemetry%2Fwrite"
    assert tuple(request.headers["Authorization"] for request in export_requests) == (
        "Bearer access-token",
        "Bearer access-token",
    )


def test_bearer_auth_refreshes_before_token_expiry() -> None:
    token_url: Final = "https://auth.example/oauth2/token"
    export_url: Final = "https://collector.example/v1/metrics"
    clock: Final = _FakeClock(1000.0)
    auth: Final = bm.BillingMetricsBearerAuth(
        client_id="client-id",
        client_secret="client-secret",
        token_url=token_url,
        clock=clock,
    )
    session: Final = requests.Session()
    session.auth = auth
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(
            token_url,
            [
                {"json": {"access_token": "first-token", "expires_in": 120}},
                {"json": {"access_token": "second-token", "expires_in": 120}},
            ],
        )
        http_mock.post(export_url, status_code=200)
        session.post(export_url, data=b"metrics")
        clock.set(1059.0)
        session.post(export_url, data=b"metrics")
        clock.set(1060.0)
        session.post(export_url, data=b"metrics")

    request_records: Final = _request_history(http_mock)
    token_requests: Final = tuple(request for request in request_records if request.url == token_url)
    export_requests: Final = tuple(request for request in request_records if request.url == export_url)
    assert len(token_requests) == 2
    assert tuple(request.headers["Authorization"] for request in export_requests) == (
        "Bearer first-token",
        "Bearer first-token",
        "Bearer second-token",
    )


def test_bearer_export_sets_authorization_header() -> None:
    config: Final = _config()
    token_url: Final = config.token_url
    export_url: Final = f"{config.endpoint}/v1/metrics"
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(token_url, json={"access_token": "export-token", "expires_in": 3600})
        http_mock.post(export_url, status_code=200)
        provider: Final = bm.build_bearer_meter_provider(config)
        try:
            provider.get_meter("test").create_counter("requests").add(1)
            assert provider.force_flush(timeout_millis=1000)
        finally:
            provider.shutdown(timeout_millis=1000)

    export_requests: Final = tuple(request for request in _request_history(http_mock) if request.url == export_url)
    assert export_requests
    assert all(request.headers["Authorization"] == "Bearer export-token" for request in export_requests)


def test_unauthorized_export_refetches_token_before_next_export() -> None:
    config: Final = _config()
    token_url: Final = config.token_url
    export_url: Final = f"{config.endpoint}/v1/metrics"
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(
            token_url,
            [
                {"json": {"access_token": "first-token", "expires_in": 3600}},
                {"json": {"access_token": "second-token", "expires_in": 3600}},
            ],
        )
        http_mock.post(export_url, [{"status_code": 401}, {"status_code": 200}])
        provider: Final = bm.build_bearer_meter_provider(config)
        try:
            counter: Final = provider.get_meter("test").create_counter("requests")
            counter.add(1)
            provider.force_flush(timeout_millis=1000)
            counter.add(1)
            provider.force_flush(timeout_millis=1000)
        finally:
            provider.shutdown(timeout_millis=1000)

    requests: Final = _request_history(http_mock)
    token_requests: Final = tuple(request for request in requests if request.url == token_url)
    export_requests: Final = tuple(request for request in requests if request.url == export_url)
    assert len(token_requests) == 2
    assert tuple(request.headers["Authorization"] for request in export_requests[:2]) == (
        "Bearer first-token",
        "Bearer second-token",
    )


def test_failed_bearer_token_fetch_does_not_raise_from_export() -> None:
    config: Final = _config()
    token_url: Final = config.token_url
    export_url: Final = f"{config.endpoint}/v1/metrics"
    http_mock: Final = requests_mock.Mocker()
    with http_mock:
        http_mock.post(token_url, status_code=400)
        provider: Final = bm.build_bearer_meter_provider(config)
        try:
            provider.get_meter("test").create_counter("requests").add(1)
            assert provider.force_flush(timeout_millis=1000)
        finally:
            provider.shutdown(timeout_millis=1000)

    request_urls: Final = tuple(request.url for request in _request_history(http_mock))
    assert token_url in request_urls
    assert export_url not in request_urls


def test_export_interval_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_full_env(monkeypatch)
    default_config: Final = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")
    assert default_config is not None
    assert default_config.export_interval_ms == bm.DEFAULT_EXPORT_INTERVAL_MS
    monkeypatch.setenv(bm.EXPORT_INTERVAL_ENV, "5000")
    override_config: Final = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")
    assert override_config is not None
    assert override_config.export_interval_ms == 5000
    monkeypatch.setenv(bm.EXPORT_INTERVAL_ENV, "not-a-number")
    invalid_config: Final = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")
    assert invalid_config is not None
    assert invalid_config.export_interval_ms == bm.DEFAULT_EXPORT_INTERVAL_MS


def test_resource_attributes_include_license_id() -> None:
    attributes: Final = bm.resource_attributes(_config(license_id="org-7"))
    assert attributes == {
        "service.name": "litellm-proxy",
        "litellm.version": "1.2.3",
        "litellm.license.id": "org-7",
    }


def test_resource_attributes_omit_license_id_when_absent() -> None:
    assert "litellm.license.id" not in bm.resource_attributes(_config(license_id=None))


def test_billable_attributes_include_model_id() -> None:
    assert bm.billable_attributes(
        BillableCategory.LLM, "/chat/completions", 200, "deploy-3"
    ) == {
        "litellm.endpoint.category": "llm",
        "http.route": "/chat/completions",
        "http.response.status_code": 200,
        "litellm.model_id": "deploy-3",
    }


def test_billable_attributes_omit_model_id_when_none() -> None:
    attributes: Final = bm.billable_attributes(
        BillableCategory.MCP, "/mcp", 200, None
    )
    assert "litellm.model_id" not in attributes


def _counter_points(reader: InMemoryMetricReader) -> tuple[tuple[str, int], ...]:
    data: Final = reader.get_metrics_data()
    if data is None:
        return ()
    return tuple(
        (_metric_category(point.attributes), int(point.value))
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
        if metric.name == bm.METRIC_NAME and isinstance(metric.data, Sum)
        for point in metric.data.data_points
    )


def test_record_increments_counter_with_attributes() -> None:
    reader: Final = InMemoryMetricReader()
    recorder: Final = bm.BillingMetricsRecorder(MeterProvider(metric_readers=[reader]))

    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200, model_id="m1")
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200, model_id="m1")
    recorder.record(category=BillableCategory.MCP, route="/mcp", status_code=200, model_id=None)

    points: Final = _counter_points(reader)
    by_category: Final = dict(points)
    assert by_category == {"llm": 2, "mcp": 1}
