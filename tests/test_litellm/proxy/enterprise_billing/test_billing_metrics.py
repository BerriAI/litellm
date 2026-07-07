"""
Tests for the enterprise billing-metrics recorder and its factory.

These verify the license gate, the missing-config and missing-cert disable
paths, the OTLP/HTTP exporter wiring (client cert+key authenticate us to the
collector's mTLS-terminating front end; CA override optional for private
collectors), the metric attribute mapping, and that recording produces the
expected OTLP counter via an in-memory reader.
"""

from pathlib import Path
from typing import Dict, Optional

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm.proxy.enterprise_billing import billing_metrics as bm
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory

_ENV_VARS = (
    bm.ENDPOINT_ENV,
    bm.CLIENT_CERT_ENV,
    bm.CLIENT_KEY_ENV,
    bm.CA_CERT_ENV,
    bm.EXPORT_INTERVAL_ENV,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_certs(tmp_path: Path) -> Dict[str, str]:
    files = {
        bm.CA_CERT_ENV: ("ca.pem", b"ca-bytes"),
        bm.CLIENT_CERT_ENV: ("client.pem", b"client-cert-bytes"),
        bm.CLIENT_KEY_ENV: ("client.key", b"client-key-bytes"),
    }
    paths = {}
    for env_name, (filename, content) in files.items():
        path = tmp_path / filename
        path.write_bytes(content)
        paths[env_name] = str(path)
    return paths


def _set_full_env(monkeypatch, tmp_path: Path) -> Dict[str, str]:
    paths = _write_certs(tmp_path)
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CA_CERT_ENV, paths[bm.CA_CERT_ENV])
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, paths[bm.CLIENT_CERT_ENV])
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, paths[bm.CLIENT_KEY_ENV])
    return paths


def _config(tmp_path: Path, license_id: Optional[str] = "org-1") -> bm.BillingMetricsConfig:
    paths = _write_certs(tmp_path)
    return bm.BillingMetricsConfig(
        endpoint="https://collector.example:4317",
        client_cert_path=paths[bm.CLIENT_CERT_ENV],
        client_key_path=paths[bm.CLIENT_KEY_ENV],
        ca_cert_path=paths[bm.CA_CERT_ENV],
        export_interval_ms=60_000,
        litellm_version="1.2.3",
        license_id=license_id,
    )


# ── Factory gating ────────────────────────────────────────────────────────────


def test_not_premium_returns_none(tmp_path, monkeypatch):
    _set_full_env(monkeypatch, tmp_path)
    assert bm.build_billing_metrics_recorder(premium=False, license_data=None, litellm_version="1.0") is None


def test_premium_without_config_returns_none(monkeypatch):
    assert bm.build_billing_metrics_recorder(premium=True, license_data={"user_id": "x"}, litellm_version="1.0") is None


def test_premium_with_missing_cert_files_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CA_CERT_ENV, str(tmp_path / "missing-ca.pem"))
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, str(tmp_path / "missing-cert.pem"))
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, str(tmp_path / "missing-key.pem"))
    assert bm.build_billing_metrics_recorder(premium=True, license_data=None, litellm_version="1.0") is None


def test_premium_with_full_config_builds_recorder(monkeypatch, tmp_path):
    _set_full_env(monkeypatch, tmp_path)
    recorder = bm.build_billing_metrics_recorder(
        premium=True, license_data={"user_id": "org-1"}, litellm_version="1.0"
    )
    assert isinstance(recorder, bm.BillingMetricsRecorder)


# ── Config loading ────────────────────────────────────────────────────────────


def test_load_config_carries_license_id(monkeypatch, tmp_path):
    _set_full_env(monkeypatch, tmp_path)
    config = bm.load_billing_metrics_config(license_data={"user_id": "org-42"}, litellm_version="9.9")
    assert config is not None and config.license_id == "org-42" and config.litellm_version == "9.9"


def test_export_interval_default_and_override(monkeypatch):
    assert bm._export_interval_ms() == bm.DEFAULT_EXPORT_INTERVAL_MS
    monkeypatch.setenv(bm.EXPORT_INTERVAL_ENV, "5000")
    assert bm._export_interval_ms() == 5000
    monkeypatch.setenv(bm.EXPORT_INTERVAL_ENV, "not-a-number")
    assert bm._export_interval_ms() == bm.DEFAULT_EXPORT_INTERVAL_MS


# ── OTLP/HTTP exporter wiring ─────────────────────────────────────────────────


def test_metrics_endpoint_appends_signal_path():
    assert bm._metrics_endpoint("https://telemetry.example.com") == "https://telemetry.example.com/v1/metrics"
    assert bm._metrics_endpoint("https://telemetry.example.com/") == "https://telemetry.example.com/v1/metrics"
    assert bm._metrics_endpoint("https://telemetry.example.com/v1/metrics") == "https://telemetry.example.com/v1/metrics"


def test_meter_provider_wires_client_cert_into_http_exporter(tmp_path, monkeypatch):
    """Client cert+key authenticate us at the collector's mTLS front end; CA override rides certificate_file."""
    captured = {}

    class _FakeExporter:
        # PeriodicExportingMetricReader probes these on the exporter it wraps.
        _preferred_temporality: dict = {}
        _preferred_aggregation: dict = {}

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def export(self, *args, **kwargs):
            return None

        def shutdown(self, *args, **kwargs):
            return None

        def force_flush(self, *args, **kwargs):
            return True

    monkeypatch.setattr(bm, "OTLPMetricExporter", _FakeExporter)
    config = _config(tmp_path)
    provider = bm.build_mtls_meter_provider(config)
    provider.shutdown()

    assert captured["endpoint"] == "https://collector.example:4317/v1/metrics"
    assert captured["client_certificate_file"] == config.client_cert_path
    assert captured["client_key_file"] == config.client_key_path
    assert captured["certificate_file"] == config.ca_cert_path


def test_load_config_without_ca_is_valid(monkeypatch, tmp_path):
    """The production collector presents a public web-PKI cert: no CA override required."""
    paths = _write_certs(tmp_path)
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://telemetry.example.com")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, paths[bm.CLIENT_CERT_ENV])
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, paths[bm.CLIENT_KEY_ENV])
    config = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")
    assert config is not None and config.ca_cert_path is None


# ── Resource and metric attributes ────────────────────────────────────────────


def test_resource_attributes_include_license_id(tmp_path):
    attrs = bm._resource_attributes(_config(tmp_path, license_id="org-7"))
    assert attrs["service.name"] == "litellm-proxy"
    assert attrs["litellm.version"] == "1.2.3"
    assert attrs["litellm.license.id"] == "org-7"


def test_resource_attributes_omit_license_id_when_absent(tmp_path):
    attrs = bm._resource_attributes(_config(tmp_path, license_id=None))
    assert "litellm.license.id" not in attrs


def test_billable_attributes_with_model_id():
    attrs = bm._billable_attributes(BillableCategory.LLM, "/chat/completions", 200, "deploy-3")
    assert attrs == {
        "litellm.endpoint.category": "llm",
        "http.route": "/chat/completions",
        "http.response.status_code": 200,
        "litellm.model_id": "deploy-3",
    }


def test_billable_attributes_omit_model_id_when_none():
    attrs = bm._billable_attributes(BillableCategory.MCP, "/mcp", 200, None)
    assert "litellm.model_id" not in attrs


# ── End-to-end recording via in-memory reader ─────────────────────────────────


def _counter_points(reader: InMemoryMetricReader):
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == bm.METRIC_NAME:
                    return list(metric.data.data_points)
    return []


def test_record_increments_counter_with_attributes():
    reader = InMemoryMetricReader()
    recorder = bm.BillingMetricsRecorder(MeterProvider(metric_readers=[reader]))

    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200, model_id="m1")
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200, model_id="m1")
    recorder.record(category=BillableCategory.MCP, route="/mcp", status_code=200, model_id=None)

    points = _counter_points(reader)
    by_category = {point.attributes["litellm.endpoint.category"]: point.value for point in points}
    assert by_category["llm"] == 2
    assert by_category["mcp"] == 1
