"""
Tests for the enterprise billing-metrics recorder and its factory.

These verify the license gate, the missing-config and missing-cert disable
paths, the OTLP/HTTP exporter wiring (client cert+key authenticate us to the
collector's mTLS-terminating front end; CA override optional for private
collectors), the metric attribute mapping, and that recording produces the
expected OTLP counter via an in-memory reader.
"""

import os
import socket
import stat
from pathlib import Path
from typing import Dict, List, Optional

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
    yield
    bm.shutdown_billing_metrics_recorder()


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
    """Builds a real MeterProvider, so the exporter is stubbed: the live one
    resolves the collector and opens a TLS connection during the shutdown flush.
    The getaddrinfo spy keeps that stub from being quietly dropped later."""
    _set_full_env(monkeypatch, tmp_path)
    monkeypatch.setattr(bm, "OTLPMetricExporter", _fake_exporter_class({}))

    resolved: List[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def _spy_getaddrinfo(host, port, *args, **kwargs):
        resolved.append(str(host))
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _spy_getaddrinfo)

    recorder = bm.build_billing_metrics_recorder(
        premium=True, license_data={"user_id": "org-1"}, litellm_version="1.0"
    )
    assert isinstance(recorder, bm.BillingMetricsRecorder)
    recorder.record(category=BillableCategory.LLM, route="/chat/completions", status_code=200, model_id=None)
    bm.shutdown_billing_metrics_recorder()

    assert [host for host in resolved if "collector.example" in host] == []


def test_building_the_recorder_logs_an_affirmative_line(monkeypatch, tmp_path):
    """
    Every disable path logs; a successful build must log too. Otherwise an
    operator cannot tell a metering component from one that silently returned
    None, which is how an unlicensed component looks healthy while exporting
    nothing.
    """
    _set_full_env(monkeypatch, tmp_path)
    monkeypatch.setenv(bm.EXPORT_INTERVAL_ENV, "5000")
    monkeypatch.setattr(bm, "OTLPMetricExporter", _fake_exporter_class({}))

    infos: List[str] = []
    monkeypatch.setattr(bm.verbose_proxy_logger, "info", lambda msg, *args: infos.append(msg % args if args else msg))

    recorder = bm.build_billing_metrics_recorder(premium=True, license_data={"user_id": "org-1"}, litellm_version="1.0")

    assert recorder is not None
    joined = "\n".join(infos)
    assert "https://collector.example:4317" in joined
    assert "5000" in joined


def test_unlicensed_build_does_not_warn(monkeypatch, tmp_path):
    """Unlicensed is the common OSS case; warning there would be pure noise."""
    _set_full_env(monkeypatch, tmp_path)

    warnings: List[str] = []
    monkeypatch.setattr(bm.verbose_proxy_logger, "warning", lambda msg, *args: warnings.append(str(msg)))

    assert bm.build_billing_metrics_recorder(premium=False, license_data=None, litellm_version="1.0") is None
    assert warnings == []


def test_shutdown_flushes_active_recorder_once(monkeypatch, tmp_path):
    """The shutdown hook must flush the recorder the factory built (buffered
    counts are lost on restart otherwise) and be idempotent for repeat calls."""
    _set_full_env(monkeypatch, tmp_path)
    shutdowns = []

    class _SpyProvider:
        def get_meter(self, name):
            return MeterProvider().get_meter(name)

        def shutdown(self, timeout_millis=None):
            shutdowns.append(timeout_millis)

    monkeypatch.setattr(bm, "build_mtls_meter_provider", lambda config: _SpyProvider())
    recorder = bm.build_billing_metrics_recorder(premium=True, license_data=None, litellm_version="1.0")
    assert recorder is not None

    bm.shutdown_billing_metrics_recorder()
    bm.shutdown_billing_metrics_recorder()
    assert shutdowns == [bm.SHUTDOWN_FLUSH_TIMEOUT_MS]


def test_shutdown_without_active_recorder_is_noop():
    bm.shutdown_billing_metrics_recorder()


# ── Config loading ────────────────────────────────────────────────────────────


def test_load_config_carries_license_id(monkeypatch, tmp_path):
    _set_full_env(monkeypatch, tmp_path)
    config = bm.load_billing_metrics_config(license_data={"user_id": "org-42"}, litellm_version="9.9")
    assert config is not None and config.license_id == "org-42" and config.litellm_version == "9.9"


def test_load_config_with_empty_string_env_is_disabled(monkeypatch, tmp_path):
    """An env var set to the empty string is as unusable as an unset one and
    must disable metering rather than produce a config with a blank endpoint."""
    paths = _write_certs(tmp_path)
    monkeypatch.setenv(bm.ENDPOINT_ENV, "")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, paths[bm.CLIENT_CERT_ENV])
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, paths[bm.CLIENT_KEY_ENV])
    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None


_CLIENT_CERT_PEM = "-----BEGIN CERTIFICATE-----\nclient-cert-body\n-----END CERTIFICATE-----"
_CLIENT_KEY_PEM = "-----BEGIN PRIVATE KEY-----\nclient-key-body\n-----END PRIVATE KEY-----"
_CA_CERT_PEM = "-----BEGIN CERTIFICATE-----\nca-body\n-----END CERTIFICATE-----"


def test_load_config_materializes_inline_pem_content(monkeypatch):
    """
    ECS and Cloud Run inject secrets as env content, not as mounted files, so the
    cert env vars must accept PEM directly. The exporter takes paths, so the PEM
    is written to disk and the config points at those files.
    """
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, _CLIENT_CERT_PEM)
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, _CLIENT_KEY_PEM)
    monkeypatch.setenv(bm.CA_CERT_ENV, _CA_CERT_PEM)

    config = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")

    assert config is not None
    assert config.ca_cert_path is not None
    written = {
        config.client_cert_path: _CLIENT_CERT_PEM,
        config.client_key_path: _CLIENT_KEY_PEM,
        config.ca_cert_path: _CA_CERT_PEM,
    }
    for path, pem in written.items():
        assert path != pem, "config must carry a file path, not the PEM itself"
        assert os.path.isfile(path)
        assert Path(path).read_text(encoding="utf-8") == f"{pem}\n"

    # The private key must not be world- or group-readable.
    assert stat.S_IMODE(os.stat(config.client_key_path).st_mode) == 0o600


def test_load_config_accepts_a_mix_of_pem_content_and_file_paths(monkeypatch, tmp_path):
    """A deployment may mount the CA but inject the client credentials inline."""
    paths = _write_certs(tmp_path)
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, _CLIENT_CERT_PEM)
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, _CLIENT_KEY_PEM)
    monkeypatch.setenv(bm.CA_CERT_ENV, paths[bm.CA_CERT_ENV])

    config = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")

    assert config is not None
    assert config.ca_cert_path == paths[bm.CA_CERT_ENV]
    assert Path(config.client_cert_path).read_text(encoding="utf-8") == f"{_CLIENT_CERT_PEM}\n"


def test_load_config_leaves_file_paths_untouched(monkeypatch, tmp_path):
    """Path-valued env vars keep working; nothing is copied or rewritten."""
    paths = _set_full_env(monkeypatch, tmp_path)

    config = bm.load_billing_metrics_config(license_data=None, litellm_version="1.0")

    assert config is not None
    assert config.client_cert_path == paths[bm.CLIENT_CERT_ENV]
    assert config.client_key_path == paths[bm.CLIENT_KEY_ENV]
    assert config.ca_cert_path == paths[bm.CA_CERT_ENV]


def test_load_config_with_inline_pem_disabled_when_unwritable(monkeypatch):
    """A failure to materialize the PEM disables metering instead of raising."""
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, _CLIENT_CERT_PEM)
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, _CLIENT_KEY_PEM)

    def _explode(prefix=None):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(bm.tempfile, "mkdtemp", _explode)

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None


def test_load_config_never_logs_credential_values(monkeypatch):
    """
    A value that is neither a readable path nor `-----BEGIN`-prefixed PEM is
    still secret material. The disable warning must name the env vars, never
    echo their contents, or a malformed key lands in the proxy logs.
    """
    secret_material = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ-not-pem-prefixed"
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, secret_material)
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, secret_material)

    logged: List[str] = []

    def _capture(msg, *args):
        logged.append(msg % args if args else msg)

    monkeypatch.setattr(bm.verbose_proxy_logger, "warning", _capture)

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None

    joined = "\n".join(logged)
    assert secret_material not in joined
    assert bm.CLIENT_CERT_ENV in joined and bm.CLIENT_KEY_ENV in joined


def test_load_config_with_empty_pem_env_is_disabled(monkeypatch):
    """Empty stays empty: an unset secret must not be mistaken for inline PEM."""
    monkeypatch.setenv(bm.ENDPOINT_ENV, "https://collector.example:4317")
    monkeypatch.setenv(bm.CLIENT_CERT_ENV, "")
    monkeypatch.setenv(bm.CLIENT_KEY_ENV, "")

    assert bm.load_billing_metrics_config(license_data=None, litellm_version="1.0") is None


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


def _fake_exporter_class(captured: Dict[str, object]) -> type:
    """A no-network stand-in for OTLPMetricExporter. Tests that build a real
    MeterProvider must install this: the real exporter resolves the collector
    host and opens a TLS connection on the reader's first export and on the
    shutdown flush."""

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

    return _FakeExporter


def test_meter_provider_wires_client_cert_into_http_exporter(tmp_path, monkeypatch):
    """Client cert+key authenticate us at the collector's mTLS front end; CA override rides certificate_file."""
    captured: Dict[str, object] = {}

    monkeypatch.setattr(bm, "OTLPMetricExporter", _fake_exporter_class(captured))
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
