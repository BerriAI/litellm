import builtins

import pytest

from litellm.integrations.opentelemetry import OpenTelemetry


def _make_otel(exporter: str) -> OpenTelemetry:
    otel = OpenTelemetry.__new__(OpenTelemetry)
    otel.OTEL_EXPORTER = exporter
    otel.OTEL_ENDPOINT = None
    otel.OTEL_HEADERS = None
    return otel


def _block_grpc_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("opentelemetry.exporter.otlp.proto.grpc"):
            raise ImportError("grpc exporter missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_should_raise_helpful_error_when_grpc_exporter_missing_for_traces(
    monkeypatch: pytest.MonkeyPatch,
):
    _block_grpc_imports(monkeypatch)
    otel = _make_otel("otlp_grpc")

    with pytest.raises(ImportError, match=r"litellm\[grpc\]"):
        otel._get_span_processor()


def test_should_raise_helpful_error_when_grpc_exporter_missing_for_logs(
    monkeypatch: pytest.MonkeyPatch,
):
    _block_grpc_imports(monkeypatch)
    otel = _make_otel("otlp_grpc")

    with pytest.raises(ImportError, match=r"litellm\[grpc\]"):
        otel._get_log_exporter()
