"""Preset tests. Focused on the AgentOps JWT fetch, which must never block the
event loop: the preset does no network I/O, and a custom exporter mints the JWT
lazily on its first export (in the BatchSpanProcessor worker thread)."""

import httpx
import pytest

from litellm.integrations.otel.plumbing import providers
from litellm.integrations.otel.model.config import ExporterSpec
from litellm.integrations.otel.presets import agentops as agentops_mod
from litellm.integrations.otel.presets.agentops import (
    _AGENTOPS_ENDPOINT,
    _AGENTOPS_EXPORTER_KIND,
    _build_agentops_exporter,
    _fetch_agentops_jwt,
    agentops_preset,
)


def test_agentops_preset_does_no_network_io(monkeypatch):
    # The preset must not fetch the JWT at build time — that would block the
    # event loop during callback construction. It only describes the exporter.
    def _boom(*_a, **_k):
        raise AssertionError("agentops_preset must not fetch the JWT eagerly")

    monkeypatch.setattr(agentops_mod, "_fetch_agentops_jwt", _boom)
    monkeypatch.setenv("AGENTOPS_API_KEY", "ak-123")
    cfg = agentops_preset()
    agentops_exporters = [e for e in cfg.exporters if e.kind == _AGENTOPS_EXPORTER_KIND]
    assert len(agentops_exporters) == 1
    spec = agentops_exporters[0]
    assert spec.endpoint == _AGENTOPS_ENDPOINT
    assert spec.options == {"api_key": "ak-123"}  # carried to the lazy exporter


def test_agentops_preset_without_key_omits_options(monkeypatch):
    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)
    cfg = agentops_preset()
    spec = next(e for e in cfg.exporters if e.kind == _AGENTOPS_EXPORTER_KIND)
    assert spec.options is None


def test_agentops_exporter_factory_is_registered():
    assert _AGENTOPS_EXPORTER_KIND in providers._EXPORTER_FACTORIES


def test_agentops_exporter_mints_jwt_lazily(monkeypatch):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    monkeypatch.setattr(
        agentops_mod, "_fetch_agentops_jwt", lambda _k: {"token": "jwt-xyz"}
    )
    spec = ExporterSpec(
        kind=_AGENTOPS_EXPORTER_KIND,
        endpoint=_AGENTOPS_ENDPOINT,
        options={"api_key": "ak"},
    )
    exporter = _build_agentops_exporter(spec)

    # No auth header until the first export triggers the (off-loop) fetch.
    assert "Authorization" not in exporter._session.headers
    exporter._ensure_authenticated()
    assert exporter._session.headers["Authorization"] == "Bearer jwt-xyz"

    # Cached: a second resolution does not re-fetch.
    calls = []
    monkeypatch.setattr(
        agentops_mod,
        "_fetch_agentops_jwt",
        lambda k: calls.append(k) or {"token": "again"},
    )
    exporter._ensure_authenticated()
    assert calls == []


def test_agentops_exporter_tolerates_fetch_failure(monkeypatch):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    def _raise(_k):
        raise RuntimeError("auth down")

    monkeypatch.setattr(agentops_mod, "_fetch_agentops_jwt", _raise)
    exporter = _build_agentops_exporter(
        ExporterSpec(
            kind=_AGENTOPS_EXPORTER_KIND,
            endpoint=_AGENTOPS_ENDPOINT,
            options={"api_key": "ak"},
        )
    )
    exporter._ensure_authenticated()  # must not raise
    assert "Authorization" not in exporter._session.headers


def test_fetch_jwt_uses_owned_client_not_shared_pool(monkeypatch):
    """The fetch owns a short-lived client and closes it, rather than closing
    the process-wide cached ``_get_httpx_client`` pool shared by other callers."""
    closed = {"n": 0}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "jwt-123"}

    class _FakeClient:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            closed["n"] += 1

        def post(self, *_a, **_k):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    assert not hasattr(agentops_mod, "_get_httpx_client")

    result = _fetch_agentops_jwt("api-key")
    assert result == {"token": "jwt-123"}
    assert closed["n"] == 1  # the owned client was closed
