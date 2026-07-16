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


def test_dynamic_cred_presets_tag_exporter_with_matching_owner(monkeypatch):
    """Each dynamic-credential preset must tag the exporter it contributes with
    its own callback name, so per-request tenant routing
    (``TenantTracerCache``) applies that integration's credentials only to its
    own exporter and never bleeds them onto a co-configured backend.
    """
    from litellm.integrations.otel.presets import (
        DYNAMIC_HEADERS_BY_CALLBACK,
        PRESET_BY_CALLBACK,
    )

    monkeypatch.setenv("ARIZE_SPACE_ID", "S")
    monkeypatch.setenv("ARIZE_API_KEY", "K")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("WANDB_API_KEY", "w")
    monkeypatch.setenv("WANDB_PROJECT_ID", "entity/project")

    from litellm.integrations.otel.model.config import ExporterOwner

    for callback_name in DYNAMIC_HEADERS_BY_CALLBACK:
        cfg = PRESET_BY_CALLBACK[callback_name]()
        owners = {e.owner for e in cfg.exporters}
        assert ExporterOwner(callback_name) in owners, (
            f"{callback_name} preset did not tag its exporter with "
            f"owner={callback_name!r}; tenant credentials would leak across "
            f"exporters. owners present: {owners}"
        )


def test_langfuse_preset_builds_otlp_exporter_without_env_creds(monkeypatch):
    """Regression: dynamic team/key/org Langfuse (no proxy ``LANGFUSE_*`` env)
    must still yield an ``otlp_http`` exporter so per-request credentials have an
    OTLP destination to be stamped onto. The preset previously required env creds
    (it raised without them), so the logger fell back to the console exporter and
    dynamic Langfuse spans were printed to the proxy instead of delivered."""
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.langfuse import langfuse_preset

    for var in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_OTEL_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = langfuse_preset()
    langfuse_exporters = [
        e for e in cfg.exporters if e.owner == ExporterOwner.LANGFUSE_OTEL
    ]
    assert len(langfuse_exporters) == 1
    spec = langfuse_exporters[0]
    assert spec.kind == "otlp_http"
    assert spec.endpoint == "https://us.cloud.langfuse.com/api/public/otel"
    assert spec.headers is None


def test_langfuse_preset_uses_env_creds_for_static_headers(monkeypatch):
    """The static/global path (proxy ``LANGFUSE_*`` env) keeps working: the
    exporter carries the env-derived endpoint and Basic auth header."""
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.langfuse import langfuse_preset

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.delenv("LANGFUSE_OTEL_HOST", raising=False)

    cfg = langfuse_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.LANGFUSE_OTEL)
    assert spec.kind == "otlp_http"
    assert spec.endpoint == "https://cloud.langfuse.com/api/public/otel"
    assert spec.headers is not None and spec.headers.startswith("Authorization=Basic ")


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


def test_agentops_endpoint_points_at_live_host():
    # Regression: the OTLP endpoint must be the resolvable AgentOps host. The
    # deprecated otlp.agentops.cloud domain no longer resolves (NXDOMAIN), so
    # spans silently failed to export with a NameResolutionError. Pin the host
    # so a typo or stale domain can never ship again.
    assert _AGENTOPS_ENDPOINT == "https://otlp.agentops.ai/v1/traces"
    assert "agentops.cloud" not in _AGENTOPS_ENDPOINT
