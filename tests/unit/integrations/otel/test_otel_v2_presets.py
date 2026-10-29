"""Preset tests. Focused on the AgentOps JWT fetch, which must never block the
event loop: the preset does no network I/O, and a custom exporter mints the JWT
lazily on its first export (in the BatchSpanProcessor worker thread)."""

import json
from collections.abc import Callable, Mapping
from io import StringIO
from typing import Final

import httpx
import pytest
import requests_mock
import respx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult

from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.plumbing import providers
from litellm.integrations.otel.presets import agentops as agentops_mod
from litellm.integrations.otel.presets.agentops import (
    _AGENTOPS_AUTH_ENDPOINT,
    _AGENTOPS_ENDPOINT,
    _AGENTOPS_EXPORTER_KIND,
    _build_agentops_exporter,
    _fetch_agentops_jwt,
    agentops_preset,
)
from litellm.integrations.otel.presets.arize import arize_preset
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.newrelic import newrelic_preset
from litellm.integrations.otel.presets.phoenix import phoenix_preset
from litellm.integrations.otel.presets.signoz import signoz_preset
from litellm.integrations.otel.presets.weave import weave_preset

_PRESET_ENV_VARS: Final = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_OTEL_HOST",
    "ARIZE_SPACE_ID",
    "ARIZE_SPACE_KEY",
    "ARIZE_API_KEY",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "WANDB_API_KEY",
    "WANDB_PROJECT_ID",
    "LEVOAI_API_KEY",
    "LEVOAI_ORG_ID",
    "LEVOAI_WORKSPACE_ID",
    "LEVOAI_COLLECTOR_URL",
    "PHOENIX_API_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_COLLECTOR_PROJECT_NAME",
    "SIGNOZ_INGESTION_ENDPOINT",
    "SIGNOZ_INGESTION_KEY",
    "NEW_RELIC_LICENSE_KEY",
    "NEW_RELIC_REGION",
    "NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED",
    "AGENTOPS_API_KEY",
    "AGENTOPS_SERVICE_NAME",
    "AGENTOPS_ENVIRONMENT",
    "OTEL_EXPORTER",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_HEADERS",
    "OTEL_EXPORTER_OTLP_HEADERS",
)

_PRESET_CASES: Final = (
    pytest.param(
        langfuse_preset,
        {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        id="langfuse",
    ),
    pytest.param(arize_preset, {"ARIZE_SPACE_ID": "space", "ARIZE_API_KEY": "key"}, id="arize"),
    pytest.param(weave_preset, {"WANDB_API_KEY": "key", "WANDB_PROJECT_ID": "entity/project"}, id="weave"),
    pytest.param(
        levo_preset,
        {
            "LEVOAI_API_KEY": "key",
            "LEVOAI_ORG_ID": "org",
            "LEVOAI_WORKSPACE_ID": "workspace",
            "LEVOAI_COLLECTOR_URL": "http://levo.test/v1/traces",
        },
        id="levo",
    ),
    pytest.param(phoenix_preset, {"PHOENIX_API_KEY": "key"}, id="phoenix"),
    pytest.param(
        signoz_preset,
        {"SIGNOZ_INGESTION_ENDPOINT": "http://signoz.test/v1/traces", "SIGNOZ_INGESTION_KEY": "key"},
        id="signoz",
    ),
    pytest.param(newrelic_preset, {"NEW_RELIC_LICENSE_KEY": "key"}, id="newrelic"),
    pytest.param(agentops_preset, {"AGENTOPS_API_KEY": "key"}, id="agentops"),
)


def _clear_preset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PRESET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _capture_console_output(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output: Final = StringIO()

    class _CapturingConsoleSpanExporter(ConsoleSpanExporter):
        def __init__(self) -> None:
            super().__init__(out=output)

    monkeypatch.setattr(providers, "ConsoleSpanExporter", _CapturingConsoleSpanExporter)
    return output


def _provider_for_console_behavior(preset: Callable[[], OpenTelemetryV2Config]) -> TracerProvider:
    config: Final = preset()
    exporters: Final = tuple(
        spec if spec.kind.lower() == "console" else ExporterSpec(kind="in_memory") for spec in config.exporters
    )
    safe_config: Final = config.model_copy(update={"exporters": exporters})
    return providers.build_tracer_provider(safe_config)


@pytest.mark.parametrize(("preset", "credentials"), _PRESET_CASES)
def test_credentialed_presets_do_not_export_to_default_console(
    preset: Callable[[], OpenTelemetryV2Config],
    credentials: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_preset_env(monkeypatch)
    console_output: Final = _capture_console_output(monkeypatch)
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    provider: Final = _provider_for_console_behavior(preset)
    with provider.get_tracer("litellm").start_as_current_span("preset") as span:
        span.set_attribute("preset", "credentialed")
    assert console_output.getvalue() == ""


def test_credentialed_preset_keeps_explicit_console_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_preset_env(monkeypatch)
    console_output: Final = _capture_console_output(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("OTEL_EXPORTER", "console")
    provider: Final = _provider_for_console_behavior(langfuse_preset)
    with provider.get_tracer("litellm").start_as_current_span("explicit-console") as span:
        span.set_attribute("preset", "console")
    assert "explicit-console" in console_output.getvalue()


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


def _agentops_exporter():
    return _build_agentops_exporter(
        ExporterSpec(
            kind=_AGENTOPS_EXPORTER_KIND,
            endpoint=_AGENTOPS_ENDPOINT,
            options={"api_key": "ak"},
        )
    )


def test_agentops_exporter_mints_the_jwt_once_on_first_export_and_sends_it_with_every_batch():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    with respx.mock() as auth, requests_mock.Mocker() as collector:
        token_route = auth.post(_AGENTOPS_AUTH_ENDPOINT).respond(200, json={"token": "jwt-xyz"})
        collector.post(_AGENTOPS_ENDPOINT, status_code=200)
        exporter = _agentops_exporter()
        assert token_route.call_count == 0, "building the exporter must not fetch the JWT"

        results = [exporter.export(()), exporter.export(())]

    assert results == [SpanExportResult.SUCCESS, SpanExportResult.SUCCESS]
    assert [json.loads(call.request.content) for call in token_route.calls] == [{"api_key": "ak"}]
    assert [request.headers.get("Authorization") for request in collector.request_history] == [
        "Bearer jwt-xyz",
        "Bearer jwt-xyz",
    ]


def test_agentops_exporter_still_exports_without_auth_when_the_jwt_fetch_fails():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    with respx.mock() as auth, requests_mock.Mocker() as collector:
        auth.post(_AGENTOPS_AUTH_ENDPOINT).mock(side_effect=httpx.ConnectError("auth down"))
        collector.post(_AGENTOPS_ENDPOINT, status_code=200)

        result = _agentops_exporter().export(())

    assert result is SpanExportResult.SUCCESS
    assert [request.headers.get("Authorization") for request in collector.request_history] == [None]


def test_fetch_jwt_uses_owned_client_not_shared_pool(monkeypatch):
    """The fetch owns a short-lived client and closes it, rather than closing
    the process-wide cached ``get_httpx_client`` pool shared by other callers."""
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
    assert not hasattr(agentops_mod, "get_httpx_client")

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


def test_newrelic_preset_reads_env_license_key(monkeypatch):
    monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "env-license-key")
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.newrelic import newrelic_preset

    cfg = newrelic_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.NEWRELIC)
    assert spec.kind == "otlp_http"
    assert spec.endpoint == "https://otlp.nr-data.net"
    assert spec.headers == "api-key=env-license-key"
    assert spec.requires_headers is True
    assert "genai" in cfg.mapper_names


def test_newrelic_preset_without_key_still_contributes_owned_spec(monkeypatch):
    # The owned spec is the stamping target for per-team credentials, so it must
    # exist even with no operator env key; requires_headers keeps the keyless
    # copy from ever exporting.
    monkeypatch.delenv("NEW_RELIC_LICENSE_KEY", raising=False)
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.newrelic import newrelic_preset

    cfg = newrelic_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.NEWRELIC)
    assert spec.headers is None
    assert spec.requires_headers is True


def test_newrelic_preset_operator_region_and_content_knob(monkeypatch):
    monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "env-license-key")
    monkeypatch.setenv("NEW_RELIC_REGION", "EU")
    monkeypatch.setenv("NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED", "true")
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.newrelic import newrelic_preset

    cfg = newrelic_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.NEWRELIC)
    assert spec.endpoint == "https://otlp.eu01.nr-data.net"
    assert cfg.capture_span_content is True

    monkeypatch.setenv("NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED", "false")
    assert newrelic_preset().capture_span_content is False


def test_newrelic_preset_unset_content_knob_keeps_default(monkeypatch):
    monkeypatch.delenv("NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    from litellm.integrations.otel.presets.newrelic import newrelic_preset

    assert newrelic_preset().capture_span_content is False


def test_signoz_preset_reads_env_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("SIGNOZ_INGESTION_ENDPOINT", "https://ingest.eu.signoz.cloud:443")
    monkeypatch.setenv("SIGNOZ_INGESTION_KEY", "env-ingestion-key")
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.signoz import signoz_preset

    cfg = signoz_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.SIGNOZ)
    assert spec.kind == "otlp_http"
    assert spec.endpoint == "https://ingest.eu.signoz.cloud:443"
    assert spec.headers == "signoz-ingestion-key=env-ingestion-key"
    assert spec.requires_headers is True
    assert "genai" in cfg.mapper_names


def test_signoz_preset_without_key_is_self_hosted(monkeypatch):
    # A self-hosted collector accepts unauthenticated OTLP, so requiring headers
    # would drop exports that would have succeeded.
    monkeypatch.setenv("SIGNOZ_INGESTION_ENDPOINT", "http://signoz-collector.internal:4318")
    monkeypatch.delenv("SIGNOZ_INGESTION_KEY", raising=False)
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.signoz import signoz_preset

    cfg = signoz_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.SIGNOZ)
    assert spec.endpoint == "http://signoz-collector.internal:4318"
    assert spec.headers is None
    assert spec.requires_headers is False


def test_signoz_preset_has_no_default_endpoint(monkeypatch):
    # No region table and no default host: the preset never invents a destination.
    monkeypatch.delenv("SIGNOZ_INGESTION_ENDPOINT", raising=False)
    monkeypatch.delenv("SIGNOZ_INGESTION_KEY", raising=False)
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.signoz import signoz_preset

    cfg = signoz_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.SIGNOZ)
    assert spec.endpoint is None


def test_signoz_preset_endpoint_passed_through_verbatim(monkeypatch):
    # The plumbing appends the signal path, so pre-appending would double it.
    monkeypatch.setenv("SIGNOZ_INGESTION_ENDPOINT", "https://ingest.us.signoz.cloud:443/v1/traces")
    monkeypatch.delenv("SIGNOZ_INGESTION_KEY", raising=False)
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.plumbing.providers import _otlp_traces_endpoint
    from litellm.integrations.otel.presets.signoz import signoz_preset

    cfg = signoz_preset()
    spec = next(e for e in cfg.exporters if e.owner == ExporterOwner.SIGNOZ)
    assert spec.endpoint == "https://ingest.us.signoz.cloud:443/v1/traces"
    assert _otlp_traces_endpoint(spec.endpoint) == "https://ingest.us.signoz.cloud:443/v1/traces"


def test_signoz_preset_accepts_the_factory_call_shape(monkeypatch):
    monkeypatch.setenv("SIGNOZ_INGESTION_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.delenv("SIGNOZ_INGESTION_KEY", raising=False)
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets import PRESET_BY_CALLBACK

    cfg = PRESET_BY_CALLBACK["signoz"](allow_missing_credentials=True)
    assert any(e.owner == ExporterOwner.SIGNOZ for e in cfg.exporters)
