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


def test_agentops_preset_without_key_omits_exporter(monkeypatch):
    # With no API key the lazy-auth exporter has nothing to mint a JWT from, so
    # the preset must not contribute a global exporter at all (it would otherwise
    # fail every export). Admin-owned destinations carry their own credentials.
    monkeypatch.delenv("AGENTOPS_API_KEY", raising=False)
    cfg = agentops_preset()
    assert [e for e in cfg.exporters if e.kind == _AGENTOPS_EXPORTER_KIND] == []


def test_arize_preset_without_credentials_omits_exporter(monkeypatch):
    # Arize's OTLP ingestion rejects unauthenticated exports (PERMISSION_DENIED),
    # so with no Arize credentials the preset must not contribute a credential-less
    # global exporter pointed at the Arize cloud.
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.arize import arize_preset

    for var in (
        "ARIZE_SPACE_ID",
        "ARIZE_SPACE_KEY",
        "ARIZE_API_KEY",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = arize_preset()
    assert [e for e in cfg.exporters if e.owner == ExporterOwner.ARIZE_AX] == []

    monkeypatch.setenv("ARIZE_SPACE_ID", "S")
    monkeypatch.setenv("ARIZE_API_KEY", "K")
    cfg = arize_preset()
    assert [e for e in cfg.exporters if e.owner == ExporterOwner.ARIZE_AX] != []


def test_phoenix_preset_without_config_omits_exporter(monkeypatch):
    # Unconfigured Phoenix defaults to http://localhost:6006; the preset must not
    # contribute that exporter unless Phoenix is actually configured (cloud key or
    # collector endpoint), so admin-owned-only setups don't export to localhost.
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.phoenix import (
        _PHOENIX_ENV_VARS,
        phoenix_preset,
    )

    for var in _PHOENIX_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cfg = phoenix_preset()
    assert [e for e in cfg.exporters if e.owner == ExporterOwner.ARIZE_PHOENIX] == []

    monkeypatch.setenv("PHOENIX_API_KEY", "px-key")
    cfg = phoenix_preset()
    assert [e for e in cfg.exporters if e.owner == ExporterOwner.ARIZE_PHOENIX] != []


def test_agentops_exporter_factory_is_registered():
    assert _AGENTOPS_EXPORTER_KIND in providers._EXPORTER_FACTORIES


def test_destination_routable_presets_tag_exporter_with_matching_owner(monkeypatch):
    """Each destination-routable preset must tag the exporter it contributes with
    its own callback name, so per-tenant routing (``TenantTracerCache``) points
    that integration's admin destination at its own exporter only and never
    rewrites a co-configured backend's exporter.
    """
    from litellm.integrations.otel.presets.destinations import (
        OTEL_V2_DESTINATION_CALLBACKS,
    )
    from litellm.integrations.otel.presets import PRESET_BY_CALLBACK

    monkeypatch.setenv("ARIZE_SPACE_ID", "S")
    monkeypatch.setenv("ARIZE_API_KEY", "K")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("WANDB_API_KEY", "w")
    monkeypatch.setenv("WANDB_PROJECT_ID", "entity/project")

    from litellm.integrations.otel.model.config import ExporterOwner

    for callback_name in OTEL_V2_DESTINATION_CALLBACKS:
        cfg = PRESET_BY_CALLBACK[callback_name]()
        owners = {e.owner for e in cfg.exporters}
        assert ExporterOwner(callback_name) in owners, (
            f"{callback_name} preset did not tag its exporter with "
            f"owner={callback_name!r}; tenant credentials would leak across "
            f"exporters. owners present: {owners}"
        )


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


def test_credential_mandatory_presets_raise_without_creds_by_default(monkeypatch):
    # weave/langfuse/levo are credential-mandatory: with no global env creds and no
    # opt-in to degrade, the preset must RAISE so a misconfigured global callback
    # (e.g. ``callbacks: ["weave_otel"]`` with no keys) fails loud at startup, the
    # same error story as before V2 landed. _maybe_construct_otel_v2 relies on this
    # raise to defer to the legacy path.
    from litellm.integrations.otel.presets.langfuse import langfuse_preset
    from litellm.integrations.otel.presets.levo import levo_preset
    from litellm.integrations.otel.presets.weave import weave_preset

    for var in (
        "WANDB_API_KEY",
        "WANDB_PROJECT_ID",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LEVOAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    for preset in (weave_preset, langfuse_preset, levo_preset):
        with pytest.raises(Exception):
            preset()


def test_credential_mandatory_presets_degrade_when_allowed(monkeypatch):
    # When an admin-owned destination is the reason for construction it carries its
    # own per-tenant credentials, so the preset is called with
    # allow_missing_credentials=True and must degrade to an exporter-less config
    # (keeping its mappers) instead of raising -- otherwise the destination never gets
    # a v2 logger and its gen-AI span falls to the generic global logger.
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.langfuse import langfuse_preset
    from litellm.integrations.otel.presets.levo import levo_preset
    from litellm.integrations.otel.presets.weave import weave_preset

    for var in (
        "WANDB_API_KEY",
        "WANDB_PROJECT_ID",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LEVOAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    for preset, owner in (
        (weave_preset, ExporterOwner.WEAVE_OTEL),
        (langfuse_preset, ExporterOwner.LANGFUSE_OTEL),
        (levo_preset, ExporterOwner.LEVO),
    ):
        cfg = preset(allow_missing_credentials=True)  # must not raise
        assert [e for e in cfg.exporters if e.owner == owner] == []


def test_weave_preset_with_creds_contributes_exporter(monkeypatch):
    from litellm.integrations.otel.model.config import ExporterOwner
    from litellm.integrations.otel.presets.weave import weave_preset

    monkeypatch.setenv("WANDB_API_KEY", "w-key")
    monkeypatch.setenv("WANDB_PROJECT_ID", "entity/project")
    cfg = weave_preset()
    assert [e for e in cfg.exporters if e.owner == ExporterOwner.WEAVE_OTEL] != []


def test_generic_preset_needs_no_global_env_and_emits_genai(monkeypatch):
    # Acceptance #2: the generic preset is vendor-neutral and admin-destination-only --
    # it must build with NO global OTEL env vars, never raise, carry the standard genai
    # (+legacy) mappers, and contribute NO vendor exporter (the per-destination exporter
    # is appended by the router).
    from litellm.integrations.otel.presets.generic import generic_preset

    for var in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_ENDPOINT", "OTEL_EXPORTER"):
        monkeypatch.delenv(var, raising=False)
    cfg = generic_preset()  # must not raise, no env needed
    assert "genai" in cfg.mapper_names
    # no vendor (owned) exporter contributed -- only the base/global passthrough, if any
    assert all(e.owner is None for e in cfg.exporters)
    # the degrade flag is accepted (Preset protocol) and irrelevant -- still builds
    assert generic_preset(allow_missing_credentials=True) is not None
