"""Per-request multi-tenant credential routing (V1 parity)."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from opentelemetry.trace import NoOpTracer

from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets import dynamic_otlp_headers
from litellm.integrations.otel.plumbing.routing import TenantTracerCache


def _cache(callback_name, exporters=None):
    cfg = OpenTelemetryV2Config(exporters=exporters or [ExporterSpec(kind="in_memory")])
    return TenantTracerCache(cfg, callback_name, "litellm")


# --- header builders mirror the V1 construct_dynamic_otel_headers overrides --- #


def test_arize_dynamic_headers():
    headers = dynamic_otlp_headers(
        "arize", {"arize_space_id": "S", "arize_api_key": "K"}
    )
    assert headers == {"arize-space-id": "S", "api_key": "K"}


def test_arize_space_key_overrides_space_id():
    headers = dynamic_otlp_headers(
        "arize", {"arize_space_id": "S", "arize_space_key": "SK"}
    )
    assert headers == {"arize-space-id": "SK"}


def test_langfuse_dynamic_headers_need_both_keys():
    assert dynamic_otlp_headers("langfuse_otel", {"langfuse_public_key": "pk"}) is None
    headers = dynamic_otlp_headers(
        "langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )
    assert headers is not None and "Authorization" in headers


def test_weave_dynamic_headers():
    headers = dynamic_otlp_headers(
        "weave_otel", {"wandb_api_key": "w", "weave_project_id": "p"}
    )
    assert headers is not None
    assert "Authorization" in headers and headers["project_id"] == "p"


def test_non_participating_callbacks_have_no_routing():
    # Phoenix subclasses the base in V1 (no override) → no dynamic routing.
    assert dynamic_otlp_headers("arize_phoenix", {"arize_api_key": "K"}) is None
    assert dynamic_otlp_headers("langtrace", {"arize_api_key": "K"}) is None
    assert dynamic_otlp_headers(None, {"arize_api_key": "K"}) is None


def test_no_dynamic_params_is_no_routing():
    assert dynamic_otlp_headers("arize", None) is None
    assert dynamic_otlp_headers("arize", {}) is None


# --- TenantTracerCache routes + caches a TracerProvider per credential set --- #


def test_provider_cached_per_credential_set():
    cache = _cache("arize")
    default = NoOpTracer()
    creds_a = {"arize_space_id": "S", "arize_api_key": "K"}
    creds_b = {"arize_space_id": "S2", "arize_api_key": "K2"}

    cache.tracer_for(default, creds_a)
    cache.tracer_for(default, creds_a)  # same set → reuse, no new provider
    assert len(cache._providers) == 1
    cache.tracer_for(default, creds_b)  # new set → new provider
    assert len(cache._providers) == 2


def test_provider_cache_is_bounded_and_evicts_lru(monkeypatch):
    # The cache key derives from request-supplied dynamic credentials, so it
    # must be bounded — an unbounded cache lets a caller spawn one provider (and
    # its background exporter thread) per unique credential set. On overflow the
    # least-recently-used provider is evicted and shut down.
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 2)
    shut_down = []
    monkeypatch.setattr(
        routing_mod, "_shutdown_provider", lambda p: shut_down.append(p)
    )

    cache = _cache("arize")
    default = NoOpTracer()

    def creds(space):
        return {"arize_space_id": space, "arize_api_key": "K"}

    cache.tracer_for(default, creds("1"))
    cache.tracer_for(default, creds("2"))
    cache.tracer_for(default, creds("1"))  # touch "1" → "2" is now LRU
    cache.tracer_for(default, creds("3"))  # overflow → evict "2"

    assert len(cache._providers) == 2
    assert len(shut_down) == 1  # exactly the evicted provider was shut down


def test_no_dynamic_params_uses_default_tracer():
    cache = _cache("arize")
    default = NoOpTracer()
    assert cache.tracer_for(default, {}) is default
    assert cache._providers == {}


def test_non_participating_callback_uses_default_tracer():
    cache = _cache("arize_phoenix")
    default = NoOpTracer()
    assert cache.tracer_for(default, {"arize_api_key": "K"}) is default
    assert cache._providers == {}


def test_dynamic_headers_applied_to_otlp_exporter_only():
    cache = _cache(
        "arize",
        exporters=[
            ExporterSpec(kind="otlp_http", owner="arize"),
            ExporterSpec(kind="in_memory", owner="arize"),
        ],
    )
    new_cfg = cache._config_with_headers({"arize-space-id": "S", "api_key": "K"})
    otlp, in_mem = new_cfg.exporters
    assert otlp.headers == "arize-space-id=S,api_key=K"
    assert in_mem.headers is None  # console/in_memory left untouched


def test_dynamic_headers_do_not_leak_to_other_owners_exporter():
    """A tenant's Arize credentials must never be stamped onto a co-configured
    exporter owned by a different backend (a self-hosted collector, Langfuse).

    Regression for the cross-backend credential leak: ``_config_with_headers``
    used to rewrite the headers of every OTLP exporter, so one request carrying
    a team's Arize key clobbered the base collector's and Langfuse's headers
    with that key.
    """
    cache = _cache(
        "arize",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://self-hosted-collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="https://cloud.langfuse.com/api/public/otel",
                headers="Authorization=Basic base-langfuse",
                owner="langfuse_otel",
            ),
            ExporterSpec(
                kind="otlp_grpc",
                endpoint="https://otlp.arize.com/v1",
                headers="space_id=base,api_key=base",
                owner="arize",
            ),
        ],
    )
    new_cfg = cache._config_with_headers(
        {"arize-space-id": "TEAMX", "api_key": "TEAMX_KEY"}
    )
    by_owner = {e.owner: e.headers for e in new_cfg.exporters}
    assert by_owner["arize"] == "arize-space-id=TEAMX,api_key=TEAMX_KEY"
    assert by_owner[None] == "x=base-collector"
    assert by_owner["langfuse_otel"] == "Authorization=Basic base-langfuse"
