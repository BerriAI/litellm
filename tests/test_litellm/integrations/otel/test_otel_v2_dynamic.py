"""Per-request multi-tenant credential routing (V1 parity)."""

import base64


from opentelemetry.trace import NoOpTracer

from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.presets import (
    dynamic_otlp_endpoint,
    dynamic_otlp_headers,
    project_routing_headers,
)
from litellm.integrations.otel.plumbing.providers import parse_headers
from litellm.integrations.otel.plumbing.routing import TenantTracerCache


def _cache(callback_name, exporters=None):
    cfg = OpenTelemetryV2Config(exporters=exporters or [ExporterSpec(kind="in_memory")])
    return TenantTracerCache(cfg, callback_name, "litellm")


# --- header builders mirror the V1 construct_dynamic_otel_headers overrides --- #


def test_arize_dynamic_headers():
    headers = dynamic_otlp_headers("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    assert headers == {"arize-space-id": "S", "api_key": "K"}


def test_arize_space_key_overrides_space_id():
    headers = dynamic_otlp_headers("arize", {"arize_space_id": "S", "arize_space_key": "SK"})
    assert headers == {"arize-space-id": "SK"}


def test_langfuse_dynamic_headers_need_both_keys():
    assert dynamic_otlp_headers("langfuse_otel", {"langfuse_public_key": "pk"}) is None
    headers = dynamic_otlp_headers("langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"})
    assert headers is not None and "Authorization" in headers


def test_langfuse_dynamic_headers_carry_v4_ingestion_version():
    headers = dynamic_otlp_headers("langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"})
    expected_auth = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert headers == {
        "Authorization": expected_auth,
        "x-langfuse-ingestion-version": "4",
    }


def test_weave_dynamic_headers():
    headers = dynamic_otlp_headers("weave_otel", {"wandb_api_key": "w", "weave_project_id": "p"})
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

    cache.route_for(default, creds_a)
    cache.route_for(default, creds_a)  # same set → reuse, no new provider
    assert len(cache._providers) == 1
    cache.route_for(default, creds_b)  # new set → new provider
    assert len(cache._providers) == 2


def test_provider_cache_is_bounded_and_evicts_lru(monkeypatch):
    # The cache key derives from request-supplied dynamic credentials, so it
    # must be bounded — an unbounded cache lets a caller spawn one provider (and
    # its background exporter thread) per unique credential set. On overflow the
    # least-recently-used provider is evicted and shut down.
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 2)
    shut_down = []
    monkeypatch.setattr(routing_mod, "_shutdown_provider", lambda p: shut_down.append(p))

    cache = _cache("arize")
    default = NoOpTracer()

    def creds(space):
        return {"arize_space_id": space, "arize_api_key": "K"}

    # route_for returns a held provider; release models the span closing.
    cache.release(cache.route_for(default, creds("1")).provider)
    cache.release(cache.route_for(default, creds("2")).provider)
    cache.release(cache.route_for(default, creds("1")).provider)  # touch "1" → "2" is now LRU
    cache.release(cache.route_for(default, creds("3")).provider)  # overflow → evict "2"

    assert len(cache._providers) == 2
    assert len(shut_down) == 1  # exactly the evicted provider was shut down


def test_no_dynamic_params_uses_default_tracer():
    cache = _cache("arize")
    default = NoOpTracer()
    assert cache.route_for(default, {}).tracer is default
    assert cache._providers == {}


def test_non_participating_callback_uses_default_tracer():
    cache = _cache("arize_phoenix")
    default = NoOpTracer()
    assert cache.route_for(default, {"arize_api_key": "K"}).tracer is default
    assert cache._providers == {}


def test_dynamic_headers_applied_to_otlp_exporter_only():
    cache = _cache(
        "arize",
        exporters=[
            ExporterSpec(kind="otlp_http", owner="arize"),
            ExporterSpec(kind="in_memory", owner="arize"),
        ],
    )
    new_cfg = cache._routed_config({"arize-space-id": "S", "api_key": "K"}, {})
    otlp, in_mem = new_cfg.exporters
    assert otlp.headers == "arize-space-id=S,api_key=K"
    assert in_mem.headers is None  # console/in_memory left untouched


def test_dynamic_headers_do_not_leak_to_other_owners_exporter():
    """A tenant's Arize credentials must never be stamped onto a co-configured
    exporter owned by a different backend (a self-hosted collector, Langfuse).

    Regression for the cross-backend credential leak: the header rewrite used
    to hit every OTLP exporter, so one request carrying a team's Arize key
    clobbered the base collector's and Langfuse's headers with that key.
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
    new_cfg = cache._routed_config({"arize-space-id": "TEAMX", "api_key": "TEAMX_KEY"}, {})
    by_owner = {e.owner: e.headers for e in new_cfg.exporters}
    assert by_owner["arize"] == "arize-space-id=TEAMX,api_key=TEAMX_KEY"
    assert by_owner[None] == "x=base-collector"
    assert by_owner["langfuse_otel"] == "Authorization=Basic base-langfuse"


# --- per-request Phoenix project routing from trusted key/team config --- #


def _phoenix_cache(kind="otlp_http"):
    return _cache(
        "arize_phoenix",
        exporters=[
            ExporterSpec(
                kind=kind,
                endpoint="http://phoenix:6006",
                headers="Authorization=Bearer phoenix-key",
                owner="arize_phoenix",
            ),
        ],
    )


def test_phoenix_project_headers_precedence_and_blanks():
    assert project_routing_headers("arize_phoenix", {"phoenix_project_name": "team-proj"}) == {
        "x-project-name": "team-proj"
    }
    assert project_routing_headers(
        "arize_phoenix",
        {"phoenix_project_name_override": "override", "phoenix_project_name": "base"},
    ) == {"x-project-name": "override"}
    assert project_routing_headers("arize_phoenix", {"phoenix_project_name": "   "}) == {}
    assert project_routing_headers("arize_phoenix", None) == {}
    # Only Phoenix participates in project routing.
    assert project_routing_headers("arize", {"phoenix_project_name": "p"}) == {}


def test_project_header_appends_and_preserves_phoenix_auth():
    """Regression: routing to a project must not drop the preset's static
    ``Authorization`` header — a replace would break Phoenix auth entirely."""
    cache = _phoenix_cache()
    cfg = cache._routed_config({}, {"x-project-name": "team-proj"})
    (spec,) = cfg.exporters
    parsed = parse_headers(spec.headers)
    assert parsed["authorization"] == "Bearer phoenix-key"
    assert parsed["x-project-name"] == "team-proj"


def test_project_name_with_header_separators_round_trips():
    cache = _phoenix_cache()
    cfg = cache._routed_config({}, {"x-project-name": "my proj, prod=1"})
    (spec,) = cfg.exporters
    parsed = parse_headers(spec.headers)
    assert parsed["x-project-name"] == "my proj, prod=1"
    assert parsed["authorization"] == "Bearer phoenix-key"


def test_project_header_does_not_touch_other_exporters():
    cache = _cache(
        "arize_phoenix",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://phoenix:6006",
                headers="Authorization=Bearer phoenix-key",
                owner="arize_phoenix",
            ),
        ],
    )
    cfg = cache._routed_config({}, {"x-project-name": "team-proj"})
    by_owner = {e.owner: e.headers for e in cfg.exporters}
    assert by_owner[None] == "x=base-collector"
    assert parse_headers(by_owner["arize_phoenix"])["x-project-name"] == "team-proj"


def test_provider_cached_per_project():
    cache = _phoenix_cache()
    default = NoOpTracer()
    routed = cache.route_for(default, None, {"phoenix_project_name": "proj-a"})
    assert routed.tracer is not default
    assert routed.detached is True  # project spans must root their own trace
    cache.route_for(default, None, {"phoenix_project_name": "proj-a"})
    assert len(cache._providers) == 1
    cache.route_for(default, None, {"phoenix_project_name": "proj-b"})
    assert len(cache._providers) == 2
    for provider in cache._providers.values():
        provider.shutdown()


def test_client_dynamic_params_cannot_choose_phoenix_project():
    # ``StandardCallbackDynamicParams`` is populated from client-supplied
    # request metadata; the project may only come from server-set key/team
    # config (the ``auth_metadata`` argument).
    cache = _phoenix_cache()
    default = NoOpTracer()
    assert cache.route_for(default, {"phoenix_project_name": "attacker"}).tracer is default
    assert cache.route_for(default, {"phoenix_project_name_override": "attacker"}).tracer is default
    assert cache._providers == {}


def test_auth_metadata_without_project_uses_default_tracer():
    cache = _phoenix_cache()
    default = NoOpTracer()
    assert cache.route_for(default, None, {"logging_setting": "x"}).tracer is default
    assert cache._providers == {}


def test_grpc_exporter_gets_no_project_routing():
    # ``x-project-name`` is only honored on the OTLP/HTTP endpoint, so a
    # gRPC-only Phoenix exporter stays on the default project (warned once).
    cache = _phoenix_cache(kind="otlp_grpc")
    default = NoOpTracer()
    assert cache.route_for(default, None, {"phoenix_project_name": "proj"}).tracer is default
    assert cache._providers == {}
    assert cache._warned_project_unroutable is True


def test_eviction_defers_shutdown_while_a_span_is_open(monkeypatch):
    # An LLM span opened at pre_call stays open until the close callback; LRU
    # eviction in that window must not stop the provider's processors, or the
    # span is silently dropped at end instead of exported. route_for itself
    # takes the hold, atomically with the cache update, so a concurrent
    # eviction can never shut a just-selected provider down before the caller
    # records its span.
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 1)
    shut_down = []
    monkeypatch.setattr(routing_mod, "_shutdown_provider", lambda p: shut_down.append(p))
    cache = _cache("arize")
    default = NoOpTracer()

    route_a = cache.route_for(default, {"arize_space_id": "A", "arize_api_key": "K"})
    assert route_a.provider is not None
    cache.route_for(default, {"arize_space_id": "B", "arize_api_key": "K"})  # evicts A
    assert shut_down == []  # deferred: A is still held by route_a
    cache.release(route_a.provider)
    assert shut_down == [route_a.provider]


def test_retired_providers_are_capped(monkeypatch):
    # Retiring an evicted provider keeps it, and its exporter thread, alive
    # while a span is open, so retirees need a cap of their own: a caller
    # cycling unique credential sets across calls that never close would
    # otherwise pin one live provider per open call, far past the cache bound.
    # Past the cap the stalest retiree is shut down and its later release is a
    # no-op, while the ones still within the cap keep draining.
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 1)
    monkeypatch.setattr(routing_mod, "_MAX_RETIRED_PROVIDERS", 2)
    shut_down = []
    monkeypatch.setattr(routing_mod, "_shutdown_provider", lambda p: shut_down.append(p))
    cache = _cache("arize")
    default = NoOpTracer()

    # Every route stays held (no release), so each one evicts and retires its
    # predecessor instead of shutting it down.
    routes = [cache.route_for(default, {"arize_space_id": str(i), "arize_api_key": "K"}) for i in range(5)]

    assert len(cache._providers) == 1
    assert len(cache._retired) == 2  # capped, not one retiree per open call
    assert shut_down == [routes[0].provider, routes[1].provider]

    cache.release(routes[0].provider)  # already shut down: no second shutdown
    assert shut_down == [routes[0].provider, routes[1].provider]
    cache.release(routes[2].provider)  # still draining: drains and shuts down
    assert shut_down[-1] is routes[2].provider


def test_release_without_eviction_keeps_provider_alive(monkeypatch):
    from litellm.integrations.otel.plumbing import routing as routing_mod

    shut_down = []
    monkeypatch.setattr(routing_mod, "_shutdown_provider", lambda p: shut_down.append(p))
    cache = _cache("arize")
    route = cache.route_for(NoOpTracer(), {"arize_space_id": "A", "arize_api_key": "K"})
    cache.release(route.provider)
    assert shut_down == []  # still cached, never retired
    cache.release(None)  # default-route release is a no-op


# --- per-request service.name routing from trusted key/team config --- #


def test_tenant_service_name_precedence_and_blanks():
    from litellm.integrations.otel.plumbing.routing import tenant_service_name

    assert tenant_service_name({"otel_service_name": "team-svc"}) == "team-svc"
    assert tenant_service_name({"otel_service_name_override": "override", "otel_service_name": "base"}) == "override"
    assert tenant_service_name({"otel_service_name": "   "}) is None
    assert tenant_service_name({"logging_setting": "x"}) is None
    assert tenant_service_name(None) is None


def test_key_override_survives_team_metadata_merge():
    from litellm.integrations.otel.plumbing.routing import tenant_service_name

    # Request setup merges team metadata over key metadata (last writer wins),
    # so a key keeps its own destination via ``otel_service_name_override``,
    # which a team defining only ``otel_service_name`` never touches.
    merged = {"otel_service_name_override": "key-svc"}
    merged.update({"otel_service_name": "team-svc"})
    assert tenant_service_name(merged) == "key-svc"


def test_provider_cached_per_service_name():
    cache = _cache("otel")
    default = NoOpTracer()
    routed = cache.route_for(default, None, {"otel_service_name": "payments-gateway"})
    assert routed.tracer is not default
    assert routed.detached is False  # stays parented into the request trace
    assert routed.provider is not None
    assert routed.provider.resource.attributes["service.name"] == "payments-gateway"
    cache.route_for(default, None, {"otel_service_name": "payments-gateway"})
    assert len(cache._providers) == 1
    cache.route_for(default, None, {"otel_service_name": "search-gateway"})
    assert len(cache._providers) == 2
    for provider in cache._providers.values():
        provider.shutdown()


def test_service_name_routed_span_carries_team_service_name(monkeypatch):
    # The artifact the exporter receives: the finished span's Resource must
    # carry the team's service.name, not the env-configured default.
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proxy-default")
    cache = _cache("otel")
    default = NoOpTracer()
    route = cache.route_for(default, None, {"otel_service_name": "payments-gateway"})
    with route.tracer.start_as_current_span("chat gpt-4o-mini") as span:
        pass
    assert span.resource.attributes["service.name"] == "payments-gateway"
    cache.release(route.provider)

    unrouted = cache.route_for(default, None, {"logging_setting": "x"})
    assert unrouted.tracer is default  # env fallback: no scoped provider built


def test_client_dynamic_params_cannot_choose_service_name():
    # ``StandardCallbackDynamicParams`` is populated from client-supplied
    # request metadata; the service name may only come from server-set
    # key/team config (the ``auth_metadata`` argument).
    cache = _cache("otel")
    default = NoOpTracer()
    assert cache.route_for(default, {"otel_service_name": "attacker"}).tracer is default
    assert cache.route_for(default, {"otel_service_name_override": "attacker"}).tracer is default
    assert cache._providers == {}


def test_service_name_override_leaves_exporters_untouched():
    cache = _cache(
        "otel",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
        ],
    )
    cfg = cache._routed_config({}, {}, None, "payments-gateway")
    assert cfg.service_name == "payments-gateway"
    (spec,) = cfg.exporters
    assert spec.headers == "x=base-collector"
    assert spec.endpoint == "http://collector:4318"


# --- New Relic: per-team api-key header + fixed-table region endpoint --- #


def test_newrelic_dynamic_headers():
    assert dynamic_otlp_headers("newrelic", {"newrelic_api_key": "NRAL-KEY"}) == {"api-key": "NRAL-KEY"}
    assert dynamic_otlp_headers("newrelic", {"newrelic_region": "eu"}) is None


def test_newrelic_dynamic_endpoint_resolves_from_fixed_table():
    from litellm.integrations.otel.presets import dynamic_otlp_endpoint

    assert dynamic_otlp_endpoint("newrelic", {"newrelic_region": "eu"}) == "https://otlp.eu01.nr-data.net"
    assert dynamic_otlp_endpoint("newrelic", {"newrelic_region": "US"}) == "https://otlp.nr-data.net"
    # A key-only team (no region) resolves to the fixed US default deterministically,
    # never the operator's NEW_RELIC_REGION-configured preset endpoint.
    assert dynamic_otlp_endpoint("newrelic", {"newrelic_api_key": "k"}) == "https://otlp.nr-data.net"
    # An unknown region also resolves to the documented US default, not a guess.
    assert dynamic_otlp_endpoint("newrelic", {"newrelic_region": "mars"}) == "https://otlp.nr-data.net"
    # Callbacks without an endpoint resolver keep their preset endpoint.
    assert dynamic_otlp_endpoint("arize", {"newrelic_region": "eu"}) is None


def test_newrelic_endpoint_stamped_onto_owned_exporter_only():
    cache = _cache(
        "newrelic",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://self-hosted-collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="https://otlp.nr-data.net",
                owner="newrelic",
                requires_headers=True,
            ),
        ],
    )
    new_cfg = cache._routed_config({"api-key": "TEAM-EU-KEY"}, {}, "https://otlp.eu01.nr-data.net")
    by_owner = {e.owner: e for e in new_cfg.exporters}
    assert by_owner["newrelic"].endpoint == "https://otlp.eu01.nr-data.net"
    assert by_owner["newrelic"].headers == "api-key=TEAM-EU-KEY"
    assert by_owner[None].endpoint == "http://self-hosted-collector:4318"
    assert by_owner[None].headers == "x=base-collector"


def test_newrelic_provider_cached_per_key_and_region():
    cache = _cache(
        "newrelic",
        exporters=[ExporterSpec(kind="in_memory"), ExporterSpec(kind="otlp_http", owner="newrelic")],
    )
    default = NoOpTracer()
    cache.route_for(default, {"newrelic_api_key": "K1", "newrelic_region": "us"})
    cache.route_for(default, {"newrelic_api_key": "K1", "newrelic_region": "us"})
    assert len(cache._providers) == 1
    # Same key, different region → distinct provider (distinct endpoint).
    cache.route_for(default, {"newrelic_api_key": "K1", "newrelic_region": "eu"})
    assert len(cache._providers) == 2
    cache.route_for(default, {"newrelic_api_key": "K2", "newrelic_region": "eu"})
    assert len(cache._providers) == 3


def test_requires_headers_spec_skipped_without_headers():
    from litellm.integrations.otel.plumbing.providers import build_tracer_provider

    cfg = OpenTelemetryV2Config(
        exporters=[ExporterSpec(kind="otlp_http", endpoint="https://otlp.nr-data.net", requires_headers=True)]
    )
    provider = build_tracer_provider(cfg)
    processors = provider._active_span_processor._span_processors
    # Only the baggage processor: the keyless spec must not export (New Relic
    # rejects unauthenticated posts with a 4xx per span batch).
    assert [type(p).__name__ for p in processors] == ["LiteLLMBaggageSpanProcessor"]

    keyed = OpenTelemetryV2Config(
        exporters=[
            ExporterSpec(
                kind="otlp_http", endpoint="https://otlp.nr-data.net", headers="api-key=k", requires_headers=True
            )
        ]
    )
    keyed_provider = build_tracer_provider(keyed)
    assert len(keyed_provider._active_span_processor._span_processors) == 2


def test_newrelic_key_only_team_routes_to_us_not_operator_region(monkeypatch):
    """A team that saves a key but no region must export to the fixed US default,
    independent of the operator's NEW_RELIC_REGION, so its spans are never
    silently dropped by an operator-configured region its key does not match."""
    monkeypatch.setenv("NEW_RELIC_REGION", "eu")
    cache = _cache(
        "newrelic",
        exporters=[ExporterSpec(kind="otlp_http", endpoint="https://otlp.eu01.nr-data.net", owner="newrelic")],
    )
    new_cfg = cache._routed_config(
        {"api-key": "US-KEY"}, {}, dynamic_otlp_endpoint("newrelic", {"newrelic_api_key": "US-KEY"})
    )
    owned = next(e for e in new_cfg.exporters if e.owner == "newrelic")
    assert owned.endpoint == "https://otlp.nr-data.net"
