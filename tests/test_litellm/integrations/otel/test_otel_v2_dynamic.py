"""Per-tenant tracer routing on admin-owned OTEL destinations, with fan-out, plus the
per-request credential routing layered over it.

A request's identity chain is assigned a set of admin-owned exporters; the v2 logger fans
the trace out to all of them (plus the configured/global exporter). Separately, a request's
``standard_callback_dynamic_params`` (team/key OTLP credentials) route the gen-AI span to a
credential-scoped tracer for the integrations that support it (V1 parity). These tests lock
both: each destination's endpoint follows its resolved host (cross-host fix), the configured
exporters are kept (global also receives), a logger only exports the destinations tagged with
its own backend, and request credentials rewrite only their own backend's exporter headers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from opentelemetry.trace import NoOpTracer

from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.model.metadata import LLMCallEvent
from litellm.integrations.otel.plumbing.context import (
    _request_destinations,
    set_request_destinations,
)
from litellm.integrations.otel.plumbing.routing import TenantTracerCache


@pytest.fixture(autouse=True)
def _reset_request_destinations():
    """The v2 router reads destinations from a server-only ContextVar. Reset it around
    each test so a prior test's anchored destinations never leak into the next."""
    token = _request_destinations.set(())
    try:
        yield
    finally:
        _request_destinations.reset(token)


def _cache(callback_name, exporters=None):
    cfg = OpenTelemetryV2Config(
        exporters=exporters or [ExporterSpec(kind="in_memory", owner=callback_name)]
    )
    return TenantTracerCache(cfg, callback_name, "litellm")


def _dest(endpoint, auth="Basic AAAA", backend="langfuse_otel"):
    return OtelDestination(
        endpoint=endpoint, headers={"Authorization": auth}, callback_name=backend
    )


def _event(destinations):
    """Anchor destinations on the server-only ContextVar (the sole source the v2 router
    reads) and build the call event from it, as the proxy does at request time."""
    set_request_destinations(
        tuple(d if isinstance(d, OtelDestination) else OtelDestination.model_validate(d) for d in destinations)
    )
    return LLMCallEvent.from_dict({"call_type": "acompletion", "model": "gpt-4o"})


# --- routing only happens for admin destinations --------------------------- #


def test_no_destinations_uses_default_tracer():
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    assert cache.tracers_for(default, ()) == (default,)
    assert cache._providers == {}


def test_provider_cached_per_destination_set():
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    a = (_dest("https://eu.example/v1", "Basic A"),)
    b = (_dest("https://eu.example/v1", "Basic B"),)

    cache.tracers_for(default, a)
    cache.tracers_for(default, a)  # same set -> reuse
    assert len(cache._providers) == 1
    cache.tracers_for(default, b)  # different creds -> new provider
    assert len(cache._providers) == 2


def test_different_host_is_a_distinct_provider():
    """Two destinations with identical headers but different hosts must not collide;
    the cache key includes each endpoint."""
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    eu = (_dest("https://cloud.langfuse.com/api/public/otel", "Basic X"),)
    us = (_dest("https://us.cloud.langfuse.com/api/public/otel", "Basic X"),)
    cache.tracers_for(default, eu)
    cache.tracers_for(default, us)
    assert len(cache._providers) == 2


def test_destination_set_is_order_independent():
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    a = _dest("https://a/v1", "Basic A")
    b = _dest("https://b/v1", "Basic B")
    cache.tracers_for(default, (a, b))
    cache.tracers_for(default, (b, a))  # same set, different order -> one provider
    assert len(cache._providers) == 1


def test_provider_cache_evicts_lru_and_shuts_it_down_off_hot_path(monkeypatch):
    """The provider cache stays bounded, and eviction shuts the LRU provider down so its
    ``BatchSpanProcessor`` worker thread is reclaimed instead of leaked for the process
    lifetime. ``shutdown`` force-flushes/can block, so it runs on a background daemon
    thread rather than the request path: the evicted provider is shut down, the survivors
    are not. Dropping the shutdown (the old leak) fails this."""
    import time
    from unittest.mock import MagicMock

    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 2)
    real_build = routing_mod.build_tracer_provider
    created = []

    def spying_build(config):
        provider = real_build(config)
        provider.shutdown = MagicMock(wraps=provider.shutdown)
        created.append(provider)
        return provider

    monkeypatch.setattr(routing_mod, "build_tracer_provider", spying_build)

    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    cache.tracers_for(default, (_dest("https://1/v1"),))  # created[0]
    cache.tracers_for(default, (_dest("https://2/v1"),))  # created[1]
    cache.tracers_for(default, (_dest("https://1/v1"),))  # touch "1" -> "2" is LRU
    cache.tracers_for(default, (_dest("https://3/v1"),))  # created[2]: overflow -> evict "2"

    assert len(cache._providers) == 2
    # eviction shuts down off the hot path, so wait for the background daemon thread
    deadline = time.time() + 5
    while not created[1].shutdown.called and time.time() < deadline:
        time.sleep(0.02)
    created[1].shutdown.assert_called()  # the evicted LRU ("2") is reclaimed
    created[0].shutdown.assert_not_called()  # survivor
    created[2].shutdown.assert_not_called()  # survivor


# --- fan-out: keep the configured exporters, append one per destination ----- #


@pytest.mark.parametrize("owner", ["langfuse_otel", "arize", "weave_otel"])
def test_fan_out_appends_destination_with_resolved_endpoint(owner):
    # The configured (global) exporter is kept; each destination is appended with its
    # OWN resolved endpoint + headers (the cross-host fix, per owner).
    cache = _cache(
        owner,
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="https://env-host.example/v1",
                headers="Authorization=Basic ENV",
                owner=owner,
            )
        ],
    )
    new = cache._config_with_destinations(
        (_dest("https://resolved.example/v1", "Basic TEAM", backend=owner),)
    )
    # global kept verbatim
    assert new.exporters[0].endpoint == "https://env-host.example/v1"
    assert new.exporters[0].headers == "Authorization=Basic ENV"
    # destination appended at the resolved host with its own auth
    assert new.exporters[-1].endpoint == "https://resolved.example/v1"
    assert new.exporters[-1].headers == "Authorization=Basic TEAM"
    assert len(new.exporters) == 2


def test_fan_out_preserves_co_configured_exporters():
    cache = _cache(
        "langfuse_otel",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://self-hosted-collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="https://us.cloud.langfuse.com/api/public/otel",
                headers="Authorization=Basic ENV",
                owner="langfuse_otel",
            ),
        ],
    )
    new = cache._config_with_destinations(
        (_dest("https://cloud.langfuse.com/api/public/otel", "Basic TEAM"),)
    )
    # both originals preserved unchanged (no rewrite/leak)
    assert new.exporters[0].endpoint == "http://self-hosted-collector:4318"
    assert new.exporters[0].headers == "x=base-collector"
    assert new.exporters[1].headers == "Authorization=Basic ENV"
    # exactly one appended
    assert new.exporters[-1].endpoint == "https://cloud.langfuse.com/api/public/otel"
    assert len(new.exporters) == 3


def test_fan_out_to_many_destinations_is_one_provider_with_all_exporters():
    cache = _cache(
        "langfuse_otel",
        exporters=[
            ExporterSpec(
                kind="otlp_http", endpoint="https://env/v1", owner="langfuse_otel"
            )
        ],
    )
    new = cache._config_with_destinations(
        (_dest("https://a/v1", "Basic A"), _dest("https://b/v1", "Basic B"))
    )
    # global + 2 destinations -> 3 span processors, one provider, one span copied to all
    assert [e.endpoint for e in new.exporters] == [
        "https://env/v1",
        "https://a/v1",
        "https://b/v1",
    ]
    cache.tracers_for(NoOpTracer(), (_dest("https://a/v1"), _dest("https://b/v1")))
    assert len(cache._providers) == 1


# --- gen-AI span Resource must match its fanned-out parents ----------------- #
#
# The gen-AI LLM-call span is emitted through the TenantTracerCache clone
# provider, while the proxy-internal spans (server/auth/db) are forwarded by
# TenantFanOutSpanProcessor, which wraps each with the destination's backend-
# required Resource attrs (Arize needs model_id / arize.project.name). If the
# clone provider does NOT also carry those attrs, the gen-AI span reaches Arize
# with only service.name and Arize renders it as an orphaned subtree. These pin
# that the clone Resource carries the same attrs the fan-out wrap injects.


def _dest_with_resource(endpoint, backend, resource_attributes):
    return OtelDestination(
        endpoint=endpoint,
        headers={"Authorization": "Basic AAAA"},
        callback_name=backend,
        resource_attributes=resource_attributes,
    )


def test_clone_config_carries_destination_resource_attrs():
    cache = _cache("arize")
    new = cache._config_with_destinations(
        (
            _dest_with_resource(
                "https://otlp.arize.com/v1",
                "arize",
                {"model_id": "team-b-proj", "arize.project.name": "team-b-proj"},
            ),
        )
    )
    assert new.resource_attributes["model_id"] == "team-b-proj"
    assert new.resource_attributes["arize.project.name"] == "team-b-proj"


def test_clone_config_carries_builder_declared_resource_attrs(monkeypatch):
    """End-to-end: an arize credential that omits the project gets ARIZE_PROJECT_NAME
    folded in at BUILD time (build_destination), and the clone config then carries
    those Resource attrs generically -- the gen-AI span lands in the same project as
    its fan-out'd parents. The clone path itself is backend-agnostic; it reads
    whatever the builder declared."""
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "env-proj")
    from litellm.integrations.otel.presets.destinations import build_destination

    dest = build_destination("arize", {"arize_space_id": "s", "arize_api_key": "k"})
    assert dest is not None
    cache = _cache("arize")
    new = cache._config_with_destinations((dest,))
    assert new.resource_attributes["model_id"] == "env-proj"
    assert new.resource_attributes["arize.project.name"] == "env-proj"


def test_clone_provider_emits_genai_span_with_destination_resource():
    """End-to-end regression: the span the clone provider actually exports must
    carry the destination's Resource attrs. Pre-fix this Resource was
    service.name only, orphaning the gen-AI span in Arize."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from litellm.integrations.otel.plumbing.providers import (
        build_tracer_provider,
        get_tracer,
    )

    cfg = OpenTelemetryV2Config(
        service_name="litellm-proxy",
        exporters=[ExporterSpec(kind="in_memory", owner="arize")],
    )
    cache = TenantTracerCache(cfg, "arize", "litellm")
    dest = _dest_with_resource(
        "https://otlp.arize.com/v1",
        "arize",
        {"model_id": "team-b-proj", "arize.project.name": "team-b-proj"},
    )
    tracers = cache.tracers_for(get_tracer(build_tracer_provider(cfg)), (dest,))
    tracer = tracers[-1]
    with tracer.start_as_current_span("chat anthropic-haiku") as span:
        span.set_attribute("gen_ai.operation.name", "chat")

    # The destination group carries the destination's Resource and only the
    # destination's exporter; the configured in-memory exporter rides the base tracer
    # (``tracers[0]``) on a clean Resource, so the tenant's project can never stamp it.
    provider = next(iter(cache._providers.values()))
    provider.force_flush()
    resource_attrs = dict(provider.resource.attributes)
    assert resource_attrs.get("model_id") == "team-b-proj"
    assert not any(
        isinstance(getattr(proc, "span_exporter", None), InMemorySpanExporter)
        for proc in provider._active_span_processor._span_processors
    ), "the configured exporter must not ride a destination group"
    assert resource_attrs.get("arize.project.name") == "team-b-proj"


# --- request credentials route the gen-AI span (V1 parity) ------------------ #


_DYNAMIC_CREDS = [
    ("langfuse_otel", {"langfuse_public_key": "pk-teamA", "langfuse_secret_key": "sk-teamA"}, "Authorization="),
    ("arize", {"arize_space_id": "space-teamA", "arize_api_key": "key-teamA"}, "arize-space-id=space-teamA"),
    ("weave_otel", {"wandb_api_key": "wandb-teamA", "weave_project_id": "proj-teamA"}, "project_id=proj-teamA"),
]


@pytest.mark.parametrize("backend, creds, owned_fragment", _DYNAMIC_CREDS)
def test_request_credentials_rewrite_only_the_owned_exporter(backend, creds, owned_fragment):
    """A request's team/key OTLP credentials rewrite the headers of THIS backend's own
    exporter and nothing else, so the gen-AI span exports to that team's account while a
    co-configured backend's exporter is untouched (no cross-backend key leak)."""
    from litellm.integrations.otel.presets import dynamic_otlp_headers

    cache = _cache(
        backend,
        exporters=[
            ExporterSpec(kind="otlp_http", endpoint="http://collector/otel", headers="Authorization=Basic GLOBAL", owner=backend),
            ExporterSpec(kind="otlp_http", endpoint="http://other/otel", headers="x=coconfigured", owner="levo"),
        ],
    )
    headers = dynamic_otlp_headers(backend, creds)
    assert headers, f"{backend} must support dynamic credentials"
    scoped = cache._config_with_headers(headers)
    assert owned_fragment in (scoped.exporters[0].headers or "")
    assert "GLOBAL" not in (scoped.exporters[0].headers or "")  # the global header was rewritten away
    assert scoped.exporters[1].headers == "x=coconfigured"  # a different backend's exporter is untouched


@pytest.mark.parametrize("backend, creds, owned_fragment", _DYNAMIC_CREDS)
def test_genai_tracers_for_spawns_credential_scoped_provider(backend, creds, owned_fragment):
    """genai_tracers_for with request creds and no admin destination returns exactly the
    credential-scoped tracer (not the default) and caches its provider under a dynamic key."""
    cache = _cache(
        backend,
        exporters=[ExporterSpec(kind="otlp_http", endpoint="http://c/otel", headers="Authorization=Basic GLOBAL", owner=backend)],
    )
    default = NoOpTracer()
    tracers = cache.genai_tracers_for(default, (), creds)
    assert len(tracers) == 1
    assert tracers[0] is not default
    assert len(cache._providers) == 1
    assert next(iter(cache._providers))[0] == "dynamic"  # namespaced key, never aliases a destination group


def test_genai_tracers_for_without_creds_is_plain_default():
    """No dynamic creds and no destinations -> the logger's default tracer, no provider spawned."""
    cache = _cache(
        "langfuse_otel",
        exporters=[ExporterSpec(kind="otlp_http", endpoint="http://c/otel", headers="", owner="langfuse_otel")],
    )
    default = NoOpTracer()
    assert cache.genai_tracers_for(default, (), None) == (default,)
    assert cache._providers == {}


def test_genai_tracers_compose_credential_scoped_plus_destination():
    """Request creds AND an admin destination: the span exports via the credential-scoped
    tracer (which carries the global exporter) plus one per-destination tracer that omits the
    base exporters, so the global collector receives the span exactly once."""
    cache = _cache(
        "langfuse_otel",
        exporters=[ExporterSpec(kind="otlp_http", endpoint="http://global/otel", headers="Authorization=Basic GLOBAL", owner="langfuse_otel")],
    )
    creds = {"langfuse_public_key": "pk-teamA", "langfuse_secret_key": "sk-teamA"}
    destinations = (_dest("https://cloud.langfuse.com/api/public/otel", auth="Basic ADMIN"),)
    tracers = cache.genai_tracers_for(NoOpTracer(), destinations, creds)
    assert len(tracers) == 2  # credential-scoped (global) + one destination group
    # the destination group provider carries no base exporter (base rides the dynamic tracer)
    dest_only = cache._config_with_destinations(destinations, include_base_exporters=False)
    assert all(spec.endpoint != "http://global/otel" for spec in dest_only.exporters)


def test_admin_destinations_route():
    event = _event(
        [
            {
                "callback_name": "langfuse_otel",
                "endpoint": "https://cloud.langfuse.com/api/public/otel",
                "headers": {"Authorization": "Basic ADMIN"},
            }
        ]
    )
    assert len(event.otel_destinations) == 1
    cache = _cache("langfuse_otel")
    cache.tracers_for(NoOpTracer(), event.otel_destinations)
    assert len(cache._providers) == 1


# --- a logger only exports the destinations tagged with its own backend ----- #


def test_logger_filters_destinations_to_its_backend():
    from litellm.integrations.otel.logger import OpenTelemetryV2

    event = _event(
        [
            {
                "callback_name": "langfuse_otel",
                "endpoint": "https://lf/api/public/otel",
                "headers": {"Authorization": "Basic A"},
            },
            {
                "callback_name": "arize",
                "endpoint": "https://otlp.arize.com/v1",
                "headers": {"space_id": "S"},
            },
        ]
    )

    class _Shim:
        callback_name = "langfuse_otel"

    got = OpenTelemetryV2._destinations_for_backend(_Shim(), event)
    assert [d.endpoint for d in got] == ["https://lf/api/public/otel"]


# --- multi-destination, same-backend: group by Resource so each gets its span -- #
#
# A backend that selects its target FROM the Resource (Arize's project) needs a
# differently-tagged span per destination, because a span carries exactly one
# Resource (a TracerProvider property). Pre-fix the gen-AI clone folded every
# destination into ONE last-wins Resource, so two Arize projects collapsed to one
# and only that project received the gen-AI span. ``tracers_for`` groups by
# ``destination_resource_attrs`` and returns one tracer (one provider, one Resource)
# per distinct group. Header-routed backends declare no Resource attrs, so they stay
# in one group with multiple exporters and keep routing by per-exporter auth.


def _arize_dest(project, space="S", key="K"):
    return OtelDestination(
        endpoint="https://otlp.arize.com/v1",
        headers={"space_id": space, "api_key": key},
        callback_name="arize",
        resource_attributes={"model_id": project, "arize.project.name": project},
    )


def _provider_project(provider):
    return provider.resource.attributes.get("arize.project.name")


def test_tracers_for_empty_returns_default_only():
    cache = _cache("arize")
    default = NoOpTracer()
    assert cache.tracers_for(default, ()) == (default,)
    assert cache._providers == {}


def test_tracers_for_single_destination_one_group():
    """A single Arize destination yields the clean base tracer (``default``) plus one
    destination provider carrying its project; base is never folded into the project group."""
    default = NoOpTracer()
    cache = _cache("arize")
    tracers = cache.tracers_for(default, (_arize_dest("solo"),))
    assert tracers[0] is default  # base rides its own clean tracer, returned first
    assert len(tracers) == 2  # default + the one project group
    assert {_provider_project(p) for p in cache._providers.values()} == {"solo"}


def test_tracers_for_two_arize_projects_split_into_separate_groups():
    """The fix: two Arize destinations with different Resource attrs must NOT
    last-wins merge -- each project gets its own provider/Resource so each receives a
    correctly-tagged gen-AI span."""
    cache = _cache("arize")
    tracers = cache.tracers_for(
        NoOpTracer(), (_arize_dest("projA"), _arize_dest("projB"))
    )
    assert len(tracers) == 3  # default (clean base) + one tracer per project group
    assert {_provider_project(p) for p in cache._providers.values()} == {
        "projA",
        "projB",
    }


def test_two_arize_projects_each_provider_has_its_own_single_project():
    """Each group's Resource carries exactly its own project (not the other's, not a
    merge)."""
    cache = _cache("arize")
    cache.tracers_for(NoOpTracer(), (_arize_dest("projA"), _arize_dest("projB")))
    by_project = {
        _provider_project(p): p.resource.attributes for p in cache._providers.values()
    }
    assert by_project["projA"]["model_id"] == "projA"
    assert by_project["projB"]["model_id"] == "projB"


def test_header_routed_destinations_stay_one_group_with_two_exporters():
    """Langfuse/Weave declare no Resource attrs, so two distinct destinations collapse
    into ONE group (one provider) with one exporter each -- they route by per-exporter
    auth, so no per-Resource split is needed or wanted."""
    cache = _cache(
        "langfuse_otel",
        exporters=[
            ExporterSpec(kind="otlp_http", endpoint="https://env/v1", owner=None)
        ],
    )
    tracers = cache.tracers_for(
        NoOpTracer(),
        (_dest("https://a/v1", "Basic A"), _dest("https://b/v1", "Basic B")),
    )
    assert len(tracers) == 2  # default (clean base) + the single empty-Resource group
    assert len(cache._providers) == 1
    (provider,) = cache._providers.values()
    endpoints = " ".join(
        str(getattr(getattr(sp, "span_exporter", None), "_endpoint", ""))
        for sp in provider._active_span_processor._span_processors
    )
    # both destinations live on the one group provider (the global exporter rides ``default``,
    # not this group); OTLP normalizes the endpoint by appending /v1/traces, so match on prefix
    assert "https://a/v1" in endpoints and "https://b/v1" in endpoints


def test_base_exporters_never_ride_a_destination_group():
    """The configured/global exporters must ride their own clean-Resource tracer
    (``default``), never a destination group: folding them into a group made the global
    export inherit that destination's Resource (e.g. Arize's project), so an operator's
    own collector saw spans stamped with a tenant's project. The global still receives the
    gen-AI span exactly once (via ``default``), not once per project."""
    cache = _cache(
        "arize",
        exporters=[ExporterSpec(kind="in_memory", endpoint=None, owner=None)],
    )
    default = NoOpTracer()
    tracers = cache.tracers_for(default, (_arize_dest("projA"), _arize_dest("projB")))
    assert tracers[0] is default  # global rides its own clean tracer, exactly once
    for provider in cache._providers.values():
        base_count = sum(
            type(getattr(sp, "span_exporter", sp)).__name__ == "InMemorySpanExporter"
            for sp in provider._active_span_processor._span_processors
        )
        assert base_count == 0, "no destination group may carry the configured/global exporters"
        # and each group's Resource stays its own project only
        assert _provider_project(provider) in {"projA", "projB"}


def test_generic_backend_resolves_generic_destination():
    """A Generic OTLP destination (callback_name='generic') must be picked up by the
    generic OpenTelemetryV2 logger, so its gen-AI span routes to the destination's
    otel_endpoint. Regression: 'generic' had no preset, so no generic logger existed and
    the gen-AI span was dropped (only proxy-internal spans fanned out)."""
    from litellm.integrations.otel.logger import OpenTelemetryV2

    event = _event(
        [
            {
                "callback_name": "generic",
                "endpoint": "http://collector:4318",
                "headers": {"x-tenant": "t1"},
            },
            {
                "callback_name": "arize",
                "endpoint": "https://otlp.arize.com/v1",
                "headers": {"space_id": "S"},
            },
        ]
    )

    class _Shim:
        callback_name = "generic"

    got = OpenTelemetryV2._destinations_for_backend(_Shim(), event)
    assert [d.endpoint for d in got] == ["http://collector:4318"]


def test_otel_destination_is_frozen_and_renders_header_string():
    """OtelDestination is the immutable value the resolver hands to the runtime;
    mutating one after resolution must fail (a request can never rewrite where its
    traces go), and header_string renders the k=v,k2=v2 form the exporter expects."""
    import pytest
    from pydantic import ValidationError

    dest = OtelDestination(
        endpoint="https://c/v1", headers={"Authorization": "Bearer x", "x-k": "y"}
    )
    with pytest.raises(ValidationError):
        dest.endpoint = "https://evil/v1"
    assert dest.header_string() == "Authorization=Bearer x,x-k=y"


@pytest.mark.parametrize(
    "backend, params, expected_endpoint",
    [
        (
            "langfuse_otel",
            {"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_host": "https://lf.internal"},
            "https://lf.internal/api/public/otel",
        ),
        ("arize", {"arize_space_key": "S", "arize_api_key": "K"}, "https://otlp.arize.com/v1"),
        ("weave_otel", {"wandb_api_key": "wk"}, "https://trace.wandb.ai/otel/v1/traces"),
    ],
)
def test_team_credentials_still_export_when_the_preset_degraded(backend, params, expected_endpoint):
    """Regression: a preset built with ``allow_missing_credentials`` contributes no owned
    exporter, so a team's own ``callback_vars`` had nothing to be stamped onto and its
    spans were dropped. The exporter is synthesized from the request's own credentials
    instead, resolved through the same builder an equivalent admin destination uses.
    """
    from litellm.integrations.otel.presets import dynamic_otlp_headers

    degraded = TenantTracerCache(OpenTelemetryV2Config(exporters=[]), backend, "litellm")
    headers = dynamic_otlp_headers(backend, params)
    assert headers, "the backend must recognise these per-team credentials"

    owned = [spec for spec in degraded._config_with_headers(headers, params).exporters if spec.owner == backend]

    assert len(owned) == 1, "a degraded preset must still export the team's own traces"
    assert owned[0].endpoint == expected_endpoint
    assert owned[0].headers == ",".join(f"{k}={v}" for k, v in headers.items())


def test_degraded_preset_synthesizes_nothing_without_team_credentials():
    """No per-request credentials means no synthesized exporter, so a degraded backend
    stays silent rather than inventing an uncredentialled vendor export."""
    degraded = TenantTracerCache(OpenTelemetryV2Config(exporters=[]), "arize", "litellm")
    assert degraded._config_with_headers({}, None).exporters == []


# --- transport is part of a destination's identity ------------------------- #
# Two destinations can agree on endpoint, headers and Resource attrs and still
# disagree on OTLP transport (an Arize credential naming ``arize_http_endpoint``
# resolves to otlp_http where the backend's intrinsic default is gRPC). The
# provider cache and the synthesized exporter both have to carry that.


def _arize_dest_with_protocol(protocol):
    return OtelDestination(
        endpoint="https://collector.internal/v1",
        headers={"space_id": "S", "api_key": "K"},
        callback_name="arize",
        resource_attributes={"model_id": "proj", "arize.project.name": "proj"},
        protocol=protocol,
    )


def _exporter_transports(provider):
    """The OTLP transport of each of a provider's exporters. Both exporters are named
    ``OTLPSpanExporter``; only the defining module tells gRPC from HTTP apart."""
    return {
        "http" if ".http." in type(sp.span_exporter).__module__ else "grpc"
        for sp in provider._active_span_processor._span_processors
        if getattr(sp, "span_exporter", None) is not None
        and "otlp" in type(sp.span_exporter).__module__
    }


def test_same_endpoint_and_headers_but_different_protocol_are_distinct_providers():
    """Two destinations differing only in OTLP transport must not share a provider;
    reusing the first one's exports the second's spans over the wrong transport."""
    cache = _cache("arize")
    cache.tracers_for(NoOpTracer(), (_arize_dest_with_protocol(None),))
    cache.tracers_for(NoOpTracer(), (_arize_dest_with_protocol("otlp_http"),))

    assert len(cache._providers) == 2
    transports = sorted(t for p in cache._providers.values() for t in _exporter_transports(p))
    assert transports == ["grpc", "http"]


def test_synthesized_exporter_uses_the_transport_the_credentials_pin():
    """A team whose Arize credentials name an HTTP collector gets an HTTP exporter, not
    the backend's intrinsic gRPC default."""
    cache = _cache("arize", exporters=[ExporterSpec(kind="otlp_grpc", endpoint="https://env/v1", owner=None)])
    params = {
        "arize_space_id": "S",
        "arize_api_key": "K",
        "arize_http_endpoint": "https://collector.internal/v1",
    }

    config = cache._config_with_headers({"space_id": "S"}, params)

    (synthesized,) = [spec for spec in config.exporters if spec.owner == "arize"]
    assert synthesized.endpoint == "https://collector.internal/v1"
    assert synthesized.kind == "otlp_http"


def test_synthesized_exporter_keeps_the_backend_default_when_credentials_pin_nothing():
    """Credentials that name no transport still get the backend's intrinsic one."""
    cache = _cache("arize", exporters=[ExporterSpec(kind="otlp_grpc", endpoint="https://env/v1", owner=None)])

    config = cache._config_with_headers({"space_id": "S"}, {"arize_space_id": "S", "arize_api_key": "K"})

    (synthesized,) = [spec for spec in config.exporters if spec.owner == "arize"]
    assert synthesized.kind == "otlp_grpc"
