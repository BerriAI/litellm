"""Per-tenant tracer routing on admin-owned OTEL destinations, with fan-out.

A request's identity chain is assigned a set of admin-owned exporters; the v2 logger
fans the trace out to all of them (plus the configured/global exporter), and never
routes on request-supplied vendor credentials. These tests lock the contract: the
request cannot route a trace, each destination's endpoint follows its resolved host
(cross-host fix), the configured exporters are kept (global also receives), and a
logger only exports the destinations tagged with its own backend.
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
    assert cache.tracer_for(default, ()) is default
    assert cache._providers == {}


def test_provider_cached_per_destination_set():
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    a = (_dest("https://eu.example/v1", "Basic A"),)
    b = (_dest("https://eu.example/v1", "Basic B"),)

    cache.tracer_for(default, a)
    cache.tracer_for(default, a)  # same set -> reuse
    assert len(cache._providers) == 1
    cache.tracer_for(default, b)  # different creds -> new provider
    assert len(cache._providers) == 2


def test_different_host_is_a_distinct_provider():
    """Two destinations with identical headers but different hosts must not collide;
    the cache key includes each endpoint."""
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    eu = (_dest("https://cloud.langfuse.com/api/public/otel", "Basic X"),)
    us = (_dest("https://us.cloud.langfuse.com/api/public/otel", "Basic X"),)
    cache.tracer_for(default, eu)
    cache.tracer_for(default, us)
    assert len(cache._providers) == 2


def test_destination_set_is_order_independent():
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    a = _dest("https://a/v1", "Basic A")
    b = _dest("https://b/v1", "Basic B")
    cache.tracer_for(default, (a, b))
    cache.tracer_for(default, (b, a))  # same set, different order -> one provider
    assert len(cache._providers) == 1


def test_provider_cache_is_bounded_and_evicts_lru(monkeypatch):
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 2)
    shut_down = []
    monkeypatch.setattr(
        routing_mod, "_shutdown_provider", lambda p: shut_down.append(p)
    )
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    cache.tracer_for(default, (_dest("https://1/v1"),))
    cache.tracer_for(default, (_dest("https://2/v1"),))
    cache.tracer_for(default, (_dest("https://1/v1"),))  # touch "1" -> "2" is LRU
    cache.tracer_for(default, (_dest("https://3/v1"),))  # overflow -> evict "2"
    assert len(cache._providers) == 2
    assert len(shut_down) == 1


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
    cache.tracer_for(NoOpTracer(), (_dest("https://a/v1"), _dest("https://b/v1")))
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
    tracer = cache.tracer_for(get_tracer(build_tracer_provider(cfg)), (dest,))
    with tracer.start_as_current_span("chat anthropic-haiku") as span:
        span.set_attribute("gen_ai.operation.name", "chat")

    provider = next(iter(cache._providers.values()))
    provider.force_flush()
    captured = []
    for proc in provider._active_span_processor._span_processors:
        exporter = getattr(proc, "span_exporter", None)
        if isinstance(exporter, InMemorySpanExporter):
            captured = exporter.get_finished_spans()
    assert captured, "clone provider exported no span to its in-memory exporter"
    resource_attrs = dict(captured[0].resource.attributes)
    assert resource_attrs.get("model_id") == "team-b-proj"
    assert resource_attrs.get("arize.project.name") == "team-b-proj"


# --- security: request credentials never route a trace --------------------- #


@pytest.mark.parametrize(
    "request_creds",
    [
        {
            "langfuse_public_key": "pk-attacker",
            "langfuse_secret_key": "sk-attacker",
            "langfuse_host": "https://attacker.example",
        },
        {"arize_api_key": "K-attacker", "arize_space_id": "S-attacker"},
        {"wandb_api_key": "w-attacker", "weave_endpoint": "https://attacker/otel"},
    ],
)
def test_request_credentials_are_inert_on_v2(request_creds):
    """Any backend's credentials in the request's dynamic params (no admin
    destinations) produce no per-tenant routing."""
    event = LLMCallEvent.from_dict(
        {
            "standard_callback_dynamic_params": request_creds,
            "call_type": "acompletion",
            "model": "gpt-4o",
        }
    )
    assert event.otel_destinations == ()
    cache = _cache("langfuse_otel")
    default = NoOpTracer()
    assert cache.tracer_for(default, event.otel_destinations) is default
    assert cache._providers == {}


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
    cache.tracer_for(NoOpTracer(), event.otel_destinations)
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
    """No regression: a single Arize destination yields one tracer/provider carrying
    its project (same as the old single-merged path)."""
    cache = _cache("arize")
    tracers = cache.tracers_for(NoOpTracer(), (_arize_dest("solo"),))
    assert len(tracers) == 1
    assert {_provider_project(p) for p in cache._providers.values()} == {"solo"}


def test_tracers_for_two_arize_projects_split_into_separate_groups():
    """The fix: two Arize destinations with different Resource attrs must NOT
    last-wins merge -- each project gets its own provider/Resource so each receives a
    correctly-tagged gen-AI span."""
    cache = _cache("arize")
    tracers = cache.tracers_for(
        NoOpTracer(), (_arize_dest("projA"), _arize_dest("projB"))
    )
    assert len(tracers) == 2  # one tracer per project group
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
    assert len(tracers) == 1  # single empty-Resource group
    assert len(cache._providers) == 1
    (provider,) = cache._providers.values()
    endpoints = " ".join(
        str(getattr(getattr(sp, "span_exporter", None), "_endpoint", ""))
        for sp in provider._active_span_processor._span_processors
    )
    # global + both destinations all live on the one provider (OTLP normalizes the
    # endpoint by appending /v1/traces, so match on the host+path prefix)
    assert "https://a/v1" in endpoints and "https://b/v1" in endpoints


def test_base_exporters_attach_to_first_group_only():
    """When a backend splits into multiple Resource groups, the configured/global
    exporters must ride exactly ONE group, so the global receives the gen-AI span once
    rather than once per project."""
    cache = _cache(
        "arize",
        exporters=[ExporterSpec(kind="in_memory", endpoint=None, owner=None)],
    )
    cache.tracers_for(NoOpTracer(), (_arize_dest("projA"), _arize_dest("projB")))
    base_counts = {}
    for provider in cache._providers.values():
        project = _provider_project(provider)
        base_counts[project] = sum(
            type(getattr(sp, "span_exporter", sp)).__name__ == "InMemorySpanExporter"
            for sp in provider._active_span_processor._span_processors
        )
    # exactly one group carries the global in_memory exporter; the other carries none
    assert sorted(base_counts.values()) == [0, 1]


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
