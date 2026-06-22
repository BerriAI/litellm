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
from litellm.integrations.otel.plumbing.routing import TenantTracerCache


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
    return LLMCallEvent.from_dict(
        {
            "standard_callback_dynamic_params": {"otel_destinations": destinations},
            "call_type": "acompletion",
            "model": "gpt-4o",
        }
    )


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
