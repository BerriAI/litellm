"""Key/team OTLP destinations override the operator's exporters for that backend."""

import contextvars
from collections.abc import Mapping

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.plumbing.context import (
    overridden_backends,
    request_destinations,
    set_request_destinations,
)
from litellm.integrations.otel.plumbing.providers import (
    TenantFanOutSpanProcessor,
    _OverriddenBackendFilter,
    build_tracer_provider,
)
from litellm.integrations.otel.plumbing.routing import TenantTracerCache, get_tracer
from litellm.integrations.otel.presets.destinations import (
    destination_capable_backends,
    destination_for,
)
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import resolve_tenant_otel_destinations

LANGFUSE_DEST = OtelDestination(
    endpoint="http://tenant.local/api/public/otel",
    headers={"Authorization": "Basic dGVuYW50"},
    callback_name="langfuse_otel",
)


def in_fresh_context(fn, *args):
    """Run ``fn`` in its own context so one test's destinations never leak."""
    return contextvars.copy_context().run(fn, *args)


def emit(provider: TracerProvider, name: str = "chat gpt-4") -> None:
    with get_tracer(provider, "litellm").start_as_current_span(name):
        pass


def wired_provider(dest_exporter: InMemorySpanExporter, global_exporter: InMemorySpanExporter) -> TracerProvider:
    """The operator's provider: one owned exporter plus the tenant fan-out."""
    provider = TracerProvider()
    provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(global_exporter), "langfuse_otel"))
    provider.add_span_processor(
        TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
    )
    return provider


class TestOverrideSuppression:
    def test_operator_exporter_keeps_the_span_when_no_destination_is_resolved(self):
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        in_fresh_context(emit, provider)

        assert [s.name for s in global_exporter.get_finished_spans()] == ["chat gpt-4"]
        assert dest_exporter.get_finished_spans() == ()

    def test_operator_exporter_is_skipped_once_the_backend_is_overridden(self):
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            emit(provider)

        in_fresh_context(run)

        assert global_exporter.get_finished_spans() == ()
        assert [s.name for s in dest_exporter.get_finished_spans()] == ["chat gpt-4"]

    def test_a_backend_the_request_did_not_override_still_exports(self):
        arize_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(arize_exporter), "arize"))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            emit(provider)

        in_fresh_context(run)

        assert [s.name for s in arize_exporter.get_finished_spans()] == ["chat gpt-4"]


class TestFanOut:
    def test_every_span_of_the_request_reaches_the_destination_in_one_trace(self):
        """The whole tree, gen-AI span included, parented as the operator would see it."""
        dest_exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                with tracer.start_as_current_span("auth /v1/chat/completions"):
                    pass
                with tracer.start_as_current_span("chat gpt-4"):
                    pass

        in_fresh_context(run)

        spans = dest_exporter.get_finished_spans()
        by_name = {s.name: s for s in spans}
        assert set(by_name) == {"POST /v1/chat/completions", "auth /v1/chat/completions", "chat gpt-4"}
        root = by_name["POST /v1/chat/completions"]
        assert len({s.context.trace_id for s in spans}) == 1, "the tenant must receive one connected trace"
        for child in ("auth /v1/chat/completions", "chat gpt-4"):
            assert by_name[child].parent.span_id == root.context.span_id

    def test_two_destinations_each_receive_their_own_copy(self):
        first, second = InMemorySpanExporter(), InMemorySpanExporter()
        by_endpoint = {"http://a.local": first, "http://b.local": second}
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda d: SimpleSpanProcessor(by_endpoint[d.endpoint]),
            )
        )

        def run():
            set_request_destinations(
                (
                    OtelDestination(endpoint="http://a.local", callback_name="langfuse_otel"),
                    OtelDestination(endpoint="http://b.local", callback_name="arize"),
                )
            )
            emit(provider)

        in_fresh_context(run)

        assert [s.name for s in first.get_finished_spans()] == ["chat gpt-4"]
        assert [s.name for s in second.get_finished_spans()] == ["chat gpt-4"]

    def test_a_destination_that_cannot_build_a_processor_is_skipped_quietly(self):
        """An unbuildable destination must not cost the caller its request."""
        attempts = []
        reached_the_end = []

        def factory(destination):
            attempts.append(destination.endpoint)

        provider = TracerProvider()
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=factory))

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            emit(provider)
            reached_the_end.append(True)

        in_fresh_context(run)

        assert attempts == [LANGFUSE_DEST.endpoint]
        assert reached_the_end == [True]

    def test_one_processor_is_reused_across_spans_of_the_same_destination(self):
        built = []

        def factory(_destination):
            processor = SimpleSpanProcessor(InMemorySpanExporter())
            built.append(processor)
            return processor

        provider = TracerProvider()
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=factory))

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            emit(provider, "one")
            emit(provider, "two")

        in_fresh_context(run)

        assert len(built) == 1


class TestProviderWiring:
    def test_build_tracer_provider_only_filters_when_asked(self):
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        operator = build_tracer_provider(config, tenant_overrides=True)
        tenant = build_tracer_provider(config)

        def kinds(provider):
            return [type(p).__name__ for p in provider._active_span_processor._span_processors]

        assert "_OverriddenBackendFilter" in kinds(operator)
        assert "TenantFanOutSpanProcessor" in kinds(operator)
        assert "_OverriddenBackendFilter" not in kinds(tenant), "a per-tenant provider must not filter itself out"
        assert "TenantFanOutSpanProcessor" not in kinds(tenant)


class TestRouting:
    def test_an_overridden_backend_is_not_detached_onto_a_second_provider(self):
        config = OpenTelemetryV2Config(
            exporters=[ExporterSpec(kind="otlp_http", endpoint="http://op.local", owner=ExporterOwner.LANGFUSE_OTEL)]
        )
        cache = TenantTracerCache(config, "langfuse_otel", "litellm")
        default = get_tracer(TracerProvider(), "litellm")
        params = {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}

        assert cache.route_for(default, params).detached is True

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return cache.route_for(default, params)

        route = in_fresh_context(run)
        assert route.detached is False
        assert route.tracer is default
        assert route.provider is None


class TestDestinationResolution:
    def test_a_langfuse_key_pair_and_host_become_a_destination(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "langfuse_otel",
                        "callback_type": "success",
                        "callback_vars": {
                            "langfuse_public_key": "pk-team",
                            "langfuse_secret_key": "sk-team",
                            "langfuse_host": "http://team.local",
                        },
                    }
                ]
            }
        )

        destinations = resolve_tenant_otel_destinations(auth)

        assert [d.endpoint for d in destinations] == ["http://team.local/api/public/otel"]
        assert destinations[0].callback_name == "langfuse_otel"

    def test_the_key_wins_over_the_team_for_the_same_backend(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()

        def entry(host: str) -> Mapping[str, object]:
            return {
                "callback_name": "langfuse_otel",
                "callback_type": "success",
                "callback_vars": {
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                    "langfuse_host": host,
                },
            }

        auth = UserAPIKeyAuth(
            metadata={"logging": [entry("http://key.local")]},
            team_metadata={"logging": [entry("http://team.local")]},
        )

        assert [d.endpoint for d in resolve_tenant_otel_destinations(auth)] == ["http://key.local/api/public/otel"]

    def test_nothing_resolves_while_otel_v2_is_off(self, monkeypatch):
        monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "langfuse_otel",
                        "callback_type": "success",
                        "callback_vars": {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
                    }
                ]
            }
        )

        assert resolve_tenant_otel_destinations(auth) == ()

    def test_a_host_without_its_key_pair_resolves_to_nothing(self):
        assert destination_for("langfuse_otel", {"langfuse_host": "http://team.local"}) is None

    def test_a_backend_with_no_dynamic_credentials_has_no_destination(self):
        assert "arize_phoenix" not in destination_capable_backends()
        assert destination_for("arize_phoenix", {"arize_api_key": "k"}) is None

    def test_the_destination_header_string_survives_the_exporter_round_trip(self):
        from litellm.integrations.otel.plumbing.providers import parse_headers

        destination = destination_for(
            "langfuse_otel",
            {"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_host": "http://x"},
        )
        assert parse_headers(destination.header_string())["authorization"] == destination.headers["Authorization"]


class TestPresetDegradation:
    def test_a_credential_less_langfuse_exports_nowhere_instead_of_to_the_console(self, monkeypatch, capsys):
        """``_normalize`` folds a console exporter in for an empty list, which would
        print every span on a proxy whose teams bring their own credentials."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        config = langfuse_preset(allow_missing_credentials=True)
        provider = build_tracer_provider(config, tenant_overrides=True)
        capsys.readouterr()
        in_fresh_context(emit, provider)
        provider.force_flush()

        assert capsys.readouterr().out == ""
        assert "langfuse" in config.mapper_names

    def test_langfuse_still_raises_for_a_global_callback_with_no_credentials(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            langfuse_preset()

    def test_a_credential_less_proxy_still_builds_the_v2_logger(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        is_otel_v2_enabled.cache_clear()
        logger = _maybe_construct_otel_v2("langfuse_otel", [])
        is_otel_v2_enabled.cache_clear()

        assert logger is not None, "team-only deployments must not fall back to the legacy integration"
        assert all(spec.requires_headers and not spec.headers for spec in logger.config.exporters)


class TestContextIsolation:
    def test_destinations_do_not_leak_between_requests(self):
        def first():
            set_request_destinations((LANGFUSE_DEST,))
            return overridden_backends()

        assert in_fresh_context(first) == frozenset({"langfuse_otel"})
        assert in_fresh_context(request_destinations) == ()
