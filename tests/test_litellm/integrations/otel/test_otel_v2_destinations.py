"""Key/team OTLP destinations override the operator's exporters for that backend."""

import contextvars
import time
from collections.abc import Mapping

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.logger import (
    OpenTelemetryV2,
    publish_global_otel_v2_provider,
)
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


@pytest.fixture
def allow_test_hosts(monkeypatch):
    """A tenant-supplied host must be allowlisted by the operator. Allowlist the ones
    these fixtures name so the resolution tests stay about resolution;
    ``TestTenantHostSsrfGuard`` covers the guard itself."""
    monkeypatch.setattr(
        litellm, "provider_url_destination_allowed_hosts", ["team.local", "key.local", "x"], raising=False
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

    def test_a_team_naming_two_backends_gets_the_trace_at_both(self):
        """The fan-out rides one provider, so it cannot skip a destination on the
        grounds that some other backend owns it: nothing else would deliver it."""
        langfuse, arize = InMemorySpanExporter(), InMemorySpanExporter()
        by_endpoint = {"http://a.local": langfuse, "http://b.local": arize}
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda d: SimpleSpanProcessor(by_endpoint[d.endpoint]))
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

        assert [s.name for s in langfuse.get_finished_spans()] == ["chat gpt-4"]
        assert [s.name for s in arize.get_finished_spans()] == ["chat gpt-4"]

    def test_a_destination_carries_the_tenants_service_name(self):
        """An overridden backend skips per-request tracer routing, so the service name
        that route used to apply has to travel on the destination instead."""
        dest = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest)))

        def run():
            set_request_destinations(
                (
                    OtelDestination(
                        endpoint="http://a.local",
                        callback_name="langfuse_otel",
                        resource_attributes={"service.name": "team-checkout"},
                    ),
                )
            )
            emit(provider)

        in_fresh_context(run)

        assert {s.resource.attributes["service.name"] for s in dest.get_finished_spans()} == {"team-checkout"}

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
        assert "_OverriddenBackendFilter" not in kinds(tenant), "a per-tenant provider must not filter itself out"
        assert "TenantFanOutSpanProcessor" not in kinds(operator), "delivery belongs to the published global alone"
        assert "TenantFanOutSpanProcessor" not in kinds(tenant)

    def test_only_the_published_global_provider_delivers_to_tenants(self):
        """A second v2 logger's provider never sees the server, auth or database spans,
        so fanning out from it would hand the tenant a one-span trace. Publishing is
        what picks the one provider the whole request tree passes through."""
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.ARIZE_AX)])
        published, other = OpenTelemetryV2(config=config, callback_name="arize"), OpenTelemetryV2(config=config)

        publish_global_otel_v2_provider([other], lambda _p: None, registered=published)

        def kinds(logger):
            return [type(p).__name__ for p in logger._tracer_provider._active_span_processor._span_processors]

        assert kinds(published).count("TenantFanOutSpanProcessor") == 1
        assert "TenantFanOutSpanProcessor" not in kinds(other)

    def test_publishing_twice_does_not_double_export(self):
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.ARIZE_AX)])
        logger = OpenTelemetryV2(config=config, callback_name="arize")

        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)
        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)

        kinds = [type(p).__name__ for p in logger._tracer_provider._active_span_processor._span_processors]
        assert kinds.count("TenantFanOutSpanProcessor") == 1


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

    def test_an_overridden_backend_does_not_detach_on_a_service_name_either(self):
        """A key or team service name is its own reason to build a second provider, so
        clearing only the credentials would still take the model call out of the tree."""
        config = OpenTelemetryV2Config(
            exporters=[ExporterSpec(kind="otlp_http", endpoint="http://op.local", owner=ExporterOwner.LANGFUSE_OTEL)]
        )
        cache = TenantTracerCache(config, "langfuse_otel", "litellm")
        default = get_tracer(TracerProvider(), "litellm")
        auth_metadata = {"otel_service_name": "team-checkout"}

        assert cache.route_for(default, None, auth_metadata).detached is False
        assert cache.route_for(default, None, auth_metadata).tracer is not default

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return cache.route_for(default, None, auth_metadata)

        route = in_fresh_context(run)
        assert route.tracer is default, "the fan-out carries the service name on the destination instead"
        assert route.provider is None


@pytest.mark.usefixtures("allow_test_hosts")
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

    def test_a_keys_service_name_outranks_its_teams_on_the_destination(self, monkeypatch):
        """The key/team ``otel_service_name`` used to reach the backend through
        per-request tracer routing, which an overridden backend skips."""
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            metadata={"otel_service_name": "key-svc"},
            team_metadata={
                "otel_service_name": "team-svc",
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
                ],
            },
        )

        destinations = resolve_tenant_otel_destinations(auth)

        assert dict(destinations[0].resource_attributes) == {"service.name": "key-svc"}

    def test_a_team_that_named_no_service_name_gets_no_resource_override(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            team_metadata={
                "otel_service_name": "   ",
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
                ],
            }
        )

        destinations = resolve_tenant_otel_destinations(auth)

        assert dict(destinations[0].resource_attributes) == {}

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


#: Anything that makes ``OpenTelemetryV2Config`` synthesize a real operator destination.
_OTEL_SHORTHAND_ENV = (
    "OTEL_ENDPOINT",
    "OTEL_HEADERS",
    "OTEL_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
)


def credential_less_proxy(monkeypatch) -> None:
    """An operator with no Langfuse account and no generic OTLP collector."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", *_OTEL_SHORTHAND_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        langfuse_preset()


class TestPresetDegradation:
    def test_a_credential_less_langfuse_exports_nowhere_instead_of_to_the_console(self, monkeypatch, capsys):
        """``_normalize`` folds a console exporter in for an empty list, which would
        print every span on a proxy whose teams bring their own credentials."""
        credential_less_proxy(monkeypatch)

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

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

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


class TestOperatorShorthandSurvivesDegradation:
    def test_a_generic_otlp_collector_keeps_receiving_when_langfuse_has_no_credentials(self, monkeypatch):
        """Only the stdout placeholder is dropped. An operator who set the standard
        OTLP env vars configured a real destination and must keep it."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")

        config = langfuse_preset(allow_missing_credentials=True)

        assert [spec.endpoint for spec in config.exporters] == ["http://collector.local:4318", None]
        assert [spec.kind for spec in config.exporters] == ["otlp_http", "console"]

    def test_the_stdout_placeholder_is_still_dropped_when_it_is_the_only_exporter(self, monkeypatch):
        credential_less_proxy(monkeypatch)

        config = langfuse_preset(allow_missing_credentials=True)

        assert all(spec.requires_headers and not spec.headers for spec in config.exporters)


class TestBackendEndpointParity:
    def test_arize_follows_its_own_http_endpoint_instead_of_the_grpc_default(self, monkeypatch):
        monkeypatch.delenv("ARIZE_ENDPOINT", raising=False)
        monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "https://otlp.arize.com/v1/traces")

        destination = destination_for("arize", {"arize_space_id": "s", "arize_api_key": "k"})

        assert destination.endpoint == "https://otlp.arize.com/v1/traces"
        assert destination.protocol == "otlp_http"

    def test_arize_uses_grpc_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.delenv("ARIZE_ENDPOINT", raising=False)
        monkeypatch.delenv("ARIZE_HTTP_ENDPOINT", raising=False)

        destination = destination_for("arize", {"arize_space_id": "s", "arize_api_key": "k"})

        assert destination.endpoint == "https://otlp.arize.com/v1"
        assert destination.protocol == "otlp_grpc"

    def test_weave_follows_a_self_hosted_wandb_host(self, monkeypatch):
        monkeypatch.setenv("WANDB_HOST", "weave.internal.example")

        destination = destination_for("weave_otel", {"wandb_api_key": "k", "weave_project_id": "e/p"})

        assert destination.endpoint == "https://weave.internal.example/otel/v1/traces"

    def test_weave_uses_the_cloud_endpoint_without_a_host(self, monkeypatch):
        monkeypatch.delenv("WANDB_HOST", raising=False)

        destination = destination_for("weave_otel", {"wandb_api_key": "k", "weave_project_id": "e/p"})

        assert destination.endpoint == "https://trace.wandb.ai/otel/v1/traces"


class TestIncompleteCredentials:
    """Half a credential set builds a non-empty but unusable header dict. Accepting it
    would suppress the operator's exporter and send the trace where it cannot land."""

    @pytest.mark.parametrize(
        "callback_name,callback_vars",
        [
            ("arize", {"arize_api_key": "k"}),
            ("arize", {"arize_space_id": "s"}),
            ("weave_otel", {"wandb_api_key": "k"}),
            ("weave_otel", {"weave_project_id": "e/p"}),
            ("langfuse_otel", {"langfuse_public_key": "pk"}),
        ],
    )
    def test_a_partial_credential_set_resolves_to_nothing(self, callback_name, callback_vars):
        assert destination_for(callback_name, callback_vars) is None

    @pytest.mark.parametrize(
        "callback_name,callback_vars",
        [
            ("arize", {"arize_space_id": "s", "arize_api_key": "k"}),
            ("weave_otel", {"wandb_api_key": "k", "weave_project_id": "e/p"}),
            ("newrelic", {"newrelic_api_key": "k"}),
        ],
    )
    def test_a_complete_credential_set_resolves(self, callback_name, callback_vars):
        assert destination_for(callback_name, callback_vars) is not None


@pytest.mark.usefixtures("allow_test_hosts")
class TestCallbackTypeFilter:
    @staticmethod
    def _auth(callback_type: str | None) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "langfuse_otel",
                        "callback_type": callback_type,
                        "callback_vars": {
                            "langfuse_public_key": "pk",
                            "langfuse_secret_key": "sk",
                            "langfuse_host": "http://team.local",
                        },
                    }
                ]
            }
        )

    @pytest.mark.parametrize("callback_type", ["success", "success_and_failure", None])
    def test_an_entry_that_wants_success_traces_gets_the_whole_trace(self, monkeypatch, callback_type):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()

        assert resolve_tenant_otel_destinations(self._auth(callback_type)) != ()

    def test_a_failure_only_entry_does_not_take_over_the_trace(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()

        assert resolve_tenant_otel_destinations(self._auth("failure")) == ()


class TestTenantConfigAgreement:
    """The destination resolver and ``convert_key_logging_metadata_to_callback`` read
    the same stored config, so they must not read it two different ways."""

    @pytest.fixture(autouse=True)
    def _v2_on(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        monkeypatch.setattr(
            litellm, "provider_url_destination_allowed_hosts", ["team.local", "key.local"], raising=False
        )
        is_otel_v2_enabled.cache_clear()
        yield
        is_otel_v2_enabled.cache_clear()

    @staticmethod
    def _entry(host, **extra):
        return {
            "callback_name": "langfuse_otel",
            "callback_vars": {"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_host": host, **extra},
        }

    def test_a_key_that_disabled_its_callbacks_does_not_fall_back_to_the_team(self):
        """Disabling a key's callbacks stores an empty list, which the sibling parser
        reads as 'the key configured none'."""
        auth = UserAPIKeyAuth(
            metadata={"logging": []},
            team_metadata={"logging": [self._entry("http://team.local")]},
        )

        assert resolve_tenant_otel_destinations(auth) == ()

    def test_two_entries_for_one_backend_merge_their_vars_last_wins(self):
        auth = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    self._entry("http://team.local"),
                    {"callback_name": "langfuse_otel", "callback_vars": {"langfuse_host": "http://key.local"}},
                ]
            }
        )

        destinations = resolve_tenant_otel_destinations(auth)

        assert [d.endpoint for d in destinations] == ["http://key.local/api/public/otel"]


class TestEvictionSafety:
    class Recording(SimpleSpanProcessor):
        def __init__(self):
            super().__init__(InMemorySpanExporter())
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    def _fan_out(self):
        built = []

        def factory(_destination):
            built.append(self.Recording())
            return built[-1]

        return TenantFanOutSpanProcessor(processor_factory=factory), built

    @staticmethod
    def _dest(index):
        return LANGFUSE_DEST.model_copy(update={"endpoint": f"http://d{index}/otel"})

    @staticmethod
    def _settle(fan_out, processor=None):
        """Wait for retirement to clear and, when given, for the drain to run.

        The drain pool is shared and bounded, so a shed processor is closed once a
        worker picks it up rather than the moment it is handed over.
        """
        for _ in range(500):
            if not fan_out._retired and (processor is None or processor.shutdown_calls):
                return
            time.sleep(0.02)

    def test_a_processor_still_exporting_a_span_is_not_closed_under_it(self):
        """``on_end`` holds a processor across the export, so closing an evicted one
        there drops the span it is holding."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built = self._fan_out()
        held = fan_out._acquire(self._dest(0))
        for index in range(1, _MAX_CACHED_DESTINATION_PROCESSORS + 1):
            fan_out._acquire(self._dest(index))
            fan_out._release(built[-1])

        assert held.shutdown_calls == 0
        assert id(held) in fan_out._retired

        fan_out._release(held)
        self._settle(fan_out)

        assert held.shutdown_calls == 1

    def test_an_idle_evicted_processor_is_closed_off_the_export_path(self):
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built = self._fan_out()
        for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 1):
            fan_out._acquire(self._dest(index))
            fan_out._release(built[-1])
        self._settle(fan_out, built[0])

        assert built[0].shutdown_calls == 1
        assert len(fan_out._processors) == _MAX_CACHED_DESTINATION_PROCESSORS

    def test_a_slow_collector_does_not_hold_up_the_export_path(self):
        """``shutdown`` flushes over the network and is reached from ``on_end``, so
        closing a shed processor inline lets one unreachable tenant collector stall
        every other tenant's spans."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        class Slow(self.Recording):
            def shutdown(self):
                time.sleep(3)
                super().shutdown()

        built = []

        def factory(_destination):
            built.append(Slow())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory)
        started = time.monotonic()
        for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 1):
            fan_out._acquire(self._dest(index))
            fan_out._release(built[-1])

        assert time.monotonic() - started < 2

    def test_shedding_many_processors_does_not_spawn_a_thread_each(self):
        """A tenant that cycles its destination config sheds a processor per request,
        so a thread per shed processor is a thread per request against a slow
        collector."""
        import threading

        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        release = threading.Event()

        class Blocking(self.Recording):
            def shutdown(self):
                release.wait(timeout=10)
                super().shutdown()

        built = []

        def factory(_destination):
            built.append(Blocking())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory)
        try:
            for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 30):
                fan_out._acquire(self._dest(index))
                fan_out._release(built[-1])
            draining = [t for t in threading.enumerate() if t.name.startswith("litellm-otel-destination-drain")]
            assert len(draining) <= 2, f"one drain thread per shed processor: {len(draining)}"
        finally:
            release.set()
            self._settle(fan_out, built[0])

    def test_a_retired_processor_is_still_closed_on_shutdown(self):
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built = self._fan_out()
        held = fan_out._acquire(self._dest(0))
        for index in range(1, _MAX_CACHED_DESTINATION_PROCESSORS + 1):
            fan_out._acquire(self._dest(index))
            fan_out._release(built[-1])
        fan_out.shutdown()

        assert held.shutdown_calls == 1


class TestCredentialGatedExporters:
    def test_layering_a_second_preset_does_not_eat_the_first_gated_exporter(self, monkeypatch):
        """``base.Preset`` advertises ``config_overrides`` layering, and the gated spec
        is itself a console exporter with no endpoint."""
        credential_less_proxy(monkeypatch)
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        once = credential_gated_exporters((), ExporterOwner.LANGFUSE_OTEL)
        twice = credential_gated_exporters(once, ExporterOwner.WEAVE_OTEL)

        assert [spec.owner for spec in twice] == [ExporterOwner.LANGFUSE_OTEL, ExporterOwner.WEAVE_OTEL]

    def test_an_exporter_the_operator_configured_survives(self):
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        operator_console = ExporterSpec(kind="console", use_simple_processor=True)

        kept = credential_gated_exporters((operator_console,), ExporterOwner.LANGFUSE_OTEL)

        assert kept[0] == operator_console

    def test_an_otlp_exporter_on_its_default_endpoint_survives(self):
        """``OTEL_EXPORTER=otlp_http`` with no endpoint is a real collector on the SDK's
        default port, not the placeholder, so the transport is what tells them apart."""
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        operator_otlp = ExporterSpec(kind="otlp_http", endpoint=None, headers=None)

        kept = credential_gated_exporters((operator_otlp,), ExporterOwner.LANGFUSE_OTEL)

        assert kept[0] == operator_otlp

    def test_an_in_memory_exporter_the_operator_asked_for_survives(self):
        """``OTEL_EXPORTER=in_memory`` stores spans, so it is a destination the operator
        chose, not the placeholder that stands in for choosing nothing."""
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        operator_memory = ExporterSpec(kind="in_memory", endpoint=None, headers=None)

        kept = credential_gated_exporters((operator_memory,), ExporterOwner.LANGFUSE_OTEL)

        assert kept[0] == operator_memory

    def test_the_synthesized_stdout_placeholder_is_dropped(self):
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        placeholder = ExporterSpec(kind="console", endpoint=None, headers=None)

        kept = credential_gated_exporters((placeholder,), ExporterOwner.LANGFUSE_OTEL)

        assert [spec.owner for spec in kept] == [ExporterOwner.LANGFUSE_OTEL]


class TestTenantHostSsrfGuard:
    """Anyone who can mint a key can write ``langfuse_host``, so the host it names has
    to be one the operator approved."""

    @pytest.fixture(autouse=True)
    def _guard_on(self, monkeypatch):
        from litellm.integrations.otel.presets.destinations import _warn_host_not_allowlisted

        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", [], raising=False)
        _warn_host_not_allowlisted.cache_clear()
        yield
        _warn_host_not_allowlisted.cache_clear()

    @staticmethod
    def _langfuse(host: str) -> Mapping[str, str]:
        return {"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_host": host}

    @pytest.mark.parametrize(
        "host",
        [
            "http://127.0.0.1:9111",
            "http://169.254.169.254",
            "http://10.0.0.5:3000",
            "https://collector.example.com",
            "https://langfuse.corp:99999",
            "ftp://collector.example.com",
        ],
    )
    def test_a_host_the_operator_never_approved_resolves_to_nothing(self, host):
        assert destination_for("langfuse_otel", self._langfuse(host)) is None

    def test_userinfo_naming_an_allowlisted_host_does_not_smuggle_a_second_one(self, monkeypatch):
        """``https://allowed@10.0.0.5`` reads as the allowlisted host to the eye and
        posts to 10.0.0.5 on the wire."""
        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", ["collector.example.com"], raising=False)

        assert destination_for("langfuse_otel", self._langfuse("https://collector.example.com@10.0.0.5")) is None

    def test_a_malformed_host_does_not_take_the_other_backends_with_it(self, monkeypatch):
        """``urlparse(...).port`` raises a bare ValueError, which would escape
        ``destination_for`` and kill the whole resolution."""
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        monkeypatch.setenv("NEW_RELIC_OTEL_ENDPOINT", "https://otlp.nr-data.net")
        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", ["collector.example.com"], raising=False)
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            token="hashed",
            team_metadata={
                "logging": [
                    {"callback_name": "langfuse_otel", "callback_vars": self._langfuse("https://lf.corp:99999")},
                    {"callback_name": "newrelic", "callback_vars": {"newrelic_api_key": "nr"}},
                ]
            },
        )

        assert [d.callback_name for d in resolve_tenant_otel_destinations(auth)] == ["newrelic"]

    def test_the_operator_can_allowlist_its_teams_internal_langfuse(self, monkeypatch):
        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", ["127.0.0.1:9111"], raising=False)

        destination = destination_for("langfuse_otel", self._langfuse("http://127.0.0.1:9111"))

        assert destination.endpoint == "http://127.0.0.1:9111/api/public/otel"

    def test_the_operators_own_internal_host_is_never_blocked(self, monkeypatch):
        """The operator configures ``LANGFUSE_HOST`` themselves, so an internal
        collector there is a deployment choice rather than caller-supplied input."""
        monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9111")

        destination = destination_for("langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"})

        assert destination.endpoint == "http://127.0.0.1:9111/api/public/otel"

    def test_an_allowlisted_host_is_taken_without_resolving_it(self, monkeypatch):
        """The check runs on the asyncio auth path, so it must not block on a name the
        caller chose. ``.invalid`` never resolves, and it is still accepted."""
        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", ["lf.invalid"], raising=False)

        destination = destination_for("langfuse_otel", self._langfuse("https://lf.invalid"))

        assert destination.endpoint == "https://lf.invalid/api/public/otel"

    def test_a_rejected_host_is_warned_about_once(self, caplog):
        with caplog.at_level("WARNING", logger="LiteLLM"):
            for _ in range(3):
                destination_for("langfuse_otel", self._langfuse("http://10.0.0.5:3000"))

        assert sum("provider_url_destination_allowed_hosts" in record.message for record in caplog.records) == 1
