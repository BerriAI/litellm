"""Key/team OTLP destinations override the operator's exporters for that backend."""

import asyncio
import contextvars
import time
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import reduce
from types import MappingProxyType
from typing import Final

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel import GenAIOperation
from litellm.integrations.otel import logger as otel_logger
from litellm.integrations.otel.logger import (
    OpenTelemetryV2,
    build_otel_v2_logger,
    fan_out_provider,
    publish_global_otel_v2_provider,
)
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.model.config import (
    CaptureMessageContent,
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    MCPToolCallSpanData,
    RequestIdentity,
    ServerInfo,
    ToolDefinition,
)
from litellm.integrations.otel.plumbing import providers as otel_providers
from litellm.integrations.otel.plumbing.context import (
    destination_backends,
    request_destinations,
    set_request_destinations,
)
from litellm.integrations.otel.plumbing.providers import (
    TenantFanOutSpanProcessor,
    _OverriddenBackendFilter,
    _sink_key,
    attach_tenant_fan_out,
    build_tracer_provider,
    deliverable_destinations,
    operator_sink_scopes,
    register_exporter_factory,
)
from litellm.integrations.otel.plumbing.routing import TenantTracerCache, get_tracer
from litellm.integrations.otel.presets.arize import arize_preset
from litellm.integrations.otel.presets.destinations import (
    destination_capable_backends,
    destination_for,
)
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.proxy._types import AddTeamCallback, TeamCallbackMetadata, UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    convert_key_logging_metadata_to_callback,
    resolve_tenant_otel_destinations,
)
from litellm.types.utils import StandardCallbackDynamicParams

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


@pytest.fixture(autouse=True)
def isolate_published_provider(monkeypatch):
    """Publishing records the fan-out carrier in module state; one test's publish must
    not become the next test's provider."""
    monkeypatch.setattr(otel_logger, "_published_v2_provider", None)


@pytest.fixture(autouse=True)
def forget_otel_v2_flag_after_each_test():
    yield
    is_otel_v2_enabled.cache_clear()


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


class TestRoutingMode:
    """The operator's choice between replacing its own exporter and exporting alongside it.

    One org-wide backend across every team is a real deployment, and losing it the
    moment a team configures its own is what ``additive`` exists to prevent.
    """

    OPERATOR_SINK = ("https://cloud.langfuse.com/api/public/otel/v1/traces", (("authorization", "Basic op"),))
    #: What a tenant destination for that same project looks like before normalizing:
    #: no signal path yet, and the header name cased the way the backend writes it.
    SAME_ACCOUNT_ENDPOINT = "https://cloud.langfuse.com/api/public/otel"

    @staticmethod
    def _additive(monkeypatch):
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", "additive", raising=False)

    @staticmethod
    def _tree(provider):
        tracer = get_tracer(provider, "litellm")
        with tracer.start_as_current_span("POST /v1/chat/completions"):
            with tracer.start_as_current_span("auth /v1/chat/completions"):
                pass
            with tracer.start_as_current_span("chat gpt-4"):
                pass

    def _run(self, provider, destinations=(LANGFUSE_DEST,)):
        def run():
            set_request_destinations(destinations)
            self._tree(provider)

        in_fresh_context(run)

    def test_global_only_keeps_every_span_and_delivers_to_nobody(self):
        """No team destination resolved, so the operator's backbone is untouched."""
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        self._run(provider, destinations=())

        assert len(global_exporter.get_finished_spans()) == 3
        assert dest_exporter.get_finished_spans() == ()

    def test_team_only_gets_the_whole_tree_with_no_operator_exporter(self):
        """A deployment with no operator credentials still gives the team its trace."""
        dest_exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )

        self._run(provider)

        assert {s.name for s in dest_exporter.get_finished_spans()} == {
            "POST /v1/chat/completions",
            "auth /v1/chat/completions",
            "chat gpt-4",
        }

    def test_additive_gives_the_operator_and_the_team_the_same_tree(self, monkeypatch):
        self._additive(monkeypatch)
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        self._run(provider)

        names = {"POST /v1/chat/completions", "auth /v1/chat/completions", "chat gpt-4"}
        assert {s.name for s in global_exporter.get_finished_spans()} == names
        assert {s.name for s in dest_exporter.get_finished_spans()} == names
        assert len(global_exporter.get_finished_spans()) == 3, "the operator must not get a span twice"

    def test_override_moves_the_tree_off_the_operator(self):
        """The default, unchanged: the tenant's traffic reaches the tenant and nowhere else."""
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        self._run(provider)

        assert global_exporter.get_finished_spans() == ()
        assert len(dest_exporter.get_finished_spans()) == 3

    def test_a_team_naming_the_operators_own_project_is_written_once(self, monkeypatch):
        """Fanning out to two accounts is the point. Writing the same account twice
        is a duplicate the operator would see in their own project."""
        self._additive(monkeypatch)
        shared = InMemorySpanExporter()
        same = OtelDestination(
            endpoint=self.SAME_ACCOUNT_ENDPOINT,
            headers=MappingProxyType({"Authorization": "Basic op"}),
            callback_name="langfuse_otel",
        )
        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(shared), "langfuse_otel"))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=MappingProxyType({self.OPERATOR_SINK: "full"}),
            )
        )

        self._run(provider, destinations=(same,))

        assert len(shared.get_finished_spans()) == 3, "the same account received the trace twice"

    def test_in_override_a_team_naming_the_operators_project_still_gets_the_trace(self):
        """Override suppresses the operator's own exporter, so the fan-out is the only
        thing left delivering. Skipping it on a matching account leaves the team with
        nothing at all."""
        shared = InMemorySpanExporter()
        same = OtelDestination(
            endpoint=self.SAME_ACCOUNT_ENDPOINT,
            headers=MappingProxyType({"Authorization": "Basic op"}),
            callback_name="langfuse_otel",
        )
        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(shared), "langfuse_otel"))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=MappingProxyType({self.OPERATOR_SINK: "full"}),
            )
        )

        self._run(provider, destinations=(same,))

        assert len(shared.get_finished_spans()) == 3, "the team's own destination received nothing"

    def test_a_team_naming_a_different_project_still_gets_its_copy(self, monkeypatch):
        """The dedup keys on the account, so a second project is still a second copy."""
        self._additive(monkeypatch)
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(global_exporter), "langfuse_otel"))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter),
                operator_sinks=MappingProxyType({self.OPERATOR_SINK: "full"}),
            )
        )

        self._run(provider)

        assert len(global_exporter.get_finished_spans()) == 3
        assert len(dest_exporter.get_finished_spans()) == 3

    @pytest.mark.parametrize("additive", [True, False])
    def test_a_failing_team_destination_leaves_the_operator_alone(self, monkeypatch, additive):
        """A tenant collector that raises on every span must not cost the operator
        its own telemetry, nor take the request down with it."""
        if additive:
            self._additive(monkeypatch)
        global_exporter, arize_exporter = InMemorySpanExporter(), InMemorySpanExporter()

        class Exploding(SimpleSpanProcessor):
            def on_end(self, span):
                raise RuntimeError("tenant collector is down")

        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(global_exporter), "langfuse_otel"))
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(arize_exporter), "arize"))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: Exploding(InMemorySpanExporter()))
        )

        self._run(provider)

        assert len(arize_exporter.get_finished_spans()) == 3, "an unrelated backend lost spans"
        assert len(global_exporter.get_finished_spans()) == (3 if additive else 0)

    def test_the_env_var_turns_additive_on_without_a_config_file(self, monkeypatch):
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", None, raising=False)
        monkeypatch.setenv("LITELLM_OTEL_TENANT_DESTINATION_MODE", "Additive")
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        self._run(provider)

        assert len(global_exporter.get_finished_spans()) == 3
        assert len(dest_exporter.get_finished_spans()) == 3

    def test_an_unrecognized_mode_stays_on_override(self, monkeypatch):
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", "both", raising=False)
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        self._run(provider)

        assert global_exporter.get_finished_spans() == ()

    def test_operator_sink_scopes_skips_an_exporter_with_no_endpoint_of_its_own(self):
        """Such an exporter resolves its endpoint from the environment at export
        time, so it has no identity to compare a destination against."""
        config = OpenTelemetryV2Config(
            exporters=(
                ExporterSpec(kind="otlp_http", endpoint=self.OPERATOR_SINK[0], headers="authorization=Basic op"),
                ExporterSpec(kind="otlp_http", endpoint=None, headers="authorization=Basic other"),
            )
        )

        assert dict(operator_sink_scopes(config)) == {self.OPERATOR_SINK: "full"}

    def test_operator_sink_scopes_skips_exporters_that_never_reach_the_wire(self):
        """A console kind ignores the endpoint and a header-gated spec with no
        credentials is dropped when the provider is built, so treating either as an
        account the operator writes to would silently withhold a team's own spans
        under additive."""
        config = OpenTelemetryV2Config(
            exporters=(
                ExporterSpec(kind="otlp_http", endpoint=self.OPERATOR_SINK[0], headers="authorization=Basic op"),
                ExporterSpec(kind="console", endpoint="http://team.local/v1/traces"),
                ExporterSpec(kind="otlp_http", endpoint="http://gated.local/v1/traces", requires_headers=True),
            )
        )

        assert dict(operator_sink_scopes(config)) == {self.OPERATOR_SINK: "full"}

    def test_operator_sink_scopes_spans_every_config_it_is_handed(self):
        first = OpenTelemetryV2Config(
            exporters=(
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=self.OPERATOR_SINK[0],
                    headers="authorization=Basic op",
                ),
            )
        )
        second = OpenTelemetryV2Config(
            exporters=(
                ExporterSpec(
                    kind="otlp_http",
                    endpoint="https://otlp.arize.com/v1/traces",
                    headers="space_id=s,api_key=k",
                ),
            )
        )

        assert dict(operator_sink_scopes(first, second)) == {
            self.OPERATOR_SINK: "full",
            _sink_key("https://otlp.arize.com/v1/traces", {"space_id": "s", "api_key": "k"}): "full",
        }

    @pytest.mark.parametrize("langfuse_first", [False, True])
    def test_two_operator_exporters_on_one_account_record_the_wider_scope(self, langfuse_first):
        langfuse = ExporterSpec(
            kind="otlp_http",
            endpoint=self.OPERATOR_SINK[0],
            headers="authorization=Basic op",
            owner=ExporterOwner.LANGFUSE_OTEL,
        )
        collector = ExporterSpec(kind="otlp_http", endpoint=self.OPERATOR_SINK[0], headers="authorization=Basic op")
        config = OpenTelemetryV2Config(
            langfuse_span_scope="llm_only",
            exporters=(langfuse, collector) if langfuse_first else (collector, langfuse),
        )

        assert dict(operator_sink_scopes(config)) == {self.OPERATOR_SINK: "full"}

    def test_a_team_pointing_at_a_credential_less_operator_exporter_still_gets_its_spans(self, monkeypatch):
        """Under additive the fan-out skips a destination the operator already writes
        to. An exporter the provider never built writes nothing, so skipping it would
        cost the team every span."""
        monkeypatch.setenv("LITELLM_OTEL_TENANT_DESTINATION_MODE", "additive")
        gated_endpoint = "http://gated.local/v1/traces"
        destination = OtelDestination(endpoint=gated_endpoint, callback_name="newrelic")
        config = OpenTelemetryV2Config(
            exporters=(ExporterSpec(kind="otlp_http", endpoint=gated_endpoint, requires_headers=True),)
        )
        dest_exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter),
                operator_sinks=operator_sink_scopes(config),
            )
        )

        def run():
            set_request_destinations((destination,))
            emit(provider)

        in_fresh_context(run)

        assert [s.name for s in dest_exporter.get_finished_spans()] == ["chat gpt-4"]

    def test_the_operators_own_langfuse_and_a_team_naming_it_are_one_account(self, monkeypatch):
        """The two sides are built by different code that writes the endpoint and the
        header names differently, so comparing them raw silently never matches."""
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.internal")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-op")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-op")
        monkeypatch.setattr(litellm, "provider_url_destination_allowed_hosts", ["lf.internal"], raising=False)
        operator = operator_sink_scopes(langfuse_preset())

        def sink(public_key, secret_key):
            destination = destination_for(
                "langfuse_otel",
                StandardCallbackDynamicParams(
                    langfuse_public_key=public_key,
                    langfuse_secret_key=secret_key,
                    langfuse_host="https://lf.internal",
                ),
            )
            assert destination is not None
            return _sink_key(destination.endpoint, destination.headers)

        assert sink("pk-op", "sk-op") in operator, "a team naming the operator's own project"
        assert sink("pk-team", "sk-team") not in operator, "a different project on the same server"

    def test_two_accounts_holding_the_same_strings_in_different_roles_are_not_one(self):
        """The values alone are not the identity. Two accounts can hold the same pair
        of strings with the space id and the api key the other way round, and folding
        them together would leave the second one's team with no trace at all."""
        endpoint = "https://otlp.arize.com/v1"

        assert _sink_key(endpoint, {"space_id": "a", "api_key": "b"}) != _sink_key(
            endpoint, {"space_id": "b", "api_key": "a"}
        )

    def test_the_operators_own_arize_space_and_a_team_naming_it_are_one_account(self, monkeypatch):
        """One account answers to two header names here: the operator's exporter sends
        ``space_id`` and a team destination sends ``arize-space-id``. Keyed on the names,
        additive would write the operator's own space twice for every request."""
        monkeypatch.setenv("ARIZE_SPACE_ID", "space-op")
        monkeypatch.setenv("ARIZE_API_KEY", "key-op")
        monkeypatch.delenv("ARIZE_SPACE_KEY", raising=False)
        operator = operator_sink_scopes(arize_preset())

        def sink(space, api_key):
            destination = destination_for(
                "arize",
                StandardCallbackDynamicParams(arize_space_key=space, arize_api_key=api_key),
            )
            assert destination is not None
            return _sink_key(destination.endpoint, destination.headers)

        assert sink("space-op", "key-op") in operator, "a team naming the operator's own space"
        assert sink("space-team", "key-team") not in operator, "a different Arize space"


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

    def test_excluded_services_drop_only_the_datastore_spans_at_the_tenant(self):
        """The exclusion is per ``db.system.*`` value: a span naming an excluded
        datastore never reaches the tenant, while every span of the request's
        own work (root, auth, guardrail, model) still does, and the operator's
        own exporter keeps the full tree."""
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter),
                excluded_db_systems=frozenset({"redis", "postgresql"}),
            )
        )
        tracer = get_tracer(provider, "litellm")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                with tracer.start_as_current_span("auth /v1/chat/completions"):
                    pass
                with tracer.start_as_current_span("execute_guardrail pii"):
                    pass
                with tracer.start_as_current_span("redis async_get_cache") as redis_span:
                    redis_span.set_attribute("db.system.name", "redis")
                with tracer.start_as_current_span("batch_write_to_db _PROXY_track_cost_callback") as spend_span:
                    spend_span.set_attribute("db.system", "postgresql")
                with tracer.start_as_current_span("chat gpt-4"):
                    pass

        in_fresh_context(run)

        assert {s.name for s in dest_exporter.get_finished_spans()} == {
            "POST /v1/chat/completions",
            "auth /v1/chat/completions",
            "execute_guardrail pii",
            "chat gpt-4",
        }
        assert {s.name for s in operator_exporter.get_finished_spans()} == {
            "POST /v1/chat/completions",
            "auth /v1/chat/completions",
            "execute_guardrail pii",
            "redis async_get_cache",
            "batch_write_to_db _PROXY_track_cost_callback",
            "chat gpt-4",
        }

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

    def test_the_operators_database_endpoint_does_not_ride_along_to_the_tenant(self):
        """A database span describes the proxy's own Postgres, so the tenant gets the
        span and its timing without the host, the port, the schema or the error text
        that names them. The operator's own copy keeps everything."""
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")
        unreachable = "Can't reach database server at db.internal.example:15400"

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("postgres get_data") as db_span:
                db_span.set_attributes(
                    {
                        "db.system.name": "postgresql",
                        "db.system": "postgresql",
                        "db.operation.name": "get_data",
                        "server.address": "db.internal.example",
                        "server.port": 15400,
                        "db.namespace": "litellm",
                        "error.type": "PrismaError",
                        "error.message": unreachable,
                        "error": unreachable,
                        "litellm.provider.error.stack_trace": f"Traceback: {unreachable}",
                    }
                )
                db_span.add_event("exception", {"exception.message": unreachable})
                db_span.set_status(Status(StatusCode.ERROR, unreachable))
            with tracer.start_as_current_span("chat claude-haiku") as llm_span:
                llm_span.set_attribute("server.address", "api.anthropic.com")

        in_fresh_context(run)

        tenant = {s.name: s for s in dest_exporter.get_finished_spans()}
        operator = {s.name: s for s in operator_exporter.get_finished_spans()}
        assert set(tenant) == {"postgres get_data", "chat claude-haiku"}, "the tenant keeps the whole tree"
        tenant_db = tenant["postgres get_data"]
        assert dict(tenant_db.attributes) == {
            "db.system.name": "postgresql",
            "db.system": "postgresql",
            "db.operation.name": "get_data",
            "error.type": "PrismaError",
        }
        assert list(tenant_db.events) == []
        assert tenant_db.status.status_code is StatusCode.ERROR, "the tenant still sees that the call failed"
        assert tenant_db.status.description is None
        assert "db.internal.example" not in tenant_db.to_json()
        assert tenant["chat claude-haiku"].attributes["server.address"] == "api.anthropic.com", (
            "only the operator's datastore is redacted, never the model endpoint"
        )
        operator_db = operator["postgres get_data"]
        assert operator_db.attributes["server.address"] == "db.internal.example"
        assert operator_db.attributes["server.port"] == 15400
        assert operator_db.attributes["db.namespace"] == "litellm"
        assert operator_db.attributes["error.message"] == unreachable
        assert operator_db.attributes["error"] == unreachable
        assert operator_db.status.description == unreachable
        assert [event.name for event in operator_db.events] == ["exception"]

    @pytest.mark.parametrize("failure_status", ["guardrail_failed_to_respond", "failure"])
    def test_a_guardrails_failure_text_does_not_ride_along_to_the_tenant(self, failure_status):
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")
        unreachable = "Cannot connect to host guardrail.internal.example:9000"
        verdict = '{"action": "block", "categories": ["pii"]}'

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                with tracer.start_as_current_span("execute_guardrail pii") as down:
                    down.set_attributes(
                        {
                            "litellm.guardrail.name": "pii",
                            "litellm.guardrail.status": failure_status,
                            "litellm.guardrail.response": unreachable,
                        }
                    )
                with tracer.start_as_current_span("execute_guardrail toxicity") as up:
                    up.set_attributes(
                        {
                            "litellm.guardrail.name": "toxicity",
                            "litellm.guardrail.status": "guardrail_intervened",
                            "litellm.guardrail.response": verdict,
                        }
                    )

        in_fresh_context(run)

        tenant = {s.name: s for s in dest_exporter.get_finished_spans()}
        operator = {s.name: s for s in operator_exporter.get_finished_spans()}
        assert dict(tenant["execute_guardrail pii"].attributes) == {
            "litellm.guardrail.name": "pii",
            "litellm.guardrail.status": failure_status,
        }
        assert "guardrail.internal.example" not in tenant["execute_guardrail pii"].to_json()
        assert tenant["execute_guardrail toxicity"].attributes["litellm.guardrail.response"] == verdict
        assert operator["execute_guardrail pii"].attributes["litellm.guardrail.response"] == unreachable

    def test_the_callers_key_in_the_query_string_does_not_ride_along_to_the_tenant(self):
        """A Google AI Studio style request authenticates with ``?key=<virtual key>``,
        and the instrumentor stamps the full request URL on the server span. The
        tenant keeps the URL up to the query string, and the operator's copy keeps it
        whole."""
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")
        path = "/v1beta/models/gemini-2.5-flash:generateContent"
        query = "key=sk-another-members-virtual-key&alt=sse"

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span(f"POST {path}") as server_span:
                server_span.set_attributes(
                    {
                        "http.method": "POST",
                        "http.route": path,
                        "http.target": f"{path}?{query}",
                        "http.url": f"http://proxy.example:4000{path}?{query}",
                        "url.path": path,
                        "url.query": query,
                        "http.status_code": 200,
                    }
                )
                with tracer.start_as_current_span("generate_content gemini-2.5-flash") as llm_span:
                    llm_span.set_attributes(
                        {
                            "gen_ai.operation.name": "generate_content",
                            "url.full": f"https://generativelanguage.googleapis.com{path}?key=AIza-operator-provider-key",
                        }
                    )

        in_fresh_context(run)

        tenant = {s.name: s for s in dest_exporter.get_finished_spans()}
        assert dict(tenant[f"POST {path}"].attributes) == {
            "http.method": "POST",
            "http.route": path,
            "http.target": path,
            "http.url": f"http://proxy.example:4000{path}",
            "url.path": path,
            "http.status_code": 200,
        }
        assert "sk-another-members-virtual-key" not in tenant[f"POST {path}"].to_json()
        assert tenant["generate_content gemini-2.5-flash"].attributes["url.full"] == (
            f"https://generativelanguage.googleapis.com{path}"
        ), "the tenant's own span keeps its error text, and still loses a query string"
        operator = {s.name: s for s in operator_exporter.get_finished_spans()}
        assert operator[f"POST {path}"].attributes["http.url"] == f"http://proxy.example:4000{path}?{query}"
        assert operator[f"POST {path}"].attributes["url.query"] == query
        assert "AIza-operator-provider-key" in operator["generate_content gemini-2.5-flash"].to_json()

    def test_captured_request_headers_do_not_ride_along_to_the_tenant(self):
        """With ``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST`` set, the
        server span carries the caller's bearer token. A team admin's collector must
        not receive it, while the operator's own copy keeps it and the tenant keeps the
        rest of the span."""
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")
        bearer = "Bearer sk-another-members-virtual-key"

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("POST /v1/chat/completions") as server_span:
                server_span.set_attributes(
                    {
                        "http.request.method": "POST",
                        "http.route": "/v1/chat/completions",
                        "http.request.header.authorization": (bearer,),
                        "http.request.header.x_litellm_api_key": (bearer,),
                        "http.response.header.set_cookie": ("session=abc",),
                    }
                )
                server_span.set_status(Status(StatusCode.ERROR))

        in_fresh_context(run)

        tenant = dest_exporter.get_finished_spans()[0]
        assert dict(tenant.attributes) == {"http.request.method": "POST", "http.route": "/v1/chat/completions"}
        assert bearer not in tenant.to_json()
        assert tenant.status.status_code is StatusCode.ERROR
        operator = operator_exporter.get_finished_spans()[0]
        assert operator.attributes["http.request.header.authorization"] == (bearer,)
        assert operator.attributes["http.response.header.set_cookie"] == ("session=abc",)

    def test_the_proxys_own_error_text_does_not_ride_along_to_the_tenant(self):
        """Postgres failing during auth surfaces as a ``ProxyException`` whose message
        quotes the Prisma error, so the auth span and the request root carry the
        operator's database endpoint in ``error.message``, in the exception event and
        in the status description. None of it is the tenant's, so it all comes off,
        while the failure itself (its type, its code, its status) stays. The tenant's
        own model call keeps its error text, less the stack trace that walks the
        operator's install. The operator's copy keeps everything."""
        dest_exporter, operator_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator_exporter))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        tracer = get_tracer(provider, "litellm")
        unreachable = "Authentication Error, Can't reach database server at db.internal.example:15400"
        install = "/srv/litellm/.venv/lib/python3.13/site-packages/opentelemetry/trace/__init__.py"
        provider_error = "AnthropicException - invalid x-api-key"

        def fail(span, message: str) -> None:
            span.set_attributes(
                {
                    "error.type": "ProxyException",
                    "error.message": message,
                    "litellm.provider.error.code": "500",
                    "litellm.provider.error.stack_trace": f"Traceback\n  File {install}\n{message}",
                }
            )
            span.add_event(
                "exception",
                {"exception.type": "ProxyException", "exception.message": message, "exception.stacktrace": install},
            )
            span.set_status(Status(StatusCode.ERROR, message))

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            with tracer.start_as_current_span("POST /v1/chat/completions") as root:
                with tracer.start_as_current_span("auth /v1/chat/completions") as auth:
                    fail(auth, unreachable)
                with tracer.start_as_current_span("chat claude-haiku") as llm:
                    llm.set_attribute("gen_ai.operation.name", "chat")
                    fail(llm, provider_error)
                fail(root, unreachable)

        in_fresh_context(run)

        tenant = {s.name: s for s in dest_exporter.get_finished_spans()}
        operator = {s.name: s for s in operator_exporter.get_finished_spans()}
        assert set(tenant) == {"POST /v1/chat/completions", "auth /v1/chat/completions", "chat claude-haiku"}
        for name in ("POST /v1/chat/completions", "auth /v1/chat/completions"):
            proxy_span = tenant[name]
            assert dict(proxy_span.attributes) == {"error.type": "ProxyException", "litellm.provider.error.code": "500"}
            assert list(proxy_span.events) == []
            assert proxy_span.status.status_code is StatusCode.ERROR
            assert proxy_span.status.description is None
            assert "db.internal.example" not in proxy_span.to_json()
            assert install not in proxy_span.to_json()
        llm_span = tenant["chat claude-haiku"]
        assert llm_span.attributes["error.message"] == provider_error, "the tenant's own call keeps its error text"
        assert "litellm.provider.error.stack_trace" not in llm_span.attributes
        assert llm_span.status.description == provider_error
        assert [dict(event.attributes) for event in llm_span.events] == [
            {"exception.type": "ProxyException", "exception.message": provider_error}
        ]
        assert install not in llm_span.to_json()
        for name, message in (("auth /v1/chat/completions", unreachable), ("chat claude-haiku", provider_error)):
            assert operator[name].attributes["error.message"] == message
            assert install in operator[name].attributes["litellm.provider.error.stack_trace"]
            assert operator[name].events[0].attributes["exception.stacktrace"] == install
            assert operator[name].status.description == message

    def test_a_tenants_service_name_is_layered_onto_the_operators_resource(self):
        """The destination's ``service.name`` replaces the operator's on the tenant's
        copy and every other resource attribute travels unchanged. Nothing is detected
        afresh per span, so no attribute the operator did not configure appears."""
        dest = InMemorySpanExporter()
        provider = TracerProvider(
            resource=Resource({"service.name": "litellm-proxy", "deployment.environment.name": "prod"})
        )
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

        (span,) = dest.get_finished_spans()
        assert dict(span.resource.attributes) == {
            "service.name": "team-checkout",
            "deployment.environment.name": "prod",
        }

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

    def test_an_unbuildable_destination_leaves_the_span_with_the_operator(self):
        """Anchoring the destination is what makes the operator's exporter stand down
        for the backend, so a destination nothing can deliver to must never be anchored,
        or the span reaches neither account."""
        global_exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(global_exporter), "langfuse_otel"))
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=lambda _d: None))

        def run():
            set_request_destinations(deliverable_destinations((LANGFUSE_DEST,), provider))
            emit(provider)
            return request_destinations()

        anchored = in_fresh_context(run)

        assert anchored == ()
        assert [s.name for s in global_exporter.get_finished_spans()] == ["chat gpt-4"]

    def test_a_buildable_destination_is_still_anchored_and_still_overrides(self):
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = wired_provider(dest_exporter, global_exporter)

        def run():
            set_request_destinations(deliverable_destinations((LANGFUSE_DEST,), provider))
            emit(provider)
            return request_destinations()

        anchored = in_fresh_context(run)

        assert anchored == (LANGFUSE_DEST,)
        assert global_exporter.get_finished_spans() == ()
        assert [s.name for s in dest_exporter.get_finished_spans()] == ["chat gpt-4"]

    def test_only_the_unbuildable_destination_is_dropped_from_a_mixed_set(self):
        dest_exporter = InMemorySpanExporter()
        other = LANGFUSE_DEST.model_copy(update={"endpoint": "http://broken.local/otel"})
        fan_out = TenantFanOutSpanProcessor(
            processor_factory=lambda d: None if d.endpoint == other.endpoint else SimpleSpanProcessor(dest_exporter)
        )

        assert fan_out.deliverable((other, LANGFUSE_DEST)) == (LANGFUSE_DEST,)

    def test_no_fan_out_means_nothing_is_anchored(self):
        """With nothing to carry the spans to the tenant, anchoring would only stop the
        operator's exporter from writing them."""
        provider = TracerProvider()

        assert deliverable_destinations((LANGFUSE_DEST,), provider) == ()

    def test_a_protocol_with_no_otlp_transport_is_not_deliverable(self):
        """An unknown exporter kind falls back to the console exporter, which ignores the
        tenant's credentials and prints its spans to the proxy's stdout. Treating that as
        deliverable would stand the operator's exporter down for spans nobody stores."""
        typo = LANGFUSE_DEST.model_copy(update={"protocol": "consle"})
        fan_out = TenantFanOutSpanProcessor()
        try:
            assert fan_out.deliverable((typo, LANGFUSE_DEST)) == (LANGFUSE_DEST,)
        finally:
            fan_out.shutdown()

    def test_a_closed_fan_out_anchors_nothing(self):
        fan_out = TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(InMemorySpanExporter()))
        provider = TracerProvider()
        provider.add_span_processor(fan_out)
        fan_out.shutdown()

        assert deliverable_destinations((LANGFUSE_DEST,), provider) == ()

    def test_the_processor_built_to_check_deliverability_is_the_one_that_exports(self):
        built = []

        def factory(_destination):
            built.append(SimpleSpanProcessor(InMemorySpanExporter()))
            return built[-1]

        provider = TracerProvider()
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=factory))

        def run():
            set_request_destinations(deliverable_destinations((LANGFUSE_DEST,), provider))
            emit(provider)

        in_fresh_context(run)

        assert len(built) == 1

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


ARIZE_TEAM_PARAMS = {"arize_space_id": "space-team", "arize_api_key": "key-team"}


def arize_destination(**rates: str) -> OtelDestination:
    destination = destination_for("arize", {**ARIZE_TEAM_PARAMS, **rates})
    assert destination is not None, "the fixture must resolve for the test to mean anything"
    return destination


class TestTenantSampling:
    """A team's ``arize_success_sampling_rate`` and ``arize_error_sampling_rate`` decide how
    much of its traffic the fan-out delivers to its space, the way they already decide what
    the legacy Arize callback exports."""

    NAMES = {"POST /v1/chat/completions", "auth /v1/chat/completions", "chat gpt-4"}

    @staticmethod
    def _fan_out(dest_exporter, draw=None, global_exporter=None):
        provider = TracerProvider()
        if global_exporter is not None:
            provider.add_span_processor(_OverriddenBackendFilter(SimpleSpanProcessor(global_exporter), "arize"))
        kwargs = {"processor_factory": lambda _d: SimpleSpanProcessor(dest_exporter)}
        if draw is not None:
            kwargs["sampling_draw"] = draw
        provider.add_span_processor(TenantFanOutSpanProcessor(**kwargs))
        return provider

    @staticmethod
    def _tree(provider, *, failed=False):
        tracer = get_tracer(provider, "litellm")
        with tracer.start_as_current_span("POST /v1/chat/completions"):
            with tracer.start_as_current_span("auth /v1/chat/completions"):
                pass
            with tracer.start_as_current_span("chat gpt-4") as span:
                if failed:
                    span.set_status(Status(StatusCode.ERROR, "upstream 500"))

    def _run(self, provider, destinations, *, failed=False, trees=1):
        def run():
            set_request_destinations(destinations)
            for _ in range(trees):
                self._tree(provider, failed=failed)

        in_fresh_context(run)

    def test_a_zero_rate_keeps_the_whole_tree_out_of_the_teams_space(self):
        """The reported case: override mode, both rates 0.0, so the team's traffic goes nowhere."""
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, global_exporter=global_exporter)

        self._run(
            provider,
            (arize_destination(arize_success_sampling_rate="0.0", arize_error_sampling_rate="0.0"),),
        )

        assert dest_exporter.get_finished_spans() == ()
        assert global_exporter.get_finished_spans() == ()

    def test_a_rate_of_one_exports_the_whole_tree(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(provider, (arize_destination(arize_success_sampling_rate="1.0"),))

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_an_unset_rate_exports_everything(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.99)

        self._run(provider, (arize_destination(),))

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_a_failed_request_is_kept_whole_by_the_error_rate_when_the_success_rate_drops_the_rest(self):
        """Mirrors the legacy callback, where a failed call answers to the error rate, at the
        size of a request tree: the failed model call arrives with its parents."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(
            provider,
            (arize_destination(arize_success_sampling_rate="0.0", arize_error_sampling_rate="1.0"),),
            failed=True,
        )

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_a_zero_error_rate_drops_the_whole_failed_request_the_success_rate_would_keep(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(
            provider,
            (arize_destination(arize_success_sampling_rate="1.0", arize_error_sampling_rate="0.0"),),
            failed=True,
        )

        assert dest_exporter.get_finished_spans() == ()

    def test_a_model_call_that_fails_after_the_server_span_ended_answers_to_the_error_rate(self):
        """The model-call span is closed in the post-call callback, after the FastAPI server
        span has ended, so a request whose only failed span ends late is still a failed
        request: it is kept by an error rate of 1.0 that a success rate of 0.0 would drop."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)
        destinations = (arize_destination(arize_success_sampling_rate="0.0", arize_error_sampling_rate="1.0"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                call = tracer.start_span("chat gpt-4")
            assert dest_exporter.get_finished_spans() == (), "undecided while the model call is open"
            call.set_status(Status(StatusCode.ERROR, "upstream 500"))
            call.end()

        in_fresh_context(run)

        assert {s.name for s in dest_exporter.get_finished_spans()} == {"POST /v1/chat/completions", "chat gpt-4"}

    def test_the_span_that_fills_a_tree_to_its_bound_still_counts_as_failed(self):
        """A tree decided early because it hit the per-tree cap answers to the error rate
        when the span that tripped the cap is the one that failed."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)
        destinations = (arize_destination(arize_success_sampling_rate="1.0", arize_error_sampling_rate="0.0"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                for index in range(otel_providers._MAX_PENDING_SPANS_PER_TREE - 1):
                    with tracer.start_as_current_span(f"chat {index}"):
                        pass
                with tracer.start_as_current_span("chat gpt-4") as call:
                    call.set_status(Status(StatusCode.ERROR, "upstream 500"))

        in_fresh_context(run)

        assert dest_exporter.get_finished_spans() == ()

    def test_a_span_that_ends_after_the_root_follows_the_requests_verdict(self):
        """A post-call database write that starts after the whole request tree has ended
        goes where the rest of the tree went, without a second draw."""
        kept_exporter, dropped_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        kept = self._fan_out(kept_exporter, draw=lambda: 0.1)
        dropped = self._fan_out(dropped_exporter, draw=lambda: 0.9)
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def late_tree(provider):
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions") as root:
                root_context = trace_api.set_span_in_context(root)
            with tracer.start_as_current_span("postgres INSERT LiteLLM_SpendLogs", context=root_context):
                pass

        for provider in (kept, dropped):

            def run(provider=provider):
                set_request_destinations(destinations)
                late_tree(provider)

            in_fresh_context(run)

        assert {s.name for s in kept_exporter.get_finished_spans()} == {
            "POST /v1/chat/completions",
            "postgres INSERT LiteLLM_SpendLogs",
        }
        assert dropped_exporter.get_finished_spans() == ()

    def test_a_late_span_whose_verdict_was_forgotten_is_decided_on_its_own_not_stranded(self):
        """Once enough other requests have been decided to evict a request's verdict, a span
        of it that starts late (a post-call database write) opens and closes its own count,
        so it is decided on a draw of its own as soon as it ends instead of waiting in a tree
        nothing would ever close."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.1)
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions") as root:
                root_context = trace_api.set_span_in_context(root)
            for index in range(otel_providers._MAX_REMEMBERED_VERDICTS):
                with tracer.start_as_current_span(f"POST {index}", context=trace_api.Context()):
                    pass
            with tracer.start_as_current_span("postgres INSERT LiteLLM_SpendLogs", context=root_context):
                pass

        in_fresh_context(run)

        names = [s.name for s in dest_exporter.get_finished_spans()]
        assert names[0] == "POST /v1/chat/completions"
        assert names[-1] == "postgres INSERT LiteLLM_SpendLogs", "exported as it ended, not held for a root"

    def test_a_callers_traceparent_neither_steers_the_draw_nor_hides_the_root(self):
        """The draw is random, not read from the trace id a caller may have chosen, and the
        server span under a remote parent still closes the tree."""
        dest_exporter = InMemorySpanExporter()
        draws = iter([0.9, 0.1])
        provider = self._fan_out(dest_exporter, draw=lambda: next(draws))
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            for trace_id in (1, 2):
                remote = trace_api.SpanContext(
                    trace_id=trace_id, span_id=1, is_remote=True, trace_flags=trace_api.TraceFlags(0x01)
                )
                parent = trace_api.set_span_in_context(trace_api.NonRecordingSpan(remote))
                with tracer.start_as_current_span("POST /v1/chat/completions", context=parent):
                    with tracer.start_as_current_span("chat gpt-4"):
                        pass

        in_fresh_context(run)

        kept = dest_exporter.get_finished_spans()
        assert [s.name for s in kept] == ["chat gpt-4", "POST /v1/chat/completions"]
        assert all(s.context.trace_id == 2 for s in kept), "the second request, whose draw of 0.1 passed"

    def test_a_root_that_never_ends_holds_the_tree_only_up_to_the_bound(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.0)
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                for index in range(otel_providers._MAX_PENDING_SPANS_PER_TREE):
                    with tracer.start_as_current_span(f"chat {index}"):
                        pass
                assert len(dest_exporter.get_finished_spans()) == otel_providers._MAX_PENDING_SPANS_PER_TREE

        in_fresh_context(run)

    def test_a_flood_of_waiting_trees_decides_the_oldest_on_what_it_has(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.0)
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            for index in range(otel_providers._MAX_PENDING_TREES + 1):
                root = tracer.start_span(f"POST {index}", context=trace_api.Context())
                tracer.start_span(f"chat {index}", context=trace_api.set_span_in_context(root)).end()
                if index < otel_providers._MAX_PENDING_TREES:
                    assert dest_exporter.get_finished_spans() == (), f"tree {index} is held while its root is open"

        in_fresh_context(run)

        assert [s.name for s in dest_exporter.get_finished_spans()] == ["chat 0"]

    def test_a_flood_of_open_traces_forgets_the_oldest_and_decides_it_as_its_spans_end(self):
        """The fan-out counts open spans for a bounded number of traces, a bound far above
        what one process keeps in flight, since every trace of the provider is counted, not
        only the sampled ones. A trace pushed out of that count is decided on what it has as
        its next span ends, root still open, rather than held until a bound gets to it."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.0)
        destinations = (arize_destination(arize_success_sampling_rate="0.5"),)

        def run():
            set_request_destinations(destinations)
            tracer = get_tracer(provider, "litellm")
            root = tracer.start_span("POST /v1/chat/completions", context=trace_api.Context())
            tracer.start_span("auth /v1/chat/completions", context=trace_api.set_span_in_context(root)).end()
            for index in range(otel_providers._MAX_OPEN_TRACES):
                tracer.start_span(f"POST {index}", context=trace_api.Context())
            assert dest_exporter.get_finished_spans() == (), "still held before the count is forgotten"
            tracer.start_span("chat gpt-4", context=trace_api.set_span_in_context(root)).end()
            assert {s.name for s in dest_exporter.get_finished_spans()} == {
                "auth /v1/chat/completions",
                "chat gpt-4",
            }
            root.end()

        in_fresh_context(run)

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_shutdown_decides_what_is_still_held(self):
        dest_exporter = InMemorySpanExporter()
        fan_out = TenantFanOutSpanProcessor(
            processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter), sampling_draw=lambda: 0.0
        )
        provider = TracerProvider()
        provider.add_span_processor(fan_out)

        def run():
            set_request_destinations((arize_destination(arize_success_sampling_rate="0.5"),))
            tracer = get_tracer(provider, "litellm")
            with tracer.start_as_current_span("POST /v1/chat/completions"):
                with tracer.start_as_current_span("chat gpt-4"):
                    pass
                assert dest_exporter.get_finished_spans() == ()
                fan_out.shutdown()

        in_fresh_context(run)

        assert [s.name for s in dest_exporter.get_finished_spans()] == ["chat gpt-4"]

    def test_a_span_that_ends_while_shutdown_flushes_is_decided_at_once_not_dropped(self):
        """Shutdown decides the trees it finds, then closes. A sampled span of a trace it
        did not find, ending while it flushes with its root still open, is decided on the
        spot rather than held by a fan-out that will never forward again."""
        dest_exporter = InMemorySpanExporter()
        stragglers = []  # mutable-ok: the span the first export ends, mid-shutdown

        class _EndsAStragglerOnExport(SimpleSpanProcessor):
            def on_end(self, span):
                for straggler in stragglers:
                    if straggler.is_recording():
                        straggler.end()
                super().on_end(span)

        fan_out = TenantFanOutSpanProcessor(
            processor_factory=lambda _d: _EndsAStragglerOnExport(dest_exporter), sampling_draw=lambda: 0.0
        )
        provider = TracerProvider()
        provider.add_span_processor(fan_out)

        def run():
            set_request_destinations((arize_destination(arize_success_sampling_rate="0.5"),))
            tracer = get_tracer(provider, "litellm")
            root = tracer.start_span("POST /v1/chat/completions", context=trace_api.Context())
            tracer.start_span("chat gpt-4", context=trace_api.set_span_in_context(root)).end()
            late_root = tracer.start_span("POST /v1/embeddings", context=trace_api.Context())
            stragglers.append(tracer.start_span("embed ada", context=trace_api.set_span_in_context(late_root)))
            fan_out.shutdown()

        in_fresh_context(run)

        assert {s.name for s in dest_exporter.get_finished_spans()} == {"chat gpt-4", "embed ada"}

    def test_a_zero_rate_drops_even_a_draw_of_zero(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.0)

        self._run(provider, (arize_destination(arize_success_sampling_rate="0.0"),))

        assert dest_exporter.get_finished_spans() == ()

    def test_the_draw_is_compared_to_the_rate_the_way_the_legacy_callback_compares_it(self):
        dropped_exporter, kept_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        destinations = (arize_destination(arize_success_sampling_rate="0.2"),)

        self._run(self._fan_out(dropped_exporter, draw=lambda: 0.3), destinations)
        self._run(self._fan_out(kept_exporter, draw=lambda: 0.2), destinations)

        assert dropped_exporter.get_finished_spans() == ()
        assert {s.name for s in kept_exporter.get_finished_spans()} == self.NAMES

    def test_a_trace_is_kept_or_dropped_whole(self):
        """One decision per request tree, not one per span, so the team never sees a trace with
        its root missing."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(provider, (arize_destination(arize_success_sampling_rate="0.5"),), trees=64)

        by_trace = {}
        for span in dest_exporter.get_finished_spans():
            by_trace.setdefault(span.context.trace_id, set()).add(span.name)
        assert all(names == self.NAMES for names in by_trace.values())
        assert 0 < len(by_trace) < 64

    def test_in_additive_mode_the_operators_own_copy_is_not_sampled(self, monkeypatch):
        """The rates are the team's setting for the team's space; the operator's backbone keeps
        every span."""
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", "additive", raising=False)
        global_exporter, dest_exporter = InMemorySpanExporter(), InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, global_exporter=global_exporter)

        self._run(provider, (arize_destination(arize_success_sampling_rate="0.0"),))

        assert dest_exporter.get_finished_spans() == ()
        assert {s.name for s in global_exporter.get_finished_spans()} == self.NAMES

    def test_a_rate_applies_only_to_the_destination_that_carries_it(self):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(provider, (arize_destination(arize_success_sampling_rate="0.0"), LANGFUSE_DEST))

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_two_views_of_one_exporter_each_draw_at_their_own_rate(self):
        """Two destinations to the same space with different rates share an exporter, not a
        verdict: the 1.0 view gets the tree once and the 0.0 view adds nothing."""
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(
            provider,
            (
                arize_destination(arize_success_sampling_rate="1.0"),
                arize_destination(arize_success_sampling_rate="0.0"),
            ),
        )

        assert sorted(s.name for s in dest_exporter.get_finished_spans()) == sorted(self.NAMES)

    def test_a_rate_does_not_split_the_teams_exporter(self):
        """Sampling decides which spans reach the processor, not how it exports, so two views of
        one account with different rates share one exporter."""
        built = []

        def factory(destination):
            built.append(destination)
            return SimpleSpanProcessor(InMemorySpanExporter())

        provider = TracerProvider()
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=factory))

        self._run(provider, (arize_destination(arize_success_sampling_rate="1.0"), arize_destination()))

        assert len(built) == 1

    @pytest.mark.parametrize("bad", ["abc", "-0.1", "1.5", "nan"])
    def test_an_unusable_rate_exports_rather_than_dropping(self, bad):
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter, draw=lambda: 0.99)

        self._run(provider, (arize_destination(arize_success_sampling_rate=bad),))

        assert {s.name for s in dest_exporter.get_finished_spans()} == self.NAMES

    def test_a_team_entry_with_rates_is_sampled_from_auth_to_the_fan_out(self, monkeypatch):
        """The whole path the ticket names: the team's callback vars resolve at auth, and the
        fan-out honours them."""
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {
                            **ARIZE_TEAM_PARAMS,
                            "arize_success_sampling_rate": "0.0",
                            "arize_error_sampling_rate": "0.0",
                        },
                    }
                ]
            }
        )
        destinations = resolve_tenant_otel_destinations(auth)
        assert destinations, "the fixture must resolve to a destination for the test to mean anything"
        dest_exporter = InMemorySpanExporter()
        provider = self._fan_out(dest_exporter)

        self._run(provider, destinations)

        assert dest_exporter.get_finished_spans() == ()


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

    @staticmethod
    def _fan_out_of(logger: OpenTelemetryV2) -> TenantFanOutSpanProcessor:
        return next(
            processor
            for processor in logger._tracer_provider._active_span_processor._span_processors
            if isinstance(processor, TenantFanOutSpanProcessor)
        )

    def test_callback_settings_excluded_services_win_over_the_published_preset_env_config(self, monkeypatch):
        """A preset builds its config env-only, so the fan-out must read
        ``callback_settings.otel.excluded_services`` itself rather than the
        published logger's config, or the env value would win."""
        monkeypatch.setattr(litellm, "callback_settings", {"otel": {"excluded_services": ["postgres"]}}, raising=False)
        preset = OpenTelemetryV2(
            config=OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory")], excluded_services=["redis"]),
            callback_name="langfuse_otel",
        )

        publish_global_otel_v2_provider([], lambda _p: None, registered=preset)

        assert self._fan_out_of(preset)._excluded_db_systems == frozenset({"postgresql"})

    def test_callback_settings_excluded_services_apply_even_when_other_otel_env_vars_are_malformed(self, monkeypatch):
        """Reading the setting must not rebuild the whole settings model, or an unrelated bad env
        value the operator overrode in config would stop publication before the fan-out is attached"""
        preset = OpenTelemetryV2(
            config=OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory")]),
            callback_name="langfuse_otel",
        )
        monkeypatch.setenv("LITELLM_OTEL_LEGACY_COMPAT", "not-a-bool")
        monkeypatch.setattr(litellm, "callback_settings", {"otel": {"excluded_services": ["postgres"]}}, raising=False)

        publish_global_otel_v2_provider([], lambda _p: None, registered=preset)

        assert self._fan_out_of(preset)._excluded_db_systems == frozenset({"postgresql"})

    def test_excluded_services_fall_back_to_the_published_logger_config_without_callback_settings(self, monkeypatch):
        monkeypatch.setattr(litellm, "callback_settings", {"otel": {"exporter": "in_memory"}}, raising=False)
        preset = OpenTelemetryV2(
            config=OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory")], excluded_services=["redis"]),
            callback_name="langfuse_otel",
        )

        publish_global_otel_v2_provider([], lambda _p: None, registered=preset)

        assert self._fan_out_of(preset)._excluded_db_systems == frozenset({"redis"})

    @pytest.mark.parametrize(
        "otel", [None, True, "on", "", []], ids=["null", "true", "on", "empty_string", "empty_list"]
    )
    def test_a_non_mapping_otel_block_falls_back_to_the_published_logger_config(self, monkeypatch, otel):
        monkeypatch.setattr(litellm, "callback_settings", {"otel": otel}, raising=False)
        preset = OpenTelemetryV2(
            config=OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory")], excluded_services=["redis"]),
            callback_name="langfuse_otel",
        )

        publish_global_otel_v2_provider([], lambda _p: None, registered=preset)

        assert self._fan_out_of(preset)._excluded_db_systems == frozenset({"redis"})

    def test_otel_after_a_preset_reuses_it_and_still_takes_callback_settings_exclusions(self, monkeypatch):
        """``callbacks: [langfuse_otel, otel]`` keeps one v2 logger, exactly as
        before ``excluded_services`` existed, and the exclusion still comes from
        ``callback_settings.otel`` rather than the preset's env-only config."""
        from litellm.litellm_core_utils import litellm_logging as logging_module

        logging_module._in_memory_loggers.clear()
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LITELLM_OTEL_EXCLUDED_SERVICES", "redis")
        is_otel_v2_enabled.cache_clear()
        monkeypatch.setattr(litellm, "callback_settings", {"otel": {"excluded_services": ["postgres"]}}, raising=False)
        try:

            def init(name: str) -> CustomLogger | None:
                return logging_module._init_custom_logger_compatible_class(
                    logging_integration=name,  # pyright: ignore[reportArgumentType]  # test passes a literal callback name
                    internal_usage_cache=None,
                    llm_router=None,
                    custom_logger_init_args={},
                )

            preset = init("langfuse_otel")
            otel_cb = init("otel")

            assert isinstance(preset, OpenTelemetryV2)
            assert otel_cb is preset
            v2_loggers = [cb for cb in logging_module._in_memory_loggers if isinstance(cb, OpenTelemetryV2)]
            assert v2_loggers == [preset], v2_loggers
            publish_global_otel_v2_provider(logging_module._in_memory_loggers, lambda _p: None, registered=preset)
            assert self._fan_out_of(preset)._excluded_db_systems == frozenset({"postgresql"})
        finally:
            logging_module._in_memory_loggers.clear()
            is_otel_v2_enabled.cache_clear()

    @pytest.mark.parametrize("canonical", ["langfuse_otel", "arize"])
    def test_publishing_tells_the_fan_out_about_every_v2_loggers_account(self, monkeypatch, canonical):
        monkeypatch.setenv("LITELLM_OTEL_TENANT_DESTINATION_MODE", "additive")
        shared = InMemorySpanExporter()
        monkeypatch.setattr(otel_providers, "_destination_processor", lambda _d: SimpleSpanProcessor(shared))
        accounts = {
            "langfuse_otel": (
                "https://cloud.langfuse.com/api/public/otel/v1/traces",
                "authorization=Basic op",
            ),
            "arize": (
                "https://otlp.arize.com/v1/traces",
                "space_id=space-op,api_key=key-op",
            ),
        }
        loggers = {
            name: OpenTelemetryV2(
                config=OpenTelemetryV2Config(
                    exporters=(ExporterSpec(kind="otlp_http", endpoint=endpoint, headers=headers),)
                ),
                callback_name=name,
                tracer_provider=TracerProvider(),
            )
            for name, (endpoint, headers) in accounts.items()
        }
        other = "arize" if canonical == "langfuse_otel" else "langfuse_otel"
        published = publish_global_otel_v2_provider(
            [loggers[other]],
            lambda _p: None,
            registered=loggers[canonical],
        )

        def destination(name, headers):
            return OtelDestination(endpoint=accounts[name][0], headers=headers, callback_name=name)

        def run(destinations):
            set_request_destinations(destinations)
            emit(published.tracer_provider)

        in_fresh_context(
            run, (destination(canonical, dict(pair.split("=") for pair in accounts[canonical][1].split(","))),)
        )
        in_fresh_context(run, (destination(other, dict(pair.split("=") for pair in accounts[other][1].split(","))),))
        assert shared.get_finished_spans() == (), "an account the operator already writes to was written twice"

        in_fresh_context(run, (destination(other, {"authorization": "Basic team"}),))
        assert [s.name for s in shared.get_finished_spans()] == ["chat gpt-4"]

    def test_publishing_twice_does_not_double_export(self):
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.ARIZE_AX)])
        logger = OpenTelemetryV2(config=config, callback_name="arize")

        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)
        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)

        kinds = [type(p).__name__ for p in logger._tracer_provider._active_span_processor._span_processors]
        assert kinds.count("TenantFanOutSpanProcessor") == 1

    def test_anchoring_reads_the_fan_out_off_the_published_provider_not_the_otel_global(self, monkeypatch):
        """``set_tracer_provider`` keeps the first provider it was handed. When
        auto-instrumentation or a legacy logger claimed it before the proxy published,
        the OTel global carries no fan-out, so reading it there would refuse every
        destination the published provider delivers."""
        from litellm.proxy import proxy_server

        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        logger = OpenTelemetryV2(config=config, callback_name="langfuse_otel")
        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)
        monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)
        claimed_first = TracerProvider()

        assert fan_out_provider() is logger.tracer_provider
        assert deliverable_destinations((LANGFUSE_DEST,), claimed_first) == ()
        assert deliverable_destinations((LANGFUSE_DEST,), fan_out_provider()) == (LANGFUSE_DEST,)

    def test_a_legacy_v1_logger_holding_the_registered_slot_does_not_hide_the_fan_out(self, monkeypatch):
        """The proxy publishes with ``registered=None`` when ``open_telemetry_logger``
        holds a v1 logger, so the fan-out lands on a v2 logger taken from
        ``_in_memory_loggers``. Reading the registered slot finds no v2 logger there and
        the OTel global belongs to v1, so both detours refuse every destination the
        published provider delivers."""
        from litellm.integrations.opentelemetry import OpenTelemetry
        from litellm.proxy import proxy_server

        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        v2 = OpenTelemetryV2(config=config, callback_name="langfuse_otel")
        publish_global_otel_v2_provider([v2], lambda _p: None, registered=None)
        monkeypatch.setattr(proxy_server, "open_telemetry_logger", OpenTelemetry())

        assert fan_out_provider() is v2.tracer_provider
        assert deliverable_destinations((LANGFUSE_DEST,), fan_out_provider()) == (LANGFUSE_DEST,)

    def test_without_a_publish_anchoring_attaches_fan_out_to_registered_v2_logger(self, monkeypatch):
        from litellm.proxy import proxy_server

        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        logger = OpenTelemetryV2(config=config, callback_name="langfuse_otel")
        monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)

        assert fan_out_provider() is logger.tracer_provider
        assert deliverable_destinations((LANGFUSE_DEST,), fan_out_provider()) == (LANGFUSE_DEST,)

    def test_concurrent_anchoring_attaches_exactly_one_fan_out(self):
        """Requests race to anchor when the startup publish never ran, and a fan-out
        attached twice delivers every tenant span twice."""
        import threading

        from litellm.integrations.otel.plumbing.providers import attach_tenant_fan_out

        class SlowAttachProvider(TracerProvider):
            def add_span_processor(self, span_processor):
                time.sleep(0.05)
                super().add_span_processor(span_processor)

        provider = SlowAttachProvider()
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        barrier = threading.Barrier(8)

        def anchor():
            barrier.wait(timeout=10)
            attach_tenant_fan_out(provider, config)

        threads = [threading.Thread(target=anchor) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        kinds = [type(p).__name__ for p in provider._active_span_processor._span_processors]
        assert kinds.count("TenantFanOutSpanProcessor") == 1, f"one fan-out per provider, got {kinds}"

    def test_without_a_publish_anchoring_falls_back_to_the_otel_global(self, monkeypatch):
        from opentelemetry import trace

        from litellm.proxy import proxy_server

        monkeypatch.setattr(proxy_server, "open_telemetry_logger", None)

        assert fan_out_provider() is trace.get_tracer_provider()

    def test_auth_seeds_the_request_with_destinations_the_registered_logger_can_deliver(
        self, monkeypatch, allow_test_hosts
    ):
        from litellm.proxy import proxy_server
        from litellm.proxy.auth.user_api_key_auth import _seed_request_destinations

        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])
        logger = OpenTelemetryV2(config=config, callback_name="langfuse_otel")
        publish_global_otel_v2_provider([], lambda _p: None, registered=logger)
        monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)
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
        expected = resolve_tenant_otel_destinations(auth)
        assert expected, "the fixture must resolve to a destination for the test to mean anything"

        def run():
            _seed_request_destinations(auth)
            return request_destinations()

        assert deliverable_destinations(expected, TracerProvider()) == ()
        assert in_fresh_context(run) == expected


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

    @pytest.mark.parametrize("callback_name", ["arize", None])
    def test_a_service_name_does_not_detach_a_backend_the_destination_does_not_name(self, callback_name):
        """The fan-out only sees spans on the published provider, so relabelling this
        logger's span onto a second provider would drop the model call out of the
        trace another backend's destination receives."""
        config = OpenTelemetryV2Config(
            exporters=[ExporterSpec(kind="otlp_http", endpoint="http://op.local", owner=ExporterOwner.ARIZE_AX)]
        )
        cache = TenantTracerCache(config, callback_name, "litellm")
        default = get_tracer(TracerProvider(), "litellm")
        auth_metadata = {"otel_service_name": "team-checkout"}

        relabelled = cache.route_for(default, None, auth_metadata)
        assert relabelled.tracer is not default
        cache.release(relabelled.provider)

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return cache.route_for(default, None, auth_metadata)

        route = in_fresh_context(run)
        assert route.tracer is default
        assert route.detached is False
        assert route.provider is None

    @pytest.mark.parametrize(
        ("owner", "params", "auth_metadata"),
        [
            (ExporterOwner.ARIZE_AX, {"arize_space_key": "space", "arize_api_key": "key"}, {}),
            (ExporterOwner.ARIZE_PHOENIX, None, {"phoenix_project_name": "team-project"}),
        ],
    )
    def test_a_backend_pointed_at_its_own_account_still_routes_next_to_another_backend_destination(
        self, owner, params, auth_metadata
    ):
        """Credentials or a project name the tenant's own account for this backend, which
        the other backend's destination cannot stand in for."""
        config = OpenTelemetryV2Config(
            exporters=[ExporterSpec(kind="otlp_http", endpoint="http://op.local", owner=owner)]
        )
        cache = TenantTracerCache(config, owner.value, "litellm")
        default = get_tracer(TracerProvider(), "litellm")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return cache.route_for(default, params, {"otel_service_name": "team-checkout", **auth_metadata})

        route = in_fresh_context(run)
        assert route.tracer is not default
        assert route.detached is True
        cache.release(route.provider)


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


LLM_ONLY_DEST = OtelDestination(
    endpoint="http://tenant.local/api/public/otel",
    headers={"Authorization": "Basic dGVuYW50"},
    callback_name="langfuse_otel",
    span_scope="llm_only",
)

#: Every span kind the proxy emits for one chat request, plus the two spans that
#: look like a model call to a naive classifier: the MCP tool call carries
#: ``gen_ai.operation.name`` too, and baggage promotes ``gen_ai.request.model``
#: onto children that are not the call.
REQUEST_TREE = frozenset(
    {
        "POST /v1/chat/completions",
        "auth /v1/chat/completions",
        "postgres SELECT",
        "redis GET",
        "execute_guardrail pii",
        "tools/call get_weather",
        "chat gpt-4",
        "chat claude-haiku",
        "cost_tracking",
    }
)
LLM_SPANS = frozenset({"chat gpt-4", "chat claude-haiku"})
TRACE_CONTROLS = MappingProxyType(
    {
        "langfuse.observation.type": "generation",
        "langfuse.trace.name": "checkout",
        "user.id": "user-7",
        "session.id": "sess-1",
        "langfuse.trace.tags": ("beta", "eu"),
    }
)


def request_tree(provider: TracerProvider) -> None:
    tracer = get_tracer(provider, "litellm")
    with tracer.start_as_current_span("POST /v1/chat/completions"):
        with tracer.start_as_current_span("auth /v1/chat/completions"):
            with tracer.start_as_current_span("postgres SELECT") as db:
                db.set_attribute("db.system", "postgresql")
            with tracer.start_as_current_span("redis GET") as cache:
                cache.set_attribute("db.system", "redis")
        with tracer.start_as_current_span("execute_guardrail pii") as guard:
            guard.set_attributes({"litellm.guardrail.name": "pii", "litellm.guardrail.status": "success"})
        with tracer.start_as_current_span("tools/call get_weather") as tool:
            tool.set_attributes({"gen_ai.operation.name": "execute_tool", "mcp.method.name": "tools/call"})
        with tracer.start_as_current_span("chat gpt-4") as llm:
            llm.set_attributes({"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4", **TRACE_CONTROLS})
            with tracer.start_as_current_span("cost_tracking") as child:
                child.set_attribute("gen_ai.request.model", "gpt-4")
        with tracer.start_as_current_span("chat claude-haiku") as retry:
            retry.set_attributes({"gen_ai.operation.name": "chat", "gen_ai.request.model": "claude-haiku"})


def names(exporter: InMemorySpanExporter) -> frozenset[str]:
    return frozenset(s.name for s in exporter.get_finished_spans())


class TestSpanScope:
    @staticmethod
    def _additive(monkeypatch):
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", "additive", raising=False)

    @staticmethod
    def _run(provider, destinations):
        def run():
            set_request_destinations(destinations)
            request_tree(provider)

        in_fresh_context(run)

    @staticmethod
    def _operator_provider(operator_exporter, dest_exporter, scope="full"):
        provider = TracerProvider()
        provider.add_span_processor(
            _OverriddenBackendFilter(SimpleSpanProcessor(operator_exporter), "langfuse_otel", scope)
        )
        provider.add_span_processor(
            TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest_exporter))
        )
        return provider

    def test_off_and_off_is_the_full_tree_on_both_sides(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant), (LANGFUSE_DEST,))

        assert names(operator) == REQUEST_TREE
        assert names(tenant) == REQUEST_TREE

    def test_a_tenant_asking_for_llm_only_gets_just_the_model_calls(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant), (LLM_ONLY_DEST,))

        assert names(tenant) == LLM_SPANS
        assert names(operator) == REQUEST_TREE, "the tenant's scope must not narrow the operator's exporter"

    def test_an_operator_asking_for_llm_only_keeps_the_tenants_tree_whole(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant, scope="llm_only"), (LANGFUSE_DEST,))

        assert names(operator) == LLM_SPANS
        assert names(tenant) == REQUEST_TREE, "the operator's scope must not narrow a tenant destination"

    def test_both_on_narrows_both(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant, scope="llm_only"), (LLM_ONLY_DEST,))

        assert names(operator) == LLM_SPANS
        assert names(tenant) == LLM_SPANS

    def test_an_operator_scope_does_not_undo_the_override(self):
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant, scope="llm_only"), (LLM_ONLY_DEST,))

        assert operator.get_finished_spans() == ()
        assert names(tenant) == LLM_SPANS

    @staticmethod
    def _same_account_provider(shared, operator_scope):
        provider = TracerProvider()
        provider.add_span_processor(
            _OverriddenBackendFilter(
                SimpleSpanProcessor(shared), "langfuse_otel", operator_scope, TestRoutingMode.OPERATOR_SINK
            )
        )
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=MappingProxyType({TestRoutingMode.OPERATOR_SINK: operator_scope}),
            )
        )
        return provider

    @staticmethod
    def _same_account_destination(span_scope):
        return OtelDestination(
            endpoint=TestRoutingMode.SAME_ACCOUNT_ENDPOINT,
            headers=MappingProxyType({"Authorization": "Basic op"}),
            callback_name="langfuse_otel",
            span_scope=span_scope,
        )

    @pytest.mark.parametrize(
        ("operator_scope", "tenant_scope", "expected"),
        [
            ("llm_only", "full", REQUEST_TREE),
            ("full", "llm_only", REQUEST_TREE),
            ("llm_only", "llm_only", LLM_SPANS),
            ("full", "full", REQUEST_TREE),
        ],
    )
    def test_a_team_naming_the_operators_project_gets_the_wider_of_the_two_scopes_once(
        self, monkeypatch, operator_scope, tenant_scope, expected
    ):
        self._additive(monkeypatch)
        shared = InMemorySpanExporter()

        self._run(self._same_account_provider(shared, operator_scope), (self._same_account_destination(tenant_scope),))

        finished = [s.name for s in shared.get_finished_spans()]
        assert frozenset(finished) == expected
        assert len(finished) == len(expected), "the same account received a span twice"

    def test_a_full_team_on_the_operators_llm_only_project_gets_one_whole_tree(self, monkeypatch):
        """The operator's exporter writes the model call, the fan-out the rest, and Langfuse
        upserts by span id: a re-rooted, self-named generation there would replace the one
        parented under the request span and rename the whole trace after itself."""
        self._additive(monkeypatch)
        shared = InMemorySpanExporter()

        self._run(self._same_account_provider(shared, "llm_only"), (self._same_account_destination("full"),))

        whole = {s.name: s for s in shared.get_finished_spans()}
        assert whole["chat claude-haiku"].parent == whole["POST /v1/chat/completions"].context
        assert "langfuse.trace.name" not in whole["chat claude-haiku"].attributes

    def test_an_llm_only_team_on_the_operators_llm_only_project_gets_re_rooted_generations(self, monkeypatch):
        self._additive(monkeypatch)
        shared = InMemorySpanExporter()

        self._run(self._same_account_provider(shared, "llm_only"), (self._same_account_destination("llm_only"),))

        kept = {s.name: s for s in shared.get_finished_spans()}["chat claude-haiku"]
        assert kept.parent is None
        assert kept.attributes["langfuse.trace.name"] == "chat claude-haiku"

    def test_a_full_team_on_another_account_does_not_widen_the_operators_llm_only_exporter(self, monkeypatch):
        self._additive(monkeypatch)
        operator = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            _OverriddenBackendFilter(
                SimpleSpanProcessor(operator), "langfuse_otel", "llm_only", TestRoutingMode.OPERATOR_SINK
            )
        )
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=lambda _d: None))

        self._run(provider, (LANGFUSE_DEST,))

        kept = {s.name: s for s in operator.get_finished_spans()}["chat claude-haiku"]
        assert names(operator) == LLM_SPANS
        assert kept.parent is None
        assert kept.attributes["langfuse.trace.name"] == "chat claude-haiku"

    def test_a_built_provider_knows_which_account_its_llm_only_exporter_writes_to(self, monkeypatch):
        self._additive(monkeypatch)
        shared = InMemorySpanExporter()
        monkeypatch.setattr(otel_providers, "_exporter_from_spec", lambda _spec: shared)
        config = OpenTelemetryV2Config(
            langfuse_span_scope="llm_only",
            exporters=[
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=TestRoutingMode.OPERATOR_SINK[0],
                    headers="authorization=Basic op",
                    owner=ExporterOwner.LANGFUSE_OTEL,
                )
            ],
        )
        provider = build_tracer_provider(config, use_simple_processor=True)
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=operator_sink_scopes(config),
            )
        )

        self._run(provider, (self._same_account_destination("full"),))

        whole = {s.name: s for s in shared.get_finished_spans()}
        assert frozenset(whole) == REQUEST_TREE
        assert whole["chat claude-haiku"].parent == whole["POST /v1/chat/completions"].context
        assert "langfuse.trace.name" not in whole["chat claude-haiku"].attributes

    def test_a_kept_generation_becomes_the_root_of_the_request_trace_with_its_trace_controls(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant), (LLM_ONLY_DEST,))

        full = {s.name: s for s in operator.get_finished_spans()}
        kept = {s.name: s for s in tenant.get_finished_spans()}["chat gpt-4"]
        assert kept.context == full["chat gpt-4"].context, "same trace id and span id as the operator's copy"
        assert kept.parent is None, "its parent is the request span the tenant never receives"
        assert {k: kept.attributes[k] for k in TRACE_CONTROLS} == dict(TRACE_CONTROLS), "the caller's trace name wins"
        assert full["chat gpt-4"].parent == full["POST /v1/chat/completions"].context, (
            "the operator's copy is untouched"
        )

    def test_a_kept_generation_with_no_trace_name_is_named_after_itself(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant, scope="llm_only"), (LLM_ONLY_DEST,))

        for exporter in (operator, tenant):
            kept = {s.name: s for s in exporter.get_finished_spans()}["chat claude-haiku"]
            assert kept.parent is None
            assert kept.attributes["langfuse.trace.name"] == "chat claude-haiku"
            assert kept.attributes["gen_ai.request.model"] == "claude-haiku", "the rest of the attributes stay"

    def test_narrowing_one_exporter_leaves_the_other_exporters_view_of_the_span_alone(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant, scope="llm_only"), (LANGFUSE_DEST,))

        whole = {s.name: s for s in tenant.get_finished_spans()}
        assert whole["chat claude-haiku"].parent == whole["POST /v1/chat/completions"].context
        assert "langfuse.trace.name" not in whole["chat claude-haiku"].attributes
        narrowed = {s.name: s for s in operator.get_finished_spans()}["chat claude-haiku"]
        assert narrowed.parent is None
        assert narrowed.attributes["langfuse.trace.name"] == "chat claude-haiku"

    def test_a_full_scope_exporter_gets_the_generation_under_its_request_span_and_unnamed(self, monkeypatch):
        self._additive(monkeypatch)
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._operator_provider(operator, tenant), (LANGFUSE_DEST,))

        for exporter in (operator, tenant):
            whole = {s.name: s for s in exporter.get_finished_spans()}
            assert whole["chat claude-haiku"].parent == whole["POST /v1/chat/completions"].context
            assert "langfuse.trace.name" not in whole["chat claude-haiku"].attributes

    def test_a_non_langfuse_destination_of_the_same_request_keeps_the_full_tree(self, monkeypatch):
        self._additive(monkeypatch)
        by_backend = {"langfuse_otel": InMemorySpanExporter(), "arize": InMemorySpanExporter()}
        provider = TracerProvider()
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda d: SimpleSpanProcessor(by_backend[d.callback_name]),
            )
        )
        arize = OtelDestination(endpoint="https://otlp.arize.com", headers={"api_key": "k"}, callback_name="arize")

        self._run(provider, (LLM_ONLY_DEST, arize))

        assert names(by_backend["langfuse_otel"]) == LLM_SPANS
        assert names(by_backend["arize"]) == REQUEST_TREE

    def test_two_views_of_one_account_share_the_exporter_but_not_the_filter(self):
        built, tenant = [], InMemorySpanExporter()
        provider = TracerProvider()

        def factory(destination):
            built.append(destination)
            return SimpleSpanProcessor(tenant)

        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=factory))

        self._run(provider, (LLM_ONLY_DEST,))
        assert names(tenant) == LLM_SPANS
        tenant.clear()

        self._run(provider, (LANGFUSE_DEST,))
        assert names(tenant) == REQUEST_TREE
        assert len(built) == 1, "the same account must not get a second exporter for a second scope"

    def test_the_config_scope_reaches_only_the_exporter_langfuse_owns(self, monkeypatch):
        exporters = {}

        def exporter_for(spec):
            return exporters.setdefault(spec.owner, InMemorySpanExporter())

        monkeypatch.setattr(otel_providers, "_exporter_from_spec", exporter_for)
        config = OpenTelemetryV2Config(
            langfuse_span_scope="llm_only",
            exporters=[
                ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL),
                ExporterSpec(kind="in_memory", owner=ExporterOwner.ARIZE_AX),
                ExporterSpec(kind="in_memory"),
            ],
        )

        self._run(build_tracer_provider(config, use_simple_processor=True), ())

        assert names(exporters[ExporterOwner.LANGFUSE_OTEL]) == LLM_SPANS
        assert names(exporters[ExporterOwner.ARIZE_AX]) == REQUEST_TREE
        assert names(exporters[None]) == REQUEST_TREE, "a bare collector must never be narrowed"

    @pytest.mark.parametrize("tenant_overrides", [False, True])
    def test_the_config_default_leaves_every_exporter_on_the_full_tree(self, monkeypatch, tenant_overrides):
        exporters = {}
        monkeypatch.setattr(
            otel_providers,
            "_exporter_from_spec",
            lambda spec: exporters.setdefault(spec.owner, InMemorySpanExporter()),
        )
        config = OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)])

        self._run(build_tracer_provider(config, use_simple_processor=True, tenant_overrides=tenant_overrides), ())

        assert names(exporters[ExporterOwner.LANGFUSE_OTEL]) == REQUEST_TREE

    def test_the_env_var_sets_the_operator_scope(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_LANGFUSE_SPAN_SCOPE", "llm_only")

        assert OpenTelemetryV2Config().langfuse_span_scope == "llm_only"

    def test_the_env_var_narrows_the_exporter_the_langfuse_preset_builds(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_LANGFUSE_SPAN_SCOPE", "llm_only")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        exporters = {}
        monkeypatch.setattr(
            otel_providers,
            "_exporter_from_spec",
            lambda spec: exporters.setdefault(spec.owner, InMemorySpanExporter()),
        )
        config = langfuse_preset(config_overrides=OpenTelemetryV2Config(exporters=[ExporterSpec(kind="in_memory")]))

        self._run(build_tracer_provider(config, use_simple_processor=True), ())

        assert names(exporters[ExporterOwner.LANGFUSE_OTEL]) == LLM_SPANS
        assert names(exporters[None]) == REQUEST_TREE

    def test_an_unknown_scope_is_rejected_by_the_config(self):
        with pytest.raises(ValueError, match="langfuse_span_scope"):
            OpenTelemetryV2Config(langfuse_span_scope="everything")

    @pytest.mark.parametrize("spelling", ["LLM_ONLY", "Llm_Only", " llm_only\n"])
    def test_the_env_var_is_read_case_and_whitespace_insensitively(self, monkeypatch, spelling):
        """A misspelt env var would otherwise fail validation inside the logger builder,
        which swallows the error and leaves the proxy up with OTel v2 silently off."""
        monkeypatch.setenv("LITELLM_OTEL_LANGFUSE_SPAN_SCOPE", spelling)

        assert OpenTelemetryV2Config().langfuse_span_scope == "llm_only"

    def test_the_operator_scope_does_not_reach_a_tenants_routed_provider(self, monkeypatch):
        """The routed clone carries the tenant's credentials on the operator's Langfuse
        exporter. The operator's ``llm_only`` is a choice about the operator's account,
        so the clone must export the full tree, as the field's contract promises."""
        tenant = InMemorySpanExporter()
        monkeypatch.setattr(otel_providers, "_exporter_from_spec", lambda _spec: tenant)
        config = OpenTelemetryV2Config(
            langfuse_span_scope="llm_only",
            exporters=[ExporterSpec(kind="otlp_http", endpoint="http://op.local", owner=ExporterOwner.LANGFUSE_OTEL)],
        )
        cache = TenantTracerCache(config, "langfuse_otel", "litellm")
        route = cache.route_for(
            get_tracer(TracerProvider(), "litellm"), {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
        )
        assert route.provider is not None

        request_tree(route.provider)
        route.provider.force_flush()

        assert names(tenant) == REQUEST_TREE

    def test_a_team_callback_var_becomes_the_destinations_scope(self, monkeypatch, allow_test_hosts):
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
                            "langfuse_span_scope": "llm_only",
                        },
                    }
                ]
            }
        )

        assert [d.span_scope for d in resolve_tenant_otel_destinations(auth)] == ["llm_only"]

    def test_a_team_that_named_no_scope_gets_the_full_tree(self, allow_test_hosts):
        creds = {"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_host": "http://x"}

        assert destination_for("langfuse_otel", creds).span_scope == "full"

    def test_only_langfuse_honours_the_scope_var(self):
        arize = destination_for(
            "arize", {"arize_api_key": "k", "arize_space_id": "s", "langfuse_span_scope": "llm_only"}
        )

        assert arize is not None and arize.span_scope == "full"

    @pytest.mark.parametrize("scope", ["everything", "LLM_ONLY", ""])
    def test_an_unknown_scope_is_rejected_when_the_callback_is_saved(self, scope):
        with pytest.raises(ValueError, match=r"Invalid langfuse_span_scope .*must be one of \['full', 'llm_only'\]"):
            AddTeamCallback(
                callback_name="langfuse_otel",
                callback_type="success",
                callback_vars={"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_span_scope": scope},
            )

    def test_a_known_scope_is_accepted_when_the_callback_is_saved(self):
        saved = AddTeamCallback(
            callback_name="langfuse_otel",
            callback_type="success",
            callback_vars={"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "langfuse_span_scope": "llm_only"},
        )

        assert saved.callback_vars["langfuse_span_scope"] == "llm_only"


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


def _closed_chat_call_kwargs() -> dict[str, object]:
    """The callback kwargs of one completed chat call, as both an operator and a destination logger see them."""
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "response": {"id": "resp_1", "model": "gpt-4o", "choices": [{"finish_reason": "stop"}]},
        "metadata": {"team_id": "t1", "user_api_key_hash": "hsh"},
        "status": "success",
        "litellm_call_id": "call_dup_1",
    }
    return {
        "standard_logging_object": payload,
        "litellm_params": {"metadata": {}},
        "api_call_start_time": datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
    }


class TestPresetDegradation:
    def test_a_credential_less_langfuse_exports_nowhere_instead_of_to_the_console(self, monkeypatch, capfd):
        """``_normalize`` folds a console exporter in for an empty list, which would
        print every span on a proxy whose teams bring their own credentials."""
        credential_less_proxy(monkeypatch)

        config = langfuse_preset(allow_missing_credentials=True)
        provider = build_tracer_provider(config, tenant_overrides=True)
        capfd.readouterr()
        in_fresh_context(emit, provider)
        provider.force_flush()

        assert '"name": "chat gpt-4"' not in capfd.readouterr().out
        assert "langfuse" in config.mapper_names

    def test_langfuse_still_raises_for_a_global_callback_with_no_credentials(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            langfuse_preset()

    def test_a_credential_less_proxy_builds_the_gated_logger_beside_a_v2_carrier(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        carrier = build_otel_v2_logger(OpenTelemetryV2Config(exporter="in_memory"))

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return _maybe_construct_otel_v2("langfuse_otel", [carrier])

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()

        assert logger is not None
        assert all(spec.requires_headers and not spec.headers for spec in logger.config.exporters)

    def test_a_credential_less_proxy_with_no_destinations_falls_back_to_the_legacy_path(self, monkeypatch):
        """Nothing can use a credential-less langfuse here, so the operator has to get
        the same story as before v2: the legacy integration, not a global provider
        that exports nowhere."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "langfuse_otel", [])
        is_otel_v2_enabled.cache_clear()

        assert logger is None

    def test_a_valid_newrelic_base_exporter_survives_without_a_license_key(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.delenv("NEW_RELIC_LICENSE_KEY", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "newrelic", [])
        is_otel_v2_enabled.cache_clear()

        assert logger is not None
        assert [spec.endpoint for spec in logger.config.exporters] == [
            "http://collector.local:4318",
            "https://otlp.nr-data.net",
        ]

    def test_a_credentialless_newrelic_without_a_base_exporter_falls_back(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.delenv("NEW_RELIC_LICENSE_KEY", raising=False)
        for name in _OTEL_SHORTHAND_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "newrelic", [])
        is_otel_v2_enabled.cache_clear()

        assert logger is None

    def test_an_explicit_console_exporter_keeps_a_credentialless_preset_on_v2(self, monkeypatch, capfd):
        """``OTEL_EXPORTER=console`` reads exactly like the placeholder ``_normalize``
        folds in, but the operator asked for it, so a credential-less New Relic keeps
        the V2 logger and its spans reach stdout instead of the legacy path."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.delenv("NEW_RELIC_LICENSE_KEY", raising=False)
        for name in _OTEL_SHORTHAND_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OTEL_EXPORTER", "console")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "newrelic", [])
        is_otel_v2_enabled.cache_clear()

        assert logger is not None
        assert logger.config.exporters[0].kind == "console"
        assert not logger.config.exporters[0].requires_headers

    def test_a_destination_for_one_backend_does_not_degrade_another(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return _maybe_construct_otel_v2("weave_otel", [])

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()

        assert logger is None

    def test_the_exporter_less_logger_is_not_reused_by_a_request_without_destinations(self, monkeypatch):
        """Reusing it would let one team's destination decide how every later request
        without one is logged, long after the degrade was justified."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        loggers = [build_otel_v2_logger(OpenTelemetryV2Config(exporter="in_memory"))]

        def with_destination():
            set_request_destinations((LANGFUSE_DEST,))
            return _maybe_construct_otel_v2("langfuse_otel", loggers)

        is_otel_v2_enabled.cache_clear()
        degraded = in_fresh_context(with_destination)
        plain = in_fresh_context(_maybe_construct_otel_v2, "langfuse_otel", loggers)
        is_otel_v2_enabled.cache_clear()

        assert degraded is not None
        assert plain is None

    def test_a_credentialed_logger_is_still_reused_across_requests(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-1")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        loggers = []

        is_otel_v2_enabled.cache_clear()
        first = in_fresh_context(_maybe_construct_otel_v2, "langfuse_otel", loggers)
        second = in_fresh_context(_maybe_construct_otel_v2, "langfuse_otel", loggers)
        is_otel_v2_enabled.cache_clear()

        assert first is not None
        assert second is first

    @staticmethod
    def _degraded_langfuse_beside(loggers, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return _maybe_construct_otel_v2("langfuse_otel", loggers)

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()
        assert logger is not None
        return logger

    def test_a_degraded_logger_beside_another_v2_logger_leaves_the_collector_to_it(self, monkeypatch):
        """The other logger's provider already exports every span to the operator's
        collector, so a second model span from this one would land there twice."""
        collector_logger = build_otel_v2_logger(OpenTelemetryV2Config(exporter="in_memory"))

        logger = self._degraded_langfuse_beside([collector_logger], monkeypatch)

        assert [spec.endpoint for spec in logger.config.exporters] == [None]
        assert all(spec.requires_headers and not spec.headers for spec in logger.config.exporters)

    @pytest.mark.parametrize("registered", [(), (CustomLogger(),)])
    def test_a_credential_less_proxy_with_a_destination_but_no_v2_carrier_falls_back(self, monkeypatch, registered):
        """Only a V2 logger publishes the provider the fan-out rides on, so a legacy
        callback beside this one leaves the destination just as unreachable as no
        callback at all, and the operator keeps the pre-V2 story."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_proxy(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")

        def run():
            set_request_destinations((LANGFUSE_DEST,))
            return _maybe_construct_otel_v2("langfuse_otel", list(registered))

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()

        assert logger is None

    @pytest.mark.parametrize("anchored", [True, False])
    def test_a_logger_built_beside_another_v2_logger_keeps_only_its_backends_exporter(self, monkeypatch, anchored):
        """Operator credentials for the backend do not make the collector safe to copy: the
        registered logger already exports every call there, so a copy of ``chat`` riding the
        preset's base exporters lands in the operator's sink a second time. A key's ``logging``
        entry reaches this builder as a plain dynamic callback too, with no destination anchored."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        for name in ("ARIZE_SPACE_KEY", "ARIZE_ENDPOINT", "ARIZE_HTTP_ENDPOINT", "ARIZE_PROJECT_NAME"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ARIZE_SPACE_ID", "space-operator")
        monkeypatch.setenv("ARIZE_API_KEY", "ak-operator")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        operator_exporter = InMemorySpanExporter()
        operator_cfg = OpenTelemetryV2Config(exporter="in_memory")
        operator = build_otel_v2_logger(
            operator_cfg, tracer_provider=otel_providers.build_tracer_provider(operator_cfg, exporter=operator_exporter)
        )

        def run():
            if anchored:
                set_request_destinations(
                    (
                        OtelDestination(
                            endpoint="https://otlp.arize.com/v1", headers={"space_id": "t"}, callback_name="arize"
                        ),
                    )
                )
            return _maybe_construct_otel_v2("arize", [operator])

        is_otel_v2_enabled.cache_clear()
        tenant = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()

        assert tenant is not None
        assert [spec.owner for spec in tenant.config.exporters] == [ExporterOwner.ARIZE_AX]
        assert "http://collector.local:4318" not in {spec.endpoint for spec in tenant.config.exporters}

        tenant_exporter = InMemorySpanExporter()
        tenant_twin = build_otel_v2_logger(
            tenant.config,
            callback_name="arize",
            tracer_provider=otel_providers.build_tracer_provider(tenant.config, exporter=tenant_exporter),
        )
        kwargs = _closed_chat_call_kwargs()
        operator.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        asyncio.run(operator.async_log_success_event(kwargs, None, None, None))
        asyncio.run(tenant_twin.async_log_success_event(kwargs, None, None, None))
        assert [span.name for span in operator_exporter.get_finished_spans()] == ["chat gpt-4o"]
        assert [span.name for span in tenant_exporter.get_finished_spans()] == ["chat gpt-4o"]

    def test_a_preset_that_owns_no_exporter_keeps_the_collector_it_was_built_on(self, monkeypatch):
        """Langtrace is a mapper over the operator's own OTLP collector and contributes no exporter
        of its own, so filtering to owned exporters would register it with nowhere to deliver."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        operator_cfg = OpenTelemetryV2Config(exporter="in_memory")
        operator = build_otel_v2_logger(
            operator_cfg,
            tracer_provider=otel_providers.build_tracer_provider(operator_cfg, exporter=InMemorySpanExporter()),
        )

        is_otel_v2_enabled.cache_clear()
        langtrace = in_fresh_context(lambda: _maybe_construct_otel_v2("langtrace", [operator]))
        is_otel_v2_enabled.cache_clear()

        assert langtrace is not None
        assert "langtrace" in langtrace.config.mapper_names
        assert "http://collector.local:4318" in {spec.endpoint for spec in langtrace.config.exporters}


def credential_less_arize(monkeypatch) -> None:
    """An operator with no Arize account and no generic OTLP collector or headers."""
    for name in (
        "ARIZE_SPACE_ID",
        "ARIZE_SPACE_KEY",
        "ARIZE_API_KEY",
        "ARIZE_ENDPOINT",
        "ARIZE_HTTP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        *_OTEL_SHORTHAND_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


ARIZE_DEST = OtelDestination(
    endpoint="https://otlp.arize.com/v1",
    headers={"arize-space-id": "space-team", "api_key": "key-team"},
    callback_name="arize",
    protocol="otlp_grpc",
)


class _ExporterCapture:
    """Stands an in-memory exporter in for every exporter the provider builds, keyed
    by the headers it would have sent (``None`` for the stdout placeholder), so a test
    reads what each account received rather than which processors were wired."""

    def __init__(self) -> None:
        self.built: tuple[tuple[str | None, InMemorySpanExporter], ...] = ()

    def build(self, spec: ExporterSpec) -> InMemorySpanExporter:
        exporter: Final = InMemorySpanExporter()
        self.built = (*self.built, (spec.headers, exporter))
        return exporter

    def received(self) -> Mapping[str | None, tuple[str, ...]]:
        """Every span name each set of headers received; an exporter that got nothing is absent."""
        return MappingProxyType(
            {
                headers: tuple(span.name for span in exporter.get_finished_spans())
                for headers, exporter in self.built
                if exporter.get_finished_spans()
            }
        )


class TestArizeTenantOnly:
    """An operator whose teams each bring their own Arize space keeps no Arize
    credentials of their own; the preset then exports nowhere for traffic without a
    team destination instead of posting it keyless to Arize."""

    @staticmethod
    def _capture_exporters(monkeypatch) -> "_ExporterCapture":
        capture: Final = _ExporterCapture()
        for kind in ("otlp_grpc", "otlp_http", "console"):
            monkeypatch.setitem(otel_providers._EXPORTER_FACTORIES, kind, capture.build)
        return capture

    def test_a_credential_less_arize_exports_nowhere(self, monkeypatch):
        credential_less_arize(monkeypatch)
        capture = self._capture_exporters(monkeypatch)
        config = arize_preset(allow_missing_credentials=True)
        provider = build_tracer_provider(config, tenant_overrides=True)

        emit(provider)
        provider.force_flush()

        assert capture.received() == {}, "not to Arize, and not to the stdout placeholder either"
        assert "openinference" in config.mapper_names

    def test_a_blank_otlp_headers_variable_is_no_credential_either(self, monkeypatch):
        """``OTEL_EXPORTER_OTLP_TRACES_HEADERS=`` left empty in a compose file must not
        turn into an Arize exporter that posts keyless."""
        credential_less_arize(monkeypatch)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", " ")
        capture = self._capture_exporters(monkeypatch)
        provider = build_tracer_provider(arize_preset(allow_missing_credentials=True), tenant_overrides=True)

        emit(provider)
        provider.force_flush()

        assert capture.received() == {}, "a blank header string is no credential: nothing leaves"

    def test_the_operators_own_credentials_still_reach_the_operators_space(self, monkeypatch):
        credential_less_arize(monkeypatch)
        monkeypatch.setenv("ARIZE_SPACE_ID", "space-operator")
        monkeypatch.setenv("ARIZE_API_KEY", "key-operator")
        capture = self._capture_exporters(monkeypatch)
        provider = build_tracer_provider(arize_preset(allow_missing_credentials=True), tenant_overrides=True)

        emit(provider)
        provider.force_flush()

        assert capture.received() == {
            "space_id=space-operator,api_key=key-operator": ("chat gpt-4",),
            None: ("chat gpt-4",),
        }, "the operator's space, plus the stdout placeholder every credentialed preset keeps today"

    def test_the_standard_otlp_headers_still_reach_the_operators_space(self, monkeypatch):
        credential_less_arize(monkeypatch)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "space_id=space-operator,api_key=key-operator")
        capture = self._capture_exporters(monkeypatch)
        provider = build_tracer_provider(arize_preset(allow_missing_credentials=True), tenant_overrides=True)

        emit(provider)
        provider.force_flush()

        assert capture.received() == {
            "space_id=space-operator,api_key=key-operator": ("chat gpt-4",),
            None: ("chat gpt-4",),
        }, "the operator's space, plus the stdout placeholder every credentialed preset keeps today"

    def test_a_credential_less_arize_still_delivers_a_team_destination(self, monkeypatch):
        from litellm.integrations.otel.plumbing.providers import attach_tenant_fan_out

        credential_less_arize(monkeypatch)
        capture = self._capture_exporters(monkeypatch)
        config = arize_preset(allow_missing_credentials=True)
        provider = build_tracer_provider(config, tenant_overrides=True)
        attach_tenant_fan_out(provider, config)

        def run():
            set_request_destinations(deliverable_destinations((ARIZE_DEST,), provider))
            emit(provider)

        in_fresh_context(run)
        provider.force_flush()

        assert capture.received() == {ARIZE_DEST.header_string(): ("chat gpt-4",)}

    def test_a_credential_less_proxy_builds_the_gated_arize_logger_beside_a_v2_carrier(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_arize(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        carrier = build_otel_v2_logger(OpenTelemetryV2Config(exporter="in_memory"))

        def run():
            set_request_destinations((ARIZE_DEST,))
            return _maybe_construct_otel_v2("arize", [carrier])

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(run)
        is_otel_v2_enabled.cache_clear()

        assert logger is not None
        assert all(spec.requires_headers and not spec.headers for spec in logger.config.exporters)

    def test_a_credential_less_arize_stays_on_v2_at_startup_instead_of_the_keyless_legacy_logger(self, monkeypatch):
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_arize(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        capture = self._capture_exporters(monkeypatch)

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "arize", [])
        is_otel_v2_enabled.cache_clear()

        assert isinstance(logger, OpenTelemetryV2), "None hands 'arize' to the legacy logger, which posts keyless"
        emit(logger.tracer_provider)
        logger.tracer_provider.force_flush()
        assert capture.received() == {}, "no operator credentials: nothing leaves until a team destination exists"

    def test_the_startup_logger_is_reused_by_later_requests_without_a_destination(self, monkeypatch):
        """Every master-key request re-initialises the callback; building a fresh
        provider each time would grow ``_in_memory_loggers`` for the life of the proxy."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        credential_less_arize(monkeypatch)
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        loggers = []

        is_otel_v2_enabled.cache_clear()
        at_startup = in_fresh_context(_maybe_construct_otel_v2, "arize", loggers)
        later = in_fresh_context(_maybe_construct_otel_v2, "arize", loggers)
        is_otel_v2_enabled.cache_clear()

        assert later is at_startup
        assert loggers == [at_startup]


class TestContextIsolation:
    def test_destinations_do_not_leak_between_requests(self):
        def first():
            set_request_destinations((LANGFUSE_DEST,))
            return destination_backends()

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


class TestArizeOtlpProtocol:
    @staticmethod
    def _arize(monkeypatch, endpoint_env, **env):
        monkeypatch.delenv("ARIZE_ENDPOINT", raising=False)
        monkeypatch.delenv("ARIZE_HTTP_ENDPOINT", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return destination_for("arize", {"arize_space_id": "s", "arize_api_key": "k", **endpoint_env})

    def test_http_transport_rewrites_the_grpc_endpoint_to_the_traces_path(self, monkeypatch):
        destination = self._arize(
            monkeypatch,
            {"arize_otlp_protocol": "http/protobuf"},
            ARIZE_ENDPOINT="https://arize.internal.example/v1",
        )

        assert destination.protocol == "otlp_http"
        assert destination.endpoint == "https://arize.internal.example/v1/traces"

    def test_http_transport_prefers_the_http_endpoint(self, monkeypatch):
        destination = self._arize(
            monkeypatch,
            {"arize_otlp_protocol": "http/protobuf"},
            ARIZE_ENDPOINT="https://arize.internal.example/v1",
            ARIZE_HTTP_ENDPOINT="https://http.arize.internal.example/v1/traces",
        )

        assert destination.protocol == "otlp_http"
        assert destination.endpoint == "https://http.arize.internal.example/v1/traces"

    def test_http_transport_falls_back_to_the_arize_cloud_http_endpoint(self, monkeypatch):
        destination = self._arize(monkeypatch, {"arize_otlp_protocol": "http/protobuf"})

        assert destination.protocol == "otlp_http"
        assert destination.endpoint == "https://otlp.arize.com/v1/traces"

    def test_grpc_transport_accepts_the_http_endpoint_as_the_grpc_target(self, monkeypatch):
        destination = self._arize(
            monkeypatch,
            {"arize_otlp_protocol": "grpc"},
            ARIZE_HTTP_ENDPOINT="https://http.arize.internal.example/v1/traces",
        )

        assert destination.protocol == "otlp_grpc"
        assert destination.endpoint == "https://http.arize.internal.example/v1/traces"

    def test_no_transport_var_keeps_the_grpc_env_default(self, monkeypatch):
        destination = self._arize(monkeypatch, {}, ARIZE_ENDPOINT="https://arize.internal.example")

        assert destination.protocol == "otlp_grpc"
        assert destination.endpoint == "https://arize.internal.example"

    def test_the_http_destination_builds_an_http_exporter_on_the_traces_path(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from litellm.integrations.otel.plumbing.providers import _destination_processor

        destination = self._arize(
            monkeypatch,
            {"arize_otlp_protocol": "http/protobuf"},
            ARIZE_ENDPOINT="https://arize.internal.example/v1",
        )
        processor = _destination_processor(destination)
        try:
            assert isinstance(processor, BatchSpanProcessor)
            assert isinstance(processor.span_exporter, HTTPSpanExporter)
            assert processor.span_exporter._endpoint == "https://arize.internal.example/v1/traces"
        finally:
            processor.shutdown()

    def test_the_grpc_destination_builds_a_grpc_exporter(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from litellm.integrations.otel.plumbing.providers import _destination_processor

        destination = self._arize(
            monkeypatch,
            {"arize_otlp_protocol": "grpc"},
            ARIZE_ENDPOINT="https://arize.internal.example",
        )
        processor = _destination_processor(destination)
        try:
            assert isinstance(processor, BatchSpanProcessor)
            assert isinstance(processor.span_exporter, GRPCSpanExporter)
        finally:
            processor.shutdown()

    def test_a_team_metadata_protocol_reaches_the_destination(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        monkeypatch.setenv("ARIZE_ENDPOINT", "https://arize.internal.example/v1")
        monkeypatch.delenv("ARIZE_HTTP_ENDPOINT", raising=False)
        is_otel_v2_enabled.cache_clear()
        auth = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {
                            "arize_space_id": "s",
                            "arize_api_key": "k",
                            "arize_otlp_protocol": "http/protobuf",
                        },
                    }
                ]
            }
        )

        destinations = resolve_tenant_otel_destinations(auth)

        assert [d.protocol for d in destinations] == ["otlp_http"]
        assert destinations[0].endpoint == "https://arize.internal.example/v1/traces"


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

    def test_a_failure_entry_still_wins_the_merge_next_to_a_success_entry(self):
        entries = [
            {**self._entry("http://team.local"), "callback_type": "success"},
            {
                **self._entry("http://key.local", langfuse_public_key="pk-failure", langfuse_secret_key="sk-failure"),
                "callback_type": "failure",
            },
        ]
        runtime = reduce(
            lambda merged, entry: convert_key_logging_metadata_to_callback(AddTeamCallback(**entry), merged),
            entries,
            None,
        )

        destinations = resolve_tenant_otel_destinations(UserAPIKeyAuth(team_metadata={"logging": entries}))

        assert runtime.callback_vars["langfuse_host"] == "http://key.local"
        assert [d.endpoint for d in destinations] == ["http://key.local/api/public/otel"]
        assert destinations[0].headers["Authorization"] == f"Basic {b64encode(b'pk-failure:sk-failure').decode()}"

    @pytest.fixture
    def premium(self, monkeypatch):
        from litellm.proxy import proxy_server

        monkeypatch.setattr(proxy_server, "premium_user", True)
        monkeypatch.setattr(litellm, "allow_dynamic_callback_disabling", True)

    @pytest.mark.usefixtures("premium")
    def test_a_backend_the_key_disabled_resolves_to_no_destination(self):
        """Dispatch skips a callback named in the key's ``litellm_disabled_callbacks``,
        so the fan-out must not deliver to it either."""
        auth = UserAPIKeyAuth(
            metadata={"litellm_disabled_callbacks": ["Langfuse_OTEL"]},
            team_metadata={"logging": [self._entry("http://team.local")]},
        )

        assert resolve_tenant_otel_destinations(auth) == ()

    @pytest.mark.usefixtures("premium")
    @pytest.mark.parametrize(
        ("header", "resolved"),
        [
            ("langfuse_otel", False),
            (" LANGFUSE_OTEL ,arize", False),
            ("arize", True),
        ],
    )
    def test_the_disable_header_wins_over_the_key_list(self, header, resolved):
        """Same precedence as dispatch: a header that names other backends re-enables
        the one the key stored."""
        auth = UserAPIKeyAuth(
            metadata={"litellm_disabled_callbacks": ["langfuse_otel"]},
            team_metadata={"logging": [self._entry("http://team.local")]},
        )

        destinations = resolve_tenant_otel_destinations(auth, {"x-litellm-disable-callbacks": header})

        assert bool(destinations) is resolved

    def test_a_non_premium_proxy_ignores_the_disabled_list_like_dispatch_does(self, monkeypatch):
        from litellm.proxy import proxy_server

        monkeypatch.setattr(proxy_server, "premium_user", False)
        auth = UserAPIKeyAuth(
            metadata={"litellm_disabled_callbacks": ["langfuse_otel"]},
            team_metadata={"logging": [self._entry("http://team.local")]},
        )

        assert resolve_tenant_otel_destinations(auth, {"x-litellm-disable-callbacks": "langfuse_otel"}) != ()


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
        self._settle(fan_out, held)

        assert held.shutdown_calls == 1

    def test_a_recently_used_destination_is_not_the_one_evicted(self):
        """Without the refresh the cache sheds by insertion order, so the busiest
        destination is the one whose exporter is rebuilt on every overflow."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built = self._fan_out()
        for index in range(_MAX_CACHED_DESTINATION_PROCESSORS):
            fan_out._release(fan_out._acquire(self._dest(index)))
        fan_out._release(fan_out._acquire(self._dest(0)))
        fan_out._release(fan_out._acquire(self._dest(_MAX_CACHED_DESTINATION_PROCESSORS)))
        self._settle(fan_out, built[1])

        assert built[1].shutdown_calls == 1
        assert built[0].shutdown_calls == 0, "the destination used most recently was the one shed"

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
        before = self._drain_workers()
        try:
            for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 30):
                fan_out._acquire(self._dest(index))
                fan_out._release(built[-1])
            grew = self._drain_workers() - before
            assert grew == 0, f"one drain thread per shed processor: {grew} new threads"
        finally:
            release.set()
            self._settle(fan_out, built[0])

    def test_a_saturated_drain_leaves_new_destinations_with_the_operator(self):
        """A shed processor keeps its batch thread until its close returns, and against
        a collector that never answers every close waits out the exporter's timeout.
        Tenants rotating past the cache cap would otherwise queue one more processor,
        and one more thread, per request for as long as the outage lasts."""
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

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, pending_drains=3)
        try:
            anchored = tuple(
                fan_out.deliverable((self._dest(index),)) for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 40)
            )

            assert len(built) == _MAX_CACHED_DESTINATION_PROCESSORS + 3, "a processor per request during the outage"
            assert sum(1 for accepted in anchored if accepted) == len(built), "anchored what it could not build"
            assert fan_out.deliverable((self._dest(999),)) == (), (
                "the span would vanish instead of staying with the operator"
            )
        finally:
            release.set()
        for _ in range(500):
            if not fan_out._drain.saturated():
                break
            time.sleep(0.02)

        assert fan_out.deliverable((self._dest(999),)) == (self._dest(999),), "the fan-out never recovered"

    def test_an_anchored_destination_evicted_under_a_saturated_drain_still_gets_the_span(self):
        """``deliverable`` accepted the destination, so the operator's exporter has stood
        down for it. Other tenants' auths can then evict it, and the eviction is what
        tips the drain into saturation, so refusing the rebuild at ``on_end`` would drop
        the span outright."""
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

        fan_out = TenantFanOutSpanProcessor(
            processor_factory=factory, pending_drains=_MAX_CACHED_DESTINATION_PROCESSORS + 1
        )
        provider = TracerProvider()
        provider.add_span_processor(fan_out)
        tracer = get_tracer(provider, "litellm")
        anchored = self._dest(0)
        try:
            for index in range(1, _MAX_CACHED_DESTINATION_PROCESSORS + 1):
                assert fan_out.deliverable((self._dest(index),))
            assert fan_out.deliverable((anchored,)) == (anchored,)
            first = built[-1]
            for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + 1, 2 * _MAX_CACHED_DESTINATION_PROCESSORS + 1):
                assert fan_out.deliverable((self._dest(index),))
            assert fan_out._drain.saturated(), "the anchored destination's own eviction saturates the drain"
            assert first not in fan_out._processors.values(), "the anchored destination was not evicted"

            def run():
                set_request_destinations((anchored,))
                with tracer.start_as_current_span("chat anthropic"):
                    pass

            before = len(built)
            in_fresh_context(run)
            assert len(built) == before + 1, "the anchored destination was not rebuilt, so its span went nowhere"
            assert [span.name for span in built[-1].span_exporter.get_finished_spans()] == ["chat anthropic"]
            assert first.span_exporter.get_finished_spans() == (), "the shed processor was handed out again"
        finally:
            release.set()

    def _saturated_by_anchoring(self, pending_drains, extra):
        """A fan-out whose drain ``extra`` anchorings past the cache cap have saturated.

        Returns it with the processors built, the destinations that anchored, and the
        event that lets the blocked closes finish.
        """
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

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, pending_drains=pending_drains)
        destinations = tuple(self._dest(index) for index in range(_MAX_CACHED_DESTINATION_PROCESSORS + extra))
        anchored = tuple(destination for destination in destinations if fan_out.deliverable((destination,)))
        assert fan_out._drain.saturated(), "anchoring past the cap did not saturate the drain"
        assert len(anchored) > _MAX_CACHED_DESTINATION_PROCESSORS, "not enough destinations in flight to churn"
        return fan_out, built, anchored, release

    def test_anchored_rebuilds_under_a_saturated_drain_do_not_grow_with_the_spans(self):
        """Every anchored rebuild past the cap evicts another anchored destination, whose
        next span rebuilds it in turn. With more destinations in flight than the cache
        holds, each span would then cost one more processor, one more batch thread and
        one more close queued behind a collector that never answers."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built, anchored, release = self._saturated_by_anchoring(pending_drains=4, extra=8)
        try:
            after_anchoring = len(built)
            for _ in range(5):
                for destination in anchored:
                    fan_out._release(fan_out._acquire(destination))

            rebuilt = len(built) - after_anchoring
            assert rebuilt == len(anchored) - _MAX_CACHED_DESTINATION_PROCESSORS, (
                f"{rebuilt} rebuilds over 5 rounds of {len(anchored)} anchored destinations: one per evicted one expected"
            )
            assert len(fan_out._processors) == len(anchored), "an anchored destination was shed under a saturated drain"
            assert all(destination in fan_out.deliverable((destination,)) for destination in anchored)
        finally:
            release.set()

    def test_the_cache_returns_to_its_cap_once_the_drain_has_room(self):
        """Holding above the cap is for the outage only: with the drain caught up, the
        entries kept for the destinations in flight are the ones to shed."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built, anchored, release = self._saturated_by_anchoring(pending_drains=4, extra=8)
        for destination in anchored:
            fan_out._release(fan_out._acquire(destination))
        assert len(fan_out._processors) > _MAX_CACHED_DESTINATION_PROCESSORS

        release.set()
        for _ in range(500):
            for destination in anchored[-4:]:
                fan_out._release(fan_out._acquire(destination))
            if len(fan_out._processors) <= _MAX_CACHED_DESTINATION_PROCESSORS:
                break
            time.sleep(0.02)

        assert len(fan_out._processors) == _MAX_CACHED_DESTINATION_PROCESSORS, "the cache never came back to its cap"
        shed = len(built) - _MAX_CACHED_DESTINATION_PROCESSORS
        for _ in range(500):
            if sum(processor.shutdown_calls for processor in built) == shed:
                break
            time.sleep(0.02)

        assert sum(processor.shutdown_calls for processor in built) == shed, "a shed processor was never closed"

    def test_concurrent_eviction_cannot_build_between_retirement_and_drain_submission(self):
        """A second request cannot build while the first eviction is being handed to
        the drain, or concurrent churn can outrun the pending-drain limit."""
        import threading

        from litellm.integrations.otel.plumbing.providers import (
            _MAX_CACHED_DESTINATION_PROCESSORS,
            _DrainPool,
        )

        class GatedDrain(_DrainPool):
            def __init__(self):
                super().__init__(workers=0)
                self.started = threading.Event()
                self.release = threading.Event()

            def saturated(self):
                return False

            def submit(self, processor):
                if not self.started.is_set():
                    self.started.set()
                    self.release.wait(timeout=5)

        built = []

        def factory(_destination):
            built.append(self.Recording())
            return built[-1]

        drain = GatedDrain()
        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, drain_pool=drain)
        for index in range(_MAX_CACHED_DESTINATION_PROCESSORS):
            fan_out._release(fan_out._acquire(self._dest(index)))

        first = threading.Thread(target=lambda: fan_out._release(fan_out._acquire(self._dest(32))))
        first.start()
        assert drain.started.wait(timeout=5)
        second = threading.Thread(target=lambda: fan_out._release(fan_out._acquire(self._dest(33))))
        second.start()
        time.sleep(0.1)

        assert len(built) == _MAX_CACHED_DESTINATION_PROCESSORS + 1

        drain.release.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not first.is_alive() and not second.is_alive()
        assert len(built) == _MAX_CACHED_DESTINATION_PROCESSORS + 2

    def test_drain_workers_are_daemons(self):
        """Python joins a ThreadPoolExecutor's workers at interpreter exit, so one
        unreachable tenant collector would hold the proxy open for its export
        timeout on the way down."""
        import threading

        self._fan_out()
        workers = [t for t in threading.enumerate() if t.name.startswith("litellm-otel-destination-drain")]

        assert workers, "no drain worker was started"
        assert all(t.daemon for t in workers), "a non-daemon drain worker blocks interpreter exit"

    def test_a_burst_of_first_evictions_starts_one_set_of_drain_workers(self):
        """A drain pool built lazily on first use is not built once: several threads
        can each finish the build, and every pool but the winner is left with its
        workers blocked on a queue nothing will ever feed again."""
        import threading

        from litellm.integrations.otel.plumbing.providers import (
            _DRAIN_WORKERS,
            _MAX_CACHED_DESTINATION_PROCESSORS,
        )

        for _ in range(3):
            before = self._drain_workers()
            fan_out, built = self._fan_out()
            for index in range(_MAX_CACHED_DESTINATION_PROCESSORS):
                fan_out._release(fan_out._acquire(self._dest(index)))
            barrier = threading.Barrier(16)

            def shed(index, fan_out=fan_out, barrier=barrier):
                barrier.wait(timeout=10)
                fan_out._release(fan_out._acquire(self._dest(index)))

            threads = [
                threading.Thread(target=shed, args=(_MAX_CACHED_DESTINATION_PROCESSORS + index,)) for index in range(16)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self._settle(fan_out)

            assert self._drain_workers() - before == _DRAIN_WORKERS

    @staticmethod
    def _drain_workers():
        import threading

        return len([t for t in threading.enumerate() if t.name.startswith("litellm-otel-destination-drain")])

    def test_shutdown_does_not_close_a_processor_under_an_in_flight_export(self):
        """``on_end`` runs on whichever thread ends a span, so it reaches the fan-out
        while the SDK tears the provider down."""
        import threading

        fan_out, _ = self._fan_out()
        held = fan_out._acquire(self._dest(0))
        closed = threading.Thread(target=fan_out.shutdown)
        closed.start()
        try:
            time.sleep(0.3)

            assert held.shutdown_calls == 0, "closed a processor with a span still being forwarded"
        finally:
            fan_out._release(held)
            closed.join(timeout=10)

        assert held.shutdown_calls == 1

    def test_a_closed_fan_out_builds_no_new_processor(self):
        """A processor built after shutdown is one nothing will ever close, and it
        exports to a tenant on a provider the SDK has already torn down."""
        fan_out, built = self._fan_out()
        fan_out.shutdown()

        assert fan_out._acquire(self._dest(0)) is None
        assert built == []

    def test_shutdown_gives_up_on_an_export_that_never_finishes(self):
        """The wait is bounded: an exporter stuck on a dead collector must not hold
        the proxy open on the way down."""
        import threading

        fan_out = TenantFanOutSpanProcessor(processor_factory=lambda _d: self.Recording(), shutdown_drain_seconds=0.2)
        fan_out._acquire(self._dest(0))
        closed = threading.Thread(target=fan_out.shutdown)
        closed.start()
        closed.join(timeout=5)

        assert not closed.is_alive(), "shutdown blocked on an export that never finished"

    def test_shutdown_retires_the_drain_workers(self):
        """A proxy that rebuilds its telemetry builds another fan-out, so workers that
        outlive the one that started them are two more threads per reload."""
        from litellm.integrations.otel.plumbing.providers import _DRAIN_WORKERS

        before = self._drain_workers()
        fan_out, _ = self._fan_out()
        assert self._drain_workers() - before == _DRAIN_WORKERS

        fan_out.shutdown()
        for _ in range(500):
            if self._drain_workers() == before:
                break
            time.sleep(0.02)

        assert self._drain_workers() == before, "the drain workers outlived their fan-out"

    def test_a_processor_shed_after_shutdown_is_still_closed(self):
        """``close`` retires the workers, so anything handed to the pool afterwards
        would sit in a queue nobody reads."""
        fan_out, _ = self._fan_out()
        stray = self.Recording()
        fan_out.shutdown()
        fan_out._drain.submit(stray)

        for _ in range(500):
            if stray.shutdown_calls:
                break
            time.sleep(0.02)

        assert stray.shutdown_calls == 1

    def test_releasing_a_straggler_after_shutdown_does_not_block_the_span_thread(self):
        """The teardown deadline has already expired by then, so closing the straggler
        inline would park whichever thread just ended a span on the very flush the
        deadline gave up waiting for."""
        import threading

        never = threading.Event()

        class Stuck(self.Recording):
            def shutdown(self):
                never.wait()

        def factory(_destination):
            return Stuck()

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, shutdown_drain_seconds=0.05)
        held = fan_out._acquire(self._dest(0))
        fan_out.shutdown()

        released = threading.Event()
        caller = threading.Thread(target=lambda: (fan_out._release(held), released.set()), daemon=True)
        caller.start()
        came_back = released.wait(timeout=5)
        never.set()

        assert came_back, "the thread that ended the span was left holding a stuck teardown"

    def test_shutdown_waits_out_an_export_that_lands_inside_the_bound(self):
        """Without the wait the closing is left to a daemon thread, which the
        interpreter can retire before it runs, so the last spans never reach the
        tenant."""
        import threading

        fan_out, built = self._fan_out()
        held = fan_out._acquire(self._dest(0))
        threading.Timer(0.2, lambda: fan_out._release(held)).start()

        fan_out.shutdown()

        assert held.shutdown_calls == 1, "shutdown returned before the export it should have waited out"

    def test_a_straggler_past_the_drain_bound_is_closed_by_its_own_thread(self):
        """The wait is bounded so one dead collector cannot hold the proxy open, which
        means a processor still exporting when it expires has to be left to the thread
        holding it rather than closed under the span it is carrying."""
        built = []

        def factory(_destination):
            built.append(self.Recording())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, shutdown_drain_seconds=0.05)
        held = fan_out._acquire(self._dest(0))

        fan_out.shutdown()

        assert held.shutdown_calls == 0

        fan_out._release(held)
        self._settle(fan_out, held)

        assert held.shutdown_calls == 1

    def test_a_processor_built_while_shutdown_waits_is_still_closed(self):
        """Shutdown cannot slip between the build and the insert, which would leave a
        live exporter, with its batch thread and its connection pool, in a map nothing
        will read again."""
        import threading

        built = []

        def slow(_destination):
            time.sleep(0.4)
            built.append(self.Recording())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=slow, shutdown_drain_seconds=0.05)
        acquired = []
        caller = threading.Thread(target=lambda: acquired.append(fan_out._acquire(self._dest(0))))
        caller.start()
        time.sleep(0.1)
        fan_out.shutdown()
        caller.join(timeout=10)

        assert acquired == built, "the build shutdown waited out was thrown away"

        fan_out._release(built[0])
        self._settle(fan_out, built[0])

        assert built[0].shutdown_calls == 1, "the exporter outlived the fan-out"
        assert fan_out._processors == {}, "an exporter was left in a cleared cache"

    def test_shutdown_returns_when_a_destination_never_finishes_closing(self):
        """Closing an exporter flushes over the network and the SDK joins its own
        worker with no timeout, so a tenant collector that answers but never finishes
        a response would hold process teardown open for as long as it likes."""
        import threading

        never = threading.Event()

        class Stuck(self.Recording):
            def shutdown(self):
                never.wait()

        built = []

        def factory(_destination):
            built.append(Stuck())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory, shutdown_drain_seconds=0.3)
        fan_out._release(fan_out._acquire(self._dest(0)))
        returned = threading.Event()
        threading.Thread(target=lambda: (fan_out.shutdown(), returned.set()), daemon=True).start()

        came_back = returned.wait(timeout=8)
        never.set()

        assert came_back, "shutdown never returned while a collector held its exporter open"

    def test_a_cold_cache_met_by_a_burst_builds_one_processor_per_destination(self):
        """Building outside the cache lock let every thread of the burst construct its
        own exporter, each with a batch thread and a connection pool, and shed all but
        one into the drain."""
        import threading

        built = []

        def factory(_destination):
            time.sleep(0.01)
            built.append(self.Recording())
            return built[-1]

        fan_out = TenantFanOutSpanProcessor(processor_factory=factory)
        ready = threading.Barrier(8)

        def acquire():
            ready.wait()
            fan_out._release(fan_out._acquire(self._dest(0)))

        callers = [threading.Thread(target=acquire) for _ in range(8)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=10)

        assert len(built) == 1, f"one destination, {len(built)} exporters built"

    def test_a_submit_racing_close_is_never_stranded_behind_the_sentinels(self):
        """A submit that read the closed state and then let ``close`` run queues its
        processor after every sentinel, where the workers have already exited."""
        import queue
        import threading

        from litellm.integrations.otel.plumbing.providers import _DrainPool

        at_the_put, close_returned = threading.Event(), threading.Event()

        class Gated(queue.Queue):
            def put(self, item, *args, **kwargs):
                if item is not None:
                    at_the_put.set()
                    close_returned.wait(timeout=1)
                super().put(item, *args, **kwargs)

        pool = _DrainPool(pending=Gated())
        submitted = self.Recording()
        submitter = threading.Thread(target=pool.submit, args=(submitted,))
        submitter.start()
        assert at_the_put.wait(timeout=5)
        closer = threading.Thread(target=pool.close)
        closer.start()
        closer.join(timeout=1.5)
        close_returned.set()
        submitter.join(timeout=5)
        closer.join(timeout=5)
        for _ in range(250):
            if submitted.shutdown_calls:
                break
            time.sleep(0.02)

        assert submitted.shutdown_calls == 1, "a processor was queued behind the sentinels and never closed"

    def test_a_retired_processor_is_still_closed_after_shutdown(self):
        """Eviction and shutdown can both land while a span is being forwarded, and the
        evicted processor still has to be closed once that export returns."""
        from litellm.integrations.otel.plumbing.providers import _MAX_CACHED_DESTINATION_PROCESSORS

        fan_out, built = self._fan_out()
        held = fan_out._acquire(self._dest(0))
        for index in range(1, _MAX_CACHED_DESTINATION_PROCESSORS + 1):
            fan_out._acquire(self._dest(index))
            fan_out._release(built[-1])
        fan_out.shutdown()

        assert held.shutdown_calls == 0

        fan_out._release(held)
        self._settle(fan_out, held)

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

    def test_the_synthesized_stdout_placeholder_is_dropped(self, monkeypatch):
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        for name in _OTEL_SHORTHAND_ENV:
            monkeypatch.delenv(name, raising=False)
        placeholder = OpenTelemetryV2Config().exporters[0]

        kept = credential_gated_exporters((placeholder,), ExporterOwner.LANGFUSE_OTEL)

        assert [spec.owner for spec in kept] == [ExporterOwner.LANGFUSE_OTEL]

    def test_a_console_exporter_the_operator_named_survives(self, monkeypatch):
        """Same kind, endpoint and headers as the placeholder; only the fact that the
        operator set ``OTEL_EXPORTER`` tells them apart."""
        from litellm.integrations.otel.presets.utils import credential_gated_exporters

        for name in _OTEL_SHORTHAND_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OTEL_EXPORTER", "console")
        operator_console = OpenTelemetryV2Config().exporters[0]

        kept = credential_gated_exporters((operator_console,), ExporterOwner.LANGFUSE_OTEL)

        assert kept[0] is operator_console


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


_ALL_MAPPERS: Final = ("genai", "legacy", "openinference", "langfuse", "weave", "langtrace")
_SECRET: Final = "SECRET-MARKER"
_MODEL_CALL_WITH_CONTENT: Final = LLMCallSpanData(
    operation=GenAIOperation.CHAT,
    provider="openai",
    request_model="gpt-4o",
    response_model="gpt-4o-2024",
    response_id="resp_1",
    request_params=LLMRequestParams(temperature=0.5, max_tokens=128),
    usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
    finish_reasons=("tool_calls",),
    error=None,
    response_cost=0.001,
    server=ServerInfo("api.openai.com", 443),
    identity=RequestIdentity(call_id="c1", team_id="t1"),
    tools=(ToolDefinition(name="lookup", description="Find a city", parameters_json='{"type":"object"}'),),
    messages_in=(
        {"role": "system", "content": f"system {_SECRET}"},
        {"role": "user", "content": f"prompt {_SECRET}"},
    ),
    choices_out=(
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": f"answer {_SECRET}",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": f'{{"city": "{_SECRET}"}}'},
                    }
                ],
            },
        },
    ),
)
_MODEL_CALL_WITHOUT_CONTENT: Final = replace(_MODEL_CALL_WITH_CONTENT, messages_in=(), choices_out=())
_TOOL_CALL_WITH_CONTENT: Final = MCPToolCallSpanData(
    operation=GenAIOperation.EXECUTE_TOOL,
    method="tools/call",
    tool_name="lookup",
    server_name="maps",
    server_address="maps.local",
    server_port=443,
    session_id="s1",
    arguments_json=f'{{"city": "{_SECRET}"}}',
    result_json=f'{{"weather": "{_SECRET}"}}',
    error=None,
    response_cost=None,
    identity=RequestIdentity(call_id="c2", team_id="t1"),
)
_TOOL_CALL_WITHOUT_CONTENT: Final = replace(_TOOL_CALL_WITH_CONTENT, arguments_json=None, result_json=None)

NO_CONTENT_DEST = OtelDestination(
    endpoint="http://team-a.local/api/public/otel",
    headers=MappingProxyType({"Authorization": "Basic YQ=="}),
    callback_name="langfuse_otel",
    capture_message_content="no_content",
)
INHERITING_DEST = OtelDestination(
    endpoint="http://team-b.local/v1",
    headers=MappingProxyType({"api_key": "k", "arize-space-id": "s"}),
    callback_name="arize",
)


def mapped(data: LLMCallSpanData | MCPToolCallSpanData) -> Mapping[str, object]:
    """What the configured mappers write on the span, the way the logger stamps it."""
    return MappingProxyType(reduce(lambda acc, mapper: {**acc, **mapper.map(data)}, resolve_mappers(_ALL_MAPPERS), {}))


def recorded(attributes: Mapping[str, object]) -> dict[str, object]:
    """``attributes`` the way the SDK records them, sequences frozen into tuples."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in attributes.items()}


def model_call_tree(provider: TracerProvider, attributes: Mapping[str, object]) -> None:
    tracer: Final = get_tracer(provider, "litellm")
    with tracer.start_as_current_span("POST /v1/chat/completions"):
        with tracer.start_as_current_span("chat gpt-4o") as llm:
            llm.set_attributes(attributes)
            llm.add_event("gen_ai.content.first_chunk", {"gen_ai.response.model": "gpt-4o-2024"})


def by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def carries_content(span: ReadableSpan) -> bool:
    return any(_SECRET in str(value) for value in span.attributes.values())


class TestCaptureMessageContent:
    """A destination's explicit capture mode overrides the global default."""

    @staticmethod
    def _fan_out(
        operator: InMemorySpanExporter,
        exporters: Mapping[str, InMemorySpanExporter],
        default_capture: CaptureMessageContent = CaptureMessageContent.SPAN_ONLY,
    ) -> TracerProvider:
        provider: Final = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(operator))
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda d: SimpleSpanProcessor(exporters[d.endpoint]),
                default_capture=default_capture,
            )
        )
        return provider

    @staticmethod
    def _run(
        provider: TracerProvider, destinations: tuple[OtelDestination, ...], attributes: Mapping[str, object]
    ) -> None:
        def run() -> None:
            set_request_destinations(destinations)
            model_call_tree(provider, attributes)

        in_fresh_context(run)

    @pytest.mark.parametrize(
        ("setting", "content_exported"),
        [
            (None, True),
            ("no_content", False),
            ("span_only", True),
            ("event_only", False),
            ("span_and_event", True),
        ],
    )
    def test_a_destination_setting_controls_its_copy_and_omission_uses_the_global_default(
        self, setting: str | None, content_exported: bool
    ) -> None:
        attributes: Final = mapped(_MODEL_CALL_WITH_CONTENT)
        destination: Final = OtelDestination(
            endpoint="http://team.local/api/public/otel",
            headers=MappingProxyType({"Authorization": "Basic dA=="}),
            callback_name="langfuse_otel",
            capture_message_content=setting,
        )
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._fan_out(operator, {destination.endpoint: tenant}), (destination,), attributes)

        exported: Final = by_name(tenant)["chat gpt-4o"]
        assert carries_content(exported) is content_exported
        assert carries_content(by_name(operator)["chat gpt-4o"])
        assert exported.attributes["gen_ai.request.model"] == "gpt-4o"
        assert exported.attributes["gen_ai.usage.input_tokens"] == 12
        assert exported.attributes["gen_ai.usage.output_tokens"] == 8

    def test_a_span_only_team_lifts_global_no_content_only_for_its_destination(self) -> None:
        operator_exporter, team_exporter, sibling_exporter = (
            InMemorySpanExporter(),
            InMemorySpanExporter(),
            InMemorySpanExporter(),
        )
        kind: Final = "lit8244_global_no_content_capture"
        register_exporter_factory(kind, lambda _spec: operator_exporter)
        config: Final = OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.NO_CONTENT,
            exporters=[ExporterSpec(kind=kind)],
        )
        provider: Final = build_tracer_provider(config, use_simple_processor=True, tenant_overrides=True)
        team: Final = OtelDestination(
            endpoint="http://team.local/api/public/otel",
            headers=MappingProxyType({"Authorization": "Basic dA=="}),
            callback_name="langfuse_otel",
            capture_message_content="span_only",
        )
        sibling: Final = INHERITING_DEST
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda destination: SimpleSpanProcessor(
                    team_exporter if destination.endpoint == team.endpoint else sibling_exporter
                ),
                default_capture=config.capture_message_content,
            )
        )

        self._run(
            provider,
            (team, sibling),
            mapped(_MODEL_CALL_WITH_CONTENT),
        )

        assert carries_content(by_name(team_exporter)["chat gpt-4o"])
        assert not any(carries_content(span) for span in operator_exporter.get_finished_spans())
        assert not any(carries_content(span) for span in sibling_exporter.get_finished_spans())

    @pytest.mark.parametrize(
        ("with_content", "without_content"),
        [
            (_MODEL_CALL_WITH_CONTENT, _MODEL_CALL_WITHOUT_CONTENT),
            (_TOOL_CALL_WITH_CONTENT, _TOOL_CALL_WITHOUT_CONTENT),
        ],
        ids=["model_call", "mcp_tool_call"],
    )
    def test_no_content_removes_exactly_what_global_capture_adds_in_every_mapper_vocabulary(
        self,
        with_content: LLMCallSpanData | MCPToolCallSpanData,
        without_content: LLMCallSpanData | MCPToolCallSpanData,
    ) -> None:
        captured, uncaptured = mapped(with_content), mapped(without_content)
        assert frozenset(captured) - frozenset(uncaptured), "the fixture must exercise captured content"
        operator, tenant = InMemorySpanExporter(), InMemorySpanExporter()

        self._run(self._fan_out(operator, {NO_CONTENT_DEST.endpoint: tenant}), (NO_CONTENT_DEST,), captured)

        exported: Final = by_name(tenant)["chat gpt-4o"]
        assert dict(exported.attributes) == recorded(uncaptured), "only content goes, every other attribute stays"
        assert not carries_content(exported)

    def test_restricting_one_destination_leaves_the_original_span_and_the_other_destination_alone(self) -> None:
        attributes: Final = mapped(_MODEL_CALL_WITH_CONTENT)
        operator, restricted, inheriting = InMemorySpanExporter(), InMemorySpanExporter(), InMemorySpanExporter()
        exporters: Final = {NO_CONTENT_DEST.endpoint: restricted, INHERITING_DEST.endpoint: inheriting}

        self._run(self._fan_out(operator, exporters), (NO_CONTENT_DEST, INHERITING_DEST), attributes)

        original: Final = by_name(operator)
        kept: Final = by_name(inheriting)
        stripped: Final = by_name(restricted)
        assert dict(original["chat gpt-4o"].attributes) == recorded(attributes)
        assert dict(kept["chat gpt-4o"].attributes) == recorded(attributes)
        assert not carries_content(stripped["chat gpt-4o"])
        for copy in (kept, stripped):
            for name, span in copy.items():
                assert span.context == original[name].context, "same trace id and span id"
                assert span.parent == original[name].parent, "same place in the tree"
            assert [e.name for e in copy["chat gpt-4o"].events] == [e.name for e in original["chat gpt-4o"].events]

    def test_an_omitted_setting_and_span_only_export_the_same_span(self) -> None:
        attributes: Final = mapped(_MODEL_CALL_WITH_CONTENT)
        span_only: Final = INHERITING_DEST.model_copy(
            update={"capture_message_content": CaptureMessageContent.SPAN_ONLY}
        )
        omitted_exporter, span_only_exporter = InMemorySpanExporter(), InMemorySpanExporter()

        for destination, exporter in ((INHERITING_DEST, omitted_exporter), (span_only, span_only_exporter)):
            self._run(
                self._fan_out(InMemorySpanExporter(), {destination.endpoint: exporter}), (destination,), attributes
            )

        assert {n: dict(s.attributes) for n, s in by_name(omitted_exporter).items()} == {
            n: dict(s.attributes) for n, s in by_name(span_only_exporter).items()
        }

    @pytest.mark.parametrize(
        ("globally_captured", "setting", "content_exported"),
        [
            (True, "no_content", False),
            (True, None, True),
            (False, "span_only", True),
            (False, None, False),
        ],
    )
    def test_the_operators_exporter_on_the_same_account_honors_the_teams_setting_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        globally_captured: bool,
        setting: str | None,
        content_exported: bool,
    ) -> None:
        """The fan-out skips a destination the operator already writes to, so the operator's
        copy is the one that account receives and must not bypass the team's restriction."""
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", "additive", raising=False)
        shared: Final = InMemorySpanExporter()
        provider: Final = TracerProvider()
        provider.add_span_processor(
            _OverriddenBackendFilter(
                SimpleSpanProcessor(shared),
                "langfuse_otel",
                "full",
                TestRoutingMode.OPERATOR_SINK,
                global_captures=globally_captured,
            )
        )
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=MappingProxyType({TestRoutingMode.OPERATOR_SINK: "full"}),
            )
        )
        destination: Final = OtelDestination(
            endpoint=TestRoutingMode.SAME_ACCOUNT_ENDPOINT,
            headers=MappingProxyType({"Authorization": "Basic op"}),
            callback_name="langfuse_otel",
            capture_message_content=setting,
        )

        self._run(provider, (destination,), mapped(_MODEL_CALL_WITH_CONTENT))

        model_calls: Final = [s for s in shared.get_finished_spans() if s.name == "chat gpt-4o"]
        assert len(model_calls) == 1, "the same account received the span twice"
        assert carries_content(model_calls[0]) is content_exported

    @pytest.mark.parametrize("mode", ["override", "additive"])
    def test_an_operator_collector_on_the_teams_account_does_not_bypass_no_content(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.setattr(litellm, "otel_tenant_destination_mode", mode, raising=False)
        shared: Final = InMemorySpanExporter()
        kind: Final = f"lit8244_collector_{mode}"
        register_exporter_factory(kind, lambda _spec: shared)
        config: Final = OpenTelemetryV2Config(
            exporters=[
                ExporterSpec(
                    kind=kind,
                    endpoint=TestRoutingMode.OPERATOR_SINK[0],
                    headers="authorization=Basic op",
                )
            ]
        )
        provider: Final = build_tracer_provider(config, use_simple_processor=True, tenant_overrides=True)
        provider.add_span_processor(
            TenantFanOutSpanProcessor(
                processor_factory=lambda _d: SimpleSpanProcessor(shared),
                operator_sinks=operator_sink_scopes(config),
            )
        )
        destination: Final = OtelDestination(
            endpoint=TestRoutingMode.SAME_ACCOUNT_ENDPOINT,
            headers=MappingProxyType({"Authorization": "Basic op"}),
            callback_name="langfuse_otel",
            capture_message_content="no_content",
        )

        self._run(provider, (destination,), mapped(_MODEL_CALL_WITH_CONTENT))

        model_calls: Final = [s for s in shared.get_finished_spans() if s.name == "chat gpt-4o"]
        assert model_calls, "the account must receive the model call before its content can be judged absent"
        assert not any(carries_content(span) for span in model_calls)

    @pytest.mark.parametrize(
        ("global_capture", "content_exported"),
        [(CaptureMessageContent.NO_CONTENT, False), (CaptureMessageContent.SPAN_ONLY, True)],
    )
    def test_a_routed_provider_exports_content_only_when_the_global_setting_captures(
        self, global_capture: str, content_exported: bool
    ) -> None:
        """A sibling destination's span_only makes the logger collect content, and a request
        routed to this callback's team credentials must not carry it under a global no_content."""
        exporters: dict[ExporterOwner | None, InMemorySpanExporter] = {}
        kind: Final = f"lit8244_routed_{global_capture}"
        register_exporter_factory(kind, lambda spec: exporters.setdefault(spec.owner, InMemorySpanExporter()))
        config: Final = OpenTelemetryV2Config(
            capture_message_content=global_capture,
            exporters=[
                ExporterSpec(
                    kind=kind,
                    endpoint="http://op.local",
                    owner=ExporterOwner.LANGFUSE_OTEL,
                    use_simple_processor=True,
                ),
                ExporterSpec(kind=kind),
            ],
        )
        cache: Final = TenantTracerCache(config, "langfuse_otel", "litellm")

        route: Final = cache.route_for(
            get_tracer(TracerProvider(), "litellm"),
            {"langfuse_public_key": "pk-team", "langfuse_secret_key": "sk-team"},
        )
        with route.tracer.start_as_current_span("chat gpt-4o") as llm:
            llm.set_attributes(mapped(_MODEL_CALL_WITH_CONTENT))
        cache.release(route.provider)

        assert route.detached is True
        for exporter in (exporters[ExporterOwner.LANGFUSE_OTEL], exporters[None]):
            copy = by_name(exporter)["chat gpt-4o"]
            assert carries_content(copy) is content_exported
            assert copy.attributes["gen_ai.usage.input_tokens"] == 12

    @pytest.mark.parametrize("first", ["newrelic", "langfuse_otel"])
    def test_an_omitted_setting_follows_its_own_callbacks_mode_whatever_the_callback_order(
        self, monkeypatch: pytest.MonkeyPatch, first: str
    ) -> None:
        monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        tenant: Final = InMemorySpanExporter()
        monkeypatch.setitem(otel_providers._EXPORTER_FACTORIES, "otlp_http", lambda _spec: tenant)
        langfuse: Final = OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
            exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.LANGFUSE_OTEL)],
        )
        newrelic: Final = OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.NO_CONTENT,
            exporters=[ExporterSpec(kind="in_memory", owner=ExporterOwner.NEWRELIC)],
        )
        provider: Final = TracerProvider()
        attach_tenant_fan_out(provider, *((newrelic, langfuse) if first == "newrelic" else (langfuse, newrelic)))
        destination: Final = OtelDestination(
            endpoint="http://team.local/api/public/otel",
            headers=MappingProxyType({"Authorization": "Basic dA=="}),
            callback_name="langfuse_otel",
        )

        self._run(provider, (destination,), mapped(_MODEL_CALL_WITH_CONTENT))
        provider.force_flush()

        assert carries_content(by_name(tenant)["chat gpt-4o"])

    def test_an_omitted_setting_follows_the_capture_mode_set_in_callback_settings_otel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        tenant: Final = InMemorySpanExporter()
        monkeypatch.setitem(otel_providers._EXPORTER_FACTORIES, "otlp_http", lambda _spec: tenant)
        operator: Final = OpenTelemetryV2Config(**{"capture_message_content": "span_only", "exporter": "in_memory"})
        provider: Final = TracerProvider()
        attach_tenant_fan_out(provider, operator)
        destination: Final = OtelDestination(
            endpoint="http://team.local/api/public/otel",
            headers=MappingProxyType({"Authorization": "Basic dA=="}),
            callback_name="langfuse_otel",
        )

        self._run(provider, (destination,), mapped(_MODEL_CALL_WITH_CONTENT))
        provider.force_flush()

        assert carries_content(by_name(tenant)["chat gpt-4o"]), "the YAML value is the global default, not only the env"

    def test_a_published_presets_capture_mode_is_the_default_for_an_unowned_destination(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A destination for a backend the proxy does not export to itself falls back to the published logger's mode,
        here New Relic with record_content on, so it keeps the content."""
        monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        tenant: Final = InMemorySpanExporter()
        monkeypatch.setitem(otel_providers._EXPORTER_FACTORIES, "otlp_http", lambda _spec: tenant)
        newrelic: Final = OpenTelemetryV2Config(
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
            exporters=[
                ExporterSpec(kind="console"),
                ExporterSpec(kind="in_memory", owner=ExporterOwner.NEWRELIC),
            ],
        )
        provider: Final = TracerProvider()
        attach_tenant_fan_out(provider, newrelic)
        destination: Final = OtelDestination(
            endpoint="http://team.local/api/public/otel",
            headers=MappingProxyType({"Authorization": "Basic dA=="}),
            callback_name="langfuse_otel",
        )

        self._run(provider, (destination,), mapped(_MODEL_CALL_WITH_CONTENT))
        provider.force_flush()

        assert carries_content(by_name(tenant)["chat gpt-4o"])

    @pytest.mark.usefixtures("allow_test_hosts")
    @pytest.mark.parametrize("setting", ["no_content", "span_only", None])
    def test_the_setting_rides_the_teams_destination_and_omission_stays_omitted(
        self, monkeypatch: pytest.MonkeyPatch, setting: str | None
    ) -> None:
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        callback_vars: Final = {
            "langfuse_public_key": "pk-team",
            "langfuse_secret_key": "sk-team",
            "langfuse_host": "http://team.local",
        }
        if setting is not None:
            callback_vars["capture_message_content"] = setting
        auth: Final = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": callback_vars},
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {"arize_api_key": "k", "arize_space_id": "s"},
                    },
                ]
            }
        )

        destinations: Final = {d.callback_name: d for d in resolve_tenant_otel_destinations(auth)}

        assert destinations["langfuse_otel"].capture_message_content == setting
        assert destinations["arize"].capture_message_content is None, "the setting stays on its own callback"

    @pytest.mark.usefixtures("allow_test_hosts")
    @pytest.mark.parametrize("setting", ["event_only", "span_and_event"])
    def test_a_stored_event_mode_is_skipped_like_any_invalid_entry(
        self, monkeypatch: pytest.MonkeyPatch, setting: str
    ) -> None:
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        is_otel_v2_enabled.cache_clear()
        auth: Final = UserAPIKeyAuth(
            team_metadata={
                "logging": [
                    {
                        "callback_name": "langfuse_otel",
                        "callback_type": "success",
                        "callback_vars": {
                            "langfuse_public_key": "pk-team",
                            "langfuse_secret_key": "sk-team",
                            "langfuse_host": "http://team.local",
                            "capture_message_content": setting,
                        },
                    },
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {"arize_api_key": "k", "arize_space_id": "s"},
                    },
                ]
            }
        )

        destinations: Final = [d.callback_name for d in resolve_tenant_otel_destinations(auth)]

        assert destinations == ["arize"]

    @pytest.mark.parametrize("value", ["full", "NO_CONTENT", "", "true", "event_only", "span_and_event"])
    def test_an_unsupported_value_fails_registration(self, value: str) -> None:
        with pytest.raises(
            ValueError,
            match=r"Invalid capture_message_content .*\['no_content', 'span_only'\]",
        ):
            AddTeamCallback(
                callback_name="langfuse_otel",
                callback_type="success",
                callback_vars={
                    "langfuse_public_key": "pk",
                    "langfuse_secret_key": "sk",
                    "capture_message_content": value,
                },
            )

    @pytest.mark.parametrize("value", ["no_content", "span_only"])
    def test_a_supported_value_is_stored_as_given(self, value: str) -> None:
        saved: Final = AddTeamCallback(
            callback_name="langfuse_otel",
            callback_type="success",
            callback_vars={"langfuse_public_key": "pk", "langfuse_secret_key": "sk", "capture_message_content": value},
        )

        assert saved.callback_vars["capture_message_content"] == value

    def test_a_team_wide_callback_vars_map_cannot_carry_the_setting(self) -> None:
        """The flattened map is shared by every callback, so a value there would apply to all of them."""
        with pytest.raises(ValueError, match="Invalid callback variable: capture_message_content"):
            TeamCallbackMetadata(
                success_callback=["langfuse_otel", "signoz"], callback_vars={"capture_message_content": "span_only"}
            )

    def test_flattening_the_entries_leaves_the_setting_on_its_own_entry(self) -> None:
        flattened: Final = convert_key_logging_metadata_to_callback(
            AddTeamCallback(
                callback_name="langfuse_otel",
                callback_type="success",
                callback_vars={"langfuse_public_key": "pk", "capture_message_content": "span_only"},
            ),
            None,
        )

        assert flattened.callback_vars == {"langfuse_public_key": "pk"}


class TestArizeProjectRouting:
    """Arize rejects an export that names no project (400: set x-project-name,
    arize.project.name, openinference.project.name, or model_id), so an Arize
    destination carries ``model_id`` the way the operator's own preset does."""

    TEAM_PARAMS: Final = MappingProxyType({"arize_space_id": "space-team", "arize_api_key": "key-team"})

    @pytest.fixture(autouse=True)
    def _operator_arize(self, monkeypatch):
        for name in ("ARIZE_SPACE_KEY", "ARIZE_ENDPOINT", "ARIZE_HTTP_ENDPOINT", "ARIZE_PROJECT_NAME"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ARIZE_SPACE_ID", "space-operator")
        monkeypatch.setenv("ARIZE_API_KEY", "key-operator")

    @classmethod
    def _team_destination(cls, service_name: str | None = None) -> OtelDestination:
        destination: Final = destination_for("arize", dict(cls.TEAM_PARAMS), service_name=service_name)
        assert destination is not None, "the team names a space and a key, so it must resolve"
        return destination

    def test_a_team_destination_lands_in_the_operators_arize_project(self, monkeypatch):
        monkeypatch.setenv("ARIZE_PROJECT_NAME", "gateway-prod")

        assert dict(self._team_destination().resource_attributes) == {"model_id": "gateway-prod"}

    @classmethod
    def _forwarded_projects(cls, resource: Resource | None = None) -> set[str]:
        dest = InMemorySpanExporter()
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(TenantFanOutSpanProcessor(processor_factory=lambda _d: SimpleSpanProcessor(dest)))

        def run():
            set_request_destinations((cls._team_destination(),))
            emit(provider)

        in_fresh_context(run)
        spans = dest.get_finished_spans()
        assert spans, "the team destination must receive the request"
        return {s.resource.attributes["model_id"] for s in spans}

    def test_a_team_export_names_a_project_even_when_the_operator_set_none(self):
        assert self._forwarded_projects() == {"litellm"}

    def test_a_project_the_operators_own_resource_names_is_kept_for_the_team(self):
        assert self._forwarded_projects(Resource({"model_id": "chosen"})) == {"chosen"}

    def test_arize_project_name_wins_over_the_project_the_operators_resource_names(self, monkeypatch):
        monkeypatch.setenv("ARIZE_PROJECT_NAME", "gateway-prod")

        assert self._forwarded_projects(Resource({"model_id": "chosen"})) == {"gateway-prod"}

    def test_the_teams_service_name_rides_beside_the_project(self, monkeypatch):
        monkeypatch.setenv("ARIZE_PROJECT_NAME", "gateway-prod")

        destination = self._team_destination(service_name="team-checkout")

        assert dict(destination.resource_attributes) == {"service.name": "team-checkout", "model_id": "gateway-prod"}

    def test_the_operators_exporter_names_a_project_even_without_arize_project_name(self):
        assert arize_preset().resource_attributes["model_id"] == "litellm"

    def test_the_operators_exporter_keeps_the_project_its_config_overrides_name(self):
        chosen = OpenTelemetryV2Config(resource_attributes={"model_id": "chosen"})

        assert arize_preset(config_overrides=chosen).resource_attributes["model_id"] == "chosen"

    def test_every_span_forwarded_to_the_team_carries_the_project(self, monkeypatch):
        monkeypatch.setenv("ARIZE_PROJECT_NAME", "gateway-prod")

        assert self._forwarded_projects() == {"gateway-prod"}
