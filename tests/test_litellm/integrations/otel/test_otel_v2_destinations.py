"""Key/team OTLP destinations override the operator's exporters for that backend."""

import contextvars
import time
from collections.abc import Mapping
from types import MappingProxyType

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel import logger as otel_logger
from litellm.integrations.otel.logger import (
    OpenTelemetryV2,
    build_otel_v2_logger,
    fan_out_provider,
    publish_global_otel_v2_provider,
)
from litellm.integrations.otel.model.config import (
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.destination import OtelDestination
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
    build_tracer_provider,
    deliverable_destinations,
    operator_sink_keys,
)
from litellm.integrations.otel.plumbing.routing import TenantTracerCache, get_tracer
from litellm.integrations.otel.presets.arize import arize_preset
from litellm.integrations.otel.presets.destinations import (
    destination_capable_backends,
    destination_for,
)
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import resolve_tenant_otel_destinations
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
                operator_sinks=frozenset({self.OPERATOR_SINK}),
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
                operator_sinks=frozenset({self.OPERATOR_SINK}),
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
                operator_sinks=frozenset({self.OPERATOR_SINK}),
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

    def test_operator_sink_keys_skips_an_exporter_with_no_endpoint_of_its_own(self):
        """Such an exporter resolves its endpoint from the environment at export
        time, so it has no identity to compare a destination against."""
        config = OpenTelemetryV2Config(
            exporters=(
                ExporterSpec(kind="otlp_http", endpoint=self.OPERATOR_SINK[0], headers="authorization=Basic op"),
                ExporterSpec(kind="otlp_http", endpoint=None, headers="authorization=Basic other"),
            )
        )

        assert operator_sink_keys(config) == frozenset({self.OPERATOR_SINK})

    def test_operator_sink_keys_skips_exporters_that_never_reach_the_wire(self):
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

        assert operator_sink_keys(config) == frozenset({self.OPERATOR_SINK})

    def test_operator_sink_keys_spans_every_config_it_is_handed(self):
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

        assert operator_sink_keys(first, second) == {
            self.OPERATOR_SINK,
            _sink_key("https://otlp.arize.com/v1/traces", {"space_id": "s", "api_key": "k"}),
        }

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
                operator_sinks=operator_sink_keys(config),
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
        operator = operator_sink_keys(langfuse_preset())

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
        operator = operator_sink_keys(arize_preset())

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

        in_fresh_context(run, (destination(canonical, dict(pair.split("=") for pair in accounts[canonical][1].split(","))),))
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

    def test_a_credentialed_logger_beside_another_v2_logger_keeps_every_exporter(self, monkeypatch):
        """Only a degraded preset gives the collector up; an operator who configured
        both the backend and the collector still exports to both, as on base."""
        from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-1")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.local:4318")
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        collector_logger = build_otel_v2_logger(OpenTelemetryV2Config(exporter="in_memory"))

        is_otel_v2_enabled.cache_clear()
        logger = in_fresh_context(_maybe_construct_otel_v2, "langfuse_otel", [collector_logger])
        is_otel_v2_enabled.cache_clear()

        assert logger is not None
        assert [spec.endpoint for spec in logger.config.exporters] == [
            "http://collector.local:4318",
            "https://cloud.langfuse.com/api/public/otel",
        ]
        assert all(spec.headers for spec in logger.config.exporters if spec.requires_headers)


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
            assert fan_out.deliverable((self._dest(999),)) == (), "the span would vanish instead of staying with the operator"
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
            _DrainPool,
            _MAX_CACHED_DESTINATION_PROCESSORS,
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
