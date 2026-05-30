"""Coverage for the engine-layer components: providers/exporters, context +
baggage helpers, metrics, the typed coercion helpers, mapper branches, span-name
builders, and the registry validator's failure paths. Needs the OTel SDK."""

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.otel.plumbing import context as ctx_mod  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config  # noqa: E402
from litellm.integrations.otel.mappers.genai import GenAIMapper  # noqa: E402
from litellm.integrations.otel.mappers.legacy import LegacyMapper  # noqa: E402
from litellm.integrations.otel.plumbing.metrics import (
    create_genai_metrics,
)  # noqa: E402
from litellm.integrations.otel.model.payloads import (  # noqa: E402
    GuardrailSpanData,
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    ProxyRequestSpanData,
    RequestIdentity,
    ServerInfo,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.model.semconv import GenAI, GenAIOperation
from litellm.integrations.otel.model.spans import (  # noqa: E402
    SPAN_REGISTRY,
    LiteLLMSpanKind,
    SpanRole,
    SpanSpec,
    db_system,
    guardrail_span_name,
    proxy_request_span_name,
    service_span_name,
    span_role_for_service,
    validate_registry,
)
from litellm.integrations.otel.model.utils import (  # noqa: E402
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_tuple,
)

# --- typed coercion helpers ------------------------------------------------- #


def test_as_str():
    assert as_str(None) is None
    assert as_str("x") == "x"
    assert as_str(5) == "5"


def test_as_int():
    assert as_int(True) == 1
    assert as_int(3) == 3
    assert as_int(3.9) == 3
    assert as_int("7") == 7
    assert as_int("nope") is None
    assert as_int(None) is None


def test_as_float():
    assert as_float(True) == 1.0
    assert as_float(2) == 2.0
    assert as_float("1.5") == 1.5
    assert as_float("nope") is None
    assert as_float(None) is None


def test_as_bool():
    assert as_bool(None) is None
    assert as_bool(True) is True
    assert as_bool(1) is True
    assert as_bool(0) is False


def test_as_str_tuple():
    assert as_str_tuple(None) is None
    assert as_str_tuple("a") == ("a",)
    assert as_str_tuple(["a", 2]) == ("a", "2")
    assert as_str_tuple(123) is None


def test_request_params_max_completion_tokens_fallback():
    params = LLMRequestParams.from_model_parameters({"max_completion_tokens": 99})
    assert params.max_tokens == 99


def test_server_info_from_api_base():
    assert ServerInfo.from_api_base(None) is None
    assert ServerInfo.from_api_base("api.host.com:8080") == ServerInfo(
        "api.host.com", 8080
    )
    assert ServerInfo.from_api_base("https://h.com/v1") == ServerInfo("h.com", None)
    # scheme present but empty netloc -> no hostname
    assert ServerInfo.from_api_base("http:///v1") is None


def test_service_span_data_from_payload():
    class _Service:
        value = "redis"

    class _Payload:
        service = _Service()
        call_type = "async_set_cache"
        error = None

    data = ServiceSpanData.from_payload(_Payload())
    assert data.service_name == "redis"
    assert data.call_type == "async_set_cache"
    assert data.error is None

    class _FailPayload:
        service = _Service()
        call_type = "async_set_cache"
        error = "boom"

    failed = ServiceSpanData.from_payload(_FailPayload())
    assert failed.error is not None
    assert failed.error.message == "boom"


# --- span name builders ----------------------------------------------------- #


def test_name_builders():
    assert (
        proxy_request_span_name(ProxyRequestSpanData("POST", "/chat/completions"))
        == "POST /chat/completions"
    )
    # "{service} {call_type}" so same-service calls stay distinguishable; the
    # service name alone when there's no call type.
    assert service_span_name(ServiceSpanData("redis", call_type="set")) == "redis set"
    assert service_span_name(ServiceSpanData("redis")) == "redis"
    assert (
        guardrail_span_name(GuardrailSpanData("presidio"))
        == "execute_guardrail presidio"
    )


# --- registry validator failure paths --------------------------------------- #


def test_validate_registry_detects_role_mismatch():
    bad = {SpanRole.LLM_CALL: SpanSpec(SpanRole.SERVICE, LiteLLMSpanKind.CLIENT, None)}
    with pytest.raises(ValueError, match="mismatched role"):
        validate_registry(bad)


def test_validate_registry_detects_unknown_parent():
    bad = {
        SpanRole.LLM_CALL: SpanSpec(
            SpanRole.LLM_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST
        )
    }
    with pytest.raises(ValueError, match="unknown parent"):
        validate_registry(bad)


def test_validate_registry_detects_missing_roles():
    partial = {
        SpanRole.PROXY_REQUEST: SPAN_REGISTRY[SpanRole.PROXY_REQUEST],
    }
    with pytest.raises(ValueError, match="missing roles"):
        validate_registry(partial)


# --- mappers (full branch coverage) ----------------------------------------- #


def _full_llm_call():
    return LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model="gpt-4o-2024",
        response_id="resp_1",
        request_params=LLMRequestParams(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=256,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop_sequences=("STOP",),
            seed=42,
        ),
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reasons=("stop",),
        error=None,
        response_cost=0.002,
        server=ServerInfo("api.openai.com", 443),
        identity=RequestIdentity(call_id="c1"),
        is_streaming=True,
    )


def test_genai_mapper_all_request_params():
    attrs = GenAIMapper().map(_full_llm_call())
    assert attrs[GenAI.REQUEST_TOP_P] == 0.9
    assert attrs[GenAI.REQUEST_TOP_K] == 40
    assert attrs[GenAI.REQUEST_MAX_TOKENS] == 256
    assert attrs[GenAI.REQUEST_FREQUENCY_PENALTY] == 0.1
    assert attrs[GenAI.REQUEST_PRESENCE_PENALTY] == 0.2
    assert attrs[GenAI.REQUEST_STOP_SEQUENCES] == ["STOP"]
    assert attrs[GenAI.REQUEST_SEED] == 42
    assert attrs["server.port"] == 443


def test_genai_mapper_guardrail_and_service():
    from litellm.integrations.otel.model.semconv import LiteLLM

    g = GenAIMapper().map(GuardrailSpanData("presidio", mode="pre"))
    assert g[LiteLLM.GUARDRAIL_NAME] == "presidio"
    assert g[LiteLLM.GUARDRAIL_MODE] == "pre"

    # A datastore service (redis) also gets db.* semconv.
    s = GenAIMapper().map(ServiceSpanData("redis", call_type="set"))
    assert s[LiteLLM.SERVICE_NAME] == "redis"
    assert s[LiteLLM.SERVICE_CALL_TYPE] == "set"
    assert s["db.system.name"] == "redis"
    assert s["db.operation.name"] == "set"

    # An internal service (router) gets no db.* keys.
    internal = GenAIMapper().map(ServiceSpanData("router", call_type="acompletion"))
    assert internal[LiteLLM.SERVICE_NAME] == "router"
    assert "db.system.name" not in internal


def test_legacy_mapper_all_request_params():
    attrs = LegacyMapper().map(_full_llm_call())
    assert attrs["llm.top_k"] == 40
    assert attrs["llm.frequency_penalty"] == 0.1
    assert attrs["llm.presence_penalty"] == 0.2
    assert attrs["llm.chat.stop_sequences"] == ["STOP"]
    assert attrs["gen_ai.usage.total_tokens"] == 15


def test_legacy_mapper_covers_service_with_v1_bare_keys():
    """Service spans dual-emit V1's bare ``service``/``call_type``/``error`` keys."""
    attrs = LegacyMapper().map(
        ServiceSpanData("redis", call_type="set", event_metadata={"k": "v"}),
    )
    assert attrs["service"] == "redis"
    assert attrs["call_type"] == "set"
    assert attrs["k"] == "v"  # event_metadata is stamped bare (V1 behavior)


def test_legacy_mapper_skips_guardrail_role():
    """Guardrail spans never had a V1 vocabulary; legacy mapper returns ``{}``."""
    assert LegacyMapper().map(GuardrailSpanData("presidio")) == {}


# --- metrics ---------------------------------------------------------------- #


def test_create_genai_metrics_records():
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    metrics = create_genai_metrics(meter)
    metrics.token_usage.record(10, {"x": "y"})
    metrics.operation_duration.record(0.5, {"x": "y"})
    data = reader.get_metrics_data()
    assert data is not None


# --- context + baggage helpers ---------------------------------------------- #


def test_extract_traceparent():
    valid = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    assert ctx_mod.extract_traceparent(valid) is not None
    assert ctx_mod.extract_traceparent({"x": "y"}) is None


def test_set_request_baggage_empty_returns_context():
    assert ctx_mod.set_request_baggage({}) is not None


def test_get_baggage_attributes_roundtrip():
    ctx = ctx_mod.set_request_baggage({"litellm.team.id": "t1"})
    assert ctx_mod.get_baggage_attributes(ctx)["litellm.team.id"] == "t1"


# --- providers -------------------------------------------------------------- #


def test_to_otel_span_kind_covers_all():
    assert providers.to_otel_span_kind(LiteLLMSpanKind.SERVER) is SpanKind.SERVER
    assert providers.to_otel_span_kind(LiteLLMSpanKind.CLIENT) is SpanKind.CLIENT
    assert providers.to_otel_span_kind(LiteLLMSpanKind.INTERNAL) is SpanKind.INTERNAL
    assert providers.to_otel_span_kind(LiteLLMSpanKind.PRODUCER) is SpanKind.PRODUCER
    assert providers.to_otel_span_kind(LiteLLMSpanKind.CONSUMER) is SpanKind.CONSUMER


def test_parse_headers():
    assert providers.parse_headers(None) == {}
    assert providers.parse_headers("a=1,b=2") == {"a": "1", "b": "2"}
    assert providers.parse_headers("no-equals") == {}


def test_otlp_traces_endpoint_normalization():
    norm = providers._otlp_traces_endpoint
    # A base endpoint gets the signal path appended (the common OTLP env shape).
    assert norm("http://collector:4318") == "http://collector:4318/v1/traces"
    assert norm("http://collector:4318/") == "http://collector:4318/v1/traces"
    # An already-correct path is left intact.
    assert norm("http://collector:4318/v1/traces") == "http://collector:4318/v1/traces"
    # Another signal's path is rewritten to traces.
    assert norm("http://collector:4318/v1/logs") == "http://collector:4318/v1/traces"
    # Splunk's path is preserved; None passes through.
    assert (
        norm("https://x.splunk.com/v2/trace/otlp")
        == "https://x.splunk.com/v2/trace/otlp"
    )
    assert norm(None) is None


def test_build_span_exporter_variants():
    assert isinstance(
        providers.build_span_exporter(OpenTelemetryV2Config(exporter="console")),
        ConsoleSpanExporter,
    )
    assert isinstance(
        providers.build_span_exporter(OpenTelemetryV2Config(exporter="in_memory")),
        InMemorySpanExporter,
    )
    assert isinstance(
        providers.build_span_exporter(OpenTelemetryV2Config(exporter="unknown")),
        ConsoleSpanExporter,
    )
    http_exporter = providers.build_span_exporter(
        OpenTelemetryV2Config(exporter="otlp_http", endpoint="http://h:4318")
    )
    assert "OTLPSpanExporter" in type(http_exporter).__name__
    grpc_exporter = providers.build_span_exporter(
        OpenTelemetryV2Config(exporter="otlp_grpc", endpoint="http://h:4317")
    )
    assert "OTLPSpanExporter" in type(grpc_exporter).__name__


def test_build_resource_includes_deployment_environment():
    resource = providers.build_resource(
        OpenTelemetryV2Config(service_name="svc", deployment_environment="prod")
    )
    assert resource.attributes["service.name"] == "svc"
    assert resource.attributes["deployment.environment"] == "prod"


def test_build_tracer_provider_processor_selection():
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    simple = providers.build_tracer_provider(cfg, exporter=InMemorySpanExporter())
    batch = providers.build_tracer_provider(
        cfg, exporter=ConsoleSpanExporter(), use_simple_processor=False
    )
    # both build without error; assert the requested processor type was used
    simple_procs = simple._active_span_processor._span_processors
    batch_procs = batch._active_span_processor._span_processors
    assert any(isinstance(p, SimpleSpanProcessor) for p in simple_procs)
    assert any(isinstance(p, BatchSpanProcessor) for p in batch_procs)


def test_baggage_processor_lifecycle_noops():
    proc = providers.LiteLLMBaggageSpanProcessor(allowed_keys=["litellm.team.id"])
    # no-op lifecycle hooks must not raise
    assert proc.on_end(None) is None  # type: ignore[arg-type]
    assert proc.shutdown() is None
    assert proc.force_flush() is True


def test_emitter_without_call_id_is_not_deduped():
    from litellm.integrations.otel.emitter import SpanEmitter

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    provider, exporter = providers.in_memory_provider(cfg)
    engine = SpanEmitter(providers.get_tracer(provider, "t"), cfg)
    data = LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model=None,
        response_id=None,
        request_params=LLMRequestParams(),
        usage=LLMUsage(),
        finish_reasons=(),
        error=SpanError(error_type="X", message=None),
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id=None),
    )
    engine.emit(SpanRole.LLM_CALL, data)
    engine.emit(SpanRole.LLM_CALL, data)  # no call_id -> not deduped
    assert len(exporter.get_finished_spans()) == 2


# --- service taxonomy: which calls become spans, and of what kind ----------- #


def test_span_role_for_service_classifies_datastores_internal_and_metrics_only():
    # Outbound datastores -> DB_CALL (CLIENT), with a db.system.
    for name in (
        "redis",
        "postgres",
        "batch_write_to_db",
        "redis_daily_spend_update_queue",
    ):
        assert span_role_for_service(name) is SpanRole.DB_CALL
        assert db_system(name) is not None
    # Genuine internal work worth a span -> SERVICE (INTERNAL).
    assert span_role_for_service("reset_budget_job") is SpanRole.SERVICE
    assert db_system("reset_budget_job") is None
    # Framework instrumentation that duplicates a gen-AI span (or gets a live
    # phase span) -> None: never emitted as a service span.
    for name in ("self", "router", "proxy_pre_call", "auth"):
        assert span_role_for_service(name) is None


# --- event_metadata sanitization -------------------------------------------- #


def test_sanitize_event_metadata_drops_objects_dumps_and_secrets():
    from litellm.integrations.otel.model.payloads import sanitize_event_metadata

    clean = sanitize_event_metadata(
        {
            "table_name": "combined_view",  # safe primitive -> kept
            "count": 3,  # primitive -> kept (stringified)
            "function_kwargs": {"prisma_client": object()},  # denylisted key
            "function_args": (1, 2),  # denylisted key
            "user_api_key_auth": "blob",  # 'auth' substring -> dropped
            "api_key": "sk-secret",  # 'api_key' substring -> dropped
            "set-cookie": "x",  # 'cookie' substring -> dropped
            "hidden_params": "headers...",  # denylisted substring
            "obj": object(),  # non-primitive value -> dropped
            "nested": {"x": 1},  # non-primitive value -> dropped
        }
    )
    assert clean == {"table_name": "combined_view", "count": "3"}


def test_sanitize_event_metadata_caps_value_length_and_handles_none():
    from litellm.integrations.otel.model.payloads import sanitize_event_metadata

    assert sanitize_event_metadata(None) == {}
    big = sanitize_event_metadata({"k": "v" * 5000})
    assert len(big["k"]) == 1024
