"""Coverage for the engine-layer components: providers/exporters, context +
baggage helpers, metrics, the typed coercion helpers, mapper branches, span-name
builders, and the registry validator's failure paths. Needs the OTel SDK."""

import json

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
    LLMCost,
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


def test_genai_mapper_stamps_input_output_messages():
    data = LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model="gpt-4o-2024",
        response_id="resp_1",
        request_params=LLMRequestParams(),
        usage=LLMUsage(),
        finish_reasons=("stop",),
        error=None,
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id="c1"),
        messages_in=(
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What's the weather?"},
        ),
        choices_out=(
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Sunny."},
            },
        ),
    )
    attrs = GenAIMapper().map(data)
    assert json.loads(attrs[GenAI.INPUT_MESSAGES]) == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What's the weather?"},
    ]
    assert json.loads(attrs[GenAI.OUTPUT_MESSAGES]) == [
        {"role": "assistant", "content": "Sunny."}
    ]


def test_genai_mapper_omits_messages_when_content_not_captured():
    attrs = GenAIMapper().map(_full_llm_call())
    assert GenAI.INPUT_MESSAGES not in attrs
    assert GenAI.OUTPUT_MESSAGES not in attrs


def test_genai_mapper_cost_breakdown():
    from litellm.integrations.otel.model.semconv import LiteLLM

    data = LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="anthropic",
        request_model="claude-sonnet-4-6",
        response_model=None,
        response_id=None,
        request_params=LLMRequestParams(),
        usage=LLMUsage(),
        finish_reasons=(),
        error=None,
        response_cost=0.012,
        server=None,
        identity=RequestIdentity(call_id=None),
        cost=LLMCost(
            input=0.004,
            output=0.006,
            cache_read=0.001,
            cache_creation=0.0,
            tool_usage=0.0005,
            original=0.013,
            discount_amount=0.001,
            discount_percent=0.077,
            margin_total_amount=0.0,
            # margin_fixed_amount / margin_percent left unset on purpose
        ),
    )
    attrs = GenAIMapper().map(data)
    assert attrs[f"{LiteLLM.COST_PREFIX}total"] == 0.012
    assert attrs[f"{LiteLLM.COST_PREFIX}input"] == 0.004
    assert attrs[f"{LiteLLM.COST_PREFIX}output"] == 0.006
    assert attrs[f"{LiteLLM.COST_PREFIX}cache_read"] == 0.001
    assert attrs[f"{LiteLLM.COST_PREFIX}cache_creation"] == 0.0
    assert attrs[f"{LiteLLM.COST_PREFIX}tool_usage"] == 0.0005
    assert attrs[f"{LiteLLM.COST_PREFIX}original"] == 0.013
    assert attrs[f"{LiteLLM.COST_PREFIX}discount_amount"] == 0.001
    assert attrs[f"{LiteLLM.COST_PREFIX}discount_percent"] == 0.077
    assert attrs[f"{LiteLLM.COST_PREFIX}margin_total_amount"] == 0.0
    # Components the source did not report are omitted, not zero-filled.
    assert f"{LiteLLM.COST_PREFIX}margin_fixed_amount" not in attrs
    assert f"{LiteLLM.COST_PREFIX}margin_percent" not in attrs


def test_genai_mapper_cost_breakdown_absent():
    # No cost_breakdown → only the rolled-up total (from response_cost) emits.
    from litellm.integrations.otel.model.semconv import LiteLLM

    attrs = GenAIMapper().map(_full_llm_call())
    assert attrs[f"{LiteLLM.COST_PREFIX}total"] == 0.002
    assert not any(
        k.startswith(LiteLLM.COST_PREFIX) and k != f"{LiteLLM.COST_PREFIX}total"
        for k in attrs
    )


def test_llm_cost_from_breakdown_maps_costbreakdown_keys():
    cost = LLMCost.from_breakdown(
        {
            "input_cost": 0.004,
            "output_cost": 0.006,
            "cache_read_cost": 0.001,
            "cache_creation_cost": 0.002,
            "tool_usage_cost": 0.0005,
            "original_cost": 0.013,
            "discount_amount": 0.001,
            "discount_percent": 0.077,
            "margin_fixed_amount": 0.0,
            "margin_percent": 0.1,
            "margin_total_amount": 0.0011,
            "total_cost": 0.012,  # carried on response_cost, not LLMCost
        }
    )
    assert cost.input == 0.004
    assert cost.output == 0.006
    assert cost.cache_read == 0.001
    assert cost.cache_creation == 0.002
    assert cost.tool_usage == 0.0005
    assert cost.original == 0.013
    assert cost.discount_amount == 0.001
    assert cost.discount_percent == 0.077
    assert cost.margin_fixed_amount == 0.0
    assert cost.margin_percent == 0.1
    assert cost.margin_total_amount == 0.0011


def test_llm_cost_from_breakdown_none_is_empty():
    assert LLMCost.from_breakdown(None) == LLMCost()


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


def test_otlp_logs_endpoint_normalization():
    norm = providers._otlp_logs_endpoint
    # A base endpoint gets the signal path appended (the common OTLP env shape).
    assert norm("http://collector:4318") == "http://collector:4318/v1/logs"
    assert norm("http://collector:4318/") == "http://collector:4318/v1/logs"
    # An already-correct path is left intact.
    assert norm("http://collector:4318/v1/logs") == "http://collector:4318/v1/logs"
    # A sibling signal's path is rewritten to logs, so one OTEL_ENDPOINT works
    # for every signal rather than POSTing events at the traces path.
    assert norm("http://collector:4318/v1/traces") == "http://collector:4318/v1/logs"
    assert norm("http://collector:4318/v1/metrics") == "http://collector:4318/v1/logs"
    assert norm(None) is None


def test_build_log_exporter_variants():
    from opentelemetry.sdk._logs.export import ConsoleLogExporter, InMemoryLogExporter

    assert isinstance(
        providers.build_log_exporter(OpenTelemetryV2Config(exporter="console")),
        ConsoleLogExporter,
    )
    assert isinstance(
        providers.build_log_exporter(OpenTelemetryV2Config(exporter="in_memory")),
        InMemoryLogExporter,
    )
    # An unrecognized kind falls back to console rather than dropping events.
    assert isinstance(
        providers.build_log_exporter(OpenTelemetryV2Config(exporter="unknown")),
        ConsoleLogExporter,
    )
    http_exporter = providers.build_log_exporter(
        OpenTelemetryV2Config(exporter="otlp_http", endpoint="http://h:4318")
    )
    assert "OTLPLogExporter" in type(http_exporter).__name__


def test_build_logger_provider_picks_processor_by_exporter_kind():
    """Console and in-memory exporters export synchronously (tests depend on it);
    every other destination gets the batch processor."""
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
        InMemoryLogExporter,
        SimpleLogRecordProcessor,
    )

    cfg = OpenTelemetryV2Config(exporter="in_memory")

    def processor_of(provider):
        return provider._multi_log_record_processor._log_record_processors[0]

    assert isinstance(
        processor_of(providers.build_logger_provider(cfg, log_exporter=InMemoryLogExporter())),
        SimpleLogRecordProcessor,
    )
    assert isinstance(
        processor_of(providers.build_logger_provider(cfg, log_exporter=ConsoleLogExporter())),
        SimpleLogRecordProcessor,
    )
    http_exporter = providers.build_log_exporter(
        OpenTelemetryV2Config(exporter="otlp_http", endpoint="http://h:4318")
    )
    assert isinstance(
        processor_of(providers.build_logger_provider(cfg, log_exporter=http_exporter)),
        BatchLogRecordProcessor,
    )
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


def _emit_error_span(message, error_type="litellm.APIError"):
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
        error=SpanError(error_type=error_type, message=message),
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id=None),
    )
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    return span


def _exception_event(span):
    from litellm.integrations.otel.model.semconv import ExceptionEvent

    events = [e for e in span.events if e.name == ExceptionEvent.NAME]
    assert len(events) == 1, "expected exactly one exception event"
    return events[0]


def test_error_message_recorded_as_full_exception_event_untruncated():
    """The ``exception`` event carries the full untruncated message under
    ``exception.message`` so backends that dynamic-map unknown string span
    attrs to ``keyword`` (e.g. Elasticsearch with a 1024-char ``ignore_above``)
    still see it in full via the semconv-recognized event field."""
    from litellm.integrations.otel.model.semconv import Error, ExceptionEvent

    long_message = "boom: " + "x" * 5000
    span = _emit_error_span(long_message, error_type="litellm.APIError")

    event = _exception_event(span)
    assert event.attributes[ExceptionEvent.MESSAGE] == long_message
    assert len(event.attributes[ExceptionEvent.MESSAGE]) == len(long_message) > 1024
    assert event.attributes[ExceptionEvent.TYPE] == "litellm.APIError"

    # error.type stays a low-cardinality attribute; the exception EVENT field
    # ``exception.message`` never becomes a bare string attribute.
    assert span.attributes[Error.TYPE] == "litellm.APIError"
    assert ExceptionEvent.MESSAGE not in span.attributes
    assert span.status.description == long_message


def test_error_details_stamped_as_span_attributes_for_labels_ingest():
    """OTel-defined keys and litellm-specific detail keys both ride span
    attributes so backends that flatten attrs into label indexes (Elastic APM
    ``labels.*``, Datadog span tags) render them. The exception event with the
    full untruncated message stays alongside — both places, matching v1's
    shape."""
    from litellm.integrations.otel.model.semconv import Error, ExceptionEvent, LiteLLMError
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
        error=SpanError(
            error_type="litellm.BadRequestError",
            message="400: violated moderation policy",
            code="400",
            stack_trace="File proxy_server.py line 8570 ...",
            llm_provider="openai",
        ),
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id=None),
    )
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()

    # OTel-defined keys (from the ``error.*`` semconv registry).
    assert span.attributes[Error.TYPE] == "litellm.BadRequestError"
    assert span.attributes[Error.MESSAGE] == "400: violated moderation policy"
    # LiteLLM-specific detail keys — vendor-namespaced under ``error.*``
    # for v1-parity, not defined by OTel semconv.
    assert span.attributes[LiteLLMError.CODE] == "400"
    assert span.attributes[LiteLLMError.STACK_TRACE] == "File proxy_server.py line 8570 ..."
    assert span.attributes[LiteLLMError.LLM_PROVIDER] == "openai"

    # The exception event carries the same message on the span too.
    event = _exception_event(span)
    assert event.attributes[ExceptionEvent.MESSAGE] == "400: violated moderation policy"


def test_error_details_omitted_when_span_error_carries_only_message():
    """A guardrail-shape error (message only, no code/traceback/provider) must
    not pollute the span with empty-string detail attributes. Only the keys
    that carry real data land."""
    from litellm.integrations.otel.model.semconv import Error, LiteLLMError

    span = _emit_error_span("guardrail rejected", error_type="ContentFilter")

    assert span.attributes[Error.TYPE] == "ContentFilter"
    assert span.attributes[Error.MESSAGE] == "guardrail rejected"
    # LiteLLM-specific detail keys aren't stamped when the SpanError doesn't
    # carry them.
    assert LiteLLMError.CODE not in span.attributes
    assert LiteLLMError.STACK_TRACE not in span.attributes
    assert LiteLLMError.LLM_PROVIDER not in span.attributes


def test_v2_error_attribute_keys_match_v1_error_attributes_byte_for_byte():
    """v1 (``opentelemetry.py``) and v2 (``otel/`` package) stamp identical
    span-attribute keys so consumers reading ``labels.error_message`` don't
    care which integration produced the span. Renaming either side is a
    breaking change for downstream dashboards; this test locks the vocabulary."""
    from litellm.integrations._types.open_inference import ErrorAttributes
    from litellm.integrations.otel.model.semconv import Error, LiteLLMError

    assert Error.TYPE == ErrorAttributes.ERROR_TYPE
    assert Error.MESSAGE == ErrorAttributes.ERROR_MESSAGE
    assert LiteLLMError.CODE == ErrorAttributes.ERROR_CODE
    assert LiteLLMError.STACK_TRACE == ErrorAttributes.ERROR_STACK_TRACE
    assert LiteLLMError.LLM_PROVIDER == ErrorAttributes.ERROR_LLM_PROVIDER


def test_error_message_falls_back_to_error_type_when_message_absent():
    """A ``SpanError(error_type=..., message=None)`` still renders on the span:
    the resolved message is the error_type, and it lands on ``error.message``,
    the exception event, and the span-status description in lockstep so a
    single-source-of-truth view isn't inconsistent."""
    from litellm.integrations.otel.model.semconv import Error, ExceptionEvent

    span = _emit_error_span(message=None, error_type="RateLimitError")

    assert span.attributes[Error.MESSAGE] == "RateLimitError"
    assert _exception_event(span).attributes[ExceptionEvent.MESSAGE] == "RateLimitError"
    assert span.status.description == "RateLimitError"


def test_success_span_records_no_exception_event():
    from litellm.integrations.otel.emitter import SpanEmitter
    from litellm.integrations.otel.model.semconv import ExceptionEvent

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    provider, exporter = providers.in_memory_provider(cfg)
    engine = SpanEmitter(providers.get_tracer(provider, "t"), cfg)
    data = LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model="gpt-4o",
        response_id="resp-1",
        request_params=LLMRequestParams(),
        usage=LLMUsage(),
        finish_reasons=("stop",),
        error=None,
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id=None),
    )
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    assert all(e.name != ExceptionEvent.NAME for e in span.events)


def _engine_with_event_recorder():
    from opentelemetry.sdk._logs.export import InMemoryLogExporter

    from litellm.integrations.otel.emitter import SpanEmitter
    from litellm.integrations.otel.plumbing.events import GenAIEventRecorder

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_events=True)
    provider, span_exporter = providers.in_memory_provider(cfg)
    log_exporter = InMemoryLogExporter()
    logger_provider = providers.build_logger_provider(cfg, log_exporter=log_exporter)
    recorder = GenAIEventRecorder(providers.get_event_logger(logger_provider))
    engine = SpanEmitter(providers.get_tracer(provider, "t"), cfg, event_recorder=recorder)
    return engine, span_exporter, log_exporter


def _llm_call_data(error):
    return LLMCallSpanData(
        operation=GenAIOperation.CHAT,
        provider="openai",
        request_model="gpt-4o",
        response_model=None,
        response_id=None,
        request_params=LLMRequestParams(),
        usage=LLMUsage(),
        finish_reasons=(),
        error=error,
        response_cost=None,
        server=None,
        identity=RequestIdentity(call_id=None),
    )


def test_operation_exception_log_event_emitted_on_failed_llm_call():
    """A failed LLM call records the GenAI semconv ``gen_ai.client.operation.exception``
    event on the logs signal: severity WARN, the full ``exception.*`` trio (including
    the stacktrace, which span-side only exists under a vendor key), correlated to
    the failed span via trace/span ids. The span-side error surface stays intact."""
    from opentelemetry._logs.severity import SeverityNumber

    from litellm.integrations.otel.model.semconv import ExceptionEvent, GenAIEvent

    engine, span_exporter, log_exporter = _engine_with_event_recorder()
    engine.emit(
        SpanRole.LLM_CALL,
        _llm_call_data(
            SpanError(
                error_type="RateLimitError",
                message="rate limited",
                code="429",
                stack_trace="Traceback (most recent call last) ...",
                llm_provider="openai",
            )
        ),
    )
    (span,) = span_exporter.get_finished_spans()
    (log,) = log_exporter.get_finished_logs()
    record = log.log_record

    assert record.attributes["event.name"] == GenAIEvent.OPERATION_EXCEPTION
    assert record.severity_number == SeverityNumber.WARN
    assert record.attributes[ExceptionEvent.TYPE] == "RateLimitError"
    assert record.attributes[ExceptionEvent.MESSAGE] == "rate limited"
    assert record.attributes[ExceptionEvent.STACKTRACE] == "Traceback (most recent call last) ..."
    assert record.trace_id == span.context.trace_id
    assert record.span_id == span.context.span_id

    assert [e.name for e in span.events] == [ExceptionEvent.NAME]
    assert span.attributes["error.type"] == "RateLimitError"


def test_operation_exception_log_event_omits_absent_stacktrace():
    from litellm.integrations.otel.model.semconv import ExceptionEvent

    engine, _, log_exporter = _engine_with_event_recorder()
    engine.emit(SpanRole.LLM_CALL, _llm_call_data(SpanError(error_type="APIError", message="boom")))
    (log,) = log_exporter.get_finished_logs()

    assert ExceptionEvent.STACKTRACE not in log.log_record.attributes
    assert log.log_record.attributes[ExceptionEvent.MESSAGE] == "boom"


def test_operation_exception_log_event_always_carries_required_pair():
    """``exception.type`` and ``exception.message`` are the semconv-required pair:
    they ride the event even when the recorder is handed empty strings, so an
    event is never emitted with no required field. Only the stacktrace is
    conditional."""
    from opentelemetry.sdk._logs.export import InMemoryLogExporter
    from opentelemetry.trace import INVALID_SPAN_CONTEXT

    from litellm.integrations.otel.model.semconv import ExceptionEvent
    from litellm.integrations.otel.plumbing.events import GenAIEventRecorder

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_events=True)
    log_exporter = InMemoryLogExporter()
    logger_provider = providers.build_logger_provider(cfg, log_exporter=log_exporter)
    recorder = GenAIEventRecorder(providers.get_event_logger(logger_provider))

    recorder.record_operation_exception(
        span_context=INVALID_SPAN_CONTEXT,
        error_type="",
        message="",
        stack_trace="",
        timestamp_ns=None,
    )
    (log,) = log_exporter.get_finished_logs()
    attributes = log.log_record.attributes
    assert attributes[ExceptionEvent.TYPE] == ""
    assert attributes[ExceptionEvent.MESSAGE] == ""
    assert ExceptionEvent.STACKTRACE not in attributes


def test_operation_exception_log_event_not_emitted_on_success():
    engine, span_exporter, log_exporter = _engine_with_event_recorder()
    engine.emit(SpanRole.LLM_CALL, _llm_call_data(None))

    assert len(span_exporter.get_finished_spans()) == 1
    assert log_exporter.get_finished_logs() == ()


def test_operation_exception_log_event_only_for_llm_call_role():
    """The event is scoped to GenAI client operations; a failed guardrail span
    keeps its span-side error surface but records no GenAI exception event."""
    engine, span_exporter, log_exporter = _engine_with_event_recorder()
    engine.emit(
        SpanRole.GUARDRAIL,
        GuardrailSpanData("presidio", status="failure", error=SpanError(error_type="X", message="denied")),
    )
    (span,) = span_exporter.get_finished_spans()

    assert span.attributes["error.type"] == "X"
    assert log_exporter.get_finished_logs() == ()


def test_resolve_logger_provider_honors_explicit_noop_optout(monkeypatch):
    """A ``NoOpLoggerProvider`` global is an explicit operator opt-out from the logs
    signal: resolve to ``None`` so no recorder (and so no event) is ever built,
    rather than emitting into a provider that drops everything."""
    from opentelemetry import _logs
    from opentelemetry._logs import NoOpLoggerProvider

    from litellm.integrations.otel.logger import OpenTelemetryV2

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_events=True)
    tracer_provider, _ = providers.in_memory_provider(cfg)
    monkeypatch.setattr(_logs, "get_logger_provider", lambda: NoOpLoggerProvider())

    assert providers.resolve_logger_provider(cfg) is None
    logger = OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider)
    assert logger._emitter._event_recorder is None


def test_resolve_logger_provider_reuses_operator_sdk_global(monkeypatch):
    """Events ride an operator-configured logs pipeline rather than a second one
    built by litellm, so they land wherever the operator's other logs land."""
    from opentelemetry import _logs
    from opentelemetry.sdk._logs.export import InMemoryLogExporter

    cfg = OpenTelemetryV2Config(exporter="in_memory", enable_events=True)
    operator_provider = providers.build_logger_provider(cfg, log_exporter=InMemoryLogExporter())
    monkeypatch.setattr(_logs, "get_logger_provider", lambda: operator_provider)

    assert providers.resolve_logger_provider(cfg) is operator_provider


def test_operation_exception_event_keys_are_pinned():
    from litellm.integrations.otel.model.semconv import ExceptionEvent, GenAIEvent

    assert GenAIEvent.OPERATION_EXCEPTION == "gen_ai.client.operation.exception"
    assert ExceptionEvent.STACKTRACE == "exception.stacktrace"


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
