"""Golden tests for the OTel v2 engine: span shape, kinds, semconv attributes,
legacy dual-emit, hierarchy, error status, and idempotency. Needs the OTel SDK."""

import json

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.trace import SpanKind  # noqa: E402
from opentelemetry.trace.status import StatusCode  # noqa: E402

from litellm.integrations.otel import (  # noqa: E402
    GenAI,
    LiteLLM,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.plumbing import context as ctx_mod  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.emitter import SpanEmitter  # noqa: E402
from litellm.integrations.otel.emitter import stamp_error  # noqa: E402
from litellm.integrations.otel.mappers.utils import (  # noqa: E402
    MAX_MESSAGE_ATTRS_PER_SPAN,
    MAX_TOOL_DEFINITION_ATTRS_PER_SPAN,
)
from litellm.integrations.otel.model.payloads import (  # noqa: E402
    GuardrailSpanData,
    LLMCallSpanData,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.model.spans import SPAN_REGISTRY, SpanRole  # noqa: E402


def _payload(**overrides):
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "model_parameters": {"temperature": 0.7, "max_tokens": 256, "top_k": 40},
        "response": {
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [{"finish_reason": "stop"}],
        },
        "metadata": {"team_id": "t1", "team_alias": "team one"},
        "api_base": "https://api.openai.com:443/v1",
        "status": "success",
        "litellm_call_id": "call_1",
        "response_cost": 0.002,
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _engine(legacy_compat=True):
    cfg = OpenTelemetryV2Config(exporter="in_memory", legacy_compat=legacy_compat)
    provider, exporter = providers.in_memory_provider(cfg)
    tracer = providers.get_tracer(provider, "litellm-test")
    return SpanEmitter(tracer, cfg), exporter


def test_llm_call_span_cost_breakdown():
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(
        _payload(
            cost_breakdown={
                "input_cost": 0.004,
                "output_cost": 0.006,
                "cache_read_cost": 0.001,
                "total_cost": 0.011,
            }
        )
    )
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    a = span.attributes
    # The rolled-up total stays sourced from response_cost.
    assert a[f"{LiteLLM.COST_PREFIX}total"] == 0.002
    # Per-component breakdown now rides the span.
    assert a[f"{LiteLLM.COST_PREFIX}input"] == 0.004
    assert a[f"{LiteLLM.COST_PREFIX}output"] == 0.006
    assert a[f"{LiteLLM.COST_PREFIX}cache_read"] == 0.001
    # Unreported components are omitted, not zero-filled.
    assert f"{LiteLLM.COST_PREFIX}margin_total_amount" not in a


def test_tracer_scope_carries_litellm_version():
    from litellm._version import version as litellm_version

    cfg = OpenTelemetryV2Config(exporter="in_memory")
    provider, exporter = providers.in_memory_provider(cfg)
    tracer = providers.get_tracer(provider, "litellm-test")
    tracer.start_span("probe").end()
    (span,) = exporter.get_finished_spans()
    assert span.instrumentation_scope.version == litellm_version


def test_llm_call_span_golden():
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(_payload())
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.kind is SpanKind.CLIENT
    a = span.attributes
    assert a[GenAI.OPERATION_NAME] == "chat"
    assert a[GenAI.PROVIDER_NAME] == "openai"
    assert a[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert a[GenAI.RESPONSE_MODEL] == "gpt-4o-2024"
    assert a[GenAI.RESPONSE_ID] == "resp_1"
    assert a[GenAI.USAGE_INPUT_TOKENS] == 10
    assert a[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert a[GenAI.RESPONSE_FINISH_REASONS] == ("stop",)
    assert a[GenAI.REQUEST_TEMPERATURE] == 0.7
    assert a["server.address"] == "api.openai.com"
    assert a[LiteLLM.CALL_ID] == "call_1"
    assert a["litellm.cost.total"] == 0.002
    # Success leaves status UNSET (semconv default), not forced OK.
    assert span.status.status_code is StatusCode.UNSET


def test_legacy_dual_emit_on():
    engine, exporter = _engine(legacy_compat=True)
    engine.emit(
        SpanRole.LLM_CALL, LLMCallSpanData.from_standard_logging_payload(_payload())
    )
    (span,) = exporter.get_finished_spans()
    # canonical AND legacy keys are both present
    assert span.attributes[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert span.attributes["gen_ai.usage.completion_tokens"] == 5
    assert span.attributes["gen_ai.system"] == "openai"


def test_legacy_dual_emit_off():
    engine, exporter = _engine(legacy_compat=False)
    engine.emit(
        SpanRole.LLM_CALL, LLMCallSpanData.from_standard_logging_payload(_payload())
    )
    (span,) = exporter.get_finished_spans()
    # canonical present, legacy absent
    assert span.attributes[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert "gen_ai.usage.completion_tokens" not in span.attributes
    assert "gen_ai.system" not in span.attributes


def test_error_span_sets_status_and_error_type():
    engine, exporter = _engine()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_message": "429"},
    )
    engine.emit(
        SpanRole.LLM_CALL, LLMCallSpanData.from_standard_logging_payload(payload)
    )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RateLimitError"


def test_stamp_error_writes_full_attribute_set_and_event():
    engine, exporter = _engine()
    span = engine.start_span(SpanRole.PROXY_REQUEST, "POST /chat/completions")
    result = stamp_error(
        span, SpanError("ProxyException", "boom", code="401", stack_trace="tb", llm_provider="anthropic")
    )
    span.end()
    (s,) = exporter.get_finished_spans()
    assert result == ("ProxyException", "boom")
    assert s.attributes["error.type"] == "ProxyException"
    assert s.attributes["error.message"] == "boom"
    assert s.attributes["litellm.provider.error.code"] == "401"
    assert s.attributes["litellm.provider.error.stack_trace"] == "tb"
    assert s.attributes["litellm.provider.error.llm_provider"] == "anthropic"
    assert s.status.status_code is StatusCode.ERROR
    assert [e.name for e in s.events] == ["exception"]


def test_stamp_error_opt_outs_skip_status_and_event():
    engine, exporter = _engine()
    span = engine.start_span(SpanRole.PROXY_REQUEST, "POST /chat/completions")
    stamp_error(span, SpanError("ProxyException", "boom", code="401"), record_event=False, set_status=False)
    span.end()
    (s,) = exporter.get_finished_spans()
    assert s.attributes["error.type"] == "ProxyException"
    assert s.attributes["litellm.provider.error.code"] == "401"
    assert s.status.status_code is StatusCode.UNSET
    assert s.events == ()


def test_stamp_error_without_type_or_message_is_noop():
    engine, exporter = _engine()
    span = engine.start_span(SpanRole.PROXY_REQUEST, "POST /chat/completions")
    assert stamp_error(span, SpanError()) is None
    span.end()
    (s,) = exporter.get_finished_spans()
    assert "error.type" not in s.attributes
    assert s.status.status_code is StatusCode.UNSET


def test_hierarchy_and_kinds_match_registry():
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(_payload())
    root = engine.start_span(SpanRole.PROXY_REQUEST, "POST /chat/completions")
    root_ctx = ctx_mod.context_from_span(root)
    engine.emit(SpanRole.LLM_CALL, data, parent_context=root_ctx)
    engine.emit(
        SpanRole.GUARDRAIL, GuardrailSpanData("presidio", status="success"), root_ctx
    )
    # An outbound datastore call (DB_CALL) and an internal service call differ in
    # span kind; both are named "{service} {call_type}".
    engine.emit(SpanRole.DB_CALL, ServiceSpanData("redis", call_type="set"), root_ctx)
    engine.emit(
        SpanRole.SERVICE, ServiceSpanData("router", call_type="acompletion"), root_ctx
    )
    root.end()

    by_name = {s.name: s for s in exporter.get_finished_spans()}
    root_id = root.get_span_context().span_id
    assert by_name["chat gpt-4o"].parent.span_id == root_id
    assert by_name["execute_guardrail presidio"].parent.span_id == root_id
    assert by_name["redis set"].parent.span_id == root_id
    assert by_name["router acompletion"].parent.span_id == root_id
    # kinds come straight from the registry
    assert by_name["chat gpt-4o"].kind is SpanKind.CLIENT
    assert by_name["execute_guardrail presidio"].kind is SpanKind.INTERNAL
    assert by_name["redis set"].kind is SpanKind.CLIENT
    assert by_name["router acompletion"].kind is SpanKind.INTERNAL
    assert by_name["POST /chat/completions"].kind is SpanKind.SERVER


def test_idempotent_dual_fire():
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(_payload())
    first = engine.emit(SpanRole.LLM_CALL, data)
    second = engine.emit(SpanRole.LLM_CALL, data)  # same call_id -> deduped
    assert first is not None
    assert second is None
    assert len(exporter.get_finished_spans()) == 1


def test_dedup_cache_is_bounded(monkeypatch):
    """The dedup cache only needs to coalesce one request's sync+async fire, so
    it is a bounded LRU — every unique call_id must not accumulate forever on a
    long-running proxy."""
    from litellm.integrations.otel import emitter as emitter_mod

    monkeypatch.setattr(emitter_mod, "_DEDUP_CACHE_MAX", 3)
    engine, _ = _engine()
    for i in range(10):
        engine.emit(
            SpanRole.LLM_CALL,
            LLMCallSpanData.from_standard_logging_payload(
                _payload(litellm_call_id=f"call_{i}")
            ),
        )
    assert len(engine._emitted) <= 3


def test_service_error_span():
    from litellm.integrations.otel.model.payloads import SpanError

    engine, exporter = _engine()
    engine.emit(
        SpanRole.SERVICE,
        ServiceSpanData(
            "postgres", call_type="query", error=SpanError("DBError", "boom")
        ),
    )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "DBError"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "postgres"


def test_guardrail_block_span_is_error_and_carries_verdict():
    engine, exporter = _engine()
    data = GuardrailSpanData.from_logging_entry(
        {
            "guardrail_name": "openai-moderation",
            "guardrail_mode": "pre_call",
            "guardrail_status": "guardrail_intervened",
            "guardrail_provider": "openai",
            "guardrail_response": {"violated_categories": ["violence"]},
            "masked_entity_count": {"EMAIL": 2},
        }
    )
    engine.emit(SpanRole.GUARDRAIL, data)
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR  # intervention → ERROR
    a = span.attributes
    assert a[LiteLLM.GUARDRAIL_STATUS] == "guardrail_intervened"
    assert a[LiteLLM.GUARDRAIL_PROVIDER] == "openai"
    assert "violence" in a[LiteLLM.GUARDRAIL_RESPONSE]  # the verdict rides the span
    assert a[LiteLLM.GUARDRAIL_MASKED_ENTITY_COUNT] == 2


def test_guardrail_success_span_is_unset():
    """On success the status is left UNSET (semconv default) — not forced OK."""
    engine, exporter = _engine()
    engine.emit(
        SpanRole.GUARDRAIL,
        GuardrailSpanData.from_logging_entry(
            {"guardrail_name": "g", "guardrail_status": "success"}
        ),
    )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET


def _tools_payload(count):
    """A request declaring ``count`` tools, in the chat-completion shape."""
    return _payload(
        model_parameters={
            "temperature": 0.7,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"description for tool {i}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for i in range(count)
            ],
        }
    )


def test_many_tools_do_not_evict_core_attributes():
    """Tool definitions must never crowd core telemetry off the span.

    An agentic client declares hundreds of tools. Spelling each one out as
    per-index attributes overruns the OTel SDK's 128-attribute span limit,
    which evicts oldest-first and so destroys the ``gen_ai.*`` attributes
    written before it. Capping the tool family keeps the core intact.
    """
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(_tools_payload(127))
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    a = span.attributes

    assert a[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert a[GenAI.PROVIDER_NAME] == "openai"
    assert a[GenAI.USAGE_INPUT_TOKENS] == 10
    assert a[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert a[GenAI.RESPONSE_FINISH_REASONS] == ("stop",)
    assert a[f"{LiteLLM.COST_PREFIX}total"] == 0.002
    assert a["gen_ai.usage.prompt_tokens"] == 10

    assert span.dropped_attributes == 0
    assert a[LiteLLM.TOOLS_DECLARED] == 127
    assert a["gen_ai.tool.0.name"] == "tool_0"
    assert "gen_ai.tool.126.name" not in a
    assert "llm.request.functions.126.name" not in a


def test_tool_definitions_kept_in_full_below_the_cap():
    """A handful of tools keeps full per-index detail in both vocabularies."""
    engine, exporter = _engine()
    data = LLMCallSpanData.from_standard_logging_payload(_tools_payload(3))
    engine.emit(SpanRole.LLM_CALL, data)
    (span,) = exporter.get_finished_spans()
    a = span.attributes

    assert a[LiteLLM.TOOLS_DECLARED] == 3
    for idx in range(3):
        assert a[f"gen_ai.tool.{idx}.name"] == f"tool_{idx}"
        assert a[f"gen_ai.tool.{idx}.description"] == f"description for tool {idx}"
        assert a[f"gen_ai.tool.{idx}.parameters"]
        assert a[f"llm.request.functions.{idx}.name"] == f"tool_{idx}"


def _tool_span(mapper_names, tool_count):
    """The exported LLM-call span for ``mapper_names`` and ``tool_count`` tools."""
    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        legacy_compat=True,
        mapper_names=list(mapper_names),
    )
    provider, exporter = providers.in_memory_provider(cfg)
    engine = SpanEmitter(providers.get_tracer(provider, "litellm-test"), cfg)
    engine.emit(
        SpanRole.LLM_CALL,
        LLMCallSpanData.from_standard_logging_payload(_tools_payload(tool_count)),
    )
    (span,) = exporter.get_finished_spans()
    return span


def _tool_definition_keys(attributes):
    return [
        key
        for key in attributes
        if key.startswith(("gen_ai.tool.", "llm.request.functions.", "llm.tools."))
    ]


@pytest.mark.parametrize(
    "mapper_names",
    [
        ["genai"],
        ["genai", "openinference"],
        ["genai", "openinference", "langfuse", "weave", "langtrace"],
    ],
)
def test_tool_definitions_stay_within_one_span_wide_budget(mapper_names):
    """Every supported composition has to leave core telemetry on the span.

    Each vocabulary spells the same tools out under its own keys, so an
    allowance handed to each mapper separately multiplies by the number of
    configured vocabularies and reaches the attribute limit again. Arize and
    Phoenix already layer OpenInference on top of the default two, and every
    vendor vocabulary can be listed at once. One budget shared across them all
    is what keeps the total bounded.
    """
    span = _tool_span(mapper_names, 127)
    a = span.attributes

    assert span.dropped_attributes == 0
    assert a[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert a[GenAI.PROVIDER_NAME] == "openai"
    assert a[GenAI.USAGE_INPUT_TOKENS] == 10
    assert a[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert a[f"{LiteLLM.COST_PREFIX}total"] == 0.002
    assert a[LiteLLM.TOOLS_DECLARED] == 127

    emitted = _tool_definition_keys(a)
    assert emitted, "some tool detail should survive in every composition"
    assert len(emitted) <= MAX_TOOL_DEFINITION_ATTRS_PER_SPAN


def test_vendor_tool_definitions_are_truncated_not_dropped():
    """The OpenInference vocabulary keeps its leading tools and loses the tail."""
    a = _tool_span(["genai", "openinference"], 127).attributes
    assert a["llm.tools.0.tool.name"] == "tool_0"
    assert a["llm.tools.0.tool.json_schema"]
    assert "llm.tools.126.tool.name" not in a


def _conversation_payload(turns, choices=1, **overrides):
    """A ``turns``-message chat with ``choices`` response choices, content-bearing."""
    return _payload(
        messages=[{"role": ("user", "assistant")[i % 2], "content": f"turn {i}"} for i in range(turns)],
        response={
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": f"reply {i}"}}
                for i in range(choices)
            ],
        },
        **overrides,
    )


def _conversation_span(mapper_names, payload, legacy_compat=False):
    """The exported LLM-call span for ``payload`` with content capture on."""
    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        legacy_compat=legacy_compat,
        mapper_names=list(mapper_names),
        capture_message_content="span_only",
    )
    provider, exporter = providers.in_memory_provider(cfg)
    engine = SpanEmitter(providers.get_tracer(provider, "litellm-test"), cfg)
    engine.emit(
        SpanRole.LLM_CALL,
        LLMCallSpanData.from_standard_logging_payload(payload, capture_content=True),
    )
    (span,) = exporter.get_finished_spans()
    return span


def _indexed_message_count(attributes, prefix):
    return len({key.split(".")[2] for key in attributes if key.startswith(f"{prefix}.")})


@pytest.mark.parametrize("turns", [60, 200])
def test_long_conversation_does_not_evict_core_attributes(turns):
    """Per-message OpenInference attributes must never crowd core telemetry off the span."""
    span = _conversation_span(["genai", "openinference"], _conversation_payload(turns))
    a = span.attributes

    assert span.dropped_attributes == 0
    assert a[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert a[GenAI.PROVIDER_NAME] == "openai"
    assert a[GenAI.USAGE_INPUT_TOKENS] == 10
    assert a[GenAI.USAGE_OUTPUT_TOKENS] == 5
    assert a[GenAI.RESPONSE_FINISH_REASONS] == ("stop",)
    assert a[f"{LiteLLM.COST_PREFIX}total"] == 0.002

    assert a["llm.input_messages.0.message.content"] == "turn 0"
    assert a["llm.output_messages.0.message.content"] == "reply 0"
    assert a[f"llm.input_messages.{turns - 1}.message.content"] == f"turn {turns - 1}"
    assert f"llm.input_messages.{turns // 2}.message.role" not in a
    assert len(json.loads(a["input.value"])) == turns
    assert len(json.loads(a["output.value"])) == 1
    assert len(json.loads(a[GenAI.INPUT_MESSAGES])) == turns


def test_short_conversation_keeps_every_message_indexed():
    """Below the cap nothing is truncated in either direction."""
    a = _conversation_span(["genai", "openinference"], _conversation_payload(4, choices=2)).attributes
    for idx in range(4):
        assert a[f"llm.input_messages.{idx}.message.content"] == f"turn {idx}"
    for idx in range(2):
        assert a[f"llm.output_messages.{idx}.message.content"] == f"reply {idx}"


def test_indexed_prompt_keeps_opener_and_latest_turns_under_a_value_length_limit(monkeypatch):
    """The system prompt and the live turn keep their own keys once the SDK clips ``input.value``."""
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "256")
    payload = _conversation_payload(60)
    payload["messages"][0] = {"role": "system", "content": "be terse"}
    payload["messages"][-1] = {"role": "user", "content": "LATEST-TURN"}
    a = _conversation_span(["genai", "openinference"], payload).attributes

    assert len(a["input.value"]) == 256
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.0.message.content"] == "be terse"
    assert a["llm.input_messages.59.message.role"] == "user"
    assert a["llm.input_messages.59.message.content"] == "LATEST-TURN"
    assert a["llm.output_messages.0.message.content"] == "reply 0"
    assert [int(key.split(".")[2]) for key in a if key.endswith("message.content") and key.startswith("llm.input_")] == [
        0,
        *range(54, 60),
    ]


def test_message_cap_is_shared_across_input_and_output():
    """One span-wide allowance covers both directions, and the response always keeps a share."""
    long_prompt = _conversation_span(["genai", "openinference"], _conversation_payload(60, choices=1)).attributes
    many_choices = _conversation_span(["genai", "openinference"], _conversation_payload(60, choices=20)).attributes

    single_reply_indexed = _indexed_message_count(long_prompt, "llm.output_messages")
    assert single_reply_indexed == 1
    assert _indexed_message_count(long_prompt, "llm.input_messages") + single_reply_indexed == (
        MAX_MESSAGE_ATTRS_PER_SPAN // 2
    )

    assert _indexed_message_count(many_choices, "llm.input_messages") > 0
    assert _indexed_message_count(many_choices, "llm.output_messages") > single_reply_indexed
    assert _indexed_message_count(many_choices, "llm.input_messages") + _indexed_message_count(
        many_choices, "llm.output_messages"
    ) == (MAX_MESSAGE_ATTRS_PER_SPAN // 2)


def test_fully_populated_span_with_every_vocabulary_stays_within_the_attribute_limit():
    """Every capped family maxed at once still leaves the whole core intact."""
    payload = _conversation_payload(
        200,
        choices=20,
        stream=True,
        model_parameters={
            **_tools_payload(127)["model_parameters"],
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "seed": 7,
            "stop": ["\n"],
        },
        cost_breakdown={
            key: 0.001
            for key in (
                "input_cost",
                "output_cost",
                "cache_read_cost",
                "cache_creation_cost",
                "tool_usage_cost",
                "original_cost",
                "discount_amount",
                "discount_percent",
                "margin_fixed_amount",
                "margin_percent",
                "margin_total_amount",
                "total_cost",
            )
        },
    )
    span = _conversation_span(["genai", "openinference", "langfuse", "weave", "langtrace"], payload, legacy_compat=True)
    a = span.attributes

    assert span.dropped_attributes == 0
    assert a[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert a[f"{LiteLLM.COST_PREFIX}total"] == 0.002
    assert a[LiteLLM.TOOLS_DECLARED] == 127
    assert a["llm.input_messages.0.message.content"] == "turn 0"
    assert a["llm.input_messages.199.message.content"] == "turn 199"
    assert a["llm.output_messages.0.message.content"] == "reply 0"
