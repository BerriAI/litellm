"""Canonical OpenTelemetry GenAI semantic-convention mapper (always active).

Owns the attribute schema for every span kind the engine emits — LLM call,
guardrail, and service — so the engine itself never references attribute keys.

Each span kind declares its schema as a flat ``attribute key -> extractor``
table: one lambda per mapping operation, applied against the typed span data.
"""

from typing import Callable

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    collect,
    drop_none,
    output_messages,
    serialize_messages,
)
from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    MCPListToolsSpanData,
    MCPToolCallSpanData,
    ServiceSpanData,
    ToolDefinition,
)
from litellm.integrations.otel.model.semconv import (
    DB,
    MCP,
    Error,
    GenAI,
    LiteLLM,
    Server,
)
from litellm.integrations.otel.model.spans import db_system


class GenAIMapper:
    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        GenAI.OPERATION_NAME: lambda d: d.operation.value,
        GenAI.PROVIDER_NAME: lambda d: d.provider or None,
        GenAI.REQUEST_MODEL: lambda d: d.request_model or None,
        GenAI.REQUEST_TEMPERATURE: lambda d: d.request_params.temperature,
        GenAI.REQUEST_TOP_P: lambda d: d.request_params.top_p,
        GenAI.REQUEST_TOP_K: lambda d: d.request_params.top_k,
        GenAI.REQUEST_MAX_TOKENS: lambda d: d.request_params.max_tokens,
        GenAI.REQUEST_FREQUENCY_PENALTY: lambda d: d.request_params.frequency_penalty,
        GenAI.REQUEST_PRESENCE_PENALTY: lambda d: d.request_params.presence_penalty,
        GenAI.REQUEST_STOP_SEQUENCES: lambda d: (
            list(d.request_params.stop_sequences) if d.request_params.stop_sequences else None
        ),
        GenAI.REQUEST_SEED: lambda d: d.request_params.seed,
        GenAI.INPUT_MESSAGES: lambda d: serialize_messages(d.messages_in),
        GenAI.OUTPUT_MESSAGES: lambda d: serialize_messages(output_messages(d)),
        GenAI.RESPONSE_MODEL: lambda d: d.response_model,
        GenAI.RESPONSE_ID: lambda d: d.response_id,
        GenAI.RESPONSE_FINISH_REASONS: lambda d: list(d.finish_reasons) if d.finish_reasons else None,
        GenAI.USAGE_INPUT_TOKENS: lambda d: d.usage.input_tokens,
        GenAI.USAGE_OUTPUT_TOKENS: lambda d: d.usage.output_tokens,
        Error.TYPE: lambda d: d.error.error_type if d.error else None,
        Server.ADDRESS: lambda d: d.server.address if d.server else None,
        Server.PORT: lambda d: d.server.port if d.server else None,
        LiteLLM.CALL_ID: lambda d: d.identity.call_id or None,
        # The provider/underlying model is only known once routing has picked a
        # deployment, so it can't ride identity Baggage (seeded at auth, before
        # routing) onto the boundary-born LLM span — stamp it directly here.
        LiteLLM.PROVIDER_MODEL: lambda d: d.identity.provider_model or None,
        f"{LiteLLM.COST_PREFIX}total": lambda d: d.response_cost,
        # Per-component cost breakdown (from the StandardLoggingPayload
        # ``cost_breakdown``). Each component is omitted when the source didn't
        # report it, so spans stay sparse rather than carrying zeros.
        f"{LiteLLM.COST_PREFIX}input": lambda d: d.cost.input,
        f"{LiteLLM.COST_PREFIX}output": lambda d: d.cost.output,
        f"{LiteLLM.COST_PREFIX}cache_read": lambda d: d.cost.cache_read,
        f"{LiteLLM.COST_PREFIX}cache_creation": lambda d: d.cost.cache_creation,
        f"{LiteLLM.COST_PREFIX}tool_usage": lambda d: d.cost.tool_usage,
        f"{LiteLLM.COST_PREFIX}original": lambda d: d.cost.original,
        f"{LiteLLM.COST_PREFIX}discount_amount": lambda d: d.cost.discount_amount,
        f"{LiteLLM.COST_PREFIX}discount_percent": lambda d: d.cost.discount_percent,
        f"{LiteLLM.COST_PREFIX}margin_fixed_amount": lambda d: d.cost.margin_fixed_amount,
        f"{LiteLLM.COST_PREFIX}margin_percent": lambda d: d.cost.margin_percent,
        f"{LiteLLM.COST_PREFIX}margin_total_amount": lambda d: d.cost.margin_total_amount,
        LiteLLM.REQUEST_STREAMING: lambda d: d.is_streaming,
    }

    _TOOL_ATTRS: dict[str, Callable[[ToolDefinition], AttrValue | None]] = {
        "name": lambda t: t.name,
        "description": lambda t: t.description or None,
        "parameters": lambda t: t.parameters_json or None,
    }

    _MCP_ATTRS: dict[str, Callable[[MCPToolCallSpanData], AttrValue | None]] = {
        GenAI.OPERATION_NAME: lambda d: d.operation.value,
        MCP.METHOD_NAME: lambda d: d.method,
        MCP.SESSION_ID: lambda d: d.session_id,
        GenAI.TOOL_NAME: lambda d: d.tool_name or None,
        GenAI.TOOL_CALL_ARGUMENTS: lambda d: d.arguments_json,
        GenAI.TOOL_CALL_RESULT: lambda d: d.result_json,
        LiteLLM.MCP_SERVER_NAME: lambda d: d.server_name,
        LiteLLM.CALL_ID: lambda d: d.identity.call_id or None,
        f"{LiteLLM.COST_PREFIX}total": lambda d: d.response_cost,
    }

    # A tools/list discovery span: the method and session only. Per semconv it must
    # NOT carry gen_ai.operation.name (execute_tool) or gen_ai.tool.name — those are
    # for tool calls, and listing executes no tool.
    _MCP_LIST_ATTRS: dict[str, Callable[[MCPListToolsSpanData], AttrValue | None]] = {
        MCP.METHOD_NAME: lambda d: d.method,
        MCP.SESSION_ID: lambda d: d.session_id,
        LiteLLM.CALL_ID: lambda d: d.identity.call_id or None,
    }

    _GUARDRAIL_ATTRS: dict[str, Callable[[GuardrailSpanData], AttrValue | None]] = {
        LiteLLM.GUARDRAIL_NAME: lambda d: d.guardrail_name,
        LiteLLM.GUARDRAIL_MODE: lambda d: d.mode,
        LiteLLM.GUARDRAIL_STATUS: lambda d: d.status,
        LiteLLM.GUARDRAIL_PROVIDER: lambda d: d.provider,
        LiteLLM.GUARDRAIL_ACTION: lambda d: d.action,
        LiteLLM.GUARDRAIL_RESPONSE: lambda d: d.response_json,
        LiteLLM.GUARDRAIL_VIOLATION_CATEGORIES: lambda d: (
            list(d.violation_categories) if d.violation_categories else None
        ),
        LiteLLM.GUARDRAIL_CONFIDENCE_SCORE: lambda d: d.confidence_score,
        LiteLLM.GUARDRAIL_RISK_SCORE: lambda d: d.risk_score,
        LiteLLM.GUARDRAIL_MASKED_ENTITY_COUNT: lambda d: d.masked_entity_count,
        LiteLLM.GUARDRAIL_DURATION: lambda d: d.duration,
        LiteLLM.GUARDRAIL_ID: lambda d: d.guardrail_id,
        LiteLLM.GUARDRAIL_POLICY_TEMPLATE: lambda d: d.policy_template,
        LiteLLM.GUARDRAIL_DETECTION_METHOD: lambda d: d.detection_method,
    }

    _SERVICE_ATTRS: dict[str, Callable[[ServiceSpanData], AttrValue | None]] = {
        LiteLLM.SERVICE_NAME: lambda d: d.service_name,
        LiteLLM.SERVICE_CALL_TYPE: lambda d: d.call_type,
    }

    def map(self, data: SpanData) -> AttributeMap:
        match data:
            case LLMCallSpanData():
                return self._llm_call(data)
            case MCPToolCallSpanData():
                return collect(self._MCP_ATTRS, data)
            case MCPListToolsSpanData():
                return collect(self._MCP_LIST_ATTRS, data)
            case GuardrailSpanData():
                return self._guardrail(data)
            case ServiceSpanData():
                return self._service(data)
            case _:
                return {}

    @classmethod
    def _llm_call(cls, data: LLMCallSpanData) -> AttributeMap:
        attrs = collect(cls._LLM_CALL_ATTRS, data)
        attrs.update(
            drop_none(
                {
                    f"gen_ai.tool.{idx}.{suffix}": extract(tool)
                    for idx, tool in enumerate(data.tools)
                    for suffix, extract in cls._TOOL_ATTRS.items()
                }
            )
        )
        return attrs

    @classmethod
    def _guardrail(cls, data: GuardrailSpanData) -> AttributeMap:
        return collect(cls._GUARDRAIL_ATTRS, data)

    @classmethod
    def _service(cls, data: ServiceSpanData) -> AttributeMap:
        attrs = collect(cls._SERVICE_ATTRS, data)
        # An outbound datastore call (DB_CALL / CLIENT span) also carries db.*
        # semconv. Internal services (router, budget jobs, …) have no db.system,
        # so they get only the litellm.service.* keys above.
        system = db_system(data.service_name)
        if system is not None:
            attrs[DB.SYSTEM_NAME] = system
            if data.call_type:
                attrs[DB.OPERATION_NAME] = data.call_type
        attrs.update({f"{LiteLLM.METADATA_PREFIX}{key}": value for key, value in data.event_metadata.items()})
        return attrs
