"""Canonical OpenTelemetry GenAI semantic-convention mapper (always active)."""

from litellm.integrations.otel.mappers.base import AttributeMap
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.semconv import Error, GenAI, LiteLLM, Server


class GenAIMapper:
    """Emits ``gen_ai.*`` (and a few ``litellm.*`` vendor) attributes."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap:
        attrs: AttributeMap = {GenAI.OPERATION_NAME: data.operation.value}
        if data.provider:
            attrs[GenAI.PROVIDER_NAME] = data.provider
        if data.request_model:
            attrs[GenAI.REQUEST_MODEL] = data.request_model

        rp = data.request_params
        if rp.temperature is not None:
            attrs[GenAI.REQUEST_TEMPERATURE] = rp.temperature
        if rp.top_p is not None:
            attrs[GenAI.REQUEST_TOP_P] = rp.top_p
        if rp.top_k is not None:
            attrs[GenAI.REQUEST_TOP_K] = rp.top_k
        if rp.max_tokens is not None:
            attrs[GenAI.REQUEST_MAX_TOKENS] = rp.max_tokens
        if rp.frequency_penalty is not None:
            attrs[GenAI.REQUEST_FREQUENCY_PENALTY] = rp.frequency_penalty
        if rp.presence_penalty is not None:
            attrs[GenAI.REQUEST_PRESENCE_PENALTY] = rp.presence_penalty
        if rp.stop_sequences:
            attrs[GenAI.REQUEST_STOP_SEQUENCES] = list(rp.stop_sequences)
        if rp.seed is not None:
            attrs[GenAI.REQUEST_SEED] = rp.seed

        if data.response_model:
            attrs[GenAI.RESPONSE_MODEL] = data.response_model
        if data.response_id:
            attrs[GenAI.RESPONSE_ID] = data.response_id
        if data.finish_reasons:
            attrs[GenAI.RESPONSE_FINISH_REASONS] = list(data.finish_reasons)

        if data.usage.input_tokens is not None:
            attrs[GenAI.USAGE_INPUT_TOKENS] = data.usage.input_tokens
        if data.usage.output_tokens is not None:
            attrs[GenAI.USAGE_OUTPUT_TOKENS] = data.usage.output_tokens

        if data.error and data.error.error_type:
            attrs[Error.TYPE] = data.error.error_type

        if data.server is not None:
            if data.server.address:
                attrs[Server.ADDRESS] = data.server.address
            if data.server.port is not None:
                attrs[Server.PORT] = data.server.port

        if data.identity.call_id:
            attrs[LiteLLM.CALL_ID] = data.identity.call_id
        if data.response_cost is not None:
            attrs[f"{LiteLLM.COST_PREFIX}total"] = data.response_cost
        if data.is_streaming is not None:
            attrs[LiteLLM.REQUEST_STREAMING] = data.is_streaming

        return attrs
