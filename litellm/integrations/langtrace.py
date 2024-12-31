import json
from typing import TYPE_CHECKING, Any

from litellm.proxy._types import SpanAttributes

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class LangtraceAttributes:
    """
    This class is used to save trace attributes to Langtrace's spans
    """

    def set_langtrace_attributes(self, span: Span, kwargs, response_obj):
        """
        This function is used to log the event to Langtrace
        """

        vendor = kwargs.get("litellm_params").get("custom_llm_provider")
        optional_params = kwargs.get("optional_params", {})
        options = {**kwargs, **optional_params}
        self.set_request_attributes(span, options, vendor)
        self.set_response_attributes(span, response_obj)
        self.set_usage_attributes(span, response_obj)

    def set_request_attributes(self, span: Span, kwargs, vendor):
        """
        This function is used to get span attributes for the LLM request
        """
        span_attributes = {
            "gen_ai.operation.name": "chat",
            "langtrace.service.name": vendor,
            SpanAttributes.LLM_REQUEST_MODEL.value: kwargs.get("model"),
            SpanAttributes.LLM_IS_STREAMING.value: kwargs.get("stream"),
            SpanAttributes.LLM_REQUEST_TEMPERATURE.value: kwargs.get("temperature"),
            SpanAttributes.LLM_TOP_K.value: kwargs.get("top_k"),
            SpanAttributes.LLM_REQUEST_TOP_P.value: kwargs.get("top_p"),
            SpanAttributes.LLM_USER.value: kwargs.get("user"),
            SpanAttributes.LLM_REQUEST_MAX_TOKENS.value: kwargs.get("max_tokens"),
            SpanAttributes.LLM_RESPONSE_STOP_REASON.value: kwargs.get("stop"),
            SpanAttributes.LLM_FREQUENCY_PENALTY.value: kwargs.get("frequency_penalty"),
            SpanAttributes.LLM_PRESENCE_PENALTY.value: kwargs.get("presence_penalty"),
        }

        prompts = kwargs.get("messages")

        if prompts:
            span.add_event(
                name="gen_ai.content.prompt",
                attributes={SpanAttributes.LLM_PROMPTS.value: json.dumps(prompts)},
            )

        self.set_span_attributes(span, span_attributes)

    def set_response_attributes(self, span: Span, response_obj):
        """
        This function is used to get span attributes for the LLM response
        """
        response_attributes = {
            "gen_ai.response_id": response_obj.get("id"),
            "gen_ai.system_fingerprint": response_obj.get("system_fingerprint"),
            SpanAttributes.LLM_RESPONSE_MODEL.value: response_obj.get("model"),
        }
        completions = []
        for choice in response_obj.get("choices", []):
            role = choice.get("message").get("role")
            content = choice.get("message").get("content")
            completions.append({"role": role, "content": content})

        span.add_event(
            name="gen_ai.content.completion",
            attributes={SpanAttributes.LLM_COMPLETIONS: json.dumps(completions)},
        )

        self.set_span_attributes(span, response_attributes)

    def set_usage_attributes(self, span: Span, response_obj):
        """
        This function is used to get span attributes for the LLM usage
        """
        usage = response_obj.get("usage")
        if usage:
            usage_attributes = {
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS.value: usage.get(
                    "prompt_tokens"
                ),
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS.value: usage.get(
                    "completion_tokens"
                ),
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS.value: usage.get("total_tokens"),
            }
            self.set_span_attributes(span, usage_attributes)

    def set_span_attributes(self, span: Span, attributes):
        """
        This function is used to set span attributes
        """
        for key, value in attributes.items():
            if not value:
                continue
            span.set_attribute(key, value)
