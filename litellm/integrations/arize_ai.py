"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


def set_arize_ai_attributes(span: Span, kwargs, response_obj):
    from litellm.integrations._types.open_inference import (
        MessageAttributes,
        MessageContentAttributes,
        SpanAttributes,
    )

    optional_params = kwargs.get("optional_params", {})
    litellm_params = kwargs.get("litellm_params", {}) or {}

    #############################################
    ############ LLM CALL METADATA ##############
    #############################################
    metadata = litellm_params.get("metadata", {}) or {}
    span.set_attribute(SpanAttributes.METADATA, str(metadata))

    #############################################
    ########## LLM Request Attributes ###########
    #############################################

    # The name of the LLM a request is being made to
    if kwargs.get("model"):
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))

    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        f"litellm-{str(kwargs.get('call_type', None))}",
    )
    span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, str(kwargs.get("messages")))

    # The Generative AI Provider: Azure, OpenAI, etc.
    span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, str(optional_params))

    if optional_params.get("user"):
        span.set_attribute(SpanAttributes.USER_ID, optional_params.get("user"))

    #############################################
    ########## LLM Response Attributes ##########
    #############################################
    llm_output_messages = []
    for choice in response_obj.get("choices"):
        llm_output_messages.append(choice.get("message"))

    span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, str(llm_output_messages))
    usage = response_obj.get("usage")
    if usage:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            usage.get("total_tokens"),
        )

        # The number of tokens used in the LLM response (completion).
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            usage.get("completion_tokens"),
        )

        # The number of tokens used in the LLM prompt.
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            usage.get("prompt_tokens"),
        )
    pass
