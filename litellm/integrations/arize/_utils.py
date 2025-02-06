import json
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    Span = _Span
else:
    Span = Any


def set_attributes(span: Span, kwargs, response_obj):
    from openinference.semconv.trace import (
        MessageAttributes,
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    try:
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
            OpenInferenceSpanKindValues.LLM.value,
        )
        messages = kwargs.get("messages")

        # for /chat/completions
        # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
        if messages:
            span.set_attribute(
                SpanAttributes.INPUT_VALUE,
                messages[-1].get("content", ""),  # get the last message for input
            )

            # LLM_INPUT_MESSAGES shows up under `input_messages` tab on the span page
            for idx, msg in enumerate(messages):
                # Set the role per message
                span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}",
                    msg["role"],
                )
                # Set the content per message
                span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                    msg.get("content", ""),
                )

        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload and (model_params := standard_logging_payload["model_parameters"]):
            # The Generative AI Provider: Azure, OpenAI, etc.
            span.set_attribute(
                SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(model_params)
            )

            if model_params.get("user"):
                user_id = model_params.get("user")
                if user_id is not None:
                    span.set_attribute(SpanAttributes.USER_ID, user_id)

        #############################################
        ########## LLM Response Attributes ##########
        # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
        #############################################
        if hasattr(response_obj, 'get'):
            for choice in response_obj.get("choices", []):
                response_message = choice.get("message", {})
                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, response_message.get("content", "")
                )

                # This shows up under `output_messages` tab on the span page
                # This code assumes a single response
                span.set_attribute(
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                    response_message.get("role"),
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                    response_message.get("content", ""),
                )

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
    except Exception as e:
        verbose_logger.error(f"Error setting arize attributes: {e}")
