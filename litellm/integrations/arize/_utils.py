import json
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from typing_extensions import override

from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span


class ArizeOTELAttributes(BaseLLMObsOTELAttributes):

    @staticmethod
    @override
    def set_messages(span: "Span", messages: List[Dict[str, Any]]):
        from litellm.integrations._types.open_inference import (
            MessageAttributes,
            SpanAttributes,
        )

        # for /chat/completions
        # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
        if messages:
            last_message = messages[-1]
            safe_set_attribute(
                span,
                SpanAttributes.INPUT_VALUE,
                last_message.get("content", ""),
            )

            # LLM_INPUT_MESSAGES shows up under `input_messages` tab on the span page.
            for idx, msg in enumerate(messages):
                prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
                # Set the role per message.
                safe_set_attribute(
                    span, f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", msg.get("role")
                )
                # Set the content per message.
                safe_set_attribute(
                    span,
                    f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                    msg.get("content", ""),
                )

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        """
        Sets output message attributes on the span from the LLM response.

        Args:
            span: The OpenTelemetry span to set attributes on
            response_obj: The response object containing choices with messages
        """
        from litellm.integrations._types.open_inference import (
            MessageAttributes,
            SpanAttributes,
        )

        for idx, choice in enumerate(response_obj.get("choices", [])):
            response_message = choice.get("message", {})
            safe_set_attribute(
                span,
                SpanAttributes.OUTPUT_VALUE,
                response_message.get("content", ""),
            )

            # This shows up under `output_messages` tab on the span page.
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}"
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                response_message.get("role"),
            )
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                response_message.get("content", ""),
            )


def set_attributes(
    span: "Span", kwargs, response_obj, attributes: Type[BaseLLMObsOTELAttributes]
):  # noqa: PLR0915
    """
    Populates span with OpenInference-compliant LLM attributes for Arize and Phoenix tracing.
    """
    from litellm.integrations._types.open_inference import (
        MessageAttributes,
        OpenInferenceSpanKindValues,
        SpanAttributes,
        ToolCallAttributes,
    )

    try:
        optional_params = kwargs.get("optional_params", {})
        litellm_params = kwargs.get("litellm_params", {})
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        #############################################
        ############ LLM CALL METADATA ##############
        #############################################

        # Set custom metadata for observability and trace enrichment.
        metadata = (
            standard_logging_payload.get("metadata")
            if standard_logging_payload
            else None
        )
        if metadata is not None:
            safe_set_attribute(span, SpanAttributes.METADATA, safe_dumps(metadata))

        #############################################
        ########## LLM Request Attributes ###########
        #############################################

        # The name of the LLM a request is being made to.
        if kwargs.get("model"):
            safe_set_attribute(
                span,
                SpanAttributes.LLM_MODEL_NAME,
                kwargs.get("model"),
            )

        # The LLM request type.
        safe_set_attribute(
            span,
            "llm.request.type",
            standard_logging_payload["call_type"],
        )

        # The Generative AI Provider: Azure, OpenAI, etc.
        safe_set_attribute(
            span,
            SpanAttributes.LLM_PROVIDER,
            litellm_params.get("custom_llm_provider", "Unknown"),
        )

        # The maximum number of tokens the LLM generates for a request.
        if optional_params.get("max_tokens"):
            safe_set_attribute(
                span,
                "llm.request.max_tokens",
                optional_params.get("max_tokens"),
            )

        # The temperature setting for the LLM request.
        if optional_params.get("temperature"):
            safe_set_attribute(
                span,
                "llm.request.temperature",
                optional_params.get("temperature"),
            )

        # The top_p sampling setting for the LLM request.
        if optional_params.get("top_p"):
            safe_set_attribute(
                span,
                "llm.request.top_p",
                optional_params.get("top_p"),
            )

        # Indicates whether response is streamed.
        safe_set_attribute(
            span,
            "llm.is_streaming",
            str(optional_params.get("stream", False)),
        )

        # Logs the user ID if present.
        if optional_params.get("user"):
            safe_set_attribute(
                span,
                "llm.user",
                optional_params.get("user"),
            )

        # The unique identifier for the completion.
        if response_obj and response_obj.get("id"):
            safe_set_attribute(span, "llm.response.id", response_obj.get("id"))

        # The model used to generate the response.
        if response_obj and response_obj.get("model"):
            safe_set_attribute(
                span,
                "llm.response.model",
                response_obj.get("model"),
            )

        # Required by OpenInference to mark span as LLM kind.
        safe_set_attribute(
            span,
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        messages = kwargs.get("messages")

        attributes.set_messages(span, messages)

        # Capture tools (function definitions) used in the LLM call.
        tools = optional_params.get("tools")
        if tools:
            for idx, tool in enumerate(tools):
                function = tool.get("function")
                if not function:
                    continue
                prefix = f"{SpanAttributes.LLM_TOOLS}.{idx}"
                safe_set_attribute(
                    span, f"{prefix}.{SpanAttributes.TOOL_NAME}", function.get("name")
                )
                safe_set_attribute(
                    span,
                    f"{prefix}.{SpanAttributes.TOOL_DESCRIPTION}",
                    function.get("description"),
                )
                safe_set_attribute(
                    span,
                    f"{prefix}.{SpanAttributes.TOOL_PARAMETERS}",
                    json.dumps(function.get("parameters")),
                )

        # Capture tool calls made during function-calling LLM flows.
        functions = optional_params.get("functions")
        if functions:
            for idx, function in enumerate(functions):
                prefix = f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{idx}"
                safe_set_attribute(
                    span,
                    f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    function.get("name"),
                )

        # Capture invocation parameters and user ID if available.
        model_params = (
            standard_logging_payload.get("model_parameters")
            if standard_logging_payload
            else None
        )
        if model_params:
            # The Generative AI Provider: Azure, OpenAI, etc.
            safe_set_attribute(
                span,
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                safe_dumps(model_params),
            )

            if model_params.get("user"):
                user_id = model_params.get("user")
                if user_id is not None:
                    safe_set_attribute(span, SpanAttributes.USER_ID, user_id)

        #############################################
        ########## LLM Response Attributes ##########
        #############################################

        # Captures response tokens, message, and content.
        if hasattr(response_obj, "get"):
            attributes.set_response_output_messages(span, response_obj)

            # Token usage info.
            usage = response_obj and response_obj.get("usage")
            if usage:
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    usage.get("total_tokens"),
                )

                # The number of tokens used in the LLM response (completion).
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    usage.get("completion_tokens"),
                )

                # The number of tokens used in the LLM prompt.
                safe_set_attribute(
                    span,
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    usage.get("prompt_tokens"),
                )

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)
