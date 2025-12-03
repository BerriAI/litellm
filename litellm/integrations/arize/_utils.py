import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

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
    def set_messages(span: "Span", kwargs: Dict[str, Any]):
        from litellm.integrations._types.open_inference import (
            MessageAttributes,
            SpanAttributes,
        )

        messages = kwargs.get("messages")

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


def _set_tool_attributes(span: "Span", optional_params: dict):
    """Helper to set tool and function call attributes on span."""
    from litellm.integrations._types.open_inference import (
        MessageAttributes,
        SpanAttributes,
        ToolCallAttributes,
    )

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

    functions = optional_params.get("functions")
    if functions:
        for idx, function in enumerate(functions):
            prefix = f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{idx}"
            safe_set_attribute(
                span,
                f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                function.get("name"),
            )


def _set_response_attributes(span: "Span", response_obj):
    """Helper to set response output and token usage attributes on span."""
    from litellm.integrations._types.open_inference import (
        MessageAttributes,
        SpanAttributes,
    )

    if not hasattr(response_obj, "get"):
        return

    for idx, choice in enumerate(response_obj.get("choices", [])):
        response_message = choice.get("message", {})
        safe_set_attribute(
            span,
            SpanAttributes.OUTPUT_VALUE,
            response_message.get("content", ""),
        )
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

    output_items = response_obj.get("output", [])
    if output_items:
        for i, item in enumerate(output_items):
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}"
            if hasattr(item, "type"):
                item_type = item.type
                if item_type == "reasoning" and hasattr(item, "summary"):
                    for summary in item.summary:
                        if hasattr(summary, "text"):
                            safe_set_attribute(
                                span,
                                f"{prefix}.{MessageAttributes.MESSAGE_REASONING_SUMMARY}",
                                summary.text,
                            )
                elif item_type == "message" and hasattr(item, "content"):
                    message_content = ""
                    content_list = item.content
                    if content_list and len(content_list) > 0:
                        first_content = content_list[0]
                        message_content = getattr(first_content, "text", "")
                    message_role = getattr(item, "role", "assistant")
                    safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, message_content)
                    safe_set_attribute(span, f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", message_content)
                    safe_set_attribute(span, f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", message_role)

    usage = response_obj and response_obj.get("usage")
    if usage:
        safe_set_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens"))
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        if completion_tokens:
            safe_set_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        if prompt_tokens:
            safe_set_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
        reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens")
        if reasoning_tokens:
            safe_set_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, reasoning_tokens)


def set_attributes(
    span: "Span", kwargs, response_obj, attributes: Type[BaseLLMObsOTELAttributes]
):
    """
    Populates span with OpenInference-compliant LLM attributes for Arize and Phoenix tracing.
    """
    from litellm.integrations._types.open_inference import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    try:
        # Remove secret_fields to prevent leaking sensitive data (e.g., authorization headers)
        optional_params = kwargs.get("optional_params", {})
        if isinstance(optional_params, dict):
            optional_params.pop("secret_fields", None)
        litellm_params = kwargs.get("litellm_params", {})
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        metadata = (
            standard_logging_payload.get("metadata")
            if standard_logging_payload
            else None
        )
        if metadata is not None:
            safe_set_attribute(span, SpanAttributes.METADATA, safe_dumps(metadata))

        if kwargs.get("model"):
            safe_set_attribute(span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))

        safe_set_attribute(span, "llm.request.type", standard_logging_payload["call_type"])
        safe_set_attribute(span, SpanAttributes.LLM_PROVIDER, litellm_params.get("custom_llm_provider", "Unknown"))

        if optional_params.get("max_tokens"):
            safe_set_attribute(span, "llm.request.max_tokens", optional_params.get("max_tokens"))
        if optional_params.get("temperature"):
            safe_set_attribute(span, "llm.request.temperature", optional_params.get("temperature"))
        if optional_params.get("top_p"):
            safe_set_attribute(span, "llm.request.top_p", optional_params.get("top_p"))

        safe_set_attribute(span, "llm.is_streaming", str(optional_params.get("stream", False)))

        if optional_params.get("user"):
            safe_set_attribute(span, "llm.user", optional_params.get("user"))

        if response_obj and response_obj.get("id"):
            safe_set_attribute(span, "llm.response.id", response_obj.get("id"))
        if response_obj and response_obj.get("model"):
            safe_set_attribute(span, "llm.response.model", response_obj.get("model"))

        safe_set_attribute(span, SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        attributes.set_messages(span, kwargs)

        _set_tool_attributes(span=span, optional_params=optional_params)

        model_params = (
            standard_logging_payload.get("model_parameters")
            if standard_logging_payload
            else None
        )
        if model_params:
            safe_set_attribute(span, SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_dumps(model_params))
            if model_params.get("user"):
                user_id = model_params.get("user")
                if user_id is not None:
                    safe_set_attribute(span, SpanAttributes.USER_ID, user_id)

        _set_response_attributes(span=span, response_obj=response_obj)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)
