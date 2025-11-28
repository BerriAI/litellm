"""
Weave-specific OTEL attributes for LLM observability.

Based on Weave's OTEL integration documentation:
https://docs.wandb.ai/weave/guides/tracking/otel
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from typing_extensions import override

from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent, ResponsesAPIResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    RerankResponse,
    TextCompletionResponse,
    TranscriptionResponse,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span


def get_output_content_by_type(
    response_obj: None
    | dict
    | EmbeddingResponse
    | ModelResponse
    | TextCompletionResponse
    | ImageResponse
    | TranscriptionResponse
    | RerankResponse
    | HttpxBinaryResponseContent
    | ResponsesAPIResponse
    | list,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """
    Extract output content from response objects based on their type.

    This utility function handles the type-specific logic for converting
    various response objects into appropriate output formats for Weave logging.

    Args:
        response_obj: The response object returned by the function
        kwargs: Optional keyword arguments containing call_type and other metadata

    Returns:
        The formatted output content suitable for Weave logging
    """
    if response_obj is None:
        return ""

    kwargs = kwargs or {}
    call_type = kwargs.get("call_type", None)

    # Embedding responses - no output content
    if call_type == "embedding" or isinstance(response_obj, EmbeddingResponse):
        return "embedding-output"

    # Binary/Speech responses
    if isinstance(response_obj, HttpxBinaryResponseContent):
        return "speech-output"

    if isinstance(response_obj, BaseModel):
        return response_obj.model_dump_json()

    if response_obj and (isinstance(response_obj, dict) or isinstance(response_obj, list)):
        return json.dumps(response_obj)

    return ""


class WeaveLLMObsOTELAttributes(BaseLLMObsOTELAttributes):
    """
    Weave-specific LLM observability OTEL attributes.

    Weave automatically maps attributes from multiple frameworks including
    GenAI, OpenInference, Langfuse, and others. We use OpenInference conventions
    (input.value, output.value, llm.*) which Weave recognizes.
    """

    @staticmethod
    @override
    def set_messages(span: Span, kwargs: dict[str, Any]):
        """Set input messages as span attributes using OpenInference conventions."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        messages = kwargs.get("messages", [])
        optional_params = kwargs.get("optional_params", {})

        # Set input.value - recognized by Weave for inputs
        prompt: dict[str, Any] = {"messages": messages}
        functions = optional_params.get("functions")
        tools = optional_params.get("tools")
        if functions is not None:
            prompt["functions"] = functions
        if tools is not None:
            prompt["tools"] = tools
        safe_set_attribute(span, WeaveSpanAttributes.INPUT_VALUE.value, json.dumps(prompt))

        # Set individual input messages in OpenInference format
        for idx, msg in enumerate(messages):
            prefix = f"{WeaveSpanAttributes.LLM_INPUT_MESSAGES.value}.{idx}"
            safe_set_attribute(span, f"{prefix}.message.role", msg.get("role", ""))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            safe_set_attribute(span, f"{prefix}.message.content", content)

    @staticmethod
    @override
    def set_response_output_messages(span: Span, response_obj):
        """Set response output as span attributes using OpenInference conventions."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        # Set output.value - recognized by Weave for outputs
        safe_set_attribute(
            span,
            WeaveSpanAttributes.OUTPUT_VALUE.value,
            get_output_content_by_type(response_obj),
        )

        # Set individual output messages in OpenInference format
        if response_obj and hasattr(response_obj, "get"):
            for idx, choice in enumerate(response_obj.get("choices", [])):
                message = choice.get("message", {})
                prefix = f"{WeaveSpanAttributes.LLM_OUTPUT_MESSAGES.value}.{idx}"
                safe_set_attribute(span, f"{prefix}.message.role", message.get("role", ""))
                safe_set_attribute(span, f"{prefix}.message.content", message.get("content", ""))
