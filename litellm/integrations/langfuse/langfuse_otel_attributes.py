"""
If the LLM Obs has any specific attributes to log request or response, we can add them here.

Relevant Issue: https://github.com/BerriAI/litellm/issues/13764
"""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from pydantic import BaseModel
from typing_extensions import override

from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.llms.openai import (
    HttpxBinaryResponseContent,
    ResponsesAPIResponse,
)
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
    response_obj: Union[
        None,
        dict,
        EmbeddingResponse,
        ModelResponse,
        TextCompletionResponse,
        ImageResponse,
        TranscriptionResponse,
        RerankResponse,
        HttpxBinaryResponseContent,
        ResponsesAPIResponse,
        list,
    ],
    kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Extract output content from response objects based on their type.

    This utility function handles the type-specific logic for converting
    various response objects into appropriate output formats for Langfuse logging.

    Args:
        response_obj: The response object returned by the function
        kwargs: Optional keyword arguments containing call_type and other metadata

    Returns:
        The formatted output content suitable for Langfuse logging, or None
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

    if response_obj and (
        isinstance(response_obj, dict) or isinstance(response_obj, list)
    ):
        return json.dumps(response_obj)
    else:
        return ""


def get_langfuse_observation_input_by_type(
    kwargs: Dict[str, Any],
) -> Optional[Union[list[Any], dict[str, Any]]]:
    """
    Build the Langfuse observation input payload for both chat- and
    Responses-style requests.

    Chat requests preserve the existing observation shape of the raw messages
    list. Responses requests snapshot the model-visible request fields under a
    single `input` payload for prompt debugging.
    """
    messages = kwargs.get("messages")
    if messages:
        return messages

    optional_params = kwargs.get("optional_params", {}) or {}
    response_input = kwargs.get("input")
    if response_input is not None:
        prompt: dict[str, Any] = {"input": response_input}
        # Keep the Responses observation focused on request fields that affect
        # model context, tool behavior, or response shape.
        for key in (
            "instructions",
            "tools",
            "tool_choice",
            "reasoning",
            "max_output_tokens",
            "max_tool_calls",
            "text",
            "parallel_tool_calls",
            "truncation",
            "temperature",
            "top_p",
            "previous_response_id",
            "prompt",
        ):
            value = optional_params.get(key)
            if value is not None:
                prompt[key] = value
        return prompt

    prompt: dict[str, Any] = {}
    # Preserve explicit empty chat message lists without serializing `null`
    # when callers pass `messages=None`.
    if messages is not None:
        prompt["messages"] = messages
    for key in ("functions", "tools"):
        value = optional_params.get(key)
        if value is not None:
            prompt[key] = value

    return prompt or None


class LangfuseLLMObsOTELAttributes(BaseLLMObsOTELAttributes):
    @staticmethod
    @override
    def set_messages(span: "Span", kwargs: Dict[str, Any]):
        input_payload = get_langfuse_observation_input_by_type(kwargs)
        if input_payload is not None:
            safe_set_attribute(
                span,
                "langfuse.observation.input",
                safe_dumps(input_payload),
            )

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        safe_set_attribute(
            span,
            "langfuse.observation.output",
            get_output_content_by_type(response_obj),
        )
