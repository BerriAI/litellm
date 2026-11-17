"""
If the LLM Obs has any specific attributes to log request or response, we can add them here.

Relevant Issue: https://github.com/BerriAI/litellm/issues/13764
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel
from typing_extensions import override

from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.litellm_core_utils.safe_json_dumps import jsonable_or_str
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


class _SupportsGet(Protocol):
    def get(self, key: str, /) -> object: ...


def serialize_langfuse_datetime(value: datetime) -> str:
    return json.dumps(value.astimezone().isoformat())


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
    kwargs: Mapping[str, object] | None = None,
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
    call_type: Final = kwargs.get("call_type", None)

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
    else:
        return ""


class LangfuseLLMObsOTELAttributes(BaseLLMObsOTELAttributes):
    @staticmethod
    @override
    def set_messages(span: "Span", kwargs: Mapping[str, object]):
        prompt: Final = {"messages": kwargs.get("messages")}
        optional_params: Final = kwargs.get("optional_params", {})
        functions: Final = require_attributes(optional_params, _SupportsGet, "get").get("functions")
        tools: Final = require_attributes(optional_params, _SupportsGet, "get").get("tools")
        if functions is not None:
            prompt["functions"] = functions
        if tools is not None:
            prompt["tools"] = tools

        input: Final = prompt
        safe_set_attribute(span, "langfuse.observation.input", json.dumps(input, default=jsonable_or_str))

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        safe_set_attribute(
            span,
            "langfuse.observation.output",
            get_output_content_by_type(response_obj),
        )
