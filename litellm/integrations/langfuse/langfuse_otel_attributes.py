"""
If the LLM Obs has any specific attributes to log request or response, we can add them here.

Relevant Issue: https://github.com/BerriAI/litellm/issues/13764
"""

import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel
from typing_extensions import override

import litellm
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


@runtime_checkable
class Gettable(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def as_gettable(value: object) -> Gettable | None:
    return value if isinstance(value, Gettable) else None


def serialize_langfuse_datetime(value: datetime) -> str:
    return json.dumps(value.astimezone().isoformat())


def default_langfuse_tags(
    kwargs: Mapping[str, object], metadata: Mapping[str, object], *, completed: bool = True
) -> tuple[str, ...]:
    default_tags: Final = litellm.langfuse_default_tags
    if not isinstance(default_tags, list):
        return ()

    def expand(key: str) -> str | None:
        if key == "cache_hit":
            return f"cache_hit:{kwargs.get('cache_hit', False)}" if completed else None
        if key == "cache_key":
            hidden_params: Final = as_gettable(metadata.get("hidden_params"))
            hidden_key: Final = hidden_params.get("cache_key") if hidden_params else None
            cache: Final = litellm.cache
            cache_key: Final = (
                cache.get_preset_cache_key_from_kwargs(**kwargs)
                if hidden_key is None and cache is not None
                else hidden_key
            )
            return f"cache_key:{cache_key}" if completed or cache_key is not None else None
        value: Final = os.environ.get("PROXY_BASE_URL") if key == "proxy_base_url" else metadata.get(key)
        return f"{key}:{value}" if value is not None else None

    return tuple(tag for key in default_tags if (tag := expand(key)) is not None)


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
    def set_messages(span: "Span", kwargs: dict[str, Any]):
        prompt: Final = {"messages": kwargs.get("messages")}
        optional_params: Final = kwargs.get("optional_params", {})
        functions: Final = optional_params.get("functions")
        tools: Final = optional_params.get("tools")
        if functions is not None:
            prompt["functions"] = functions
        if tools is not None:
            prompt["tools"] = tools

        input: Final = prompt
        safe_set_attribute(span, "langfuse.observation.input", json.dumps(input))

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        safe_set_attribute(
            span,
            "langfuse.observation.output",
            get_output_content_by_type(response_obj),
        )
