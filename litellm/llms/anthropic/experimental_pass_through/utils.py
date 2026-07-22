import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Optional, Union

import litellm
from pydantic import TypeAdapter, ValidationError
from litellm.types.llms.anthropic import (
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesTextParam,
    AnthropicMessagesUserMessageParam,
    AnthropicSystemMessageContent,
)
from litellm.types.utils import ModelInfo
from typing_extensions import Literal, Required, TypedDict


_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_SEQUENCE_ADAPTER = TypeAdapter(Sequence[object])


class AnthropicMessagesSystemMessageParam(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[Union[str, Sequence[AnthropicSystemMessageContent]]]


AllAnthropicPassThroughMessageValues = Union[
    AnthropicMessagesUserMessageParam,
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesSystemMessageParam,
]


def _normalize_anthropic_system_text_block(block: object) -> AnthropicMessagesTextParam | None:
    if not isinstance(block, Mapping):
        return None
    try:
        typed_block = _MAPPING_ADAPTER.validate_python(block, strict=True)
    except ValidationError:
        return None
    text = typed_block.get("text")
    if typed_block.get("type") != "text" or not isinstance(text, str):
        return None
    cache_control = typed_block.get("cache_control")
    if isinstance(cache_control, Mapping):
        try:
            typed_cache_control = _MAPPING_ADAPTER.validate_python(cache_control, strict=True)
        except ValidationError:
            return AnthropicMessagesTextParam(type="text", text=text)
        return AnthropicMessagesTextParam(
            type="text",
            text=text,
            cache_control=deepcopy(typed_cache_control),
        )
    return AnthropicMessagesTextParam(type="text", text=text)


def normalize_anthropic_system_message_content(
    content: object,
) -> str | tuple[AnthropicMessagesTextParam, ...] | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or not content:
        return None
    try:
        content_blocks = _SEQUENCE_ADAPTER.validate_python(content, strict=True)
    except ValidationError:
        return None
    text_blocks = tuple(
        normalized_block
        for block in content_blocks
        if (normalized_block := _normalize_anthropic_system_text_block(block)) is not None
    )
    if not text_blocks:
        return None
    return text_blocks


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return litellm.reasoning_auto_summary or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
) -> str:
    """
    Normalize a reasoning effort value based on model capabilities.

    Degradation chains:
    - "max"     → max / xhigh / high
    - "xhigh"   → xhigh / high
    - "minimal" → minimal / low
    - other values pass through unchanged
    """
    if effort not in ("max", "xhigh", "minimal"):
        return effort

    from litellm.utils import get_model_info

    model_info: Optional[ModelInfo] = None
    try:
        model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        model_info = None

    if effort == "max":
        if model_info and model_info.get("supports_max_reasoning_effort"):
            return "max"
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "xhigh":
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "minimal":
        if model_info and model_info.get("supports_minimal_reasoning_effort"):
            return "minimal"
        return "low"
    return "medium"
