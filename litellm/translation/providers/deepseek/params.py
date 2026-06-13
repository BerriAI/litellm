"""Parameter gates for the deepseek serializer.

v1's gate is ``_check_valid_arg`` over ``DeepSeekChatConfig.
get_supported_openai_params`` — the OpenAI base list plus ``thinking`` and
``reasoning_effort``, appended UNCONDITIONALLY for every deepseek model
(deepseek/chat/transformation.py:19-25; the map's thinking rewrite is not
model-gated either, only the history fill is). Two body-rewriting v1
transforms fall back typed so v1 serves its own rewrite:

- the ALWAYS-on content-list flatten
  (``handle_messages_with_content_list_to_str_conversion`` runs before
  super, :122-136): text-only lists are the serializer's flatten delta;
  any non-text part is lossily flattened by v1 (the sambanova shape), so
  those fall back.
- thinking-mode history fill (``_fill_reasoning_content``, :67-107): when
  ``supports_reasoning(model, "deepseek")`` AND the EFFECTIVE thinking is
  enabled (the verbatim dict or the reasoning_effort rewrite), v1 patches
  EVERY assistant message missing ``reasoning_content`` (promote from
  provider_specific_fields, else a single-space placeholder). Messages
  that already carry ``reasoning_content`` fall back at the inbound
  boundary, so every v2-visible assistant message would be patched.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import BASE_LIST, unsupported_against, user_note
from ..openai_compat.params import unsupported_response_format

_DEEPSEEK_LIST = BASE_LIST | frozenset({"thinking", "reasoning_effort"})


def supports_deepseek_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"deepseek/{model}", "supports_reasoning")


def thinking_mode_requested(request: ChatRequest) -> bool:
    """The post-map ``thinking == {"type": "enabled"}`` truth: the verbatim
    dict wins; only when it is absent does ``reasoning_effort != "none"``
    rewrite into thinking-enabled (map_openai_params:49-63, elif chain)."""
    thinking = request.thinking.default_value(None)
    if thinking is not None:
        return thinking.tag == "enabled"
    effort = request.reasoning_effort.default_value(None)
    return effort is not None and effort != "none"


def _has_non_text_content_block(request: ChatRequest) -> bool:
    return any(
        block.tag not in ("text", "tool_use", "tool_result")
        for message in request.messages
        for block in message.content
    )


def _has_assistant_message(request: ChatRequest) -> bool:
    return any(message.role == "assistant" for message in request.messages)


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if _has_non_text_content_block(request):
        return (
            "non-text content block on deepseek: v1's always-on content-list "
            "flatten drops non-text parts and skips the base image transforms "
            "(deepseek/chat/transformation.py _transform_messages); v1 serves "
            "its lossy flatten"
        )
    if (
        thinking_mode_requested(request)
        and supports_deepseek_reasoning(request.model, deps)
        and _has_assistant_message(request)
    ):
        return (
            "assistant history in deepseek thinking mode: v1's "
            "_fill_reasoning_content injects reasoning_content (a "
            "single-space placeholder) into every assistant message missing "
            "it; v1 serves its rewrite"
        )
    return unsupported_against(
        request,
        provider="deepseek",
        allowed=_DEEPSEEK_LIST,
        notes={"user": user_note("deepseek")},
    ) or unsupported_response_format(request)
