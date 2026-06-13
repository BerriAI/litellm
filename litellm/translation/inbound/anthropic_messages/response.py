"""IR ``ChatResponse`` -> Anthropic Messages response body.

The reverse of the request parse: an IR ``ChatResponse`` (produced by any
provider's ``parse_response``) becomes an Anthropic Messages response body,
reproducing v1's ``translate_openai_response_to_anthropic``
(transformation.py:1374). The Anthropic Messages response shape is
provider-independent, so the ``ResponseDialect`` argument the pipeline threads
is ignored here (it parameterizes the chat-completion outbound, not this one).

Content blocks are emitted in v1's per-choice order: thinking (or
redacted_thinking) first, then text, then tool_use. ``stop_reason`` maps
stop->end_turn, length->max_tokens, tool_calls->tool_use; ``content_filter``
takes v1's default-branch ``end_turn`` (researcher-6 §1.5 item 11: v1 has no
content_filter stop_reason and falls through to end_turn).

Usage reproduces v1's chat-usage math exactly. v1 receives an OpenAI
ModelResponse whose ``prompt_tokens`` is ``input + cache_creation + cache_read``
with ``prompt_tokens_details.cached_tokens == cache_read`` (the anthropic
chat dialect's ``_usage_json``), then emits
``input_tokens = prompt_tokens - cached_tokens = input + cache_creation``.
Reconstructing that from the IR ``ResponseUsage`` (whose ``input_tokens`` is
already the uncached count) means emitting
``input + cache_creation_input_tokens`` for the Anthropic ``input_tokens``;
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are emitted
only when > 0, matching v1.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from expression.collections import Block

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    THOUGHT_SIGNATURE_SEPARATOR as _THOUGHT_SIGNATURE_SEPARATOR,
)
from ...ir import (
    Body,
    ChatResponse,
    ContentBlock,
    FinishReason,
    PlainJson,
    ResponseUsage,
)

_STOP_REASON: Mapping[FinishReason, str] = MappingProxyType(
    {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
)


def serialize_response(
    response: ChatResponse,
    deps: TranslationDeps,
    dialect: object = None,
) -> Body | TranslationError:
    """The ``dialect`` is the provider's outbound chat-completion dialect the
    pipeline threads uniformly; the Anthropic Messages response shape is
    provider-independent, so it is accepted (kept off the sibling
    openai_chat ``ResponseDialect`` import per the no-sibling tenet) and
    ignored."""
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "model": response.model or "unknown-model",
        "stop_sequence": None,
        "usage": _usage(response.usage),
        "content": _content(response.content),
        "stop_reason": _STOP_REASON[response.finish],
    }


def _content(content: Block[ContentBlock]) -> list[PlainJson]:
    return [
        *_thinking_blocks(content),
        *_text_blocks(content),
        *_tool_use_blocks(content),
    ]


def _thinking_blocks(content: Block[ContentBlock]) -> list[PlainJson]:
    blocks: list[PlainJson] = []
    for block in content:
        # blocks is a local build-then-freeze accumulator; it never escapes
        if block.tag == "thinking":
            signature = block.thinking.signature.default_value(None)
            blocks.append(  # nosemgrep: translation-no-mutation
                {
                    "type": "thinking",
                    "thinking": block.thinking.thinking,
                    "signature": signature,
                }
            )
        elif block.tag == "redacted_thinking":
            blocks.append(  # nosemgrep: translation-no-mutation
                {
                    "type": "redacted_thinking",
                    "data": block.redacted_thinking.data,
                }
            )
    return blocks


def _text_blocks(content: Block[ContentBlock]) -> list[PlainJson]:
    return [
        {"type": "text", "text": block.text.text}
        for block in content
        if block.tag == "text"
    ]


def _tool_use_blocks(content: Block[ContentBlock]) -> list[PlainJson]:
    blocks: list[PlainJson] = []
    for block in content:
        if block.tag != "tool_use":
            continue
        identifier = block.tool_use.id
        base_id = identifier.split(_THOUGHT_SIGNATURE_SEPARATOR, 1)[0]
        entry: dict[str, PlainJson] = {
            "type": "tool_use",
            "id": base_id,
            "name": block.tool_use.name,
            "input": block.tool_use.arguments.value,
        }
        blocks.append(entry)  # nosemgrep: translation-no-mutation
    return blocks


def _usage(usage: ResponseUsage) -> dict[str, PlainJson]:
    result: dict[str, PlainJson] = {
        "input_tokens": usage.input_tokens + usage.cache_creation_input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if usage.cache_creation_input_tokens > 0:
        result = {
            **result,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        }
    if usage.cache_read_input_tokens > 0:
        result = {
            **result,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        }
    return result
