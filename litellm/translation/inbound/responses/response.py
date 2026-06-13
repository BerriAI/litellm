"""IR ``ChatResponse`` -> OpenAI Responses API response body.

The reverse of the request parse: an IR ``ChatResponse`` (produced by any
provider's ``parse_response``) becomes a ``ResponsesAPIResponse`` body,
reproducing v1's ``transform_chat_completion_response_to_responses_api_response``
(transformation.py:1648). The Responses response shape is provider-independent,
so the ``dialect`` argument the pipeline threads is ignored here.

Output items in v1's order (researcher-6 §2.2): reasoning items first (from
``Thinking`` blocks, id ``rs_{hash(reasoning)}``), then a message item (from
``Text`` blocks), then function_call items (from ``ToolUse`` blocks). v1 reads
the echoed top-level request params off the chat ``ModelResponse`` via
``getattr`` with defaults; the IR ``ChatResponse`` carries none of them, so the
serializer emits the same defaults (``temperature=0.0``, ``tool_choice="auto"``,
``parallel_tool_calls=False``, ``tools=[]``, ``top_p``/``max_output_tokens``
null). The reasoning-item id is ``rs_{hash(str(reasoning))}`` where ``reasoning``
is the thinking blocks joined exactly as the chat dialect builds
``reasoning_content`` (so it matches v1 within a process, where ``hash`` of a
str is salted identically).

``created_at`` is ``None`` here: the IR ``ChatResponse`` carries no timestamp
(it is ambient envelope the seam owns, exactly as the chat-completion reverse
leaves id/created to the ``ModelResponse`` envelope), so the seam stamps it.

``status`` maps stop/tool_calls -> completed, length/content_filter ->
incomplete (v1's ``_map_chat_completion_finish_reason_to_responses_status``).
Usage maps onto ``ResponseAPIUsage``: input/output/total verbatim, plus
``input_tokens_details.cached_tokens`` (with audio/text null) when cache read is
reported and ``output_tokens_details.reasoning_tokens`` (with text null) when
the IR carries reasoning tokens. Annotations/citations, image_generation_call
and code_interpreter_call output items are IR-GAPs the reverse cannot produce
(they only arise from provider fields the IR does not carry); they are omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from expression import Option
from expression.collections import Block

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    Body,
    ChatResponse,
    ContentBlock,
    FinishReason,
    PlainJson,
    ResponseUsage,
)

_STATUS: Mapping[FinishReason, str] = MappingProxyType(
    {
        "stop": "completed",
        "tool_calls": "completed",
        "length": "incomplete",
        "content_filter": "incomplete",
    }
)


def serialize_response(
    response: ChatResponse,
    deps: TranslationDeps,
    dialect: object = None,
) -> Body | TranslationError:
    return {
        "id": response.id,
        "created_at": None,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": response.model,
        "object": "response",
        "output": _output(response),
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": None,
        "status": _STATUS[response.finish],
        "store": None,
        "temperature": 0.0,
        "text": {},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": None,
        "usage": _usage(response.usage),
        "user": None,
    }


def _output(response: ChatResponse) -> list[PlainJson]:
    items: list[PlainJson] = []
    reasoning = _reasoning_text(response.content)
    if reasoning is not None:
        items.append(  # nosemgrep: translation-no-mutation
            _reasoning_item(reasoning, response)
        )
    items.append(_message_item(response))  # nosemgrep: translation-no-mutation
    items.extend(
        _function_call_items(response.content)
    )  # nosemgrep: translation-no-mutation
    return items


def _reasoning_text(content: Block[ContentBlock]) -> str | None:
    thinking = "".join(
        block.thinking.thinking for block in content if block.tag == "thinking"
    )
    return thinking or None


def _reasoning_item(reasoning: str, response: ChatResponse) -> PlainJson:
    return {
        "type": "reasoning",
        "id": f"rs_{hash(str(reasoning))}",
        "status": _STATUS[response.finish],
        "role": "assistant",
        "phase": None,
        "content": [{"type": "output_text", "text": reasoning, "annotations": []}],
    }


def _message_item(response: ChatResponse) -> PlainJson:
    text = "".join(block.text.text for block in response.content if block.tag == "text")
    # The chat message content is `text or None` (v1's anthropic dialect), and
    # v1's OutputText.text copies it verbatim, so a tool-only / empty turn emits
    # null text rather than "".
    return {
        "type": "message",
        "id": response.id,
        "status": _STATUS[response.finish],
        "role": "assistant",
        "phase": None,
        "content": [{"type": "output_text", "text": text or None, "annotations": []}],
    }


def _function_call_items(content: Block[ContentBlock]) -> list[PlainJson]:
    items: list[PlainJson] = []
    for block in content:
        if block.tag != "tool_use":
            continue
        items.append(  # nosemgrep: translation-no-mutation
            {
                "type": "function_call",
                "name": block.tool_use.name,
                "arguments": _arguments(block),
                "call_id": block.tool_use.id,
                "id": block.tool_use.id,
                "namespace": None,
                "status": "completed",
            }
        )
    return items


def _arguments(block: ContentBlock) -> str:
    import json

    match block.tool_use.arguments_raw:
        case Option(tag="some", some=raw):
            return raw
        case _:
            return json.dumps(block.tool_use.arguments.value)


def _usage(usage: ResponseUsage) -> dict[str, PlainJson]:
    # v1's responses reverse copies the chat ModelResponse usage verbatim:
    # input_tokens is the chat prompt_tokens (cache folded in), and the token
    # details mirror the chat dialect's. Fold cache the way every chat dialect
    # does so the responses usage matches regardless of upstream provider.
    prompt_tokens = (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    total = usage.total_tokens.default_value(prompt_tokens + usage.output_tokens)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": total,
        "cost": None,
        "input_tokens_details": _input_details(usage),
        "output_tokens_details": _output_details(usage),
    }


def _input_details(usage: ResponseUsage) -> PlainJson:
    return {
        "cached_tokens": usage.cache_read_input_tokens,
        "audio_tokens": None,
        "text_tokens": usage.input_tokens,
    }


def _output_details(usage: ResponseUsage) -> PlainJson:
    reasoning_tokens = usage.reasoning_tokens.default_value(0)
    text_tokens = (
        usage.output_tokens - reasoning_tokens
        if reasoning_tokens > 0
        else usage.output_tokens
    )
    return {"reasoning_tokens": reasoning_tokens, "text_tokens": text_tokens}
