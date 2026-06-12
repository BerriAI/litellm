"""IR ``ChatResponse`` -> OpenAI chat-completion response body.

Emits the plain dict the seam feeds into ``ModelResponse`` (which owns the
ambient envelope: chatcmpl id, created timestamp). Field shapes mirror what
v1's per-provider ``transform_response`` builds, so the serializer carries a
``dialect``:

- ``anthropic`` (also bedrock_invoke, which delegates to the anthropic
  transform in v1): empty content is ``None``, every message key is always
  present (v1 passes them explicitly, ``None`` included), usage carries the
  ephemeral cache-creation detail and computes ``total_tokens``;
- ``bedrock_converse``: content is always a string (``""`` when empty),
  optional keys are OMITTED when unset (v1 builds the message dict
  conditionally), ``provider_specific_fields`` mirrors the raw
  ``reasoningContentBlocks``, and usage uses the wire ``totalTokens`` with no
  ephemeral detail.
"""

from __future__ import annotations

import json
from typing import Literal

from expression import Option
from expression.collections import Block

from ...deps import TranslationDeps
from ...ir import Body, ChatResponse, ContentBlock, PlainJson, ResponseUsage

ResponseDialect = Literal["anthropic", "bedrock_converse", "gemini"]


def serialize_response(
    response: ChatResponse,
    deps: TranslationDeps,
    dialect: ResponseDialect = "anthropic",
) -> Body:
    if dialect == "gemini":
        return _gemini_body(response)
    text = "".join(block.text.text for block in response.content if block.tag == "text")
    thinking_blocks = _thinking_blocks(response.content)
    reasoning: str | None = None
    if thinking_blocks is not None:
        reasoning = "".join(
            block.thinking.thinking
            for block in response.content
            if block.tag == "thinking"
        )
    message = (
        _anthropic_message(response, text, thinking_blocks, reasoning)
        if dialect == "anthropic"
        else _converse_message(response, text, thinking_blocks, reasoning)
    )
    usage = (
        _usage_json(response.usage, reasoning, deps)
        if dialect == "anthropic"
        else _converse_usage_json(response.usage, reasoning, deps)
    )
    return {
        "object": "chat.completion",
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": response.finish,
                "message": message,
            }
        ],
        "usage": usage,
    }


def _anthropic_message(
    response: ChatResponse,
    text: str,
    thinking_blocks: PlainJson,
    reasoning: str | None,
) -> dict[str, PlainJson]:
    if response.synthesized_json_content:
        # v1's json-mode replacement is a bare Message(content=...): no
        # provider fields, no reasoning, no thinking blocks.
        return {"role": "assistant", "content": text or None}
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": _tool_calls(response.content),
        "reasoning_content": reasoning,
        "thinking_blocks": thinking_blocks,
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": thinking_blocks,
        },
    }


def _converse_message(
    response: ChatResponse,
    text: str,
    thinking_blocks: PlainJson,
    reasoning: str | None,
) -> dict[str, PlainJson]:
    """v1's converse ``_transform_response`` builds the message dict key by
    key: content always a string, the reasoning trio only when reasoning
    blocks exist, ``tool_calls`` only when present."""
    message: dict[str, PlainJson] = {"role": "assistant", "content": text}
    if thinking_blocks is not None:
        message = {
            **message,
            "provider_specific_fields": {
                "reasoningContentBlocks": _raw_reasoning_blocks(response.content)
            },
            "reasoning_content": reasoning,
            "thinking_blocks": thinking_blocks,
        }
    tool_calls = _tool_calls(response.content)
    if tool_calls is not None:
        message = {**message, "tool_calls": tool_calls}
    return message


def _raw_reasoning_blocks(content: Block[ContentBlock]) -> PlainJson:
    blocks: list[PlainJson] = []
    for block in content:
        if block.tag == "thinking":
            text_block: dict[str, PlainJson] = {"text": block.thinking.thinking}
            match block.thinking.signature:
                case Option(tag="some", some=signature):
                    text_block = {**text_block, "signature": signature}
                case _:
                    pass
            blocks.append(  # nosemgrep: translation-no-mutation
                {"reasoningText": text_block}
            )
        elif block.tag == "redacted_thinking":
            blocks.append(  # nosemgrep: translation-no-mutation
                {"redactedContent": block.redacted_thinking.data}
            )
    return blocks


def _tool_calls(content: Block[ContentBlock]) -> PlainJson:
    calls: list[PlainJson] = [
        {
            "id": block.tool_use.id,
            "type": "function",
            "function": {
                "name": block.tool_use.name,
                "arguments": json.dumps(block.tool_use.arguments.value),
            },
            "index": index,
        }
        for index, block in enumerate(content)
        if block.tag == "tool_use"
    ]
    return calls or None


def _thinking_blocks(content: Block[ContentBlock]) -> PlainJson:
    blocks = [
        _thinking_block_json(block)
        for block in content
        if block.tag in ("thinking", "redacted_thinking")
    ]
    return blocks or None


def _thinking_block_json(block: ContentBlock) -> PlainJson:
    if block.tag == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.redacted_thinking.data}
    thinking = block.thinking
    base: dict[str, PlainJson] = {"type": "thinking", "thinking": thinking.thinking}
    match thinking.signature:
        case Option(tag="some", some=signature):
            return {**base, "signature": signature}
        case _:
            return base


_THOUGHT_SIGNATURE_SEPARATOR = "__thought__"


def _gemini_body(response: ChatResponse) -> Body:
    """Mirror v1 ``_process_candidates`` + ``_calculate_usage`` shapes: the
    message keys are set conditionally exactly like v1 (``thinking_blocks``
    and ``images`` are always-set lists, possibly empty), tool ids keep the
    empty-prefix sentinel for the seam to mint ``call_<uuid>``, and the usage
    JSON carries v1's exact wrapper-kwarg sets (the seam rebuilds ``Usage``
    from them)."""
    text = "".join(block.text.text for block in response.content if block.tag == "text")
    reasoning = (
        "".join(
            block.thinking.thinking
            for block in response.content
            if block.tag == "thinking"
        )
        or None
    )
    message: dict[str, PlainJson] = {
        "role": "assistant",
        "content": text or None,
        "thinking_blocks": _gemini_thinking_blocks(response.content),
        "images": [],
    }
    if reasoning is not None:
        message = {**message, "reasoning_content": reasoning}
    tool_calls = _gemini_tool_calls(response.content)
    if tool_calls:
        message = {**message, "tool_calls": tool_calls}
    signatures = _thought_signatures(response.content)
    if signatures:
        message = {
            **message,
            "provider_specific_fields": {"thought_signatures": signatures},
        }
    body: Body = {
        "object": "chat.completion",
        "model": response.model,
        "choices": [
            {"index": 0, "finish_reason": response.finish, "message": message}
        ],
        "usage": _gemini_usage_json(response.usage),
    }
    if response.id:
        body = {**body, "id": response.id}
    return body


def _gemini_thinking_blocks(content: Block[ContentBlock]) -> PlainJson:
    blocks: list[PlainJson] = []
    for block in content:
        if block.tag != "thinking":
            continue
        entry: dict[str, PlainJson] = {
            "type": "thinking",
            "thinking": block.thinking.thinking,
        }
        match block.thinking.signature:
            case Option(tag="some", some=signature):
                entry = {**entry, "signature": signature}
            case _:
                pass
        blocks.append(entry)  # nosemgrep: translation-no-mutation
    return blocks


def _gemini_tool_calls(content: Block[ContentBlock]) -> list[PlainJson]:
    calls: list[PlainJson] = []
    index = 0
    for block in content:
        if block.tag != "tool_use":
            continue
        identifier = block.tool_use.id
        signature = identifier.partition(_THOUGHT_SIGNATURE_SEPARATOR)[2]
        entry: dict[str, PlainJson] = {
            "id": identifier,
            "type": "function",
            "function": {
                "name": block.tool_use.name,
                "arguments": json.dumps(
                    block.tool_use.arguments.value, ensure_ascii=False
                ),
            },
            "index": index,
        }
        if signature:
            entry = {
                **entry,
                "provider_specific_fields": {"thought_signature": signature},
            }
        calls.append(entry)  # nosemgrep: translation-no-mutation
        index = index + 1
    return calls


def _thought_signatures(content: Block[ContentBlock]) -> list[PlainJson]:
    signatures: list[PlainJson] = []
    for block in content:
        if block.tag == "thinking":
            match block.thinking.signature:
                case Option(tag="some", some=signature):
                    signatures.append(  # nosemgrep: translation-no-mutation
                        signature
                    )
                case _:
                    pass
        elif block.tag == "tool_use":
            suffix = block.tool_use.id.partition(_THOUGHT_SIGNATURE_SEPARATOR)[2]
            if suffix:
                signatures.append(suffix)  # nosemgrep: translation-no-mutation
    return signatures


def _gemini_usage_json(usage: ResponseUsage) -> PlainJson:
    """v1 ``_calculate_usage`` output, with set-only completion detail keys
    (the wrapper only serializes explicitly assigned fields)."""
    cached: PlainJson = (
        usage.cache_read_input_tokens if usage.cache_read_reported else None
    )
    prompt_details: dict[str, PlainJson] = {
        "cached_tokens": cached,
        "audio_tokens": None,
        "text_tokens": None,
        "image_tokens": None,
        "video_tokens": None,
    }
    match usage.prompt_modalities:
        case Option(tag="some", some=prompt_tokens):
            prompt_details = {
                "cached_tokens": cached,
                "audio_tokens": prompt_tokens.audio.default_value(None),
                "text_tokens": prompt_tokens.text.default_value(None),
                "image_tokens": prompt_tokens.image.default_value(None),
                "video_tokens": prompt_tokens.video.default_value(None),
            }
        case _:
            pass
    completion_details: dict[str, PlainJson] = {}
    match usage.completion_modalities:
        case Option(tag="some", some=completion_tokens):
            for key, option in (
                ("text_tokens", completion_tokens.text),
                ("audio_tokens", completion_tokens.audio),
                ("image_tokens", completion_tokens.image),
                ("video_tokens", completion_tokens.video),
            ):
                match option:
                    case Option(tag="some", some=value):
                        completion_details = {**completion_details, key: value}
                    case _:
                        pass
        case _:
            pass
    reasoning_tokens = usage.reasoning_tokens.default_value(None)
    if reasoning_tokens is not None:
        completion_details = {
            **completion_details,
            "reasoning_tokens": reasoning_tokens,
        }
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens.default_value(
            usage.input_tokens + usage.output_tokens
        ),
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": completion_details,
        "cache_read_input_tokens": cached,
        "reasoning_tokens": reasoning_tokens,
    }


def _converse_usage_json(
    usage: ResponseUsage, reasoning: str | None, deps: TranslationDeps
) -> PlainJson:
    """v1 converse ``_transform_usage``: cache tokens fold into prompt_tokens,
    the wire ``totalTokens`` rides verbatim, reasoning tokens estimate is
    uncapped, and there is no ephemeral cache-creation detail."""
    prompt_tokens = (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    completion_tokens = usage.output_tokens
    reasoning_tokens = deps.count_response_tokens(reasoning) if reasoning else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.total_tokens.default_value(
            prompt_tokens + completion_tokens
        ),
        "prompt_tokens_details": {
            "cached_tokens": usage.cache_read_input_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "text_tokens": usage.input_tokens,
        },
        "completion_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
            "text_tokens": (
                completion_tokens - reasoning_tokens
                if reasoning_tokens > 0
                else completion_tokens
            ),
        },
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
    }


def _usage_json(
    usage: ResponseUsage, reasoning: str | None, deps: TranslationDeps
) -> PlainJson:
    """v1 ``calculate_usage``: cache tokens fold into prompt_tokens, the raw
    input count lands in details.text_tokens, and reasoning tokens are
    estimated by the injected token counter, capped at completion tokens."""
    prompt_tokens = (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    completion_tokens = usage.output_tokens
    estimated = deps.count_response_tokens(reasoning) if reasoning else 0
    reasoning_tokens = min(estimated, completion_tokens)
    creation_details: PlainJson = None
    match usage.cache_creation:
        case Option(tag="some", some=details):
            creation_details = {
                "ephemeral_5m_input_tokens": details.five_minute.default_value(None),
                "ephemeral_1h_input_tokens": details.one_hour.default_value(None),
            }
        case _:
            creation_details = None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": usage.cache_read_input_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "cache_creation_token_details": creation_details,
            "text_tokens": usage.input_tokens,
        },
        "completion_tokens_details": {
            "reasoning_tokens": max(0, reasoning_tokens),
            "text_tokens": (
                completion_tokens - reasoning_tokens
                if reasoning_tokens > 0
                else completion_tokens
            ),
        },
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
    }
