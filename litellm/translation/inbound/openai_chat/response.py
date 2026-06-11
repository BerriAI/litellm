"""IR ``ChatResponse`` -> OpenAI chat-completion response body.

Emits the plain dict the seam feeds into ``ModelResponse`` (which owns the
ambient envelope: chatcmpl id, created timestamp). Field shapes mirror what
v1's ``transform_parsed_response`` builds, including the usage detail
wrappers and the always-present ``provider_specific_fields`` keys.
"""

from __future__ import annotations

import json

from expression import Option
from expression.collections import Block

from ...deps import TranslationDeps
from ...ir import Body, ChatResponse, ContentBlock, PlainJson, ResponseUsage


def serialize_response(response: ChatResponse, deps: TranslationDeps) -> Body:
    text = "".join(block.text.text for block in response.content if block.tag == "text")
    tool_calls = _tool_calls(response.content)
    thinking_blocks = _thinking_blocks(response.content)
    reasoning: str | None = None
    if thinking_blocks is not None:
        reasoning = "".join(
            block.thinking.thinking
            for block in response.content
            if block.tag == "thinking"
        )
    message: dict[str, PlainJson]
    if response.synthesized_json_content:
        # v1's json-mode replacement is a bare Message(content=...): no
        # provider fields, no reasoning, no thinking blocks.
        message = {"role": "assistant", "content": text or None}
    else:
        message = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning,
            "thinking_blocks": thinking_blocks,
            "provider_specific_fields": {
                "citations": None,
                "thinking_blocks": thinking_blocks,
            },
        }
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
        "usage": _usage_json(response.usage, reasoning, deps),
    }


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
