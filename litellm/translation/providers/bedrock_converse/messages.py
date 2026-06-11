"""IR messages -> Converse message dicts, reproducing v1's quirks exactly:

- assistant text that is empty/whitespace-only is silently dropped (v1 skips
  blank blocks); an assistant turn left with no blocks vanishes and the
  surrounding same-role turns merge, exactly like v1's alternation loop;
- user/system empty text is ambiguous in v1 (string content is dropped, list
  parts are kept) and the IR cannot tell the forms apart, so it is
  ``unsupported`` -> the seam falls back to v1;
- tool names rewrite through ``make_valid_bedrock_name``; tool_use/tool_result
  ids are NOT sanitized (unlike anthropic); toolUse dedupes per merged
  assistant turn and toolResult per consecutive run;
- assistant blocks sort reasoningContent < other < toolUse/cachePoint (v1
  ``_sort_bedrock_assistant_content_blocks``); no final-assistant rstrip;
- a block's ``cache_control`` becomes a following ``cachePoint`` (never a
  ttl: the factory passes no model for message content);
- thinking history without a signature becomes plain text (v1
  ``add_thinking_blocks_to_assistant_content``).
"""

from __future__ import annotations

from expression import Option
from expression.collections import Block
from typing_extensions import assert_never

from ...errors import TranslationError
from ...ir import (
    CacheControl,
    ContentBlock,
    Image,
    Message,
    PlainJson,
    Role,
    ToolResult,
    ToolUse,
)
from .tools import content_cache_point, make_valid_bedrock_name

_Turn = tuple[Role, list[PlainJson]]


def serialize_messages(
    messages: Block[Message],
) -> list[PlainJson] | TranslationError:
    turns: list[_Turn] = []
    for message in messages:
        blocks = _content_blocks(message)
        if isinstance(blocks, TranslationError):
            return blocks
        if message.role == "assistant":
            blocks = _sort_assistant(_dedupe("toolUse", blocks))
        if not blocks:
            continue
        if turns and turns[-1][0] == message.role:
            # v1 merges the roles left adjacent after a blank turn vanished.
            role, previous = turns[-1]
            merged: _Turn = (role, [*previous, *blocks])
            turns[-1] = merged  # nosemgrep: translation-no-mutation
            continue
        turns.append((message.role, blocks))  # nosemgrep: translation-no-mutation
    return [{"role": role, "content": blocks} for role, blocks in turns]


def _content_blocks(message: Message) -> list[PlainJson] | TranslationError:
    blocks: list[PlainJson] = []
    run: list[PlainJson] = []  # consecutive toolResult run, deduped as a unit
    for block in message.content:
        if block.tag == "tool_result":
            run.extend(  # nosemgrep: translation-no-mutation
                _tool_result_json(block.tool_result)
            )
            continue
        if run:
            deduped = _dedupe("toolResult", run)
            blocks.extend(deduped)  # nosemgrep: translation-no-mutation
            run = []
        rendered = _block_json(block, message.role)
        if isinstance(rendered, TranslationError):
            return rendered
        blocks.extend(rendered)  # nosemgrep: translation-no-mutation
    if run:
        blocks.extend(_dedupe("toolResult", run))  # nosemgrep: translation-no-mutation
    return blocks


def _block_json(block: ContentBlock, role: Role) -> list[PlainJson] | TranslationError:
    match block.tag:
        case "text":
            return _text_json(block.text.text, block.text.cache, role)
        case "image":
            return _image_json(block.image)
        case "tool_use":
            return _tool_use_json(block.tool_use)
        case "thinking":
            thinking = block.thinking
            match thinking.signature:
                case Option(tag="some", some=signature):
                    text: dict[str, PlainJson] = {
                        "text": thinking.thinking,
                        "signature": signature,
                    }
                    return [{"reasoningContent": {"reasoningText": text}}]
                case _:
                    return _unsigned_thinking_json(thinking.thinking)
        case "redacted_thinking":
            return [
                {"reasoningContent": {"redactedContent": block.redacted_thinking.data}}
            ]
        case "tool_result":
            return []  # consumed by the run accumulator in _content_blocks
    assert_never(block.tag)


def _unsigned_thinking_json(thinking: str) -> list[PlainJson]:
    if not thinking.strip():
        return []
    return [{"text": thinking}]


def _text_json(
    text: str, cache: Option[CacheControl], role: Role
) -> list[PlainJson] | TranslationError:
    if role == "assistant":
        if not text.strip():
            return []
    elif not text:
        return TranslationError.of_unsupported(
            "empty text content is content-form-dependent on converse; v1 handles it"
        )
    point = content_cache_point(cache)
    base: PlainJson = {"text": text}
    return [base, point] if point is not None else [base]


def _image_json(image: Image) -> list[PlainJson] | TranslationError:
    if image.source.tag != "base64":
        return TranslationError.of_unsupported(
            "URL image sources require v1's download path on bedrock"
        )
    source = image.source.base64
    _, _, subtype = source.media_type.partition("/")
    block: PlainJson = {
        "image": {
            "format": subtype or source.media_type,
            "source": {"bytes": source.data},
        }
    }
    point = content_cache_point(image.cache)
    return [block, point] if point is not None else [block]


def _tool_use_json(tool_use: ToolUse) -> list[PlainJson]:
    arguments = tool_use.arguments.value
    block: PlainJson = {
        "toolUse": {
            "input": arguments if isinstance(arguments, dict) else {},
            "name": make_valid_bedrock_name(tool_use.name),
            "toolUseId": tool_use.id,
        }
    }
    point = content_cache_point(tool_use.cache)
    return [block, point] if point is not None else [block]


def _tool_result_json(tool_result: ToolResult) -> list[PlainJson]:
    content: list[PlainJson]
    cached = tool_result.cache.is_some()
    if tool_result.content.tag == "text":
        content = [{"text": tool_result.content.text}]
    else:
        content = [{"text": part.text} for part in tool_result.content.parts]
        cached = cached or any(
            part.cache.is_some() for part in tool_result.content.parts
        )
    block: PlainJson = {
        "toolResult": {"content": content, "toolUseId": tool_result.tool_use_id}
    }
    if cached:
        return [block, {"cachePoint": {"type": "default"}}]
    return [block]


def _dedupe(key: str, blocks: list[PlainJson]) -> list[PlainJson]:
    """v1 ``_deduplicate_bedrock_content_blocks``: first id wins."""
    seen: set[str] = set()
    kept: list[PlainJson] = []
    for block in blocks:
        keyed = block.get(key) if isinstance(block, dict) else None
        identifier = keyed.get("toolUseId") if isinstance(keyed, dict) else None
        if isinstance(identifier, str) and identifier:
            if identifier in seen:
                continue
            seen.add(identifier)  # nosemgrep: translation-no-mutation
        kept.append(block)  # nosemgrep: translation-no-mutation
    return kept


def _sort_assistant(blocks: list[PlainJson]) -> list[PlainJson]:
    def _key(block: PlainJson) -> int:
        if isinstance(block, dict):
            if "reasoningContent" in block:
                return 0
            if "toolUse" in block or "cachePoint" in block:
                return 2
        return 1

    return sorted(blocks, key=_key)
