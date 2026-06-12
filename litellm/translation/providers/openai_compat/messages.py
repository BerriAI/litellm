"""IR messages -> OpenAI chat wire messages.

The inverse of ``inbound/openai_chat/messages.py`` for the shapes the raw
guard admits: system texts re-emit as leading system messages (1:1, because
list-form system content is guarded), tool_result blocks split back into
``tool`` role messages at their original positions, a lone text block
re-emits as string content (the guarded set makes that unambiguous), and
``cache_control`` is dropped everywhere exactly like v1's transform. Thinking
blocks have no OpenAI wire form and return ``unsupported``.
"""

from __future__ import annotations

import json

from typing_extensions import assert_never

from ...errors import TranslationError
from ...ir import ChatRequest, Image, Message, PlainJson, ToolResult


def serialize_messages(request: ChatRequest) -> list[PlainJson] | TranslationError:
    out: list[PlainJson] = [
        {"role": "system", "content": system.text} for system in request.system
    ]
    for message in request.messages:
        emitted = (
            _assistant_message(message)
            if message.role == "assistant"
            else _user_messages(message)
        )
        if isinstance(emitted, TranslationError):
            return emitted
        out.extend(emitted)  # nosemgrep: translation-no-mutation
    return out


def _user_messages(message: Message) -> list[PlainJson] | TranslationError:
    out: list[PlainJson] = []
    parts: list[PlainJson] = []
    image_count = 0
    for block in message.content:
        if block.tag == "tool_result":
            _flush_user(out, parts, image_count)
            parts = []
            image_count = 0
            out.append(  # nosemgrep: translation-no-mutation
                _tool_message(block.tool_result)
            )
        elif block.tag == "text":
            parts.append(  # nosemgrep: translation-no-mutation
                {"type": "text", "text": block.text.text}
            )
        elif block.tag == "image":
            parts.append(_image_part(block.image))  # nosemgrep: translation-no-mutation
            image_count = image_count + 1
        else:
            return TranslationError.of_unsupported(
                f"{block.tag} block in a user turn has no OpenAI wire form; v1 handles it"
            )
    _flush_user(out, parts, image_count)
    return out


def _flush_user(out: list[PlainJson], parts: list[PlainJson], image_count: int) -> None:
    if not parts:
        return
    if len(parts) == 1 and image_count == 0:
        only = parts[0]
        text = only.get("text") if isinstance(only, dict) else None
        out.append(  # nosemgrep: translation-no-mutation
            {"role": "user", "content": text}
        )
        return
    out.append({"role": "user", "content": parts})  # nosemgrep: translation-no-mutation


def _image_part(image: Image) -> PlainJson:
    source = image.source
    match source.tag:
        case "base64":
            url = f"data:{source.base64.media_type};base64,{source.base64.data}"
        case "url":
            url = source.url.url
        case _:
            assert_never(source.tag)
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_message(result: ToolResult) -> PlainJson:
    content: PlainJson
    match result.content.tag:
        case "text":
            content = result.content.text
        case "parts":
            content = [
                {"type": "text", "text": part.text} for part in result.content.parts
            ]
        case _:
            assert_never(result.content.tag)
    return {"role": "tool", "tool_call_id": result.tool_use_id, "content": content}


def _assistant_message(message: Message) -> list[PlainJson] | TranslationError:
    texts: list[str] = []
    tool_calls: list[PlainJson] = []
    for block in message.content:
        if block.tag == "text":
            texts.append(block.text.text)  # nosemgrep: translation-no-mutation
        elif block.tag == "tool_use":
            tool_calls.append(  # nosemgrep: translation-no-mutation
                {
                    "id": block.tool_use.id,
                    "type": "function",
                    "function": {
                        "name": block.tool_use.name,
                        "arguments": json.dumps(block.tool_use.arguments.value),
                    },
                }
            )
        else:
            return TranslationError.of_unsupported(
                f"assistant {block.tag} block has no OpenAI wire form; v1 forwards it"
            )
    content: PlainJson
    if not texts:
        content = None
    elif len(texts) == 1:
        content = texts[0]
    else:
        content = [{"type": "text", "text": text} for text in texts]
    base: dict[str, PlainJson] = {"role": "assistant", "content": content}
    if tool_calls:
        return [{**base, "tool_calls": tool_calls}]
    return [base]
