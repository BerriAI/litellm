"""Validated Anthropic Messages wire messages -> IR messages.

The IR is Anthropic-shaped, so this is largely 1:1: ``Message.role`` is
user|assistant, system rides ``ChatRequest.system``, a tool_result is a
``tool_result`` content block inside a user message, and a tool_use is a block
inside an assistant message. Shapes the IR cannot carry (a non-text image
source, a non-text tool_result part, a non-text system block) return
``unsupported`` so the seam falls back to v1, never a silent drop
(researcher-6 §1.4: fail closed, do NOT replicate v1's silent non-text drop).

Hot path note (mirrors openai_chat/messages.py): internal helpers return
``value | TranslationError`` plain unions and build local lists, lifting into
``Result``/``Block`` once at the public boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ... import boundary
from ...errors import TranslationError
from ...ir import (
    Base64Source,
    CacheControl,
    ContentBlock,
    Image,
    ImageSource,
    JsonBlob,
    Message,
    RedactedThinking,
    Role,
    SystemText,
    Text,
    Thinking,
    ToolResult,
    ToolResultContent,
    ToolUse,
    UrlSource,
)
from .schema import (
    AssistantMessageIn,
    Base64SourceIn,
    CacheControlIn,
    ImageBlockIn,
    RedactedThinkingBlockIn,
    SystemTextBlockIn,
    TextBlockIn,
    ThinkingBlockIn,
    ToolResultBlockIn,
    ToolResultContentTextIn,
    ToolUseBlockIn,
    UrlSourceIn,
    UserMessageIn,
)

WireMessage = UserMessageIn | AssistantMessageIn

_ConvertResult = Result[Block[Message], TranslationError]

_NO_CACHE: Option[CacheControl] = Nothing


def convert_messages(items: Sequence[WireMessage]) -> _ConvertResult:
    messages: list[Message] = []
    for item in items:
        blocks = _message_blocks(item)
        if isinstance(blocks, TranslationError):
            return Error(blocks)
        # build-locally-freeze-once: `messages` never escapes this scope
        messages.append(  # nosemgrep: translation-no-mutation
            Message(role=_role(item), content=Block.of_seq(blocks))
        )
    return Ok(Block.of_seq(messages))


def convert_system(
    system: str | list[SystemTextBlockIn] | None,
) -> Result[Block[SystemText], TranslationError]:
    if system is None:
        return Ok(Block.empty())
    if isinstance(system, str):
        if not system:
            return Ok(Block.empty())
        return Ok(Block.of_seq([SystemText(text=system, cache=_NO_CACHE)]))
    texts = _system_texts(system)
    if isinstance(texts, TranslationError):
        return Error(texts)
    return Ok(Block.of_seq(texts))


def cache_of(value: CacheControlIn | None) -> Option[CacheControl]:
    if value is None:
        return _NO_CACHE
    ttl = Some(value.ttl) if value.ttl is not None else Nothing
    return Some(CacheControl(type=value.type, ttl=ttl))


def _role(message: WireMessage) -> Role:
    return "user" if isinstance(message, UserMessageIn) else "assistant"


def _system_texts(
    blocks: Iterable[SystemTextBlockIn],
) -> list[SystemText] | TranslationError:
    texts: list[SystemText] = []
    for block in blocks:
        if block.type != "text" or block.text is None:
            # v1 silently drops non-text system blocks; the IR's SystemText is
            # text-only, so fail closed instead of replicating the drop.
            return TranslationError.of_unsupported(
                "non-text system block; v1 drops it, the IR cannot carry it"
            )
        texts.append(  # nosemgrep: translation-no-mutation
            SystemText(text=block.text, cache=cache_of(block.cache_control))
        )
    return texts


def _message_blocks(
    message: WireMessage,
) -> list[ContentBlock] | TranslationError:
    if isinstance(message, UserMessageIn):
        return _user_blocks(message)
    return _assistant_blocks(message)


def _user_blocks(
    message: UserMessageIn,
) -> list[ContentBlock] | TranslationError:
    if isinstance(message.content, str):
        text = Text(text=message.content, cache=_NO_CACHE)
        return [ContentBlock.of_text(text)]
    blocks: list[ContentBlock] = []
    for part in message.content:
        block = _user_block(part)
        if isinstance(block, TranslationError):
            return block
        blocks.append(block)  # nosemgrep: translation-no-mutation
    return blocks


def _user_block(
    part: TextBlockIn | ImageBlockIn | ToolResultBlockIn,
) -> ContentBlock | TranslationError:
    if isinstance(part, TextBlockIn):
        return ContentBlock.of_text(
            Text(text=part.text, cache=cache_of(part.cache_control))
        )
    if isinstance(part, ImageBlockIn):
        return _image_block(part)
    return _tool_result(part)


def _image_block(part: ImageBlockIn) -> ContentBlock | TranslationError:
    cache = cache_of(part.cache_control)
    if isinstance(part.source, Base64SourceIn):
        source = ImageSource.of_base64(
            Base64Source(media_type=part.source.media_type, data=part.source.data)
        )
        return ContentBlock.of_image(Image(source=source, cache=cache))
    if isinstance(part.source, UrlSourceIn):
        source = ImageSource.of_url(UrlSource(url=part.source.url, format=Nothing))
        return ContentBlock.of_image(Image(source=source, cache=cache))
    return TranslationError.of_unsupported(
        "image source other than base64/url (e.g. file_id); v1 handles it"
    )


def _tool_result(part: ToolResultBlockIn) -> ContentBlock | TranslationError:
    content = _tool_result_content(part.content)
    if isinstance(content, TranslationError):
        return content
    return ContentBlock.of_tool_result(
        ToolResult(
            tool_use_id=part.tool_use_id,
            content=content,
            cache=cache_of(part.cache_control),
        )
    )


def _tool_result_content(
    content: str | list[ToolResultContentTextIn | object] | None,
) -> ToolResultContent | TranslationError:
    if content is None:
        return ToolResultContent.of_text("")
    if isinstance(content, str):
        return ToolResultContent.of_text(content)
    parts: list[Text] = []
    for item in content:
        if not isinstance(item, ToolResultContentTextIn):
            # image/document parts in a tool_result: v1 routes them via
            # image_url; the IR's ToolResult parts are text-only, fail closed.
            return TranslationError.of_unsupported(
                "non-text tool_result content (image/document); v1 handles it"
            )
        parts.append(  # nosemgrep: translation-no-mutation
            Text(text=item.text, cache=cache_of(item.cache_control))
        )
    return ToolResultContent.of_parts(Block.of_seq(parts))


def _assistant_blocks(
    message: AssistantMessageIn,
) -> list[ContentBlock] | TranslationError:
    if isinstance(message.content, str):
        text = Text(text=message.content, cache=_NO_CACHE)
        return [ContentBlock.of_text(text)]
    blocks: list[ContentBlock] = []
    for part in message.content:
        block = _assistant_block(part)
        if isinstance(block, TranslationError):
            return block
        blocks.append(block)  # nosemgrep: translation-no-mutation
    return blocks


def _assistant_block(
    part: TextBlockIn | ToolUseBlockIn | ThinkingBlockIn | RedactedThinkingBlockIn,
) -> ContentBlock | TranslationError:
    if isinstance(part, TextBlockIn):
        return ContentBlock.of_text(
            Text(text=part.text, cache=cache_of(part.cache_control))
        )
    if isinstance(part, ToolUseBlockIn):
        return _tool_use(part)
    if isinstance(part, ThinkingBlockIn):
        signature = Some(part.signature) if part.signature is not None else Nothing
        return ContentBlock.of_thinking(
            Thinking(
                thinking=part.thinking,
                signature=signature,
                cache=cache_of(part.cache_control),
            )
        )
    return ContentBlock.of_redacted_thinking(RedactedThinking(data=part.data))


def _tool_use(part: ToolUseBlockIn) -> ContentBlock | TranslationError:
    if part.caller is not None:
        return TranslationError.of_unsupported(
            "tool_use caller (code-execution tool call); v1 handles it"
        )
    match boundary.as_plain_json(part.input):
        case Result(tag="ok", ok=copied) if isinstance(copied, dict):
            arguments = JsonBlob(value=copied)
        case Result(tag="ok"):
            return TranslationError.of_unsupported(
                "non-object tool_use input; v1 handles it"
            )
        case Result(error=reason):
            return TranslationError.of_unsupported(f"tool_use input: {reason}")
    return ContentBlock.of_tool_use(
        ToolUse(
            id=part.id,
            name=part.name,
            arguments=arguments,
            cache=cache_of(part.cache_control),
        )
    )
