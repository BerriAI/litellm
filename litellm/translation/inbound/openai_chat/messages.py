"""Validated OpenAI wire messages -> IR messages.

Pure conversion over already-validated wire models. Semantic shapes v1 serves
through unported paths (http:// images v1 downloads, legacy ``function_call``,
server-tool reconstruction, malformed-JSON argument repair) return
``unsupported`` so the seam falls back to v1; everything else converts
totally. Consecutive same-role turns merge into one IR message (``tool`` rides
with ``user``), matching every Anthropic-family wire format.

Hot path note: this module runs once per message of a history that routinely
reaches hundreds of turns, so internal helpers return ``value |
TranslationError`` plain unions and build local lists, lifting into
``Result``/``Block`` once at the public boundary (the pattern-auditor perf
budget is v2 <= 1.5x v1 at 600-message histories; per-part ``Ok``/``Block``
allocation alone blew it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from itertools import chain, groupby

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    Base64Source,
    CacheControl,
    ContentBlock,
    Image,
    ImageSource,
    JsonBlob,
    Message,
    PlainJson,
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
    CacheControlIn,
    ImagePartIn,
    ImageUrlIn,
    RedactedThinkingPartIn,
    SystemMessageIn,
    TextPartIn,
    ThinkingPartIn,
    ToolCallIn,
    ToolMessageIn,
    UserMessageIn,
)

WireMessage = SystemMessageIn | UserMessageIn | AssistantMessageIn | ToolMessageIn

_ConvertResult = Result[tuple[Block[SystemText], Block[Message]], TranslationError]

_Turn = tuple[Role, list[ContentBlock]]

_NO_CACHE: Option[CacheControl] = Nothing


def convert_messages(items: Sequence[WireMessage]) -> _ConvertResult:
    systems = Block.of_seq(
        chain.from_iterable(
            _system_texts(item) for item in items if isinstance(item, SystemMessageIn)
        )
    )
    turns: list[_Turn] = []
    for item in items:
        turn = _convert_turn(item)
        if isinstance(turn, TranslationError):
            return Error(turn)
        if turn is not None:
            # build-locally-freeze-once: `turns` never escapes this scope
            turns.append(turn)  # nosemgrep: translation-no-mutation
    return Ok((systems, _merge_adjacent(turns)))


def cache_of(value: CacheControlIn | None) -> Option[CacheControl]:
    if value is None:
        return _NO_CACHE
    ttl = Some(value.ttl) if value.ttl is not None else Nothing
    return Some(CacheControl(type=value.type, ttl=ttl))


def _system_texts(message: SystemMessageIn) -> Iterable[SystemText]:
    if isinstance(message.content, str):
        if not message.content:
            return ()
        return (
            SystemText(text=message.content, cache=cache_of(message.cache_control)),
        )
    if message.content is None:
        return ()
    return tuple(
        SystemText(text=part.text, cache=cache_of(part.cache_control))
        for part in message.content
        if part.text
    )


def _convert_turn(message: WireMessage) -> _Turn | TranslationError | None:
    if isinstance(message, UserMessageIn):
        blocks = _user_blocks(message)
        if isinstance(blocks, TranslationError):
            return blocks
        return ("user", blocks)
    if isinstance(message, AssistantMessageIn):
        blocks = _assistant_blocks(message)
        if isinstance(blocks, TranslationError):
            return blocks
        return ("assistant", blocks)
    if isinstance(message, ToolMessageIn):
        return ("user", [_tool_result(message)])
    return None  # system: handled by _system_texts


def _user_blocks(
    message: UserMessageIn,
) -> list[ContentBlock] | TranslationError:
    if message.content is None:
        return []
    if isinstance(message.content, str):
        text = Text(text=message.content, cache=cache_of(message.cache_control))
        return [ContentBlock.of_text(text)]
    blocks: list[ContentBlock] = []
    for part in message.content:
        if isinstance(part, TextPartIn):
            block = ContentBlock.of_text(
                Text(text=part.text, cache=cache_of(part.cache_control))
            )
        else:
            image = _image_block(part)
            if isinstance(image, TranslationError):
                return image
            block = image
        blocks.append(block)  # nosemgrep: translation-no-mutation
    return blocks


def _image_block(part: ImagePartIn) -> ContentBlock | TranslationError:
    url = (
        part.image_url.url if isinstance(part.image_url, ImageUrlIn) else part.image_url
    )
    fmt = part.image_url.format if isinstance(part.image_url, ImageUrlIn) else None
    cache = cache_of(part.cache_control)
    if url.startswith("https://"):
        fmt_option = Some(fmt) if fmt is not None else Nothing
        return ContentBlock.of_image(
            Image(
                source=ImageSource.of_url(UrlSource(url=url, format=fmt_option)),
                cache=cache,
            )
        )
    if url.startswith("http://"):
        return TranslationError.of_unsupported(
            "http:// image URLs require a download; v1 handles them"
        )
    source = _base64_source(url, fmt)
    if isinstance(source, TranslationError):
        return source
    return ContentBlock.of_image(Image(source=source, cache=cache))


def _base64_source(url: str, fmt: str | None) -> ImageSource | TranslationError:
    prefix, separator, data = url.partition(";base64,")
    if not separator or not prefix.startswith("data:"):
        return TranslationError.of_boundary(
            BoundaryError.of(
                Block.of_seq(["image_url is not a data:<media>;base64,<data> URI"])
            )
        )
    media_type = fmt if fmt else prefix[len("data:") :].replace("\\/", "/")
    return ImageSource.of_base64(Base64Source(media_type=media_type, data=data))


def _tool_result(message: ToolMessageIn) -> ContentBlock:
    if isinstance(message.content, str):
        content = ToolResultContent.of_text(message.content)
    elif message.content is None:
        content = ToolResultContent.of_text("")
    else:
        content = ToolResultContent.of_parts(
            Block.of_seq(
                Text(text=part.text, cache=cache_of(part.cache_control))
                for part in message.content
            )
        )
    return ContentBlock.of_tool_result(
        ToolResult(
            tool_use_id=message.tool_call_id,
            content=content,
            cache=cache_of(message.cache_control),
        )
    )


def _assistant_blocks(
    message: AssistantMessageIn,
) -> list[ContentBlock] | TranslationError:
    if message.function_call is not None:
        return TranslationError.of_unsupported(
            "legacy assistant.function_call; v1 handles it"
        )
    if message.provider_specific_fields:
        return TranslationError.of_unsupported(
            "assistant.provider_specific_fields (server-tool payloads); v1 handles them"
        )
    tool_uses = _tool_use_blocks(message.tool_calls)
    if isinstance(tool_uses, TranslationError):
        return tool_uses
    return [*_thinking_blocks(message), *_assistant_content(message), *tool_uses]


def _thinking_blocks(message: AssistantMessageIn) -> list[ContentBlock]:
    """v1 prepends ``thinking_blocks`` verbatim, unless the content list
    already carries inline thinking parts (then the list order wins)."""
    if message.thinking_blocks is None:
        return []
    inline_thinking = isinstance(message.content, list) and any(
        isinstance(part, (ThinkingPartIn, RedactedThinkingPartIn))
        for part in message.content
    )
    if inline_thinking:
        return []
    return [_thinking_ir(part) for part in message.thinking_blocks]


def _thinking_ir(part: ThinkingPartIn | RedactedThinkingPartIn) -> ContentBlock:
    if isinstance(part, RedactedThinkingPartIn):
        return ContentBlock.of_redacted_thinking(RedactedThinking(data=part.data))
    signature = Some(part.signature) if part.signature is not None else Nothing
    return ContentBlock.of_thinking(
        Thinking(
            thinking=part.thinking,
            signature=signature,
            cache=cache_of(part.cache_control),
        )
    )


def _assistant_content(message: AssistantMessageIn) -> list[ContentBlock]:
    if isinstance(message.content, str):
        # An empty string is kept: v1 rewrites it to the placeholder text
        # block (the anthropic serializer owns that rewrite).
        text = Text(text=message.content, cache=cache_of(message.cache_control))
        return [ContentBlock.of_text(text)]
    if message.content is None:
        return []
    return [
        _assistant_part_ir(part)
        for part in message.content
        if _keeps_assistant_part(part)
    ]


def _keeps_assistant_part(
    part: TextPartIn | ThinkingPartIn | RedactedThinkingPartIn,
) -> bool:
    """v1 keeps non-empty text and non-empty inline thinking; it drops inline
    redacted_thinking parts (only ``thinking_blocks`` carries those through)."""
    if isinstance(part, TextPartIn):
        return True
    if isinstance(part, ThinkingPartIn):
        return len(part.thinking) > 0
    return False


def _assistant_part_ir(
    part: TextPartIn | ThinkingPartIn | RedactedThinkingPartIn,
) -> ContentBlock:
    if isinstance(part, TextPartIn):
        return ContentBlock.of_text(
            Text(text=part.text, cache=cache_of(part.cache_control))
        )
    return _thinking_ir(part)


def _tool_use_blocks(
    tool_calls: list[ToolCallIn] | None,
) -> list[ContentBlock] | TranslationError:
    if tool_calls is None:
        return []
    blocks: list[ContentBlock] = []
    for call in tool_calls:
        if call.type != "function":
            continue  # v1 silently skips non-function tool calls
        block = _tool_use(call)
        if isinstance(block, TranslationError):
            return block
        blocks.append(block)  # nosemgrep: translation-no-mutation
    return blocks


def _tool_use(call: ToolCallIn) -> ContentBlock | TranslationError:
    function = call.function
    if function is None:
        return TranslationError.of_boundary(
            BoundaryError.of(
                Block.of_seq([f"tool_call {call.id!r} is missing 'function'"])
            )
        )
    if call.id.startswith("srvtoolu_"):
        return TranslationError.of_unsupported(
            "server-tool call reconstruction (srvtoolu_*); v1 handles it"
        )
    arguments = _arguments_of(function.arguments)
    if isinstance(arguments, TranslationError):
        return arguments
    return ContentBlock.of_tool_use(
        ToolUse(
            id=call.id,
            name=function.name,
            arguments=arguments,
            cache=cache_of(call.cache_control),
            arguments_raw=(
                Some(function.arguments)
                if isinstance(function.arguments, str)
                else Nothing
            ),
        )
    )


def _arguments_of(raw: str | None) -> JsonBlob | TranslationError:
    if raw is None or not raw.strip():
        return JsonBlob(value={})
    try:
        value: PlainJson = json.loads(raw)
    except ValueError:
        return TranslationError.of_unsupported(
            "malformed tool_call arguments need v1's JSON repair"
        )
    return JsonBlob(value=value)


def _merge_adjacent(turns: Sequence[_Turn]) -> Block[Message]:
    """O(n) merge of consecutive same-role turns (the tracer's fold copied the
    accumulator per step: O(n^2), 10x v1 latency at 600 messages)."""

    def _role(turn: _Turn) -> Role:
        return turn[0]

    merged = (
        Message(
            role=role,
            content=Block.of_seq(chain.from_iterable(blocks for _, blocks in group)),
        )
        for role, group in groupby(turns, key=_role)
    )
    # v1 only emits a merged turn when it produced content, so an empty turn
    # (e.g. a content-less user message) splits adjacency the same way here.
    return Block.of_seq(message for message in merged if len(message.content) > 0)
