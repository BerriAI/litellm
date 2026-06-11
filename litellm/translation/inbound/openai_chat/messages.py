"""Validated OpenAI wire messages -> IR messages.

Pure conversion over already-validated wire models. Semantic shapes v1 serves
through unported paths (http:// images v1 downloads, legacy ``function_call``,
server-tool reconstruction, malformed-JSON argument repair) return
``unsupported`` so the seam falls back to v1; everything else converts
totally. Consecutive same-role turns merge into one IR message (``tool`` rides
with ``user``), matching every Anthropic-family wire format.
"""

from __future__ import annotations

import json
from itertools import chain, groupby
from typing import Iterable, List, Optional, Sequence, Tuple, Union

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

WireMessage = Union[SystemMessageIn, UserMessageIn, AssistantMessageIn, ToolMessageIn]

_ConvertResult = Result[Tuple[Block[SystemText], Block[Message]], TranslationError]

_RoleBlocks = Tuple[Role, Block[ContentBlock]]


def convert_messages(items: Sequence[WireMessage]) -> _ConvertResult:
    systems = Block.of_seq(
        chain.from_iterable(_system_texts(item) for item in items if isinstance(item, SystemMessageIn))
    )
    turns: List[_RoleBlocks] = []
    for item in items:
        match _convert_turn(item):
            case Result(tag="ok", ok=turn):
                if turn is not None:
                    # build-locally-freeze-once: `turns` never escapes this scope
                    turns.append(turn)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok((systems, _merge_adjacent(turns)))


def cache_of(value: Optional[CacheControlIn]) -> Option[CacheControl]:
    if value is None:
        return Nothing
    ttl = Some(value.ttl) if value.ttl is not None else Nothing
    return Some(CacheControl(type=value.type, ttl=ttl))


def _system_texts(message: SystemMessageIn) -> Iterable[SystemText]:
    if isinstance(message.content, str):
        if not message.content:
            return ()
        return (SystemText(text=message.content, cache=cache_of(message.cache_control)),)
    if message.content is None:
        return ()
    return tuple(
        SystemText(text=part.text, cache=cache_of(part.cache_control)) for part in message.content if part.text
    )


def _convert_turn(
    message: WireMessage,
) -> Result[Optional[_RoleBlocks], TranslationError]:
    if isinstance(message, SystemMessageIn):
        return Ok(None)
    if isinstance(message, UserMessageIn):
        return _user_blocks(message).map(lambda blocks: ("user", blocks))
    if isinstance(message, ToolMessageIn):
        return Ok(("user", Block.of_seq([_tool_result(message)])))
    return _assistant_blocks(message).map(lambda blocks: ("assistant", blocks))


def _user_blocks(
    message: UserMessageIn,
) -> Result[Block[ContentBlock], TranslationError]:
    if message.content is None:
        return Ok(Block.empty())
    if isinstance(message.content, str):
        text = Text(text=message.content, cache=cache_of(message.cache_control))
        return Ok(Block.of_seq([ContentBlock.of_text(text)]))
    blocks: List[ContentBlock] = []
    for part in message.content:
        if isinstance(part, TextPartIn):
            blocks.append(  # nosemgrep: translation-no-mutation
                ContentBlock.of_text(Text(text=part.text, cache=cache_of(part.cache_control)))
            )
            continue
        match _image_block(part):
            case Result(tag="ok", ok=block):
                blocks.append(block)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(blocks))


def _image_block(part: ImagePartIn) -> Result[ContentBlock, TranslationError]:
    url = part.image_url.url if isinstance(part.image_url, ImageUrlIn) else part.image_url
    fmt = part.image_url.format if isinstance(part.image_url, ImageUrlIn) else None
    cache = cache_of(part.cache_control)
    if url.startswith("https://"):
        return Ok(ContentBlock.of_image(Image(source=ImageSource.of_url(UrlSource(url=url)), cache=cache)))
    if url.startswith("http://"):
        return Error(TranslationError.of_unsupported("http:// image URLs require a download; v1 handles them"))
    return _base64_source(url, fmt).map(lambda source: ContentBlock.of_image(Image(source=source, cache=cache)))


def _base64_source(url: str, fmt: Optional[str]) -> Result[ImageSource, TranslationError]:
    prefix, separator, data = url.partition(";base64,")
    if not separator or not prefix.startswith("data:"):
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(Block.of_seq(["image_url is not a data:<media>;base64,<data> URI"]))
            )
        )
    media_type = fmt if fmt else prefix[len("data:") :].replace("\\/", "/")
    return Ok(ImageSource.of_base64(Base64Source(media_type=media_type, data=data)))


def _tool_result(message: ToolMessageIn) -> ContentBlock:
    if isinstance(message.content, str):
        content = ToolResultContent.of_text(message.content)
    elif message.content is None:
        content = ToolResultContent.of_text("")
    else:
        content = ToolResultContent.of_parts(
            Block.of_seq(Text(text=part.text, cache=cache_of(part.cache_control)) for part in message.content)
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
) -> Result[Block[ContentBlock], TranslationError]:
    if message.function_call is not None:
        return Error(TranslationError.of_unsupported("legacy assistant.function_call; v1 handles it"))
    if message.provider_specific_fields:
        return Error(
            TranslationError.of_unsupported(
                "assistant.provider_specific_fields (server-tool payloads); v1 handles them"
            )
        )
    thinking = _thinking_blocks(message)
    content = _assistant_content(message)
    return _tool_use_blocks(message.tool_calls).map(lambda tool_uses: thinking + content + tool_uses)


def _thinking_blocks(message: AssistantMessageIn) -> Block[ContentBlock]:
    """v1 prepends ``thinking_blocks`` verbatim, unless the content list
    already carries inline thinking parts (then the list order wins)."""
    if message.thinking_blocks is None:
        return Block.empty()
    inline_thinking = isinstance(message.content, list) and any(
        isinstance(part, (ThinkingPartIn, RedactedThinkingPartIn)) for part in message.content
    )
    if inline_thinking:
        return Block.empty()
    return Block.of_seq(_thinking_ir(part) for part in message.thinking_blocks)


def _thinking_ir(
    part: Union[ThinkingPartIn, RedactedThinkingPartIn],
) -> ContentBlock:
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


def _assistant_content(message: AssistantMessageIn) -> Block[ContentBlock]:
    if isinstance(message.content, str):
        # An empty string is kept: v1 rewrites it to the placeholder text
        # block (the anthropic serializer owns that rewrite).
        text = Text(text=message.content, cache=cache_of(message.cache_control))
        return Block.of_seq([ContentBlock.of_text(text)])
    if message.content is None:
        return Block.empty()
    return Block.of_seq(_assistant_part_ir(part) for part in message.content if _keeps_assistant_part(part))


def _keeps_assistant_part(
    part: Union[TextPartIn, ThinkingPartIn, RedactedThinkingPartIn],
) -> bool:
    """v1 keeps non-empty text and non-empty inline thinking; it drops inline
    redacted_thinking parts (only ``thinking_blocks`` carries those through)."""
    if isinstance(part, TextPartIn):
        return True
    if isinstance(part, ThinkingPartIn):
        return len(part.thinking) > 0
    return False


def _assistant_part_ir(
    part: Union[TextPartIn, ThinkingPartIn, RedactedThinkingPartIn],
) -> ContentBlock:
    if isinstance(part, TextPartIn):
        return ContentBlock.of_text(Text(text=part.text, cache=cache_of(part.cache_control)))
    return _thinking_ir(part)


def _tool_use_blocks(
    tool_calls: Optional[List[ToolCallIn]],
) -> Result[Block[ContentBlock], TranslationError]:
    if tool_calls is None:
        return Ok(Block.empty())
    blocks: List[ContentBlock] = []
    for call in tool_calls:
        if call.type != "function":
            continue  # v1 silently skips non-function tool calls
        match _tool_use(call):
            case Result(tag="ok", ok=block):
                blocks.append(block)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(blocks))


def _tool_use(call: ToolCallIn) -> Result[ContentBlock, TranslationError]:
    if call.function is None:
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(Block.of_seq([f"tool_call {call.id!r} is missing 'function'"]))
            )
        )
    if call.id.startswith("srvtoolu_"):
        return Error(TranslationError.of_unsupported("server-tool call reconstruction (srvtoolu_*); v1 handles it"))
    return _arguments_of(call).map(
        lambda arguments: ContentBlock.of_tool_use(
            ToolUse(
                id=call.id,
                name=call.function.name if call.function is not None else "",
                arguments=arguments,
                cache=cache_of(call.cache_control),
            )
        )
    )


def _arguments_of(call: ToolCallIn) -> Result[JsonBlob, TranslationError]:
    raw = call.function.arguments if call.function is not None else None
    if raw is None or not raw.strip():
        return Ok(JsonBlob(value={}))
    try:
        value: PlainJson = json.loads(raw)
    except ValueError:
        return Error(TranslationError.of_unsupported("malformed tool_call arguments need v1's JSON repair"))
    return Ok(JsonBlob(value=value))


def _merge_adjacent(turns: Sequence[_RoleBlocks]) -> Block[Message]:
    """O(n) merge of consecutive same-role turns (the tracer's fold copied the
    accumulator per step: O(n^2), 10x v1 latency at 600 messages)."""
    merged = (
        Message(
            role=role,
            content=Block.of_seq(chain.from_iterable(blocks for _, blocks in group)),
        )
        for role, group in groupby(turns, key=lambda turn: turn[0])
    )
    # v1 only emits a merged turn when it produced content, so an empty turn
    # (e.g. a content-less user message) splits adjacency the same way here.
    return Block.of_seq(message for message in merged if len(message.content) > 0)
