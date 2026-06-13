"""Responses ``input`` list -> IR messages.

The input list is a polymorphic item union; this module dispatches each item by
its tag (a match over the typed union, never a nested if-chain) and folds it
into IR ``Message``s. Served items: a ``message`` (text/image content), a
``function_call`` (an assistant ``ToolUse`` block; consecutive function_calls
merge into one assistant message, the Anthropic-one-turn rule the IR already
honors), and a ``function_call_output``/``tool_result`` (a ``ToolResult`` block
in a user message). Items the chat IR cannot carry fail closed so the seam
falls back to v1 (researcher-6 §2.4/§2.5):

- ``reasoning`` input items (carry summary/encrypted_content/id for replay),
- ``item_reference`` (needs ``previous_response_id`` session state),
- ``input_file`` content parts (the IR has no document/file block),
- a ``function_call_output`` whose ``call_id`` matches no tool_use in THIS
  request (v1 reconstructs the missing tool_use from the module-level
  ``TOOL_CALLS_CACHE``, cross-request state v2 must not reproduce).

Hot path note (mirrors openai_chat/messages.py): internal helpers return
``value | TranslationError`` plain unions and build local lists, lifting into
``Result``/``Block`` once at the public boundary.
"""

from __future__ import annotations

from expression import Error, Nothing, Ok, Result, Some
from expression.collections import Block

from ... import boundary
from ...errors import TranslationError
from ...ir import (
    Base64Source,
    ContentBlock,
    Image,
    ImageSource,
    JsonBlob,
    Message,
    Text,
    ToolResult,
    ToolResultContent,
    ToolUse,
    UrlSource,
)
from .schema import (
    FunctionCallItemIn,
    FunctionCallOutputItemIn,
    ImageUrlContentIn,
    InputFileContentIn,
    InputItemIn,
    InputTextContentIn,
    ItemReferenceIn,
    MessageItemIn,
    ReasoningItemIn,
)

_NO_CACHE = Nothing

_ConvertResult = Result[Block[Message], TranslationError]


def convert_string_input(text: str) -> Block[Message]:
    block = ContentBlock.of_text(Text(text=text, cache=_NO_CACHE))
    return Block.of_seq([Message(role="user", content=Block.of_seq([block]))])


def convert_input_items(items: list[InputItemIn]) -> _ConvertResult:
    messages: list[Message] = []
    tool_call_ids: set[str] = set()
    for item in items:
        step = _convert_item(item, messages, tool_call_ids)
        if isinstance(step, TranslationError):
            return Error(step)
    return Ok(Block.of_seq(messages))


def _convert_item(
    item: InputItemIn, messages: list[Message], tool_call_ids: set[str]
) -> None | TranslationError:
    if isinstance(item, ReasoningItemIn):
        return TranslationError.of_unsupported(
            "responses reasoning input items (summary/encrypted_content); v1 handles them"
        )
    if isinstance(item, ItemReferenceIn):
        return TranslationError.of_unsupported(
            "responses item_reference; needs previous_response_id session state"
        )
    if isinstance(item, FunctionCallItemIn):
        return _append_function_call(item, messages, tool_call_ids)
    if isinstance(item, FunctionCallOutputItemIn):
        return _append_function_call_output(item, messages, tool_call_ids)
    return _append_message(item, messages)


def _append_function_call(
    item: FunctionCallItemIn, messages: list[Message], tool_call_ids: set[str]
) -> None | TranslationError:
    call_id = item.call_id or item.id or ""
    tool_use = _tool_use(item, call_id)
    if isinstance(tool_use, TranslationError):
        return tool_use
    if call_id:
        tool_call_ids.add(call_id)  # nosemgrep: translation-no-mutation
    # Merge consecutive function_calls into the trailing assistant message (v1's
    # Anthropic-one-turn rule, transformation.py:400); the IR's assistant turn
    # carries every tool_use block, so an append onto the last assistant turn.
    block = ContentBlock.of_tool_use(tool_use)
    if messages and messages[-1].role == "assistant":
        last = messages[-1]
        merged = Message(role="assistant", content=last.content + Block.of_seq([block]))
        messages[-1] = merged  # nosemgrep: translation-no-mutation
        return None
    messages.append(  # nosemgrep: translation-no-mutation
        Message(role="assistant", content=Block.of_seq([block]))
    )
    return None


def _tool_use(item: FunctionCallItemIn, call_id: str) -> ToolUse | TranslationError:
    arguments = item.arguments or ""
    match boundary.as_plain_json(_loads_or_self(arguments)):
        case Result(tag="ok", ok=copied) if isinstance(copied, dict):
            blob = JsonBlob(value=copied)
        case Result(tag="ok"):
            return TranslationError.of_unsupported(
                "responses function_call with non-object arguments; v1 handles it"
            )
        case Result(error=reason):
            return TranslationError.of_unsupported(
                f"responses function_call arguments: {reason}"
            )
    return ToolUse(
        id=call_id,
        name=item.name or "",
        arguments=blob,
        cache=_NO_CACHE,
        arguments_raw=Some(arguments),
    )


def _loads_or_self(arguments: str) -> object:
    import json

    try:
        return json.loads(arguments) if arguments else {}
    except ValueError:
        return arguments


def _append_function_call_output(
    item: FunctionCallOutputItemIn, messages: list[Message], tool_call_ids: set[str]
) -> None | TranslationError:
    if not item.call_id:
        return TranslationError.of_unsupported(
            "responses function_call_output with empty call_id; v1 drops it"
        )
    if item.call_id not in tool_call_ids:
        # v1 reconstructs the matching tool_use from the cross-request
        # TOOL_CALLS_CACHE (transformation.py:1121); that module-level state is
        # banned and unreproducible, so fail closed when the pair is not local.
        return TranslationError.of_unsupported(
            "responses function_call_output without a matching function_call in"
            " this request; v1 reconstructs it from TOOL_CALLS_CACHE"
        )
    result = ToolResult(
        tool_use_id=item.call_id,
        content=ToolResultContent.of_text(item.output or ""),
        cache=_NO_CACHE,
    )
    messages.append(  # nosemgrep: translation-no-mutation
        Message(
            role="user", content=Block.of_seq([ContentBlock.of_tool_result(result)])
        )
    )
    return None


def _append_message(
    item: MessageItemIn, messages: list[Message]
) -> None | TranslationError:
    blocks = _message_blocks(item)
    if isinstance(blocks, TranslationError):
        return blocks
    if not blocks:
        return None
    role = "assistant" if item.role == "assistant" else "user"
    messages.append(  # nosemgrep: translation-no-mutation
        Message(role=role, content=Block.of_seq(blocks))
    )
    return None


def _message_blocks(item: MessageItemIn) -> list[ContentBlock] | TranslationError:
    if isinstance(item.content, str):
        return [ContentBlock.of_text(Text(text=item.content, cache=_NO_CACHE))]
    blocks: list[ContentBlock] = []
    for part in item.content:
        block = _content_part(part)
        if isinstance(block, TranslationError):
            return block
        blocks.append(block)  # nosemgrep: translation-no-mutation
    return blocks


def _content_part(
    part: InputTextContentIn | ImageUrlContentIn | InputFileContentIn,
) -> ContentBlock | TranslationError:
    if isinstance(part, InputFileContentIn):
        return TranslationError.of_unsupported(
            "responses input_file content; the IR has no document block, v1 handles it"
        )
    if isinstance(part, ImageUrlContentIn):
        return _image(part)
    if part.text is None:
        return TranslationError.of_unsupported(
            "responses message content part with null text; v1 drops it"
        )
    return ContentBlock.of_text(Text(text=part.text, cache=_NO_CACHE))


def _image(part: ImageUrlContentIn) -> ContentBlock | TranslationError:
    if part.file_id is not None:
        return TranslationError.of_unsupported(
            "responses input_image by file_id; v1 handles it"
        )
    url = part.image_url or ""
    if url.startswith("data:"):
        decoded = _data_url(url)
        if isinstance(decoded, TranslationError):
            return decoded
        media_type, data = decoded
        source = ImageSource.of_base64(Base64Source(media_type=media_type, data=data))
        return ContentBlock.of_image(Image(source=source, cache=_NO_CACHE))
    source = ImageSource.of_url(UrlSource(url=url, format=Nothing))
    return ContentBlock.of_image(Image(source=source, cache=_NO_CACHE))


def _data_url(url: str) -> tuple[str, str] | TranslationError:
    header, _, data = url.partition(",")
    if not data or ";base64" not in header:
        return TranslationError.of_unsupported(
            "responses input_image data URL not base64-encoded; v1 handles it"
        )
    media_type = header[len("data:") :].split(";", 1)[0]
    return media_type, data
