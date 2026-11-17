import base64
import binascii
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import groupby
from typing import Final, cast  # noqa: TID251  # Existing casts retained unchanged in this pure move

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.openai import (
    ChatCompletionReasoningItem,
    ChatCompletionReasoningSummaryTextBlock,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)


def _provider_text_or_none(value: object) -> str | None:
    return cast(str | None, value)  # cast-ok: reasoning fields arrive in untyped provider messages


def extract_reasoning_content(message: Mapping[str, object]) -> tuple[str | None, str | None]:
    """
    Extract reasoning content and main content from a message.

    Args:
        message (dict): The message dictionary that may contain reasoning_content

    Returns:
        tuple[Optional[str], Optional[str]]: A tuple of (reasoning_content, content)
    """
    message_content: Final = message.get("content")
    if "reasoning_content" in message:
        return _provider_text_or_none(message["reasoning_content"]), _provider_text_or_none(message_content)
    elif "reasoning" in message:
        return _provider_text_or_none(message["reasoning"]), _provider_text_or_none(message_content)
    elif isinstance(message_content, str):
        return parse_content_for_reasoning(message_content)
    return None, _provider_text_or_none(message_content)


_extract_reasoning_content = extract_reasoning_content


def _readable_thinking_text(block: Mapping[str, object]) -> str:
    """The text a chat model can read back, empty for redacted blocks and malformed ones."""
    if block.get("type") != "thinking":
        return ""
    return str(block.get("thinking") or "")


def reasoning_content_from_thinking_blocks(
    thinking_blocks: Iterable[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock],
) -> str:
    """Flatten Anthropic thinking blocks into the `reasoning_content` string chat models expect.

    Redacted blocks carry no readable text, so they contribute nothing.
    """
    return "\n".join(text for block in thinking_blocks if (text := _readable_thinking_text(block)))


ENCRYPTED_REASONING_SIGNATURE_PREFIX: Final = "litellm_encrypted_reasoning:"


def encrypted_reasoning_signature(encrypted_content: str) -> str:
    """The opaque value a Responses API reasoning item's `encrypted_content` travels in.

    Anthropic clients echo a thinking block's `signature` and a redacted block's `data`
    back verbatim, so either field can carry the encrypted reasoning across turns; the
    prefix tells the two apart from a signature Anthropic minted.
    """
    return f"{ENCRYPTED_REASONING_SIGNATURE_PREFIX}{encrypted_content}"


def _carries_encrypted_reasoning(signature: object) -> bool:
    return isinstance(signature, str) and signature.startswith(ENCRYPTED_REASONING_SIGNATURE_PREFIX)


def encrypted_content_from_signature(signature: object) -> str | None:
    if not isinstance(signature, str) or not _carries_encrypted_reasoning(signature):
        return None
    return signature.removeprefix(ENCRYPTED_REASONING_SIGNATURE_PREFIX) or None


def _encrypted_reasoning_field(block: Mapping[str, object]) -> object:
    match block.get("type"):
        case "thinking":
            return block.get("signature")
        case "redacted_thinking":
            return block.get("data")
        case _:
            return None


def encrypted_content_of_block(block: Mapping[str, object]) -> str | None:
    return encrypted_content_from_signature(_encrypted_reasoning_field(block))


_REASONING_SUMMARY_SEPARATOR: Final = "\n\n"


_OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])


def _object_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return _OBJECT_TUPLE.validate_python(value)


def _iter_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(mapping for item in _object_sequence(value) if (mapping := as_str_mapping(item)) is not None)


def _reasoning_summary_part_text(part: object) -> str:
    mapping: Final = as_str_mapping(part)
    if mapping is not None:
        text: Final = mapping.get("text")
        return text if isinstance(text, str) else ""
    text_attr: Final = getattr(part, "text", None)
    return text_attr if isinstance(text_attr, str) else ""


def _reasoning_item_summary_text(item: Mapping[str, object]) -> str:
    parts: Final = tuple(_reasoning_summary_part_text(part) for part in _object_sequence(item.get("summary")))
    return _REASONING_SUMMARY_SEPARATOR.join(part for part in parts if part)


def thinking_block_from_reasoning_item(
    item: object,
) -> ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock | None:
    """The chat thinking block that carries one Responses `reasoning` item back to an Anthropic client.

    `encrypted_content` rides the same `litellm_encrypted_reasoning:` signature the Responses
    adapter uses. A summary becomes thinking text. Bytes with no summary become `redacted_thinking`,
    because an empty thinking string is not something the client can replay.
    """
    mapping: Final = as_str_mapping(item)
    if mapping is None or mapping.get("type") not in (None, "reasoning"):
        return None
    encrypted_content: Final = mapping.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    signature: Final = encrypted_reasoning_signature(encrypted_content)
    text: Final = _reasoning_item_summary_text(mapping)
    if not text:
        return ChatCompletionRedactedThinkingBlock(type="redacted_thinking", data=signature)
    return ChatCompletionThinkingBlock(type="thinking", thinking=text, signature=signature)


def thinking_blocks_from_reasoning_items(
    items: object,
) -> tuple[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock, ...]:
    return tuple(
        block for item in _object_sequence(items) if (block := thinking_block_from_reasoning_item(item)) is not None
    )


def _block_encrypted_content(block: object) -> str | None:
    mapping: Final = as_str_mapping(block)
    if mapping is None:
        return None
    return encrypted_content_of_block(mapping)


def thinking_blocks_with_encrypted_reasoning(
    thinking_blocks: object,
    reasoning_items: object,
) -> tuple[Mapping[str, object], ...]:
    """Thinking blocks already on the message, plus any encrypted reasoning item they do not already carry."""
    direct: Final = _iter_mappings(thinking_blocks)
    represented: Final = frozenset(
        content for block in direct if (content := _block_encrypted_content(block)) is not None
    )
    extra: Final = tuple(
        block
        for block in thinking_blocks_from_reasoning_items(_iter_mappings(reasoning_items))
        if _block_encrypted_content(block) not in represented
    )
    return (*extra, *direct)


def is_encrypted_reasoning_block(block: object) -> bool:
    """A thinking or redacted_thinking block carrying Responses API encrypted reasoning.

    Only the Responses API that minted the content can read it back, so an Anthropic
    backend has to drop such a block rather than fail signature verification on it.
    """
    if not isinstance(block, Mapping):
        return False
    mapping: Final = cast(Mapping[str, object], block)  # cast-ok: narrowed by isinstance
    return _carries_encrypted_reasoning(_encrypted_reasoning_field(mapping))


def is_unsignable_thinking_block(block: object) -> bool:
    """A thinking block Anthropic cannot accept on input.

    Anthropic verifies the signature cryptographically, so a block with a null,
    empty, or missing signature (e.g. from an open-source reasoning model) is
    rejected with a 400, and so is a block whose signature or data carries
    another provider's encrypted reasoning. A signed block is kept whatever its
    text, since ``display: omitted`` returns signed blocks with empty ``thinking``.
    `redacted_thinking` blocks carry no signature and are always kept.
    """
    if is_encrypted_reasoning_block(block):
        return True
    if not isinstance(block, Mapping):
        return False
    mapping: Final = cast(Mapping[str, object], block)  # cast-ok: narrowed by isinstance
    if mapping.get("type") != "thinking":
        return False
    signature: Final = mapping.get("signature")
    return not (isinstance(signature, str) and len(signature) > 0)


def has_blank_redacted_data(block: object) -> bool:
    match block:
        case {"type": "redacted_thinking", "data": str() as data}:
            return not data.strip()
        case {"type": "redacted_thinking"}:
            return True
        case _:
            return False


def native_redacted_thinking_data(data: str) -> str:
    """Claude's own `data` string for a redacted_thinking block.

    Bedrock Converse carries it base64-wrapped as `redactedContent`, and LiteLLM passed that
    wrapper through as `data` until it started decoding Converse replies, so histories saved
    before then still hold it. Claude's own value never decodes twice, since its first byte
    decodes to a protobuf tag outside the base64 alphabet.
    """
    return _converse_wrapped_payload(data) or data


def converse_redacted_content(data: str) -> str:
    """The `redactedContent` Bedrock Converse expects for a redacted_thinking block's `data`."""
    return base64.b64encode(native_redacted_thinking_data(data).encode()).decode()


def _converse_wrapped_payload(data: str) -> str | None:
    try:
        payload: Final = base64.b64decode(data, validate=True).decode("ascii")
        _ = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    else:
        return payload


def strip_encrypted_reasoning_from_messages(
    messages: object,
    *,
    should_strip: Callable[[Mapping[str, object]], bool] | None = None,
) -> None:
    """Drop the bridge-tagged reasoning blocks a routed deployment cannot decrypt from
    Anthropic-shaped history.

    The whole block goes, the way #40280 drops undecryptable Responses ``input`` items: a
    provider that did not mint the block rejects it signed (a foreign signature) and unsigned
    (a missing signature) alike, so keeping its text as an unsigned thinking block only moves
    the 400 from the router to the provider.

    Mutates the content lists in place: the router's fallback snapshot shares these
    message objects, so a rebound list would replay the stripped blocks on the fallback hop.
    """
    if not isinstance(messages, list):
        return
    for content in anthropic_content_lists(cast(list[object], messages)):  # cast-ok: untyped client json
        _strip_encrypted_reasoning_from_blocks(content, should_strip=should_strip)


def anthropic_content_lists(messages: Sequence[object]) -> Iterator[object]:
    return (
        cast(list[object], content)  # cast-ok: narrowed by isinstance
        for message in messages
        if isinstance(message, Mapping)
        for content in (cast(Mapping[str, object], message).get("content"),)  # cast-ok: narrowed by isinstance
        if isinstance(content, list)
    )


def _strip_encrypted_reasoning_from_blocks(
    content: object,
    *,
    should_strip: Callable[[Mapping[str, object]], bool] | None = None,
) -> None:
    blocks: Final = cast(list[object], content)  # cast-ok: narrowed by the caller's isinstance
    kept: Final = tuple(
        block
        for block in blocks
        if not is_encrypted_reasoning_block(block)
        or (should_strip is not None and not should_strip(cast(Mapping[str, object], block)))
    )
    blocks[:] = kept


def _reasoning_replay_group_key(indexed_block: tuple[int, Mapping[str, object]]) -> str:
    index, block = indexed_block
    return f"encrypted:{index}" if is_encrypted_reasoning_block(block) else "summary"


def _reasoning_item_from_block_group(group: tuple[Mapping[str, object], ...]) -> ChatCompletionReasoningItem | None:
    summary: Final[list[ChatCompletionReasoningSummaryTextBlock]] = [  # mutable-ok: API message payload
        ChatCompletionReasoningSummaryTextBlock(type="summary_text", text=text)
        for block in group
        if (text := _readable_thinking_text(block))
    ]
    encrypted_content: Final = encrypted_content_of_block(group[0])
    if encrypted_content is not None:
        return ChatCompletionReasoningItem(type="reasoning", summary=summary, encrypted_content=encrypted_content)
    if not summary:
        return None
    return ChatCompletionReasoningItem(type="reasoning", summary=summary)


def responses_reasoning_items_from_thinking_blocks(
    thinking_blocks: Iterable[Mapping[str, object]],
) -> tuple[ChatCompletionReasoningItem, ...]:
    """Build Responses API `reasoning` input items from Anthropic thinking blocks.

    A block carrying encrypted reasoning replays the item it came from byte for byte;
    a run of plain thinking blocks collapses into one summary-only item. No item carries
    an `id`: the Responses API 404s on any id it did not mint itself and rejects an empty
    one, while an item without an id is always accepted.
    """
    return tuple(
        item
        for _, group in groupby(enumerate(thinking_blocks), key=_reasoning_replay_group_key)
        if (item := _reasoning_item_from_block_group(tuple(block for _, block in group))) is not None
    )


def parse_content_for_reasoning(
    message_text: str | None,
) -> tuple[str | None, str | None]:
    """
    Parse the content for reasoning

    Returns:
    - reasoning_content: The content of the reasoning
    - content: The content of the message
    """
    if not message_text:
        return None, message_text

    reasoning_match: Final = re.match(
        r"<(?:think|thinking|budget:thinking)>(.*?)</(?:think|thinking|budget:thinking)>(.*)",
        message_text,
        re.DOTALL,
    )

    if reasoning_match:
        return reasoning_match.group(1), reasoning_match.group(2)

    return None, message_text


_parse_content_for_reasoning = parse_content_for_reasoning
