"""Placement policy for ``role: "system"`` messages that appear after the first turn
of an Anthropic-shaped chat completions request.

Only the leading run of system messages belongs in the top-level ``system``
parameter. Hoisting a later one there rewrites the cached prefix, so the provider
re-bills the whole conversation at cache-write pricing on every reminder (#36559).

Models flagged ``supports_mid_conversation_system`` in the cost map accept the role
inside ``messages`` under Anthropic's placement rules: the message must directly
follow a user turn, must be the last entry or be followed by an assistant turn, and
must not sit next to another system message. OpenAI-shaped clients put system
messages anywhere, so this module places each run by its neighbours alone: a run
after a user turn stays with that turn, a run after an assistant turn slides
behind the user turn that immediately follows it, and a run that ends the array
or precedes an assistant turn becomes a user turn in place. Runs that land on the
same slot merge into one system message. No later message can move an earlier
run, so a client that replays the conversation with more turns appended sends a
byte-identical prefix and preserved thinking blocks keep their binding.

Models without the flag reject the role inside ``messages``. Their system messages
become user turns in place, prefixed with an operator note so the model can tell
the instruction apart from the user's own words. A run caught between a tool call
and its result moves to just after the result so the ``tool_result`` block stays
first in the merged user turn.

Every transformation here is a pure function of the message sequence: turn N's
output stays a prefix of turn N+1's output, which is what keeps the provider-side
prompt cache readable across turns. Messages are handled in OpenAI format; the
Anthropic wire shape is built later by ``anthropic_messages_pt``.
"""

from collections.abc import Iterator, Mapping, Sequence
from itertools import groupby
from typing import Final, Literal, TypeAlias

from litellm.types.llms.anthropic import AnthropicMessagesSystemMessageParam, AnthropicSystemMessageContent
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionUserMessage,
)

CONVERTED_SYSTEM_NOTE: Final = (
    "Operator note (not from the user): the following was originally a mid-conversation system-role reminder."
)

_USER_TYPE_ROLES: Final = frozenset({"user", "tool", "function"})
_TOOL_ROLES: Final = frozenset({"tool", "function"})
_RENDERED_PART_TYPES: Final = frozenset({"text", "image_url", "document", "file"})
_RENDERED_ASSISTANT_PART_TYPES: Final = frozenset({"text", "server_tool_use"})

_MessageKind: TypeAlias = Literal["system", "tool", "user", "other"]
_TextPart: TypeAlias = tuple[str, ChatCompletionCachedContent | None]


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def parts_of(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, Sequence) and not isinstance(value, str) else ()


def message_field(message: object, key: str) -> object:
    """A message field, whether the message is a dict or a pydantic ``Message``.

    Clients replay assistant turns straight from a response, so a history mixes
    plain dicts with ``litellm.Message`` objects; every predicate reads through here.
    """
    mapping: Final = _as_mapping(message)
    return mapping.get(key) if mapping is not None else getattr(message, key, None)


def is_system_message(message: object) -> bool:
    return message_field(message, "role") == "system"


def _is_user_type(message: object) -> bool:
    return message_field(message, "role") in _USER_TYPE_ROLES


def _kind(message: object) -> _MessageKind:
    role: Final = message_field(message, "role")
    if role == "system":
        return "system"
    if role in _TOOL_ROLES:
        return "tool"
    if role == "user":
        return "user"
    return "other"


def split_leading_system_run(
    messages: Sequence[AllMessageValues],
) -> tuple[tuple[AllMessageValues, ...], tuple[AllMessageValues, ...]]:
    """Split ``messages`` into the leading run of system messages and everything after it."""
    leading_count: Final = next(
        (index for index, message in enumerate(messages) if not is_system_message(message)),
        len(messages),
    )
    return tuple(messages[:leading_count]), tuple(messages[leading_count:])


def _cache_control(holder: object) -> ChatCompletionCachedContent | None:
    """The client's ``cache_control`` rebuilt in the only shape Anthropic accepts."""
    value: Final = _as_mapping(message_field(holder, "cache_control"))
    if value is None or value.get("type") != "ephemeral":
        return None
    ttl: Final = value.get("ttl")
    if ttl == "1h":
        one_hour: Final[ChatCompletionCachedContent] = {"type": "ephemeral", "ttl": "1h"}
        return one_hour
    if ttl == "5m":
        five_minutes: Final[ChatCompletionCachedContent] = {"type": "ephemeral", "ttl": "5m"}
        return five_minutes
    ephemeral: Final[ChatCompletionCachedContent] = {"type": "ephemeral"}
    return ephemeral


def _text_parts(message: object) -> tuple[_TextPart, ...]:
    """``(text, cache_control)`` for each non-empty text part of a system message.

    Anthropic rejects empty text blocks and only accepts text in system content. A
    ``cache_control`` on the message itself belongs to the block built from string
    content; block-level ``cache_control`` stays with its block.
    """
    content: Final = message_field(message, "content")
    if isinstance(content, str):
        return ((content, _cache_control(message)),) if content else ()
    return tuple(
        (text, _cache_control(part))
        for part in parts_of(content)
        if message_field(part, "type") == "text"
        for text in (message_field(part, "text"),)
        if isinstance(text, str) and text
    )


def _openai_text_block(part: _TextPart) -> ChatCompletionTextObject:
    text, cache_control = part
    if cache_control is None:
        plain: Final[ChatCompletionTextObject] = {"type": "text", "text": text}
        return plain
    cached: Final[ChatCompletionTextObject] = {"type": "text", "text": text, "cache_control": cache_control}
    return cached


def _anthropic_text_block(part: _TextPart) -> AnthropicSystemMessageContent:
    text, cache_control = part
    if cache_control is None:
        plain: Final[AnthropicSystemMessageContent] = {"type": "text", "text": text}
        return plain
    cached: Final[AnthropicSystemMessageContent] = {"type": "text", "text": text, "cache_control": cache_control}
    return cached


def anthropic_system_messages(message: object) -> tuple[AnthropicMessagesSystemMessageParam, ...]:
    """The Anthropic wire message for a system message, or nothing when it carries no text."""
    blocks: Final = tuple(_anthropic_text_block(part) for part in _text_parts(message))
    if not blocks:
        return ()
    wire: Final[AnthropicMessagesSystemMessageParam] = {
        "role": "system",
        "content": list(blocks),  # mutable-ok: wire payload; cache_control hooks edit content blocks in place
    }
    return (wire,)


def system_message_as_user(message: object) -> ChatCompletionUserMessage:
    """A system message re-rolled as a user turn, prefixed with the operator note."""
    note: Final[ChatCompletionTextObject] = {"type": "text", "text": CONVERTED_SYSTEM_NOTE}
    content: Final[list[ChatCompletionTextObject]] = [  # mutable-ok: anthropic_messages_pt only recognises list content
        note,
        *(_openai_text_block(part) for part in _text_parts(message)),
    ]
    turn: Final[ChatCompletionUserMessage] = {"role": "user", "content": content}
    return turn


def _merged_system_message(run: Sequence[object]) -> tuple[ChatCompletionSystemMessage, ...]:
    parts: Final = tuple(part for message in run for part in _text_parts(message))
    if not parts:
        return ()
    content: Final[list[ChatCompletionTextObject]] = [  # mutable-ok: anthropic_messages_pt only recognises list content
        _openai_text_block(part) for part in parts
    ]
    merged: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content}
    return (merged,)


def _converted_user_turns(run: Sequence[object]) -> tuple[ChatCompletionUserMessage, ...]:
    return tuple(system_message_as_user(message) for message in run if _text_parts(message))


def _runs(messages: Sequence[AllMessageValues]) -> tuple[tuple[_MessageKind, tuple[AllMessageValues, ...]], ...]:
    return tuple((kind, tuple(group)) for kind, group in groupby(messages, key=_kind))


def _converted_for_unflagged_model(messages: Sequence[AllMessageValues]) -> tuple[AllMessageValues, ...]:
    """Convert every system message to a user turn in place.

    A system run whose follower is a tool message is emitted after that tool run:
    ``tool_result`` blocks have to open the merged user turn.
    """
    runs: Final = _runs(messages)

    def emit(index: int) -> tuple[AllMessageValues, ...]:
        kind, run = runs[index]
        follower: Final = runs[index + 1][0] if index + 1 < len(runs) else None
        if kind == "system":
            return () if follower == "tool" else _converted_user_turns(run)
        if kind == "tool" and index > 0 and runs[index - 1][0] == "system":
            return (*run, *_converted_user_turns(runs[index - 1][1]))
        return run

    return tuple(message for index in range(len(runs)) for message in emit(index))


def _user_type_blocks(messages: Sequence[AllMessageValues]) -> tuple[tuple[bool, tuple[int, ...]], ...]:
    """Maximal groups of consecutive non-system messages, keyed by whether they are user-type.

    Consecutive user-type messages become one user turn on the wire, so a group is
    the unit a system message can validly follow.
    """
    indexed: Final = tuple((index, message) for index, message in enumerate(messages) if not is_system_message(message))
    return tuple(
        (is_user, tuple(index for index, _ in group))
        for is_user, group in groupby(indexed, key=lambda pair: _is_user_type(pair[1]))
    )


def _system_runs(messages: Sequence[AllMessageValues]) -> tuple[tuple[int, ...], ...]:
    """Index runs of consecutive system messages."""
    system_indices: Final = tuple(index for index, message in enumerate(messages) if is_system_message(message))
    return tuple(
        tuple(index for _, index in group)
        for _, group in groupby(enumerate(system_indices), key=lambda pair: pair[1] - pair[0])
    )


def _block_containing(message_index: int, blocks: Sequence[tuple[bool, tuple[int, ...]]]) -> int:
    return next(index for index, (_, indices) in enumerate(blocks) if message_index in indices)


def _thinking_block_renders(block: object) -> bool:
    """A thinking block the converter keeps: redacted, or signed so Anthropic can verify it."""
    block_type: Final = message_field(block, "type")
    if block_type == "redacted_thinking":
        return True
    signature: Final = message_field(block, "signature")
    return block_type == "thinking" and isinstance(signature, str) and bool(signature)


def _assistant_part_renders(part: object) -> bool:
    """A text part always renders: the converter pads empty text with a placeholder."""
    part_type: Final = message_field(part, "type")
    if part_type == "thinking":
        thinking: Final = message_field(part, "thinking")
        return isinstance(thinking, str) and bool(thinking) and _thinking_block_renders(part)
    return part_type in _RENDERED_ASSISTANT_PART_TYPES or (
        isinstance(part_type, str) and part_type.endswith("_tool_result")
    )


def _assistant_renders(message: object) -> bool:
    """Whether ``anthropic_messages_pt`` puts a block on the wire for this assistant message.

    String content (the converter pads an empty one with a placeholder), a text part,
    a signed thinking part, a server tool part, tool calls, a function call, a kept
    thinking block and compaction blocks each render. An assistant message with none
    of them, such as ``content: None`` or an empty list, vanishes from the wire.
    """
    content: Final = message_field(message, "content")
    if isinstance(content, str):
        return True
    return (
        any(_assistant_part_renders(part) for part in parts_of(content))
        or any(_thinking_block_renders(block) for block in parts_of(message_field(message, "thinking_blocks")))
        or bool(message_field(message, "tool_calls"))
        or bool(message_field(message, "function_call"))
        or bool(message_field(message_field(message, "provider_specific_fields"), "compaction_blocks"))
    )


def _renders(message: object) -> bool:
    """Whether ``anthropic_messages_pt`` puts a block on the wire for this message.

    A tool message always becomes a ``tool_result`` and a user message with string
    content always becomes a text block (empty text gets a placeholder). A user list
    renders only through parts of a type the converter emits; ``None``, an empty list,
    and a list of other parts vanish. Assistant messages follow ``_assistant_renders``.
    """
    role: Final = message_field(message, "role")
    if role in _TOOL_ROLES:
        return True
    if role == "assistant":
        return _assistant_renders(message)
    content: Final = message_field(message, "content")
    return isinstance(content, str) or any(
        message_field(part, "type") in _RENDERED_PART_TYPES for part in parts_of(content)
    )


def _rendered_block(
    message_index: int,
    messages: Sequence[AllMessageValues],
    blocks: Sequence[tuple[bool, tuple[int, ...]]],
) -> int | None:
    block_index: Final = _block_containing(message_index, blocks)
    _, indices = blocks[block_index]
    return block_index if any(_renders(messages[index]) for index in indices) else None


def _system_may_follow(
    block_index: int,
    messages: Sequence[AllMessageValues],
    blocks: Sequence[tuple[bool, tuple[int, ...]]],
) -> bool:
    """Whether a system message behind this block precedes an assistant turn or ends the array on the wire.

    Blocks alternate between user-type and assistant, so the check is whether the
    first later block that puts anything on the wire is an assistant block.
    """
    return next(
        (
            not is_user
            for is_user, indices in blocks[block_index + 1 :]
            if any(_renders(messages[index]) for index in indices)
        ),
        True,
    )


def _anchor_block(
    run: Sequence[int],
    messages: Sequence[AllMessageValues],
    blocks: Sequence[tuple[bool, tuple[int, ...]]],
) -> int | None:
    """The user-type block a system run must follow, or ``None`` when it converts in place.

    The run never starts at 0: the leading system run was split off before this
    policy runs, so the message before a run is always a non-system message. Only
    the run's neighbours decide, so a request that replays these messages with more
    turns appended places the run identically. A block that puts nothing on the wire
    cannot anchor a run: the system message would land first or behind an assistant
    turn, so the run converts in place instead. The same happens when the assistant
    turn after the anchor puts nothing on the wire and a user turn follows it: the
    system message would sit directly before that user turn, which Anthropic rejects.
    """
    previous: Final = run[0] - 1
    neighbour: Final = previous if _is_user_type(messages[previous]) else run[-1] + 1
    if neighbour >= len(messages) or not _is_user_type(messages[neighbour]):
        return None
    block_index: Final = _rendered_block(neighbour, messages, blocks)
    if block_index is None or not _system_may_follow(block_index, messages, blocks):
        return None
    return block_index


def _placed_for_flagged_model(messages: Sequence[AllMessageValues]) -> tuple[AllMessageValues, ...]:
    """Keep system messages as ``role: "system"`` at a placement Anthropic accepts.

    A run already sitting after a user-type message stays with that user turn. A
    run after an assistant turn moves behind the user turn that immediately follows
    it. A run that ends the array or is followed by an assistant turn becomes user
    turns in place, so replaying the same messages with more turns appended cannot
    move it. Runs that share a user turn merge into one system message.
    """
    blocks: Final = _user_type_blocks(messages)
    anchors: Final = tuple((run, _anchor_block(run, messages, blocks)) for run in _system_runs(messages))

    def anchored_to(block_index: int) -> tuple[AllMessageValues, ...]:
        return tuple(messages[index] for run, anchor in anchors if anchor == block_index for index in run)

    def converted_after(message_index: int) -> tuple[ChatCompletionUserMessage, ...]:
        return tuple(
            turn
            for run, anchor in anchors
            if anchor is None and run[0] == message_index + 1
            for turn in _converted_user_turns(tuple(messages[index] for index in run))
        )

    def emit(block_index: int) -> Iterator[AllMessageValues]:
        is_user, indices = blocks[block_index]
        for index in indices:
            yield messages[index]
            yield from converted_after(index)
        if is_user:
            yield from _merged_system_message(anchored_to(block_index))

    return tuple(message for block_index in range(len(blocks)) for message in emit(block_index))


def place_mid_conversation_system(
    messages: Sequence[AllMessageValues],
    *,
    supports_mid_conversation_system: bool,
) -> tuple[AllMessageValues, ...]:
    """Apply the placement policy to the messages after the leading system run."""
    if not any(is_system_message(message) for message in messages):
        return tuple(messages)
    if supports_mid_conversation_system:
        return _placed_for_flagged_model(messages)
    return _converted_for_unflagged_model(messages)
