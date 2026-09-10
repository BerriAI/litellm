"""Rebuild a replayable Anthropic Messages request from a recorded spend-log body.

A recorded `proxy_server_request` is a post-guardrail snapshot of the proxy's mutated
request dict, not the caller's wire body, so it needs three repairs before a provider
will accept it back: stray `role: "system"` messages belong in the top-level `system`
array, proxy-injected keys must go, and every string longer than
`MAX_STRING_LENGTH_PROMPT_IN_DB` was truncated on the way in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

TRUNCATION_MARKER: Final = "litellm_truncated"

REPLAYABLE_SAMPLING_KEYS: Final = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "thinking",
        "tool_choice",
        "output_config",
        "service_tier",
    }
)

UNANSWERED_TOOL_RESULT: Final = "[no recorded result]"

_SYSTEM_REMINDER: Final = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    id: str | None = None
    tool_use_id: str | None = None
    content: str | tuple[ContentBlock, ...] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None


class RecordedMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | tuple[ContentBlock, ...]

    def blocks(self) -> tuple[ContentBlock, ...]:
        if isinstance(self.content, str):
            return (ContentBlock(type="text", text=self.content),)
        return self.content

    def block_types(self) -> frozenset[str]:
        return frozenset(block.type for block in self.blocks())

    def text(self) -> str:
        return "".join(block.text or "" for block in self.blocks() if block.type == "text")


class RecordedBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: tuple[RecordedMessage, ...] = ()
    system: str | tuple[ContentBlock, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayTranscript:
    system: tuple[ContentBlock, ...]
    tools: tuple[ToolDefinition, ...]
    user_turns: tuple[RecordedMessage, ...]
    sampling_params: tuple[tuple[str, object], ...]
    human_asks: tuple[str, ...]
    truncated_strings: int
    recorded_model: str | None


def _count_truncations(value: object) -> int:
    """Walk the stored body iteratively; a recursive walk trips the CPU-spike CI gate."""
    pending: Final[list[object]] = [value]  # mutable-ok: explicit stack replaces recursion
    marked: Final[list[str]] = []  # mutable-ok: one entry per truncated string
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if TRUNCATION_MARKER in current:
                marked.append(current)
        elif isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return len(marked)


def _as_system_blocks(system: str | tuple[ContentBlock, ...]) -> tuple[ContentBlock, ...]:
    if isinstance(system, str):
        return (ContentBlock(type="text", text=system),)
    return system


def _hoisted_system_block(message: RecordedMessage) -> ContentBlock:
    return ContentBlock(type="text", text=message.text())


def human_ask(message: RecordedMessage) -> str:
    return _SYSTEM_REMINDER.sub("", message.text()).strip()


def build_transcript(recorded_body: RecordedBody) -> ReplayTranscript:
    """Split a recorded body into a fixed environment track and the arm's own policy track.

    `role: "system"` messages are hoisted rather than dropped: the Anthropic Messages API
    rejects that role inside `messages`, but the text is real context the recorded model saw.

    Sampling parameters are allowlisted, never filtered. The recorded body is caller-controlled
    and keeps transport fields the snapshot does not strip, `api_base` among them, so forwarding
    everything unrecognized would let a seeded session redirect an admin's replay traffic.
    """
    hoisted: Final = tuple(
        _hoisted_system_block(message) for message in recorded_body.messages if message.role == "system"
    )
    user_turns: Final = tuple(message for message in recorded_body.messages if message.role == "user")
    asks: Final = tuple(
        ask for message in user_turns if "tool_result" not in message.block_types() and (ask := human_ask(message))
    )
    sampling: Final = tuple(
        (key, value)
        for key, value in recorded_body.model_dump(exclude_none=True).items()
        if key in REPLAYABLE_SAMPLING_KEYS
    )
    return ReplayTranscript(
        system=_as_system_blocks(recorded_body.system) + hoisted,
        tools=recorded_body.tools,
        user_turns=user_turns,
        sampling_params=sampling,
        human_asks=asks,
        truncated_strings=_count_truncations(recorded_body.model_dump()),
        recorded_model=recorded_body.model,
    )


def tool_use_ids(assistant_blocks: Sequence[ContentBlock]) -> tuple[str, ...]:
    return tuple(block.id for block in assistant_blocks if block.type == "tool_use" and block.id is not None)


def attach_turn(turn: RecordedMessage, pending_tool_use_ids: Sequence[str]) -> RecordedMessage | None:
    """Bind a recorded user turn onto the arm's own trajectory, or drop it as unattachable.

    The Anthropic Messages API requires every `tool_use` an assistant emits to be answered by a
    `tool_result` carrying that exact id, in the very next user message. The recording answers
    the ids the ORIGINAL run emitted, so once an arm diverges the two no longer line up in
    either direction, and both mismatches are a provider 400 rather than a quality question:

    - the arm called tools the recording does not answer, including the case where the next
      recorded turn is ordinary text, so every unanswered id gets a stub
    - the recording answers tools the arm never called, so the surplus results are dropped
      rather than piled onto the last pending id, which would repeat one id

    A turn carrying only tool results while the arm called nothing has nothing to bind to, and
    replaying the recording's tool output as prose would hand the arm answers to questions it
    never asked, so that turn is dropped.
    """
    blocks: Final = turn.blocks()
    results: Final = tuple(block for block in blocks if block.type == "tool_result")
    others: Final = tuple(block for block in blocks if block.type != "tool_result")
    if not pending_tool_use_ids:
        return None if results else turn
    rebound: Final = tuple(
        result.model_copy(update={"tool_use_id": tool_id})  # mutable-ok: pydantic model_copy update mapping
        for tool_id, result in zip(pending_tool_use_ids, results)
    )
    stubs: Final = tuple(
        ContentBlock(type="tool_result", tool_use_id=tool_id, content=UNANSWERED_TOOL_RESULT)
        for tool_id in pending_tool_use_ids[len(rebound) :]
    )
    return RecordedMessage(role="user", content=rebound + stubs + others)
