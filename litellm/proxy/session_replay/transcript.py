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

_PROXY_INJECTED_KEYS: Final = frozenset(
    {
        "headers",
        "litellm_metadata",
        "litellm_session_id",
        "litellm_trace_id",
        "metadata",
        "provider_specific_header",
        "secret_fields",
    }
)

_STRUCTURAL_KEYS: Final = frozenset({"model", "messages", "system", "tools", "stream"})

_SYSTEM_REMINDER: Final = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    id: str | None = None
    tool_use_id: str | None = None
    content: str | None = None


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
    if isinstance(value, str):
        return 1 if TRUNCATION_MARKER in value else 0
    if isinstance(value, dict):
        return sum(_count_truncations(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_truncations(item) for item in value)
    return 0


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
        if key not in _PROXY_INJECTED_KEYS and key not in _STRUCTURAL_KEYS
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

    Recorded `tool_result` blocks carry the original run's `tool_use_id`s, which name
    `tool_use` blocks this arm never emitted. When the arm did call tools the results are
    rebound onto its ids in order; when it called none there is nothing to bind to, and
    injecting the recording's tool output as prose would feed the arm answers to questions
    it never asked, so the turn is dropped instead.
    """
    blocks: Final = turn.blocks()
    results: Final = tuple(block for block in blocks if block.type == "tool_result")
    if not results:
        return turn
    if not pending_tool_use_ids:
        return None
    rebound: Final = tuple(
        block.model_copy(
            update={"tool_use_id": pending_tool_use_ids[min(index, len(pending_tool_use_ids) - 1)]}
        )  # mutable-ok: pydantic model_copy update mapping
        if block.type == "tool_result"
        else block
        for index, block in enumerate(blocks)
    )
    answered: Final = frozenset(block.tool_use_id for block in rebound if block.type == "tool_result")
    stubs: Final = tuple(
        ContentBlock(type="tool_result", tool_use_id=tool_id, content="[no recorded result]")
        for tool_id in pending_tool_use_ids
        if tool_id not in answered
    )
    return RecordedMessage(role="user", content=rebound + stubs)
