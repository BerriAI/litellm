"""Public types for litellm.harness: the Harness enum, events, results and state."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from litellm.harness.errors import StateIncompatible

StopReason = Literal["done", "max_turns", "timeout", "cancelled", "runtime_error"]
PermissionMode = Literal["read-only", "ask", "edit", "full"]
FileChangeKind = Literal["created", "modified", "deleted"]

STATE_VERSION = 1


class Harness(Enum):
    """Supported agent runtimes. A plain Enum on purpose: strings are rejected."""

    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    OPENCODE = "opencode"
    DEEPAGENTS = "deepagents"
    TOOL_LOOP = "tool_loop"


def require_harness(harness: object) -> Harness:
    """Return harness if it is a Harness member, else raise TypeError with a hint."""
    if isinstance(harness, Harness):
        return harness
    hint = ""
    if isinstance(harness, str):
        normalized = harness.strip().lower().replace("-", "_")
        for member in Harness:
            if normalized in (member.value, member.name.lower()):
                hint = f" Did you mean Harness.{member.name}?"
    raise TypeError(
        f"harness must be a litellm.harness.Harness member, got {type(harness).__name__} {harness!r}.{hint}"
    )


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Text:
    delta: str


@dataclass(frozen=True)
class Reasoning:
    delta: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    native_name: str
    input: Mapping[str, Any]
    builtin: bool = True


@dataclass(frozen=True)
class ToolResult:
    id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class FileChange:
    path: str
    kind: FileChangeKind
    diff: str | None = None


@dataclass(frozen=True)
class Compaction:
    tokens_before: int | None = None
    tokens_after: int | None = None


@dataclass(frozen=True)
class Approval:
    """A request to run a tool. The turn waits until allow() or deny() is called."""

    tool: str
    input: Mapping[str, Any]
    _decision: asyncio.Future[tuple[bool, str]] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future(),
        compare=False,
        repr=False,
    )

    def allow(self) -> None:
        self._resolve(True, "")

    def deny(self, reason: str = "") -> None:
        self._resolve(False, reason)

    @property
    def answered(self) -> bool:
        return self._decision.done()

    async def wait(self) -> tuple[bool, str]:
        return await self._decision

    def _resolve(self, allowed: bool, reason: str) -> None:
        if self._decision.done():
            return
        loop = self._decision.get_loop()
        loop.call_soon_threadsafe(self._set_result, allowed, reason)

    def _set_result(self, allowed: bool, reason: str) -> None:
        if not self._decision.done():
            self._decision.set_result((allowed, reason))


@dataclass(frozen=True)
class Result:
    text: str
    output: BaseModel | None
    files: list[FileChange]  # mutable-ok: public Result field; users index/iterate it as a list
    events: list[Event]  # mutable-ok: public Result field; users index/iterate it as a list
    usage: Usage
    cost: float
    stop_reason: StopReason
    session_id: str


@dataclass(frozen=True)
class Done:
    result: Result

    @property
    def usage(self) -> Usage:
        return self.result.usage

    @property
    def cost(self) -> float:
        return self.result.cost

    @property
    def stop_reason(self) -> StopReason:
        return self.result.stop_reason


Event = Text | Reasoning | ToolCall | ToolResult | FileChange | Compaction | Approval | Done


@dataclass(frozen=True)
class Capabilities:
    structured_output: bool
    tool_approval: bool
    tool_filtering: bool
    history: bool
    custom_tools: bool
    skills: bool
    resume: bool
    permission_modes: frozenset[str]


@dataclass(frozen=True)
class State:
    """Resume state for a detached or stopped session. Contains no credentials."""

    harness: Harness
    native_session_id: str | None
    workdir: str
    model: str | None = None
    version: int = STATE_VERSION

    def dumps(self) -> bytes:
        return json.dumps(
            {  # mutable-ok: JSON payload serialized immediately by json.dumps
                "harness": self.harness.value,
                "native_session_id": self.native_session_id,
                "workdir": self.workdir,
                "model": self.model,
                "version": self.version,
            }
        ).encode("utf-8")

    @classmethod
    def loads(cls, data: bytes) -> State:
        try:
            raw = json.loads(data.decode("utf-8"))
            harness = Harness(raw["harness"])
            version = int(raw["version"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
            raise StateIncompatible(f"Unreadable harness state: {e}") from e
        if version != STATE_VERSION:
            raise StateIncompatible(f"State version {version} is not supported (expected {STATE_VERSION})")
        return cls(
            harness=harness,
            native_session_id=raw.get("native_session_id"),
            workdir=raw["workdir"],
            model=raw.get("model"),
            version=version,
        )
