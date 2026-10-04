"""Typed per-harness options. Settings that only make sense for one runtime live here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ClaudeCodeOptions:
    config: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CodexOptions:
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None
    web_search: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenCodeOptions:
    agent: str = "build"
    config: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepAgentsOptions:
    subagents: Sequence[Any] = ()
    recursion_limit: int | None = None


@dataclass(frozen=True)
class ToolLoopOptions:
    completion_kwargs: Mapping[str, Any] = field(default_factory=dict)


HarnessOptions = ClaudeCodeOptions | CodexOptions | OpenCodeOptions | DeepAgentsOptions | ToolLoopOptions
