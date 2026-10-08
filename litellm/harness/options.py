"""Typed per-harness options. Settings that only make sense for one runtime live here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

_NO_CONFIG: Final[Mapping[str, object]] = MappingProxyType({})
_NO_ENV: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True)
class ClaudeCodeOptions:
    config: Mapping[str, object] = _NO_CONFIG
    env: Mapping[str, str] = _NO_ENV


@dataclass(frozen=True)
class CodexOptions:
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None
    web_search: bool = False
    config: Mapping[str, object] = _NO_CONFIG
    env: Mapping[str, str] = _NO_ENV


@dataclass(frozen=True)
class OpenCodeOptions:
    agent: str = "build"
    config: Mapping[str, object] = _NO_CONFIG
    env: Mapping[str, str] = _NO_ENV


@dataclass(frozen=True)
class PiOptions:
    thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    config: Mapping[str, object] = _NO_CONFIG
    env: Mapping[str, str] = _NO_ENV


@dataclass(frozen=True)
class DeepAgentsOptions:
    subagents: Sequence[object] = ()
    recursion_limit: int | None = None


@dataclass(frozen=True)
class ToolLoopOptions:
    completion_kwargs: Mapping[str, object] = _NO_CONFIG


HarnessOptions: TypeAlias = (
    ClaudeCodeOptions | CodexOptions | OpenCodeOptions | PiOptions | DeepAgentsOptions | ToolLoopOptions
)
