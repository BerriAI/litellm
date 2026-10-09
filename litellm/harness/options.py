"""Typed per-harness options. Settings that only make sense for one runtime live here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

PiThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def _no_config() -> Mapping[str, object]:
    return MappingProxyType({})


def _no_env() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ClaudeCodeOptions:
    config: Mapping[str, object] = field(default_factory=_no_config)
    env: Mapping[str, str] = field(default_factory=_no_env)


@dataclass(frozen=True)
class CodexOptions:
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None
    web_search: bool = False
    config: Mapping[str, object] = field(default_factory=_no_config)
    env: Mapping[str, str] = field(default_factory=_no_env)


@dataclass(frozen=True)
class OpenCodeOptions:
    agent: str = "build"
    config: Mapping[str, object] = field(default_factory=_no_config)
    env: Mapping[str, str] = field(default_factory=_no_env)


@dataclass(frozen=True)
class PiOptions:
    thinking: PiThinkingLevel | None = None
    config: Mapping[str, object] = field(default_factory=_no_config)
    env: Mapping[str, str] = field(default_factory=_no_env)


@dataclass(frozen=True)
class DeepAgentsOptions:
    subagents: Sequence[object] = ()
    recursion_limit: int | None = None


@dataclass(frozen=True)
class ToolLoopOptions:
    completion_kwargs: Mapping[str, object] = field(default_factory=_no_config)


HarnessOptions: TypeAlias = (
    ClaudeCodeOptions | CodexOptions | OpenCodeOptions | PiOptions | DeepAgentsOptions | ToolLoopOptions
)
