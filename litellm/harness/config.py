"""Reusable agent definitions shared by runs, sessions and evaluations."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import BaseModel

from litellm.harness.context import ApprovalHandler
from litellm.harness.options import HarnessOptions
from litellm.harness.types import Harness, PermissionMode, require_harness

T = TypeVar("T")


class Unset(Enum):
    VALUE = "unset"


UNSET: Final = Unset.VALUE


def resolve(value: T | Unset, default: T) -> T:
    return default if isinstance(value, Unset) else value


@dataclass(frozen=True, slots=True)
class Agent:
    """An agent definition. Runs supply their own prompt and sandbox."""

    harness: Harness
    model: str | None = None
    _: KW_ONLY
    instructions: str | None = None
    tools: Sequence[Callable[..., object]] = ()
    skills: Sequence[str | os.PathLike[str]] = ()
    disable_tools: Sequence[str] = ()
    permissions: PermissionMode = "full"
    on_approval: ApprovalHandler | None = None
    output: type[BaseModel] | None = None
    max_turns: int | None = None
    timeout: float | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    options: HarnessOptions | None = field(default=None, repr=False)
    install: bool = False
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        require_harness(self.harness)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(self, "disable_tools", tuple(self.disable_tools))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
