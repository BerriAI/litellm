from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Protocol


class ActiveMemberRecord(Protocol):
    @property
    def member(self) -> str: ...
    @property
    def priority_class(self) -> str: ...


class ActiveTeamsHandle(Protocol):
    @property
    def has_redis(self) -> bool: ...
    def touch(
        self, model: str, member: str, priority_class: str, window_seconds: int
    ) -> Awaitable[Sequence[ActiveMemberRecord]]: ...
    def touch_local(
        self, model: str, member: str, priority_class: str, window_seconds: int
    ) -> Sequence[ActiveMemberRecord]: ...
