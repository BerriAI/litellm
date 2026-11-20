from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm._internal_context import with_service_target
from litellm.redis.stores.active_teams.handle import ActiveTeamsHandle
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET

__all__ = ["ActiveTeamStore"]


class ActiveTeamStore:
    """Which callers used a model in the current window, in Redis when the handle has one and otherwise in this
    process's memory."""

    def __init__(self, handle: ActiveTeamsHandle) -> None:
        self._handle: Final = handle

    @with_service_target(RATE_LIMITS_TARGET)
    async def touch(self, model: str, member: str, priority_class: str, window: int) -> Mapping[str, str]:
        """Record ``member`` as active on ``model`` and return every member active on it, oldest first, each with
        the priority class it last sent with. Without Redis the answer comes back without yielding."""
        listed: Final = (
            await self._handle.touch(model, member, priority_class, window)
            if self._handle.has_redis
            else self._handle.touch_local(model, member, priority_class, window)
        )
        return MappingProxyType({active.member: active.priority_class for active in listed})
