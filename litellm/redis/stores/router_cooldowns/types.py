from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeAlias

from typing_extensions import ReadOnly, TypedDict


class CooldownCacheValue(TypedDict):
    exception_received: ReadOnly[str]
    status_code: ReadOnly[str]
    timestamp: ReadOnly[float]
    cooldown_time: ReadOnly[float]


ROUTER_COOLDOWNS_TARGET: Final = "router_cooldowns"
ActiveCooldowns: TypeAlias = Sequence[tuple[str, CooldownCacheValue]]
