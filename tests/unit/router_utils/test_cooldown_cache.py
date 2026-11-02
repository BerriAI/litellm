from typing import Final

import pytest

from litellm.redis.stores.router_cooldowns import RouterCooldownStore
from litellm.router_utils.cooldown_cache import CooldownCache

_NOW: Final = 1000.0


@pytest.mark.parametrize("message", ("Short error", "Rate limit " + "secret" * 100))
@pytest.mark.parametrize("ttl", (None, 0.5, 30.0))
def test_cooldown_masks_exception_and_preserves_status_timestamp_and_ttl(message: str, ttl: float | None) -> None:
    store: Final = RouterCooldownStore(None, clock=lambda: _NOW)
    cache: Final = CooldownCache(store, default_cooldown_time=60.0, clock=lambda: _NOW)
    cache.add_deployment_to_cooldown("dep", Exception(message), 429, ttl)
    expected_ttl: Final = 60.0 if ttl is None else ttl
    expected_message: Final = message if len(message) <= 50 else message[:50] + "*" * (len(message) - 50)
    assert cache.get_active_cooldowns(["dep"], None) == [
        (
            "dep",
            {
                "exception_received": expected_message,
                "status_code": "429",
                "timestamp": _NOW,
                "cooldown_time": expected_ttl,
            },
        )
    ]


@pytest.mark.asyncio
async def test_sync_and_async_router_cooldown_surfaces_share_active_state() -> None:
    store: Final = RouterCooldownStore(None, clock=lambda: _NOW)
    cache: Final = CooldownCache(store, default_cooldown_time=60.0, clock=lambda: _NOW)
    cache.add_deployment_to_cooldown("dep", Exception("unavailable"), 503, 30.0)
    expected: Final = [
        ("dep", {"exception_received": "unavailable", "status_code": "503", "timestamp": _NOW, "cooldown_time": 30.0})
    ]
    assert cache.get_active_cooldowns(["absent", "dep"], None) == expected
    assert await cache.async_get_active_cooldowns(["absent", "dep"], None) == expected
    assert cache.get_min_cooldown(["dep"], None) == 30.0
    assert cache.get_min_cooldown(["absent"], None) == 60.0
