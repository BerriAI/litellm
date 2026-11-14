import socket
from typing import Final

import pytest

import litellm
from litellm import DualCache, Router
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    PROXY_DynamicRateLimitHandlerV3,
    _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the public name is a Final alias, not a type
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3


def _borrowing_handler(monkeypatch: pytest.MonkeyPatch) -> _PROXY_DynamicRateLimitHandlerV3:
    from litellm.types.utils import PriorityReservationSettings

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(
        litellm,
        "priority_reservation_settings",
        PriorityReservationSettings(
            saturation_threshold=0.5,
            team_model_limit_borrows_idle_capacity_from_other_teams=True,
            priority_class_share_splits_equally_among_active_teams=False,
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", ["dynamic_rate_limiter_v3"])
    router: Final = Router(
        model_list=[
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            },
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    handler = PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=router)
    return handler


def _unreachable_redis_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.asyncio
async def test_a_worker_whose_redis_read_fails_reads_the_models_saturation_from_its_own_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    handler: Final = _borrowing_handler(monkeypatch)
    unreachable: Final = RedisCache(host="127.0.0.1", port=_unreachable_redis_port())
    handler.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(DualCache(redis_cache=unreachable)))
    assert handler.v3_limiter.rate_limits.has_redis
    counter: Final = RateLimitScope("model_saturation_check", "shared-model").counter("tokens")
    handler.v3_limiter.rate_limits.set_local(counter, 700, 60)
    group: Final = handler.llm_router.registry().group("shared-model")
    assert group is not None
    assert await handler._check_model_saturation("shared-model", group) == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_a_worker_without_redis_reads_the_models_saturation_from_its_own_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    handler: Final = _borrowing_handler(monkeypatch)
    handler.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(DualCache()))
    assert not handler.v3_limiter.rate_limits.has_redis
    counter: Final = RateLimitScope("model_saturation_check", "shared-model").counter("tokens")
    handler.v3_limiter.rate_limits.set_local(counter, 700, 60)
    group: Final = handler.llm_router.registry().group("shared-model")
    assert group is not None
    assert await handler._check_model_saturation("shared-model", group) == pytest.approx(0.7)
