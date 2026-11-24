from typing import Final

import pytest

import litellm
from litellm import DualCache, Router
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3
from tests.unit.proxy.hooks.native_capacity_handler import NativeDynamicRateLimitHandler, listening


def _borrowing_handler(monkeypatch: pytest.MonkeyPatch) -> NativeDynamicRateLimitHandler:
    from litellm.types.utils import ModelLimitBorrowingConfig, PriorityReservationSettings

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(
        litellm,
        "priority_reservation_settings",
        PriorityReservationSettings(
            saturation_threshold=0.5,
            model_limit_borrowing_config=ModelLimitBorrowingConfig(borrow_idle_capacity=("team",)),
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
    handler: Final = NativeDynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=router)
    return handler


async def _model_saturation_at_the_hook(handler: NativeDynamicRateLimitHandler) -> float:
    with pytest.raises(ProxyRateLimitError) as refused:
        await handler.async_pre_call_hook(
            UserAPIKeyAuth(
                api_key="sk-borrow", team_id="team-a", team_metadata={"model_tpm_limit": {"shared-model": 100}}
            ),
            DualCache(),
            {"model": "shared-model", "messages": [{"role": "user", "content": "word " * 400}], "max_tokens": 10},
            "acompletion",
        )
    return float(str(refused.value.detail).split("Model saturation: ")[1].split("%")[0]) / 100


def _unreachable_redis_port() -> int:
    with listening() as (_, port):
        return port


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
    assert await _model_saturation_at_the_hook(handler) == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_a_worker_without_redis_reads_the_models_saturation_from_its_own_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    handler: Final = _borrowing_handler(monkeypatch)
    handler.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(DualCache()))
    assert not handler.v3_limiter.rate_limits.has_redis
    counter: Final = RateLimitScope("model_saturation_check", "shared-model").counter("tokens")
    handler.v3_limiter.rate_limits.set_local(counter, 700, 60)
    assert await _model_saturation_at_the_hook(handler) == pytest.approx(0.7)
