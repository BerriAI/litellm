from typing import Final

import pytest

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_rate_limit import RouterFallbackRateLimitCheck
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.router_api import RoutingContext

PRIMARY: Final = {"model_name": "primary", "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"}}
FALLBACK: Final = {"model_name": "fallback", "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"}}


def _router() -> Router:
    return Router(model_list=[PRIMARY, FALLBACK], fallbacks=[{"primary": ["fallback"]}])


def _limiter() -> PROXY_MaxParallelRequestsHandler_v3:
    return PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))


def _request(token: UserAPIKeyAuth) -> dict[str, object]:
    return {
        "model": "primary",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_api_key_auth": token},
    }


@pytest.mark.asyncio
async def test_fallback_target_is_admitted_until_the_keys_per_model_rpm_limit_is_spent() -> None:
    limiter: Final = _limiter()
    check: Final = RouterFallbackRateLimitCheck(limiter=lambda: limiter)
    token: Final = UserAPIKeyAuth(api_key="hashed", token="hashed", metadata={"model_rpm_limit": {"fallback": 2}})
    ctx: Final = RoutingContext(_router())
    verdicts: Final = tuple([await check(model="fallback", request_kwargs=_request(token), ctx=ctx) for _ in range(3)])
    assert verdicts == (True, True, False)


@pytest.mark.asyncio
async def test_fallback_target_without_a_per_model_limit_is_always_admitted() -> None:
    limiter: Final = _limiter()
    check: Final = RouterFallbackRateLimitCheck(limiter=lambda: limiter)
    token: Final = UserAPIKeyAuth(api_key="hashed", token="hashed", metadata={"model_rpm_limit": {"other": 1}})
    ctx: Final = RoutingContext(_router())
    verdicts: Final = tuple([await check(model="fallback", request_kwargs=_request(token), ctx=ctx) for _ in range(3)])
    assert verdicts == (True, True, True)


@pytest.mark.asyncio
async def test_request_without_a_key_is_unrestricted() -> None:
    check: Final = RouterFallbackRateLimitCheck(limiter=_limiter)
    assert await check(model="fallback", request_kwargs={"model": "primary"}, ctx=RoutingContext(_router())) is True


@pytest.mark.asyncio
async def test_proxy_without_the_v3_limiter_is_unrestricted() -> None:
    check: Final = RouterFallbackRateLimitCheck(limiter=lambda: None)
    token: Final = UserAPIKeyAuth(api_key="hashed", token="hashed", metadata={"model_rpm_limit": {"fallback": 0}})
    assert await check(model="fallback", request_kwargs=_request(token), ctx=RoutingContext(_router())) is True


@pytest.mark.asyncio
async def test_a_failing_limiter_fails_closed() -> None:
    class BrokenLimiter(PROXY_MaxParallelRequestsHandler_v3):
        async def async_fallback_target_allowed(
            self, user_api_key_dict: object, model: str, request_data: object
        ) -> bool:
            raise RuntimeError("redis down")

    limiter: Final = BrokenLimiter(internal_usage_cache=InternalUsageCache(DualCache()))
    check: Final = RouterFallbackRateLimitCheck(limiter=lambda: limiter)
    token: Final = UserAPIKeyAuth(api_key="hashed", token="hashed", metadata={"model_rpm_limit": {"fallback": 5}})
    assert await check(model="fallback", request_kwargs=_request(token), ctx=RoutingContext(_router())) is False
