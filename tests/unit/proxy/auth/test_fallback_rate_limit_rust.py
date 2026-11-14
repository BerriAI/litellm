from typing import Final

import pytest

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_rate_limit import RouterFallbackRateLimitCheck
from litellm.proxy.utils import InternalUsageCache
from litellm.router_api import RoutingContext
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3

PRIMARY: Final = {"model_name": "primary", "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"}}
FALLBACK: Final = {"model_name": "fallback", "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"}}


@pytest.fixture(autouse=True)
def _native_parallel_request_limiter(monkeypatch):
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")


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
