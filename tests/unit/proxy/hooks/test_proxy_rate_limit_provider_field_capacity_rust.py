import pytest
from fastapi import HTTPException

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.exceptions import RateLimitError
from litellm.proxy._types import UserAPIKeyAuth
from tests.unit.proxy.hooks.native_capacity_handler import (
    NativeDynamicRateLimitHandler as PROXY_DynamicRateLimitHandlerV3,
)
from litellm.redis.stores.rate_limits import RateLimitScope
from tests.unit.proxy.hooks.native_capacity_handler import pre_call_at_saturation, seed_counter


def _gpt_4o_mini(rpm: int) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "gpt-4o-mini", "api_key": "test-key", "rpm": rpm},
            }
        ]
    )


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_model_capacity_path_populates_provider():
    handler = PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=_gpt_4o_mini(rpm=100))
    seed_counter(handler, RateLimitScope("model_saturation_check", "gpt-4o-mini").counter("requests"), 100)

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn-v3")
    user_api_key_dict.metadata = {}

    with pytest.raises(HTTPException) as exc_info:
        await pre_call_at_saturation(handler, user_api_key_dict, "gpt-4o-mini", saturation=1.0)

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"
