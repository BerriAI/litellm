import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.exceptions import RateLimitError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.rate_limiter_utils import PROXY_LLM_PROVIDER_FALLBACK
from litellm.proxy.utils import InternalUsageCache, hash_token
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3


@pytest.fixture(autouse=True)
def _native_parallel_request_limiter(monkeypatch):
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")


async def _over_limit(handler: PROXY_MaxParallelRequestsHandler_v3, model: str | None) -> None:
    auth = UserAPIKeyAuth(api_key=hash_token("sk-provider-field"), rpm_limit=1)
    body = {} if model is None else {"model": model}
    await handler.async_pre_call_hook(
        user_api_key_dict=auth, cache=DualCache(), data=dict(body), call_type="acompletion"
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=auth, cache=DualCache(), data=dict(body), call_type="acompletion"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, expected_provider",
    [
        ("gpt-4o-mini", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
    ],
)
async def test_parallel_request_limiter_v3_populates_provider(model, expected_provider):
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))

    with pytest.raises(HTTPException) as exc_info:
        await _over_limit(handler, model)

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == expected_provider
    assert exc.model


@pytest.mark.asyncio
async def test_parallel_request_limiter_v3_unknown_model_falls_back():
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))

    with pytest.raises(HTTPException) as exc_info:
        await _over_limit(handler, "totally-bogus")

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == "totally-bogus"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v3_missing_model_falls_back():
    handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))

    with pytest.raises(HTTPException) as exc_info:
        await _over_limit(handler, None)

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == ""
