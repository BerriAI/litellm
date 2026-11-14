import pytest

from litellm.caching.caching import DualCache
from litellm.exceptions import RateLimitErrorCategory
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.utils import InternalUsageCache, hash_token
from tests._support.rust_parallel_request_limiter import PROXY_MaxParallelRequestsHandler_v3


@pytest.fixture(autouse=True)
def _native_parallel_request_limiter(monkeypatch):
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")


async def _admit(handler: PROXY_MaxParallelRequestsHandler_v3, auth: UserAPIKeyAuth, content: str = "hi") -> None:
    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=DualCache(),
        data={"model": "gpt-4o", "messages": [{"role": "user", "content": content}], "max_tokens": 10},
        call_type="acompletion",
    )


class TestProxyHooksActuallyRaiseProxyRateLimitError:
    @pytest.mark.asyncio
    async def test_parallel_request_limiter_v3_handle_rate_limit_error_raises(self):
        handler = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))
        auth = UserAPIKeyAuth(api_key=hash_token("sk-test"), rpm_limit=1)
        await _admit(handler, auth)
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await _admit(handler, auth)
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert e.headers is not None
        assert {"retry-after", "rate_limit_type", "reset_at"}.issubset(e.headers.keys())


class TestProxyHooksWireTypeCorrectly:
    @pytest.mark.asyncio
    async def test_v3_limiter_handle_rate_limit_error_propagates_type(self):
        handler = PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=InternalUsageCache(DualCache()),
        )
        auth = UserAPIKeyAuth(api_key=hash_token("sk-test"), tpm_limit=100)
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await _admit(handler, auth, content="x" * 2000)
        e = exc_info.value
        assert e.rate_limit_type == "tokens"
        assert e.headers is not None
        assert e.headers.get("rate_limit_type") == "tokens"
        assert e.headers.get("retry-after") is not None

    @pytest.mark.asyncio
    async def test_v3_limiter_max_parallel_requests_maps_to_concurrent(self):
        handler = PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=InternalUsageCache(DualCache()),
        )
        auth = UserAPIKeyAuth(api_key=hash_token("sk-test"), max_parallel_requests=1)
        await _admit(handler, auth)
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await _admit(handler, auth)
        assert exc_info.value.rate_limit_type == "concurrent_requests"
        assert exc_info.value.headers["rate_limit_type"] == "max_parallel_requests"
