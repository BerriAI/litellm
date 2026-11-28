import asyncio
import contextvars
import os
import uuid
from typing import Final

import pytest
import redis

import litellm
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.types.utils import CallTypes

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_a_tool_calls_tpm_check_inside_a_request_gives_its_reservation_back_before_the_tool_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=host, port=port)
    proxy_logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    proxy_logging.update_values(redis_cache=redis_cache)
    proxy_logging._add_proxy_hooks()  # pyright: ignore[reportPrivateUsage]  # the proxy's startup registration
    limiter: Final = next(
        callback for callback in litellm.callbacks if isinstance(callback, PROXY_MaxParallelRequestsHandler_v3)
    )
    key_hash: Final = hash_token(f"sk-tool-call-tpm-{uuid.uuid4()}")
    auth: Final = UserAPIKeyAuth(api_key=key_hash, tpm_limit=100_000)
    request: Final = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Add 2 and 3 with the add tool."}],
        "max_tokens": 20,
        "litellm_call_id": f"call-{uuid.uuid4()}",
    }
    tool_kwargs: Final = {
        "name": "add",
        "arguments": {"a": 2, "b": 3},
        "server_name": "math",
        "user_api_key_auth": auth,
    }
    tool_call: Final = proxy_logging.convert_mcp_to_llm_format(
        proxy_logging.create_mcp_request_object_from_kwargs(tool_kwargs), tool_kwargs
    )
    counters: Final = redis.Redis(host=host, port=port, decode_responses=True)
    tokens: Final = f"{{api_key:{key_hash}}}:tokens"

    with request_redis_batch_scope(post_call_deadline=3600):
        context: Final = contextvars.copy_context()
        await asyncio.create_task(proxy_logging.pre_call_hook(auth, request, "acompletion"), context=context)
        held_by_the_request: Final = counters.get(tokens)
        await asyncio.create_task(
            proxy_logging.pre_call_hook(auth, tool_call, CallTypes.call_mcp_tool.value), context=context
        )
        held_once_the_tool_runs: Final = counters.get(tokens)
    await redis_cache.disconnect()
    counters.close()

    assert limiter.rate_limits.has_redis
    assert held_by_the_request is not None
    assert held_once_the_tool_runs == held_by_the_request
