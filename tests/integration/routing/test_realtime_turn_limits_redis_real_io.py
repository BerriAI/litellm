import os
import uuid
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import CallTypes, Usage

pytestmark = pytest.mark.asyncio(loop_scope="session")

_TPM_LIMIT: Final = 1000
_MODEL: Final = "gpt-realtime"


@pytest.mark.parametrize(
    ("tpm_limit", "project_metadata"),
    [
        (_TPM_LIMIT, {}),
        (None, {"model_itpm_limit": {_MODEL: _TPM_LIMIT}}),
        (None, {"model_otpm_limit": {_MODEL: _TPM_LIMIT}}),
    ],
    ids=["key-tpm", "project-itpm", "project-otpm"],
)
async def test_a_realtime_turn_charged_inside_the_socket_redis_batch_refuses_the_next_response_at_once(
    monkeypatch: pytest.MonkeyPatch, tpm_limit: int | None, project_metadata: dict[str, dict[str, int]]
) -> None:
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    unique: Final = uuid.uuid4()
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token(f"sk-realtime-redis-{unique}"),
        tpm_limit=tpm_limit,
        project_id=f"project-rt-{unique}",
        project_metadata=project_metadata,
    )
    redis_cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    cache: Final = DualCache(redis_cache=redis_cache)
    handler: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache))

    with request_redis_batch_scope():
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={"model": _MODEL, "litellm_call_id": f"realtime-{uuid.uuid4()}", "metadata": {}},
            call_type=CallTypes.arealtime.value,
        )
        await handler.charge_realtime_turn_usage(
            auth, _MODEL, Usage(prompt_tokens=_TPM_LIMIT, completion_tokens=_TPM_LIMIT, total_tokens=2 * _TPM_LIMIT)
        )
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await handler.enforce_token_limits_for_frame(auth, _MODEL, call_type=CallTypes.arealtime.value)
    await redis_cache.disconnect()

    assert handler.rate_limits.has_redis
    assert exc_info.value.status_code == 429
