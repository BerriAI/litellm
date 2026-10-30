from datetime import timedelta
from typing import Final

import fakeredis
import pytest
from redis.exceptions import ResponseError

from litellm.redis.stores.mcp_single_use import AuthCodeUsed, TransportMcpSingleUseStore
from tests._support.fake_redis import fake_redis_transport


class _RedisWithoutEvalsha(fakeredis.FakeAsyncRedis):
    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: str | bytes | float) -> object:
        raise ResponseError("NOPERM this user has no permissions to run the 'evalsha' command")


@pytest.mark.asyncio
async def test_claim_counts_and_expires_with_plain_eval_when_evalsha_is_denied() -> None:
    raw: Final = _RedisWithoutEvalsha(server=fakeredis.FakeServer())
    store: Final = TransportMcpSingleUseStore(fake_redis_transport(raw, failure_threshold=1, namespace="ns"))
    marker: Final = AuthCodeUsed("jti-1")

    assert await store.claim(marker, ttl=timedelta(seconds=90)) == 1
    assert await store.claim(marker, ttl=timedelta(seconds=90)) == 2
    assert await raw.get("ns:mcp_gateway_dcr_code_used:jti-1") == b"2"
    assert 0 < await raw.pttl("ns:mcp_gateway_dcr_code_used:jti-1") <= 90_000
