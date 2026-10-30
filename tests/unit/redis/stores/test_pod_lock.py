from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import pytest

from litellm.redis.stores.pod_lock import TransportPodLockStore, Vacant
from litellm.redis.transport.contract import Arg, EvalReply, RedisFailure


@dataclass(frozen=True, slots=True)
class ExpiredLockTransport:
    async def get(self, key: str) -> bytes | None | RedisFailure:
        return None

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure:
        return not nx

    async def mget(self, keys: Sequence[str]) -> tuple[bytes | None, ...] | RedisFailure:
        raise AssertionError("unexpected mget")

    async def delete(self, *keys: str) -> int | RedisFailure:
        raise AssertionError("unexpected delete")

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure:
        raise AssertionError("unexpected eval")


@pytest.mark.asyncio
async def test_acquire_is_vacant_when_the_lock_expires_after_losing_set_nx() -> None:
    store: Final = TransportPodLockStore(ExpiredLockTransport())

    assert await store.acquire("test_job", "pod", ttl=timedelta(seconds=60)) == Vacant()
