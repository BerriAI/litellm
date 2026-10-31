from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, NoReturn

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

    async def incrby(self, key: str, amount: int) -> NoReturn:
        raise AssertionError("unexpected incrby")

    async def incrbyfloat(self, key: str, amount: float) -> NoReturn:
        raise AssertionError("unexpected incrbyfloat")

    async def expire(self, key: str, ttl: timedelta) -> NoReturn:
        raise AssertionError("unexpected expire")

    async def ttl(self, key: str) -> NoReturn:
        raise AssertionError("unexpected ttl")

    async def rpush(self, key: str, *values: Arg) -> NoReturn:
        raise AssertionError("unexpected rpush")

    async def lpop(self, key: str, count: int | None = None) -> NoReturn:
        raise AssertionError("unexpected lpop")

    async def ltrim(self, key: str, start: int, stop: int) -> NoReturn:
        raise AssertionError("unexpected ltrim")

    async def sadd(self, key: str, *members: Arg) -> NoReturn:
        raise AssertionError("unexpected sadd")

    async def scan(self, prefix: str, *, limit: int) -> NoReturn:
        raise AssertionError("unexpected scan")

    def batch(self, *, transaction: bool = False) -> NoReturn:
        raise AssertionError("unexpected batch")

    def request_batch(self) -> NoReturn:
        raise AssertionError("unexpected request_batch")

    def post_call_batch(self) -> NoReturn:
        raise AssertionError("unexpected post_call_batch")

    async def publish(self, channel: str, message: str | bytes) -> NoReturn:
        raise AssertionError("unexpected publish")

    async def publish_all(self, channel: str, messages: Sequence[str | bytes]) -> NoReturn:
        raise AssertionError("unexpected publish_all")

    async def subscribe(self, *channels: str) -> NoReturn:
        raise AssertionError("unexpected subscribe")

    async def ping(self) -> NoReturn:
        raise AssertionError("unexpected ping")

    async def info(self) -> NoReturn:
        raise AssertionError("unexpected info")

    async def client_list(self) -> NoReturn:
        raise AssertionError("unexpected client_list")

    async def flushall(self) -> NoReturn:
        raise AssertionError("unexpected flushall")


@pytest.mark.asyncio
async def test_acquire_is_vacant_when_the_lock_expires_after_losing_set_nx() -> None:
    store: Final = TransportPodLockStore(ExpiredLockTransport())

    assert await store.acquire("test_job", "pod", ttl=timedelta(seconds=60)) == Vacant()
