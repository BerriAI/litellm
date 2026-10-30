import hashlib
import json
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck
from tests._support.fake_redis import (
    fake_redis_cache,
    keys_matching,
    local_get,
    local_set,
    seed_async_client,
)


async def _pre_call(checker: DeploymentAffinityCheck, request: dict[str, object], call_type: None = None) -> None:
    await checker.async_pre_call_deployment_hook(request, call_type)  # pyright: ignore[reportUnknownMemberType]  # legacy cache API is untyped


async def _read_session(
    checker: DeploymentAffinityCheck,
    model_group: str,
    session_id: str,
    user_key: str | None,
    ttl_seconds: int,
    parent_otel_span: None,
) -> object:
    return await checker._read_session_pin(model_group, session_id, user_key, ttl_seconds, parent_otel_span)  # pyright: ignore[reportPrivateUsage]  # exercises the pin path without a full router request


@pytest.fixture(autouse=True)
def fixed_clock() -> Iterator[None]:
    with patch("time.time", return_value=1700000000.0):
        yield


def _checker(server: fakeredis.FakeServer, *, repin: bool = True) -> DeploymentAffinityCheck:
    return DeploymentAffinityCheck(
        DualCache(redis_cache=fake_redis_cache(server, namespace="session")),
        60,
        False,
        False,
        enable_session_id_affinity=True,
        session_affinity_repin_on_failover=repin,
    )


def _key(caller: str = "caller") -> str:
    return "session:deployment_affinity:v1:session:group:" + hashlib.sha256(caller.encode()).hexdigest() + ":same"


def _request(model_id: str, caller: str = "caller") -> dict[str, object]:
    return {
        "model_info": {"id": model_id},
        "metadata": {
            "deployment_model_name": "group",
            "model_group": "group",
            "session_id": "same",
            "user_api_key_hash": caller,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("repin", [False, True])
@pytest.mark.parametrize("wire", [b'"chosen"', b'{"model_id": "chosen"}', b"{'model_id': 'chosen'}"])
async def test_session_reads_legacy_values_without_refresh(wire: bytes, repin: bool) -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server, repin=repin)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), wire, ex=20)
    assert await _read_session(checker, "group", "same", "caller", 60, None) in ("chosen", {"model_id": "chosen"})
    assert await raw.get(_key()) == wire
    assert await raw.pttl(_key()) == 20000


@pytest.mark.asyncio
async def test_session_claim_bytes_scope_and_claim_time_ttl() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await _pre_call(checker, _request("部署"), None)
    assert await raw.get(_key()) == b'{"model_id": "\\u90e8\\u7f72"}'
    assert await raw.pttl(_key()) == 60000
    await raw.pexpire(_key(), 20000)
    await _pre_call(checker, _request("other"), None)
    assert await raw.pttl(_key()) == 20000
    await _pre_call(checker, _request("部署"), None)
    assert await raw.pttl(_key()) == 60000
    await _pre_call(checker, _request("other", "second"), None)
    assert await raw.get(_key("second")) == b'{"model_id": "other"}'
    assert await keys_matching(raw, "session:session:*") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,expected",
    [
        (b'{"model_id": "chosen"}', b'{"model_id": "chosen"}'),
        (b'"chosen"', b'"chosen"'),
        (b"chosen", b"chosen"),
        (b'{"model_id": "removed"}', b'{"model_id": "replacement"}'),
        (b'{"model_id": "chosen", "extra": true}', b'{"model_id": "replacement"}'),
        (b"invalid{", b'{"model_id": "replacement"}'),
    ],
)
async def test_success_claim_preserves_only_eligible_complete_values(wire: bytes, expected: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), wire, ex=20)
    await checker.async_repin_session_after_success(
        _request("replacement"), "group", frozenset(("chosen", "replacement"))
    )
    assert await raw.get(_key()) == expected
    assert await raw.pttl(_key()) == 60000


@pytest.mark.asyncio
async def test_shared_miss_clears_local_while_outage_retains_it() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    local_key: Final = _key().removeprefix("session:")
    local_set(checker.cache, local_key, {"model_id": "stale"}, ttl=60)
    assert await _read_session(checker, "group", "same", "caller", 60, None) is None
    assert local_get(checker.cache, local_key) is None
    local_set(checker.cache, local_key, {"model_id": "local"}, ttl=60)
    server.connected = False
    assert await _read_session(checker, "group", "same", "caller", 60, None) == {"model_id": "local"}


class ConcurrentRead(fakeredis.FakeAsyncRedis):
    after_read: Callable[[], None]

    async def get(self, name: str) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # narrows fakeredis for the test
        result = await super().get(name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # fakeredis is untyped
        self.after_read()
        return result  # pyright: ignore[reportUnknownVariableType]  # fakeredis is untyped


@pytest.mark.asyncio
async def test_shared_read_does_not_overwrite_a_newer_local_claim() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), json.dumps({"model_id": "shared"}), ex=20)
    local_key: Final = _key().removeprefix("session:")
    client: Final = ConcurrentRead(server=server)
    client.after_read = lambda: local_set(checker.cache, local_key, {"model_id": "newer"}, ttl=60)
    redis_cache: Final = checker.cache.redis_cache
    assert redis_cache is not None
    seed_async_client(redis_cache, client)
    assert await _read_session(checker, "group", "same", "caller", 60, None) == {"model_id": "shared"}
    assert local_get(checker.cache, local_key) == {"model_id": "newer"}


class LostClaimReply(fakeredis.FakeAsyncRedis):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | int) -> object:
        await super().eval(script, numkeys, *keys_and_args)  # pyright: ignore[reportUnknownMemberType]  # fakeredis is untyped
        raise RedisTimeoutError("reply lost after apply")


@pytest.mark.asyncio
async def test_session_unknown_write_retains_local_continuity() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    redis_cache: Final = checker.cache.redis_cache
    assert redis_cache is not None
    seed_async_client(redis_cache, LostClaimReply(server=server))
    local_key: Final = _key().removeprefix("session:")
    local_set(checker.cache, local_key, {"model_id": "local"}, ttl=60)
    await _pre_call(checker, _request("shared"), None)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get(_key()) == b'{"model_id": "shared"}'
    assert local_get(checker.cache, local_key) == {"model_id": "local"}
