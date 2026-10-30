import hashlib
import json
from collections.abc import Iterator
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
    legacy_get,
    local_get,
    local_set,
    seed_async_client,
)


async def _pre_call(checker: DeploymentAffinityCheck, request: dict[str, object], call_type: None = None) -> None:
    await checker.async_pre_call_deployment_hook(request, call_type)  # pyright: ignore[reportUnknownMemberType]  # legacy cache API is untyped


async def _filter(
    checker: DeploymentAffinityCheck,
    model: str,
    deployments: list[dict[str, object]],
    messages: None,
    request_kwargs: dict[str, object] | None,
) -> object:
    return await checker.async_filter_deployments(model, deployments, messages, request_kwargs)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy hook API is untyped


async def _claim_user(checker: DeploymentAffinityCheck, model_group: str, user_key: str, model_id: str) -> str | None:
    return await checker._claim_user_pin(model_group, user_key, model_id)  # pyright: ignore[reportPrivateUsage]  # exercises the pin path without a full router request


@pytest.fixture(autouse=True)
def fixed_clock() -> Iterator[None]:
    with patch("time.time", return_value=1700000000.0):
        yield


class LostClaimReply(fakeredis.FakeAsyncRedis):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | int) -> object:
        await super().eval(script, numkeys, *keys_and_args)  # pyright: ignore[reportUnknownMemberType]  # fakeredis is untyped
        raise RedisTimeoutError("reply lost after apply")


def _checker(server: fakeredis.FakeServer) -> DeploymentAffinityCheck:
    return DeploymentAffinityCheck(
        DualCache(redis_cache=fake_redis_cache(server, namespace="affinity")),
        ttl_seconds=60,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
    )


def _request(model_id: str) -> dict[str, object]:
    return {
        "metadata": {"deployment_model_name": "group", "user_api_key_hash": "caller"},
        "model_info": {"id": model_id},
    }


def _key() -> str:
    return "affinity:deployment_affinity:v1:group:" + hashlib.sha256(b"caller").hexdigest()


@pytest.mark.asyncio
async def test_user_claim_wire_winner_and_ttl() -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    checker: Final = _checker(server)
    await _pre_call(checker, _request("部署"), None)
    assert await raw.get(_key()) == b'{"model_id": "\\u90e8\\u7f72"}'
    assert 59000 <= await raw.pttl(_key()) <= 60000
    redis_cache: Final = checker.cache.redis_cache
    assert redis_cache is not None
    assert await legacy_get(redis_cache, _key()) == {"model_id": "部署"}
    await raw.pexpire(_key(), 20000)
    await _pre_call(checker, _request("other"), None)
    assert await raw.get(_key()) == b'{"model_id": "\\u90e8\\u7f72"}'
    assert 19000 <= await raw.pttl(_key()) <= 20000
    await _pre_call(checker, _request("部署"), None)
    assert 59000 <= await raw.pttl(_key()) <= 60000
    assert await keys_matching(raw, "affinity:affinity:*") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b'"chosen"', b'{"model_id": "chosen"}', b"{'model_id': 'chosen'}"])
async def test_user_read_accepts_legacy_wire_without_refresh(wire: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    checker: Final = _checker(server)
    await raw.set(_key(), wire, ex=20)
    deployments: Final[list[dict[str, object]]] = [
        {"model_name": "group", "litellm_params": {"model": "openai/group"}, "model_info": {"id": "chosen"}},
        {"model_name": "group", "litellm_params": {"model": "openai/group"}, "model_info": {"id": "other"}},
    ]
    result: Final = await _filter(checker, "group", deployments, None, _request("chosen"))
    assert result == deployments[:1]
    assert 19000 <= await raw.pttl(_key()) <= 20000


@pytest.mark.asyncio
async def test_user_late_attachment_and_outage_keep_local_winner() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = DeploymentAffinityCheck(DualCache(), 60, True, False)
    await _pre_call(checker, _request("local"), None)
    checker.cache.redis_cache = fake_redis_cache(server, namespace="affinity")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(_key(), json.dumps({"model_id": "shared"}), ex=60)
    await _pre_call(checker, _request("local"), None)
    assert local_get(checker.cache, _key().removeprefix("affinity:")) == {"model_id": "shared"}
    server.connected = False
    await _pre_call(checker, _request("other"), None)
    assert local_get(checker.cache, _key().removeprefix("affinity:")) == {"model_id": "shared"}


@pytest.mark.asyncio
async def test_user_unknown_write_does_not_replace_local_winner() -> None:
    server: Final = fakeredis.FakeServer()
    checker: Final = _checker(server)
    redis_cache: Final = checker.cache.redis_cache
    assert redis_cache is not None
    seed_async_client(redis_cache, LostClaimReply(server=server))
    local_set(checker.cache, _key().removeprefix("affinity:"), {"model_id": "local"}, ttl=60)
    await _pre_call(checker, _request("shared"), None)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get(_key()) == b'{"model_id": "shared"}'
    assert local_get(checker.cache, _key().removeprefix("affinity:")) == {"model_id": "local"}


@pytest.mark.asyncio
async def test_user_outage_preserves_legacy_mapping_with_non_string_keys() -> None:
    server: Final = fakeredis.FakeServer()
    server.connected = False
    checker: Final = _checker(server)
    winner: Final = {"model_id": "local", 1: "legacy"}
    local_set(checker.cache, _key().removeprefix("affinity:"), winner, ttl=60)
    assert await _claim_user(checker, "group", "caller", "shared") == "local"
    assert local_get(checker.cache, _key().removeprefix("affinity:")) == winner
