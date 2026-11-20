import asyncio
import json
import os
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import Reply, Request, wire_server
from redis import Redis

import litellm
from litellm import Router
from litellm.caching.redis_cache import RedisCache

pytestmark = pytest.mark.timeout(240)

TTL_MS: Final = 30_000
_ROUNDS: Final = 10
_EVEN_LOAD: Final = (("a", 0), ("b", 0))
_REPLY: Final = json.dumps(
    {
        "id": "shared-pick",
        "object": "chat.completion",
        "created": 1,
        "model": "shared-pick",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
).encode()


@pytest.fixture
def picks() -> Iterator[tuple[RedisCache, str, str]]:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    key: Final = f"least-busy-picks-{uuid.uuid4()}"
    counters: Final = f"least-busy-counter-{uuid.uuid4()}"
    yield cache, key, counters
    cache.redis_client.delete(cache.check_and_fix_namespace(key=key))
    cache.redis_client.delete(*(cache.check_and_fix_namespace(key=f"{counters}:{name}") for name in "abcd"))


def test_a_pick_takes_the_least_loaded_candidate_and_holds_it_for_the_next_one(
    picks: tuple[RedisCache, str, str],
) -> None:
    cache, key, counters = picks
    counter_keys: Final = tuple(f"{counters}:{name}" for name in "abcd")
    cache.redis_client.set(cache.check_and_fix_namespace(key=counter_keys[0]), 2)
    cache.redis_client.set(cache.check_and_fix_namespace(key=counter_keys[3]), 1)
    for _ in range(2):
        cache.reserve_least_loaded(key, (("c", 0),), (), True, TTL_MS)
    floors: Final = (("a", 0), ("b", 2), ("c", 0), ("d", 1))

    assert cache.reserve_least_loaded(key, floors, counter_keys, True, TTL_MS)[0] == 3
    assert cache.reserve_least_loaded(key, floors, counter_keys, True, TTL_MS)[0] == 0


def test_a_lookup_holds_nothing_and_a_reservation_holds_its_key_no_longer_than_its_ttl(
    picks: tuple[RedisCache, str, str],
) -> None:
    cache, key, _ = picks

    assert cache.reserve_least_loaded(key, _EVEN_LOAD, (), False, TTL_MS) == (0, None)
    assert cache.reserve_least_loaded(key, _EVEN_LOAD, (), False, TTL_MS) == (0, None)
    assert cache.redis_client.exists(cache.check_and_fix_namespace(key=key)) == 0

    index, reservation = cache.reserve_least_loaded(key, _EVEN_LOAD, (), True, TTL_MS)

    assert index == 0 and reservation is not None and reservation.endswith(":a")
    assert 0 < cache.redis_client.pttl(cache.check_and_fix_namespace(key=key)) <= TTL_MS


def test_a_reservation_lapses_after_its_ttl(picks: tuple[RedisCache, str, str]) -> None:
    cache, key, _ = picks
    cache.reserve_least_loaded(key, _EVEN_LOAD, (), True, 300)

    assert cache.reserve_least_loaded(key, _EVEN_LOAD, (), False, TTL_MS)[0] == 1
    assert eventually(lambda: cache.reserve_least_loaded(key, _EVEN_LOAD, (), False, TTL_MS)[0], lambda i: i == 0) == 0


def test_releasing_a_reservation_frees_its_candidate(picks: tuple[RedisCache, str, str]) -> None:
    cache, key, _ = picks
    index, reservation = cache.reserve_least_loaded(key, _EVEN_LOAD, (), True, TTL_MS)
    assert reservation is not None

    cache.release_reservation(key, reservation)

    assert cache.reserve_least_loaded(key, _EVEN_LOAD, (), True, TTL_MS)[0] == index


async def test_the_async_pick_behaves_the_same_way(picks: tuple[RedisCache, str, str]) -> None:
    cache, key, counters = picks
    counter_keys: Final = (f"{counters}:a", f"{counters}:b")
    await cache.async_increment_with_floor(counter_keys[0], 1, 600)

    assert await cache.async_reserve_least_loaded(key, _EVEN_LOAD, counter_keys, False, TTL_MS) == (1, None)
    index, reservation = await cache.async_reserve_least_loaded(key, _EVEN_LOAD, counter_keys, True, TTL_MS)
    assert index == 1 and reservation is not None
    assert (await cache.async_reserve_least_loaded(key, _EVEN_LOAD, counter_keys, False, TTL_MS))[0] == 0

    await cache.async_release_reservation(key, reservation)

    assert (await cache.async_reserve_least_loaded(key, _EVEN_LOAD, counter_keys, False, TTL_MS))[0] == 1


def _held_until_both_arrive(arrived: threading.Barrier) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        arrived.wait()
        return Reply(body=_REPLY)

    return respond


def _router(group: str, deployment_ids: tuple[str, ...], api_base: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {"model": "openai/shared-pick", "api_key": "synthetic-key", "api_base": api_base},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in deployment_ids
        ],
        routing_strategy="least-busy",
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
        num_retries=0,
        disable_cooldowns=True,
    )


def _idle(redis: Redis, group: str, deployment_ids: tuple[str, ...]) -> tuple[tuple[int, ...], int]:
    counts: Final = redis.mget([f"{group}_request_count:{deployment_id}" for deployment_id in deployment_ids])
    return tuple(int(count or 0) for count in counts), int(redis.zcard(f"least_busy_picks:{group}"))


async def _burst(
    routers: tuple[Router, ...], redis: Redis, group: str, deployment_ids: tuple[str, ...]
) -> Counter[str]:
    answered: Final = await asyncio.gather(
        *(router.acompletion(model=group, messages=[{"role": "user", "content": "Say ok"}]) for router in routers)
    )
    await asyncio.to_thread(eventually, lambda: _idle(redis, group, deployment_ids), lambda state: state == ((0, 0), 0))
    return Counter(response._hidden_params["model_id"] for response in answered)


async def test_two_routers_sharing_redis_split_every_burst_across_idle_deployments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    group: Final = f"least-busy-shared-{uuid.uuid4().hex[:8]}"
    deployment_ids: Final = (f"{group}-a", f"{group}-b")
    with (
        wire_server(_held_until_both_arrive(threading.Barrier(2, timeout=60))) as upstream,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as redis,
    ):
        routers: Final = tuple(_router(group, deployment_ids, upstream.url + "/v1") for _ in range(2))
        splits: Final = tuple([await _burst(routers, redis, group, deployment_ids) for _ in range(_ROUNDS)])
        redis.delete(*(f"{group}_request_count:{deployment_id}" for deployment_id in deployment_ids))

    assert splits == (Counter(deployment_ids),) * _ROUNDS
