"""RedisRequestPlan: one pipeline per RedisCache per round, lazily run when a declared future is awaited."""

import asyncio

import pytest

from litellm.caching.redis_request_plan import (
    RedisRequestPlan,
    active_redis_request_plan,
    redis_request_plan_scope,
)

from .redis_batch_fakes import FakeRedisCache, RecordingPipeline, RecordingRedisClient, json_value


def _cache(*pipes: RecordingPipeline) -> FakeRedisCache:
    return FakeRedisCache(RecordingRedisClient(list(pipes)))


@pytest.mark.asyncio
async def test_two_declarations_into_the_same_cache_resolve_in_one_round():
    pipe = RecordingPipeline(results=[[json_value(1)], json_value(2)])
    cache = _cache(pipe)
    plan = RedisRequestPlan()

    first = plan.batch_for(cache).mget(["a"])
    second = plan.batch_for(cache).get("b")

    assert await plan.resolve(first) == {"a": 1}
    assert second.done()
    assert await second == 2
    assert pipe.execute_count == 1
    assert plan.rounds == 1


@pytest.mark.asyncio
async def test_a_declaration_after_a_round_lands_in_the_next_round():
    pipes = [
        RecordingPipeline(results=[[json_value(1)]]),
        RecordingPipeline(results=[json_value(2)]),
    ]
    cache = _cache(*pipes)
    plan = RedisRequestPlan()

    assert await plan.resolve(plan.batch_for(cache).mget(["a"])) == {"a": 1}
    assert await plan.resolve(plan.batch_for(cache).get("b")) == 2

    assert [pipe.execute_count for pipe in pipes] == [1, 1]
    assert plan.rounds == 2


@pytest.mark.asyncio
async def test_batches_for_two_caches_execute_concurrently_in_one_round():
    pipe_a = RecordingPipeline(results=[json_value(1)])
    pipe_b = RecordingPipeline(results=[json_value(2)])
    plan = RedisRequestPlan()

    first = plan.batch_for(_cache(pipe_a)).get("a")
    second = plan.batch_for(_cache(pipe_b)).get("b")
    await plan.resolve(first)

    assert second.done()
    assert pipe_a.execute_count == 1
    assert pipe_b.execute_count == 1
    assert plan.rounds == 1


@pytest.mark.asyncio
async def test_flush_executes_anything_still_pending():
    pipe = RecordingPipeline(results=[True])
    cache = _cache(pipe)
    plan = RedisRequestPlan()

    future = plan.batch_for(cache).set("k", 1)
    assert not future.done()
    await plan.flush()

    assert future.done()
    assert pipe.calls == [("set", "k", "1", None)]
    assert plan.rounds == 1


@pytest.mark.asyncio
async def test_scope_installs_joins_and_resets_the_active_plan():
    assert active_redis_request_plan() is None
    with redis_request_plan_scope() as outer:
        assert active_redis_request_plan() is outer
        with redis_request_plan_scope() as inner:
            assert inner is outer
    assert active_redis_request_plan() is None


@pytest.mark.asyncio
async def test_sibling_tasks_each_get_their_own_plan():
    plans: list[RedisRequestPlan] = []

    async def request() -> None:
        with redis_request_plan_scope() as plan:
            plans.append(plan)
            await asyncio.sleep(0)

    await asyncio.gather(request(), request())
    assert len(plans) == 2
    assert plans[0] is not plans[1]
