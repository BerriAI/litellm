import asyncio
import json
import os
import uuid
from typing import Final
from unittest.mock import patch

import pytest
from redis import Redis
from redis.exceptions import ResponseError

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.hooks.max_budget_per_session_limiter import PROXY_MaxBudgetPerSessionHandler
from litellm.proxy.hooks.max_iterations_limiter import PROXY_MaxIterationsHandler
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse
from tests.integration._support.client import eventually


def _migration_agent(agent_id: str) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name=agent_id,
        litellm_params={"max_iterations": 200, "max_budget_per_session": 10.0},
        agent_card_params={"name": agent_id, "version": "1"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_value", (" 0.5", "0.5 ", b"0.5\x00", b"0.5\x00junk"))
@pytest.mark.parametrize("operation", ("read", "increment"))
async def test_redis_budget_rejects_invalid_numeric_text_before_mutating_the_ledger(
    agent_value: str | bytes, operation: str
) -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=host, port=port)
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(DualCache(redis_cache=redis_cache)))
    session_id: Final = f"whitespace-budget-{uuid.uuid4().hex}"
    legacy_key: Final = redis_cache.check_and_fix_namespace(handler._make_legacy_cache_key(session_id))
    scope_key: Final = redis_cache.check_and_fix_namespace(handler._make_agent_scope_cache_key(session_id))
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    with Redis(host=host, port=port) as raw:
        try:
            raw.set(legacy_key, "0.5")
            raw.hset(scope_key, mapping={"__total_new": "0.5", agent_field: agent_value})
            previous_legacy: Final = raw.get(legacy_key)
            previous_scope: Final = raw.hgetall(scope_key)

            if operation == "read":
                with pytest.raises(ResponseError):
                    await handler._get_agent_spend(session_id, "agent-a")
            else:
                with pytest.raises(ResponseError):
                    await handler._increment_agent_spend(session_id, "agent-a", 0.25)

            assert raw.get(legacy_key) == previous_legacy
            assert raw.hgetall(scope_key) == previous_scope
        finally:
            raw.delete(legacy_key, scope_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("counter", ("legacy", "total_new", "agent"))
async def test_redis_iteration_rejects_leading_zero_before_mutating_the_ledger(counter: str) -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=host, port=port)
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(DualCache(redis_cache=redis_cache)))
    session_id: Final = f"leading-zero-iterations-{uuid.uuid4().hex}"
    legacy_key: Final = redis_cache.check_and_fix_namespace(handler._make_legacy_cache_key(session_id))
    scope_key: Final = redis_cache.check_and_fix_namespace(handler._make_agent_scope_cache_key(session_id))
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    with Redis(host=host, port=port) as raw:
        try:
            raw.set(legacy_key, "01" if counter == "legacy" else "1")
            raw.hset(
                scope_key,
                mapping={
                    "__total_new": "01" if counter == "total_new" else "1",
                    agent_field: "01" if counter == "agent" else "1",
                },
            )
            previous_legacy: Final = raw.get(legacy_key)
            previous_scope: Final = raw.hgetall(scope_key)

            with pytest.raises(ResponseError):
                await handler._increment_agent_and_get(session_id, "agent-a")

            assert raw.get(legacy_key) == previous_legacy
            assert raw.hgetall(scope_key) == previous_scope
        finally:
            raw.delete(legacy_key, scope_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("counter", ("legacy", "legacy_with_scope", "total_new", "agent"))
@pytest.mark.parametrize("invalid_value", ("bad", -2.0, float("nan"), float("inf")))
async def test_redis_budget_rejects_malformed_ledger(counter: str, invalid_value: object) -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=host, port=port)
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(DualCache(redis_cache=redis_cache)))
    session_id: Final = f"malformed-budget-{uuid.uuid4().hex}"
    legacy_key: Final = redis_cache.check_and_fix_namespace(handler._make_legacy_cache_key(session_id))
    scope_key: Final = redis_cache.check_and_fix_namespace(handler._make_agent_scope_cache_key(session_id))
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    with Redis(host=host, port=port) as raw:
        try:
            raw.set(legacy_key, json.dumps(invalid_value) if counter.startswith("legacy") else "2.0")
            if counter != "legacy":
                raw.hset(
                    scope_key,
                    mapping={
                        "__total_new": json.dumps(invalid_value) if counter == "total_new" else "2.0",
                        agent_field: json.dumps(invalid_value) if counter == "agent" else "0.2",
                    },
                )
            with pytest.raises(ResponseError):
                await handler._get_agent_spend(session_id, "agent-a")
            with pytest.raises(ResponseError):
                await handler._increment_agent_spend(session_id, "agent-a", 0.1)
        finally:
            raw.delete(legacy_key, scope_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_value", ("bad", -5, 0.9, float("nan"), float("inf")))
async def test_redis_iteration_rejects_malformed_legacy_ledger(invalid_value: object) -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=host, port=port)
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(DualCache(redis_cache=redis_cache)))
    session_id: Final = f"malformed-iterations-{uuid.uuid4().hex}"
    legacy_key: Final = redis_cache.check_and_fix_namespace(handler._make_legacy_cache_key(session_id))
    scope_key: Final = redis_cache.check_and_fix_namespace(handler._make_agent_scope_cache_key(session_id))
    with Redis(host=host, port=port) as raw:
        try:
            raw.set(legacy_key, json.dumps(invalid_value))
            with pytest.raises(ResponseError):
                await handler._increment_agent_and_get(session_id, "agent-a")
        finally:
            raw.delete(legacy_key, scope_key)


@pytest.mark.asyncio
async def test_redis_budget_migration_is_atomic_and_preserves_session_ttl() -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis: Final = RedisCache(host=host, port=port)
    redis_client: Final = redis.init_async_client()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(DualCache(redis_cache=redis)))
    handler.ttl = 30
    registry: Final = AgentRegistry()
    registry.register_agent(_migration_agent("agent-a"))
    registry.register_agent(_migration_agent("agent-b"))
    session_id: Final = f"budget-migration-{uuid.uuid4().hex}"
    concurrent_session: Final = f"budget-concurrent-{uuid.uuid4().hex}"

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        try:
            legacy_key: Final = handler._make_legacy_cache_key(session_id)
            agent_scope_key: Final = handler._make_agent_scope_cache_key(session_id)
            concurrent_legacy_key: Final = handler._make_legacy_cache_key(concurrent_session)
            concurrent_scope_key: Final = handler._make_agent_scope_cache_key(concurrent_session)
            keys: Final = (legacy_key, agent_scope_key, concurrent_legacy_key, concurrent_scope_key)
            await redis.async_set_cache(key=legacy_key, value=0.5, ttl=30)
            legacy_redis_key: Final = redis.check_and_fix_namespace(legacy_key)
            initial_ttl: Final = await redis_client.pttl(legacy_redis_key)

            await handler._increment_agent_spend(session_id, "agent-a", 0.4)
            await redis.async_increment(legacy_key, 0.2, ttl=30)  # Old pod updates only the aggregate.
            await handler._increment_agent_spend(session_id, "agent-a", 0.1)
            await handler._increment_agent_spend(session_id, "agent-b", 0.1)
            assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(1.2)
            assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(0.8)

            hash_tags: Final = {key[key.index("{") : key.index("}") + 1] for key in keys[:2]}
            assert len(hash_tags) == 1
            ttls: Final = await redis_client.eval(
                "return {redis.call('PTTL', KEYS[1]), redis.call('PTTL', KEYS[2])}",
                2,
                redis.check_and_fix_namespace(agent_scope_key),
                legacy_redis_key,
            )
            assert 0 <= ttls[0] <= ttls[1]
            assert ttls[1] <= initial_ttl
            await redis_client.pexpire(legacy_redis_key, 15000)
            await redis_client.pexpire(redis.check_and_fix_namespace(agent_scope_key), 14999)
            before_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            before_increment_sidecar_ttl: Final = await redis_client.pttl(
                redis.check_and_fix_namespace(agent_scope_key)
            )
            await handler._increment_agent_spend(session_id, "agent-a", 0.01)
            after_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            after_increment_sidecar_ttl: Final = await redis_client.pttl(redis.check_and_fix_namespace(agent_scope_key))
            assert after_increment_ttl <= before_increment_ttl
            assert after_increment_sidecar_ttl <= before_increment_sidecar_ttl

            # Losing the hash evicts migration and agent counters together; the aggregate remains a safe baseline.
            await redis_client.delete(redis.check_and_fix_namespace(agent_scope_key))
            assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(1.31)
            await handler._increment_agent_spend(session_id, "agent-a", 0.01)
            assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(1.32)

            await redis.async_set_cache(key=concurrent_legacy_key, value=5.0, ttl=30)
            await asyncio.gather(
                *(handler._increment_agent_spend(concurrent_session, "agent-a", 0.01) for _ in range(100)),
                *(handler._increment_agent_spend(concurrent_session, "agent-b", 0.01) for _ in range(50)),
            )
            assert await handler._get_agent_spend(concurrent_session, "agent-a") == pytest.approx(6.0)
            assert await handler._get_agent_spend(concurrent_session, "agent-b") == pytest.approx(5.5)

            saved_total_new: Final = await redis_client.hget(
                redis.check_and_fix_namespace(agent_scope_key), "__total_new"
            )
            await redis_client.hdel(redis.check_and_fix_namespace(agent_scope_key), "__total_new")
            with pytest.raises(ResponseError, match="agent session scope is missing its migration total"):
                await handler._get_agent_spend(session_id, "agent-a")
            assert saved_total_new is not None
            await redis_client.hset(redis.check_and_fix_namespace(agent_scope_key), "__total_new", saved_total_new)

            await redis_client.pexpire(legacy_redis_key, 100)
            with Redis(host=host, port=port) as raw:
                eventually(lambda: raw.exists(legacy_redis_key), lambda value: value == 0)
            with pytest.raises(ResponseError, match="legacy session spend expired before agent scope"):
                await handler._get_agent_spend(session_id, "agent-a")
        finally:
            await redis_client.delete(*(redis.check_and_fix_namespace(key) for key in keys))


@pytest.mark.asyncio
async def test_redis_iteration_migration_is_atomic_and_preserves_session_ttl() -> None:
    host: Final = os.environ["REDIS_HOST"]
    port: Final = int(os.environ["REDIS_PORT"])
    redis: Final = RedisCache(host=host, port=port)
    redis_client: Final = redis.init_async_client()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(DualCache(redis_cache=redis)))
    handler.ttl = 30
    registry: Final = AgentRegistry()
    registry.register_agent(_migration_agent("agent-a"))
    registry.register_agent(_migration_agent("agent-b"))
    session_id: Final = f"iterations-migration-{uuid.uuid4().hex}"

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        try:
            legacy_key: Final = handler._make_legacy_cache_key(session_id)
            agent_scope_key: Final = handler._make_agent_scope_cache_key(session_id)
            keys: Final = (legacy_key, agent_scope_key)
            await redis.async_set_cache(key=legacy_key, value=3, ttl=30)
            legacy_redis_key: Final = redis.check_and_fix_namespace(legacy_key)
            initial_ttl: Final = await redis_client.pttl(legacy_redis_key)
            hash_tags: Final = {key[key.index("{") : key.index("}") + 1] for key in keys}
            assert len(hash_tags) == 1

            assert await handler._increment_agent_and_get(session_id, "agent-a") == 4
            assert await handler._increment_agent_and_get(session_id, "agent-b") == 4
            await redis_client.incr(legacy_redis_key)  # Old pod updates only the aggregate.
            assert await handler._increment_agent_and_get(session_id, "agent-a") == 6
            assert await handler._increment_agent_and_get(session_id, "agent-b") == 6

            ttls: Final = await redis_client.eval(
                "return {redis.call('PTTL', KEYS[1]), redis.call('PTTL', KEYS[2])}",
                2,
                redis.check_and_fix_namespace(agent_scope_key),
                legacy_redis_key,
            )
            assert 0 <= ttls[0] <= ttls[1]
            assert ttls[1] <= initial_ttl
            await redis_client.pexpire(legacy_redis_key, 15000)
            await redis_client.pexpire(redis.check_and_fix_namespace(agent_scope_key), 14999)
            before_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            before_increment_sidecar_ttl: Final = await redis_client.pttl(
                redis.check_and_fix_namespace(agent_scope_key)
            )
            assert await handler._increment_agent_and_get(session_id, "agent-a") == 7
            after_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            after_increment_sidecar_ttl: Final = await redis_client.pttl(redis.check_and_fix_namespace(agent_scope_key))
            assert after_increment_ttl <= before_increment_ttl
            assert after_increment_sidecar_ttl <= before_increment_sidecar_ttl

            values: Final = await asyncio.gather(
                *(handler._increment_agent_and_get(session_id, "agent-a") for _ in range(100))
            )
            assert sorted(values) == list(range(8, 108))

            # Losing the hash evicts migration and agent counters together; the aggregate remains a safe baseline.
            await redis_client.delete(redis.check_and_fix_namespace(agent_scope_key))
            assert await handler._increment_agent_and_get(session_id, "agent-b") == 110
            assert await handler._increment_agent_and_get(session_id, "agent-a") == 110

            saved_total_new: Final = await redis_client.hget(
                redis.check_and_fix_namespace(agent_scope_key), "__total_new"
            )
            await redis_client.hdel(redis.check_and_fix_namespace(agent_scope_key), "__total_new")
            with pytest.raises(ResponseError, match="agent session scope is missing its migration total"):
                await handler._increment_agent_and_get(session_id, "agent-a")
            assert saved_total_new is not None
            await redis_client.hset(redis.check_and_fix_namespace(agent_scope_key), "__total_new", saved_total_new)

            await redis_client.pexpire(legacy_redis_key, 100)
            with Redis(host=host, port=port) as raw:
                eventually(lambda: raw.exists(legacy_redis_key), lambda value: value == 0)
            with pytest.raises(ResponseError, match="legacy session count expired before agent scope"):
                await handler._increment_agent_and_get(session_id, "agent-a")
        finally:
            await redis_client.delete(*(redis.check_and_fix_namespace(key) for key in keys))
