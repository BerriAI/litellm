from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import pytest
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, wire_server

from litellm import Router
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.health_check import build_deployment_health_states
from litellm.proxy.health_check_utils.recent_traffic import RecentTrafficHealth
from litellm.proxy.health_check_utils.shared_health_check_manager import SharedHealthCheckManager


@pytest.mark.asyncio
async def test_workers_share_business_outcomes_and_reject_a_cached_success_after_a_failure(tmp_path: Path) -> None:
    clock: Final = Mock(return_value=100.0)
    first: Final = datetime.fromtimestamp(100, tz=timezone.utc)
    model: Final = {"model_info": {"id": "a", "mode": "chat"}, "litellm_params": {"model": "openai/test"}}
    event: Final = {"model_id": "a", "call_type": "acompletion", "status": "success"}
    details: Final = {
        "current_attempt_api_call_start_time": first,
        "response_headers": {"content-type": "application/json"},
    }
    with owned_redis(tmp_path) as server:
        redis: Final = RedisCache(host=server.host, port=server.port)
        writer: Final = RecentTrafficHealth(10, lambda: redis, clock)
        reader: Final = RecentTrafficHealth(10, lambda: redis, clock)
        manager: Final = SharedHealthCheckManager(redis_cache=redis, traffic_aware=True)
        try:
            await writer.async_log_success_event({**details, "standard_logging_object": event}, {}, first, first)
            recent: Final = await reader.snapshot([model])
            assert recent.successes == {"a": 100.0}
            await manager.cache_health_check_results(
                [{"model_id": "a", "health_check_source": "recent_traffic", "last_success_at": 100.0}], []
            )
            cached: Final = await manager.get_cached_health_check_results()
            assert cached is not None
            assert recent.accepts(cached["healthy_endpoints"], cached["timestamp"])
            assert await SharedHealthCheckManager(redis_cache=redis).get_cached_health_check_results() is None
            clock.return_value = 101.0
            failure: Final = {**details, "standard_logging_object": {**event, "status": "failure"}}
            await writer.async_log_failure_event(failure, None, first, datetime.fromtimestamp(101, tz=timezone.utc))
            await writer.async_log_success_event({**details, "standard_logging_object": event}, {}, first, first)
            failed: Final = await reader.snapshot([model])
            assert not failed.successes
            assert not failed.accepts(cached["healthy_endpoints"], cached["timestamp"])
        finally:
            await redis.disconnect()


@pytest.mark.asyncio
async def test_shared_recent_success_does_not_hide_an_outstanding_failed_probe(tmp_path: Path) -> None:
    reply: Final = Reply(
        body=b'{"id":"chatcmpl-probe","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
        b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
    )
    finished: Final = datetime.fromtimestamp(100, tz=timezone.utc)
    with owned_redis(tmp_path) as server, wire_server(lambda _: reply) as upstream:
        redis: Final = RedisCache(host=server.host, port=server.port)
        model: Final = {
            "model_name": "backup",
            "model_info": {"id": "served", "mode": "chat"},
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": f"{upstream.url}/v1", "api_key": "test"},
        }
        router: Final = Router(model_list=[model], enable_health_check_routing=True)
        writer: Final = RecentTrafficHealth(60, lambda: redis, lambda: 100.0)
        reader: Final = RecentTrafficHealth(60, lambda: redis, lambda: 100.0)
        manager: Final = SharedHealthCheckManager(redis_cache=redis, traffic_aware=True)
        try:
            router.health().health_checks.set_deployment_health_states(
                build_deployment_health_states([], [{"model_id": "served"}])
            )
            await writer.async_log_success_event(
                {
                    "standard_logging_object": {"model_id": "served", "call_type": "acompletion", "status": "success"},
                    "current_attempt_api_call_start_time": finished,
                    "response_headers": {"content-type": "application/json"},
                },
                {},
                finished,
                finished,
            )
            assert (await reader.snapshot([model])).successes == {"served": 100.0}
            healthy, unhealthy, _ = await manager.perform_shared_health_check(
                [model], router=router, recent_traffic=reader
            )
            assert not unhealthy
            assert [endpoint["model_id"] for endpoint in healthy] == ["served"]
            assert "health_check_source" not in healthy[0]
            assert [req.target for req in upstream.drain()] == ["/v1/chat/completions"]
        finally:
            router.retire()
            await redis.disconnect()
