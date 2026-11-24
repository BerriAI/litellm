from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import pytest
from integration._support.redis_process import owned_redis

from litellm.caching.redis_cache import RedisCache
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
