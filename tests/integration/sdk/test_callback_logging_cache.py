import asyncio
import os
from typing import Final
from uuid import uuid4

import pytest

import litellm
from litellm import Cache, acompletion
from litellm.caching.caching import LiteLLMCacheType
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm._service_logger import ServiceLogging


async def _drain_cache_writes() -> None:
    await GLOBAL_LOGGING_WORKER.flush()
    await asyncio.gather(*_PENDING_CACHE_WRITES)


@pytest.fixture
def redis_response_cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    cache: Final = Cache(
        type=LiteLLMCacheType.REDIS,
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


@pytest.mark.asyncio
async def test_redis_cache_completion_stream(redis_response_cache: Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    messages: Final = [{"role": "user", "content": f"cache stream {uuid4().hex}"}]
    first_response: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        stream=True,
        mock_response="The same response is replayed.",
    )
    first_chunks: Final = tuple([chunk async for chunk in first_response])
    for _ in range(10):
        await asyncio.sleep(0)
        await _drain_cache_writes()
    second_response: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        stream=True,
        mock_response="A cache miss would return this.",
    )
    second_chunks: Final = tuple([chunk async for chunk in second_response])

    assert first_chunks[-1].id == second_chunks[-1].id
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "The same response is replayed."
    assert "".join(chunk.choices[0].delta.content or "" for chunk in second_chunks) == "The same response is replayed."


@pytest.mark.asyncio
async def test_completion_with_caching(redis_response_cache: Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    service_logger: Final = ServiceLogging(mock_testing=True)
    service_logger.prometheusServicesLogger.mock_testing = True
    redis_response_cache.cache.service_logger_obj = service_logger
    messages: Final = [{"role": "user", "content": f"prometheus cache {uuid4().hex}"}]

    await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="cached response",
    )
    await _drain_cache_writes()
    await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="unused cache miss response",
    )
    await _drain_cache_writes()

    assert service_logger.mock_testing_async_success_hook == 2
    assert service_logger.prometheusServicesLogger.mock_testing_success_calls == 2
    assert service_logger.mock_testing_sync_failure_hook == 0
    assert service_logger.mock_testing_async_failure_hook == 0
