"""Unit tests for litellm.timeout decorator."""

import asyncio
import time
from typing import Final

import pytest

from litellm.exceptions import Timeout
from litellm.timeout import timeout


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_arg", ["request_timeout", "force_timeout"])
async def test_async_timeout_decorator_respects_per_call_timeout(timeout_arg: str):
    """An async function decorated with a large timeout must respect a shorter per-call timeout."""

    @timeout(timeout_duration=5.0)
    async def slow_async_func(**kwargs):
        await asyncio.sleep(0.5)
        return "ok"

    start_time: Final = time.monotonic()
    kwargs = {timeout_arg: 0.05, "model": "test-model"}

    with pytest.raises(Timeout) as exc_info:
        await slow_async_func(**kwargs)

    elapsed: Final = time.monotonic() - start_time
    assert elapsed < 0.3, f"Call waited {elapsed}s instead of respecting {timeout_arg}=0.05"
    assert "0.05 second(s)" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_timeout_decorator_extends_duration_with_per_call_timeout():
    """An async function with a small default timeout must not time out if request_timeout gives more time."""

    @timeout(timeout_duration=0.05)
    async def moderately_slow_async_func(**kwargs):
        await asyncio.sleep(0.1)
        return "completed"

    result = await moderately_slow_async_func(request_timeout=1.0, model="test-model")
    assert result == "completed"


@pytest.mark.asyncio
async def test_async_timeout_decorator_handles_none_force_timeout():
    """Passing force_timeout=None should not break timeout_duration fallback."""

    @timeout(timeout_duration=0.05)
    async def slow_async_func(**kwargs):
        await asyncio.sleep(0.2)
        return "ok"

    with pytest.raises(Timeout) as exc_info:
        await slow_async_func(force_timeout=None, model="test-model")

    assert "0.05 second(s)" in str(exc_info.value)


def test_sync_timeout_decorator_respects_per_call_timeout():
    """A sync function decorated with a large timeout must respect a shorter per-call timeout."""

    @timeout(timeout_duration=5.0)
    def slow_sync_func(**kwargs):
        time.sleep(0.5)
        return "ok"

    start_time: Final = time.monotonic()
    with pytest.raises(Timeout) as exc_info:
        slow_sync_func(request_timeout=0.05, model="test-model")

    elapsed: Final = time.monotonic() - start_time
    assert elapsed < 0.3, f"Call waited {elapsed}s instead of respecting request_timeout=0.05"
    assert "0.05 second(s)" in str(exc_info.value)
