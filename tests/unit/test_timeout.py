"""Unit tests for litellm.timeout decorator."""

import asyncio
import time
from typing import Final

import pytest

from litellm.exceptions import Timeout
from litellm.timeout import timeout


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_arg", ["request_timeout", "force_timeout"])
async def test_async_timeout_decorator_enforces_per_call_timeout(timeout_arg: str) -> None:
    @timeout(timeout_duration=60.0)
    async def hung_func(**kwargs):
        await asyncio.sleep(10.0)
        return "never"

    with pytest.raises(Timeout) as exc_info:
        await hung_func(**{timeout_arg: 0.001, "model": "test-model"})

    assert "0.001 second(s)" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_arg", ["request_timeout", "force_timeout"])
async def test_async_timeout_decorator_extends_duration_with_per_call_timeout(
    timeout_arg: str,
) -> None:
    @timeout(timeout_duration=0.001)
    async def delayed_func(**kwargs):
        await asyncio.sleep(0.01)
        return "completed"

    result: Final = await delayed_func(**{timeout_arg: 60.0, "model": "test-model"})
    assert result == "completed"


@pytest.mark.asyncio
async def test_async_timeout_decorator_handles_none_force_timeout() -> None:
    @timeout(timeout_duration=0.001)
    async def hung_func(**kwargs):
        await asyncio.sleep(10.0)
        return "never"

    with pytest.raises(Timeout) as exc_info:
        await hung_func(force_timeout=None, model="test-model")

    assert "0.001 second(s)" in str(exc_info.value)


def test_sync_timeout_decorator_enforces_per_call_timeout() -> None:
    @timeout(timeout_duration=60.0)
    def hung_sync_func(**kwargs):
        time.sleep(10.0)
        return "never"

    with pytest.raises(Timeout) as exc_info:
        hung_sync_func(request_timeout=0.001, model="test-model")

    assert "0.001 second(s)" in str(exc_info.value)
