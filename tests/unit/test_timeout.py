"""Unit tests for litellm.timeout decorator."""

import asyncio
from typing import Final

import pytest

from litellm.exceptions import Timeout
from litellm.timeout import timeout


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_arg", ["request_timeout", "force_timeout"])
async def test_async_timeout_decorator_passes_per_call_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout_arg: str
) -> None:
    captured_timeout: Final[list[float | None]] = []

    async def fake_wait_for(fut, timeout):
        captured_timeout.append(timeout)
        return await fut

    monkeypatch.setattr("litellm.timeout.asyncio.wait_for", fake_wait_for)

    @timeout(timeout_duration=5.0)
    async def sample_func(**kwargs):
        return "ok"

    result: Final = await sample_func(**{timeout_arg: 0.05, "model": "test-model"})
    assert result == "ok"
    assert captured_timeout == [0.05]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_arg", ["request_timeout", "force_timeout"])
async def test_async_timeout_decorator_raises_with_per_call_timeout_message(
    monkeypatch: pytest.MonkeyPatch, timeout_arg: str
) -> None:
    async def fake_wait_for_timeout(fut, timeout):
        fut.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr("litellm.timeout.asyncio.wait_for", fake_wait_for_timeout)

    @timeout(timeout_duration=5.0)
    async def sample_func(**kwargs):
        return "ok"

    with pytest.raises(Timeout) as exc_info:
        await sample_func(**{timeout_arg: 0.05, "model": "test-model"})

    assert "0.05 second(s)" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_timeout_decorator_handles_none_force_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_timeout: Final[list[float | None]] = []

    async def fake_wait_for(fut, timeout):
        captured_timeout.append(timeout)
        return await fut

    monkeypatch.setattr("litellm.timeout.asyncio.wait_for", fake_wait_for)

    @timeout(timeout_duration=5.0)
    async def sample_func(**kwargs):
        return "ok"

    await sample_func(force_timeout=None, model="test-model")
    assert captured_timeout == [5.0]


@pytest.mark.asyncio
async def test_async_timeout_decorator_uses_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_timeout: Final[list[float | None]] = []

    async def fake_wait_for(fut, timeout):
        captured_timeout.append(timeout)
        return await fut

    monkeypatch.setattr("litellm.timeout.asyncio.wait_for", fake_wait_for)

    @timeout(timeout_duration=42.0)
    async def sample_func(**kwargs):
        return "ok"

    await sample_func(model="test-model")
    assert captured_timeout == [42.0]


def test_sync_timeout_decorator_runs_without_clock() -> None:
    @timeout(timeout_duration=5.0)
    def sample_sync_func(**kwargs):
        return "sync_ok"

    result: Final = sample_sync_func(request_timeout=0.05, model="test-model")
    assert result == "sync_ok"
