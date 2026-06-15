import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.streaming_handler import (
    CustomStreamWrapper,
    _StreamInactivityGuard,
)


class _FakeAsyncStream:
    """Async iterator driven by (delay_seconds, value_or_exception) steps."""

    def __init__(self, steps):
        self._steps = list(steps)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        delay, item = self._steps.pop(0)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(item, BaseException):
            raise item
        return item


async def _drain(aiterable):
    out = []
    async for item in aiterable:
        out.append(item)
    return out


def test_first_chunk_is_not_guarded():
    # First chunk carries time-to-first-token, so a slow first chunk must not trip.
    guard = _StreamInactivityGuard(
        _FakeAsyncStream([(0.3, "a"), (0.0, "b")]), 0.05, "m", "openai"
    )
    assert asyncio.run(_drain(guard)) == ["a", "b"]


def test_mid_stream_stall_raises_timeout():
    guard = _StreamInactivityGuard(
        _FakeAsyncStream([(0.0, "a"), (0.3, "b")]), 0.05, "m", "openai"
    )

    async def _run():
        seen = []
        agen = guard.__aiter__()
        seen.append(await agen.__anext__())
        await agen.__anext__()
        return seen

    with pytest.raises(litellm.Timeout):
        asyncio.run(_run())


def test_fast_stream_passes_through():
    guard = _StreamInactivityGuard(
        _FakeAsyncStream([(0.0, "a"), (0.0, "b"), (0.0, "c")]), 0.5, "m", "openai"
    )
    assert asyncio.run(_drain(guard)) == ["a", "b", "c"]


def test_exhaustion_propagates_stop_async_iteration():
    guard = _StreamInactivityGuard(_FakeAsyncStream([(0.0, "a")]), 0.5, "m", "openai")
    assert asyncio.run(_drain(guard)) == ["a"]


def test_source_disabled_returns_raw_stream(monkeypatch):
    monkeypatch.setattr(
        "litellm.constants.LITELLM_STREAM_INACTIVITY_TIMEOUT_SECONDS", None
    )
    wrapper = CustomStreamWrapper(
        completion_stream=None, model=None, logging_obj=MagicMock(), custom_llm_provider=None
    )
    raw = _FakeAsyncStream([(0.0, "a")])
    assert wrapper._async_inactivity_source(raw) is raw


def test_source_enabled_wraps_and_caches(monkeypatch):
    monkeypatch.setattr(
        "litellm.constants.LITELLM_STREAM_INACTIVITY_TIMEOUT_SECONDS", 1.0
    )
    wrapper = CustomStreamWrapper(
        completion_stream=None, model=None, logging_obj=MagicMock(), custom_llm_provider=None
    )
    raw = _FakeAsyncStream([(0.0, "a")])
    first = wrapper._async_inactivity_source(raw)
    assert isinstance(first, _StreamInactivityGuard)
    assert wrapper._async_inactivity_source(raw) is first
