"""Tests for background cache-write task tracking in the caching handler.

asyncio only keeps a weak reference to a task created with create_task, so a
fire-and-forget cache write can be garbage-collected before it finishes,
silently dropping the entry. _track_cache_task retains a strong reference until
the task completes.
"""

import asyncio

import pytest

from litellm.caching.caching_handler import (
    _background_cache_tasks,
    _track_cache_task,
)


@pytest.mark.asyncio
async def test_track_cache_task_keeps_reference_until_done():
    async def _write():
        await asyncio.sleep(0)
        return "written"

    task = _track_cache_task(asyncio.create_task(_write()))

    # While pending, a strong reference is held in the tracking set.
    assert task in _background_cache_tasks

    result = await task

    # Once done, the task delivers its result and the reference is released.
    assert result == "written"
    await asyncio.sleep(0)  # allow the done-callback to run
    assert task not in _background_cache_tasks


@pytest.mark.asyncio
async def test_track_cache_task_releases_reference_on_exception():
    async def _boom():
        raise RuntimeError("cache backend down")

    task = _track_cache_task(asyncio.create_task(_boom()))
    assert task in _background_cache_tasks

    with pytest.raises(RuntimeError, match="cache backend down"):
        await task

    await asyncio.sleep(0)
    assert task not in _background_cache_tasks
