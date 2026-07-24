"""
Regression tests for Router.acompletion priority handling.

Covers the bug where `request_priority = kwargs.get("priority") or self.default_priority`
treated an explicit `priority=0` as falsy and silently demoted the request out of the
scheduler queue. `priority=0` is a valid lowest-priority opt-in and must reach
`schedule_acompletion`.
"""

from unittest.mock import AsyncMock

import pytest

from litellm import Router


def _router(default_priority=None):
    return Router(
        model_list=[
            {
                "model_name": "fake",
                "litellm_params": {"model": "openai/fake", "api_key": "sk-x"},
            }
        ],
        default_priority=default_priority,
    )


@pytest.mark.asyncio
async def test_priority_zero_reaches_scheduler_when_default_none():
    # default_priority=None, caller asks for priority=0 -> must schedule, not fall back
    router = _router(default_priority=None)
    router.schedule_acompletion = AsyncMock(return_value="scheduled")
    router.async_function_with_fallbacks = AsyncMock(return_value="fallback")

    result = await router.acompletion(
        model="fake", messages=[{"role": "user", "content": "hi"}], priority=0
    )

    assert result == "scheduled"
    router.schedule_acompletion.assert_awaited_once()
    router.async_function_with_fallbacks.assert_not_awaited()


@pytest.mark.asyncio
async def test_priority_zero_beats_nonzero_default():
    # default_priority=5, caller asks for priority=0 -> explicit 0 wins, not 5
    router = _router(default_priority=5)
    router.schedule_acompletion = AsyncMock(return_value="scheduled")
    router.async_function_with_fallbacks = AsyncMock(return_value="fallback")

    result = await router.acompletion(
        model="fake", messages=[{"role": "user", "content": "hi"}], priority=0
    )

    assert result == "scheduled"
    router.schedule_acompletion.assert_awaited_once()
    assert router.schedule_acompletion.await_args.kwargs["priority"] == 0


@pytest.mark.asyncio
async def test_absent_priority_uses_default():
    # no priority passed, default_priority=5 -> still routes to scheduler (unchanged behavior)
    router = _router(default_priority=5)
    router.schedule_acompletion = AsyncMock(return_value="scheduled")
    router.async_function_with_fallbacks = AsyncMock(return_value="fallback")

    result = await router.acompletion(
        model="fake", messages=[{"role": "user", "content": "hi"}]
    )

    assert result == "scheduled"
    router.schedule_acompletion.assert_awaited_once()
    router.async_function_with_fallbacks.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_none_priority_falls_back():
    # explicit priority=None, default_priority=None -> fallbacks path (unchanged behavior)
    router = _router(default_priority=None)
    router.schedule_acompletion = AsyncMock(return_value="scheduled")
    router.async_function_with_fallbacks = AsyncMock(return_value="fallback")

    result = await router.acompletion(
        model="fake", messages=[{"role": "user", "content": "hi"}], priority=None
    )

    assert result == "fallback"
    router.async_function_with_fallbacks.assert_awaited_once()
    router.schedule_acompletion.assert_not_awaited()
