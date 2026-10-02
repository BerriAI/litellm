import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

import litellm

T = TypeVar("T")


def warm_tokenizer(model: str) -> None:
    litellm.token_counter(model=model, text="load the tokenizer before anything is timed")


async def loop_wake_lags(until: asyncio.Event) -> tuple[float, ...]:
    async def wake_lag() -> float:
        started: Final = time.perf_counter()
        await asyncio.sleep(0.001)
        return time.perf_counter() - started - 0.001

    return tuple([await wake_lag() for _ in iter(until.is_set, True)])


async def timed_with_loop_lags(run: Callable[[], Awaitable[T]]) -> tuple[T, float, tuple[float, ...]]:
    finished: Final = asyncio.Event()

    async def timed() -> tuple[T, float]:
        await asyncio.sleep(0)
        started: Final = time.perf_counter()
        try:
            return await run(), time.perf_counter() - started
        finally:
            finished.set()

    (result, took), lags = await asyncio.gather(timed(), loop_wake_lags(finished))
    return result, took, lags


def assert_loop_stayed_free(took: float, lags: tuple[float, ...]) -> None:
    assert max(lags) < took / 4, f"the event loop stalled {max(lags):.3f}s during a {took:.3f}s count"
