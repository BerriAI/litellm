"""Off-the-event-loop proofs.

`recording_tokenizer_threads` records the thread every tokenizer call ran on, so tests/unit can prove
token counting left the loop without touching the clock. The lag-timing helpers below stay for the
legacy tests/test_litellm tree.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Final, TypeVar
from unittest.mock import patch

import tiktoken

import litellm

T = TypeVar("T")


class TokenizerThreads:
    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []


@contextmanager
def recording_tokenizer_threads() -> Iterator[TokenizerThreads]:
    recorded: Final = TokenizerThreads()
    real_encode: Final = tiktoken.Encoding.encode

    def encode(self: tiktoken.Encoding, *args, **kwargs):
        recorded.threads.append(threading.current_thread())
        return real_encode(self, *args, **kwargs)

    with patch.object(tiktoken.Encoding, "encode", encode):
        yield recorded


def assert_counted_off_the_event_loop(recorded: TokenizerThreads) -> None:
    loop_thread: Final = threading.current_thread()
    assert recorded.threads, "the tokenizer was never called"
    on_loop: Final = [thread for thread in recorded.threads if thread is loop_thread]
    assert not on_loop, f"{len(on_loop)} of {len(recorded.threads)} tokenizer calls ran on the event loop thread"


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
