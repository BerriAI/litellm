import asyncio
import queue
from typing import Final

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

RUNS: Final[queue.SimpleQueue[tuple[asyncio.AbstractEventLoop, asyncio.AbstractEventLoop]]] = queue.SimpleQueue()


async def record_run(queued_on: asyncio.AbstractEventLoop) -> None:
    RUNS.put((queued_on, asyncio.get_running_loop()))


async def test_1_leaves_an_event_pending() -> None:
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(record_run(asyncio.get_running_loop()))


async def test_2_never_inherits_the_pending_event() -> None:
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    queued_on, ran_on = RUNS.get_nowait()
    assert RUNS.empty()
    assert ran_on is queued_on
    assert ran_on is not asyncio.get_running_loop()
