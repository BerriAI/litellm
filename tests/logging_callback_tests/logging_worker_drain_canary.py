import asyncio
from dataclasses import dataclass
from typing import Final

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@dataclass(slots=True)
class LeakProbe:
    pending_from_loop: asyncio.AbstractEventLoop | None = None
    ran_on_loop: asyncio.AbstractEventLoop | None = None
    run_count: int = 0

    async def record_run(self) -> None:
        self.ran_on_loop = asyncio.get_running_loop()
        self.run_count += 1


PROBE: Final = LeakProbe()


async def test_1_leaves_an_event_pending() -> None:
    PROBE.pending_from_loop = asyncio.get_running_loop()
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(PROBE.record_run())


async def test_2_never_inherits_the_pending_event() -> None:
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    assert PROBE.run_count == 1
    assert PROBE.ran_on_loop is PROBE.pending_from_loop
    assert PROBE.ran_on_loop is not asyncio.get_running_loop()
