import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.scheduler import FlowItem, Scheduler


@pytest.mark.asyncio
async def test_poll_admits_request_missing_from_queue_while_a_deployment_is_healthy():
    scheduler: Final = Scheduler()

    assert await scheduler.poll(
        id="erased-by-concurrent-write", model_name="sched-model", health_deployments=[{"model_info": {"id": "a"}}]
    )


@pytest.mark.asyncio
async def test_poll_during_cooldown_admits_only_the_head_of_the_queue():
    scheduler: Final = Scheduler()
    await scheduler.add_request(FlowItem(priority=2, request_id="later", model_name="sched-model"))
    await scheduler.add_request(FlowItem(priority=1, request_id="head", model_name="sched-model"))

    assert not await scheduler.poll(id="later", model_name="sched-model", health_deployments=[])
    assert await scheduler.poll(id="head", model_name="sched-model", health_deployments=[])
    assert await scheduler.get_queue("sched-model") == [(2, "later")]


class _PausesAfterEnqueueScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()
        self.enqueued: Final = asyncio.Event()

    async def add_request(self, request: FlowItem) -> None:
        await super().add_request(request)
        self.enqueued.set()
        await asyncio.Event().wait()


async def _no_healthy_deployments() -> Sequence[object]:
    return ()


@pytest.mark.asyncio
async def test_wait_for_turn_removes_entry_when_cancelled_mid_enqueue():
    scheduler: Final = _PausesAfterEnqueueScheduler()
    waiting: Final = asyncio.create_task(
        scheduler.wait_for_turn(
            request=FlowItem(priority=1, request_id="cancelled", model_name="sched-model"),
            timeout=5,
            get_healthy_deployments=_no_healthy_deployments,
        )
    )
    await scheduler.enqueued.wait()
    assert await scheduler.get_queue("sched-model") == [(1, "cancelled")]

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert await scheduler.get_queue("sched-model") == []
