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
