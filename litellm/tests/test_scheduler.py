# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import sys, os, time, openai, uuid
import traceback, asyncio
import pytest
from typing import List

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.scheduler import FlowItem, Scheduler
from litellm import ModelResponse


@pytest.mark.asyncio
async def test_scheduler_diff_model_names():
    """
    Assert 2 requests to 2 diff model groups are top of their respective queue's
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-4")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    assert (
        await scheduler.poll(
            id="10", model_name="gpt-3.5-turbo", health_deployments=[{"key": "value"}]
        )
        == True
    )
    assert (
        await scheduler.poll(
            id="11", model_name="gpt-4", health_deployments=[{"key": "value"}]
        )
        == True
    )


@pytest.mark.parametrize("p0, p1", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.asyncio
async def test_scheduler_prioritized_requests(p0, p1):
    """
    2 requests for same model group
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=p0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=p1, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    if p0 == 0:
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=[{"key": "value"}],
            )
            == True
        )
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=[{"key": "value"}],
            )
            == False
        )
    else:
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=[{"key": "value"}],
            )
            == True
        )
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=[{"key": "value"}],
            )
            == False
        )


@pytest.mark.parametrize("p0, p1", [(0, 1), (0, 0)])  #
@pytest.mark.asyncio
async def test_aascheduler_prioritized_requests_mock_response_simplified(p0, p1):
    """
    2 requests for same model group

    if model is at rate limit, ensure the higher priority request gets done first
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hello world this is Macintosh!",
                    "rpm": 0,
                },
            },
        ],
        timeout=10,
        num_retries=3,
        cooldown_time=5,
        routing_strategy="usage-based-routing-v2",
    )

    tasks = []

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
    }

    tasks.append(router.schedule_acompletion(**data, priority=p0))
    tasks.append(router.schedule_acompletion(**data, priority=p1))

    # Running the tasks and getting responses in order of completion
    completed_responses: List[dict] = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
        except Exception as e:
            result = {"priority": e.priority, "response_completed_at": time.time()}
            completed_responses.append(result)
            print(f"Received response: {result}")

    print(f"responses: {completed_responses}")

    assert (
        completed_responses[0]["priority"] == 0
    )  # assert higher priority request got done first
    assert (
        completed_responses[0]["response_completed_at"]
        < completed_responses[1]["response_completed_at"]
    )  # higher priority request tried first
