# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import sys, os, time, openai, uuid
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.scheduler import FlowItem, Scheduler


@pytest.mark.asyncio
async def test_scheduler_diff_model_names():
    """
    Assert 2 requests to 2 diff model groups are top of their respective queue's
    """
    scheduler = Scheduler()

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
        ]
    )

    scheduler.update_variables(llm_router=router)

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-4")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    assert await scheduler.poll(id="10", model_name="gpt-3.5-turbo") == True
    assert await scheduler.poll(id="11", model_name="gpt-4") == True


@pytest.mark.parametrize("p0, p1", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.asyncio
async def test_scheduler_prioritized_requests(p0, p1):
    """
    2 requests for same model group
    """
    scheduler = Scheduler()

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
        ]
    )

    scheduler.update_variables(llm_router=router)

    item1 = FlowItem(priority=p0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=p1, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    if p0 == 0:
        assert await scheduler.peek(id="10", model_name="gpt-3.5-turbo") == True
        assert await scheduler.peek(id="11", model_name="gpt-3.5-turbo") == False
    else:
        assert await scheduler.peek(id="11", model_name="gpt-3.5-turbo") == True
        assert await scheduler.peek(id="10", model_name="gpt-3.5-turbo") == False


@pytest.mark.parametrize("p0, p1", [(0, 1), (0, 0), (1, 0)])  #
@pytest.mark.asyncio
async def test_aascheduler_prioritized_requests_mock_response(p0, p1):
    """
    2 requests for same model group

    if model is at rate limit, ensure the higher priority request gets done first
    """
    scheduler = Scheduler()

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

    scheduler.update_variables(llm_router=router)

    async def _make_prioritized_call(flow_item: FlowItem):
        ## POLL QUEUE
        default_timeout = router.timeout
        end_time = time.time() + default_timeout
        poll_interval = 0.03  # poll every 3ms
        curr_time = time.time()

        make_request = False

        if router is None:
            raise Exception("No llm router value")

        while curr_time < end_time:
            make_request = await scheduler.poll(
                id=flow_item.request_id, model_name=flow_item.model_name
            )
            print(f"make_request={make_request}, priority={flow_item.priority}")
            if make_request:  ## IF TRUE -> MAKE REQUEST
                break
            else:  ## ELSE -> loop till default_timeout
                await asyncio.sleep(poll_interval)
                curr_time = time.time()

        if make_request:
            try:
                _response = await router.acompletion(
                    model=flow_item.model_name,
                    messages=[{"role": "user", "content": "Hey!"}],
                )
            except Exception as e:
                print("Received error - {}".format(str(e)))
                return flow_item.priority, flow_item.request_id, time.time()

            return flow_item.priority, flow_item.request_id, time.time()

        raise Exception("didn't make request")

    tasks = []

    item = FlowItem(
        priority=p0, request_id=str(uuid.uuid4()), model_name="gpt-3.5-turbo"
    )
    await scheduler.add_request(request=item)
    tasks.append(_make_prioritized_call(flow_item=item))

    item = FlowItem(
        priority=p1, request_id=str(uuid.uuid4()), model_name="gpt-3.5-turbo"
    )
    await scheduler.add_request(request=item)
    tasks.append(_make_prioritized_call(flow_item=item))

    # Running the tasks and getting responses in order of completion
    completed_responses = []
    for task in asyncio.as_completed(tasks):
        result = await task
        completed_responses.append(result)
        print(f"Received response: {result}")

    print(f"responses: {completed_responses}")

    assert (
        completed_responses[0][0] == 0
    )  # assert higher priority request got done first
    assert (
        completed_responses[0][2] < completed_responses[1][2]
    )  # higher priority request tried first
