#### What this tests ####
#    This tests the router's ability to pick deployment with lowest cost

import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, copy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import Router
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.caching.caching import DualCache

### UNIT TESTS FOR cost ROUTING ###


@pytest.mark.asyncio
async def test_get_available_deployments():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "openai-gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "groq/llama-3.1-8b-instant"},
            "model_info": {"id": "groq-llama"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache,
    )
    model_group = "gpt-3.5-turbo"

    ## CHECK WHAT'S SELECTED ##
    selected_model = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print("selected model: ", selected_model)

    assert selected_model["model_info"]["id"] == "groq-llama"


@pytest.mark.asyncio
async def test_get_available_deployments_custom_price():
    from litellm._logging import verbose_router_logger
    import logging

    verbose_router_logger.setLevel(logging.DEBUG)
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-nano",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00003,
            },
            "model_info": {"id": "chatgpt-v-experimental"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-1",
                "input_cost_per_token": 0.000000001,
                "output_cost_per_token": 0.00000001,
            },
            "model_info": {"id": "chatgpt-v-1"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-5",
                "input_cost_per_token": 10,
                "output_cost_per_token": 12,
            },
            "model_info": {"id": "chatgpt-v-5"},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache,
    )
    model_group = "gpt-3.5-turbo"

    ## CHECK WHAT'S SELECTED ##
    selected_model = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print("selected model: ", selected_model)

    assert selected_model["model_info"]["id"] == "chatgpt-v-1"


@pytest.mark.asyncio
async def test_lowest_cost_routing():
    """
    Test if router, returns model with the lowest cost
    """
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "openai-gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo"},
            "model_info": {"id": "gpt-3.5-turbo"},
        },
    ]

    # init router
    router = Router(model_list=model_list, routing_strategy="cost-based-routing")
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    print(response)
    print(
        response._hidden_params["model_id"]
    )  # expect groq-llama, since groq/llama has lowest cost
    assert "gpt-3.5-turbo" == response._hidden_params["model_id"]


async def _deploy(lowest_cost_logger, deployment_id, tokens_used, duration):
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "gpt-4",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": tokens_used}}
    time.sleep(duration)
    end_time = time.time()
    await lowest_cost_logger.async_log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.mark.parametrize(
    "ans_rpm", [1, 5]
)  # 1 should produce nothing, 10 should select first
@pytest.mark.asyncio
async def test_get_available_endpoints_tpm_rpm_check_async(ans_rpm):
    """
    Pass in list of 2 valid models

    Update cache with 1 model clearly being at tpm/rpm limit

    assert that only the valid model is returned
    """
    from litellm._logging import verbose_router_logger
    import logging

    verbose_router_logger.setLevel(logging.DEBUG)
    test_cache = DualCache()
    ans = "1234"
    non_ans_rpm = 3
    assert ans_rpm != non_ans_rpm, "invalid test"
    if ans_rpm < non_ans_rpm:
        ans = None
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "1234", "rpm": ans_rpm},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "groq/llama-3.1-8b-instant"},
            "model_info": {"id": "5678", "rpm": non_ans_rpm},
        },
    ]
    lowest_cost_logger = LowestCostLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    d1 = [(lowest_cost_logger, "1234", 50, 0.01)] * non_ans_rpm
    d2 = [(lowest_cost_logger, "5678", 50, 0.01)] * non_ans_rpm

    await asyncio.gather(*[_deploy(*t) for t in [*d1, *d2]])

    asyncio.sleep(3)

    ## CHECK WHAT'S SELECTED ##
    d_ans = await lowest_cost_logger.async_get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    assert (d_ans and d_ans["model_info"]["id"]) == ans

    print("selected deployment:", d_ans)
