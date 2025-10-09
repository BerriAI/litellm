#### What this tests ####
#    This tests the router's ability to pick deployment with lowest latency

import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()
import copy
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler

### UNIT TESTS FOR LATENCY ROUTING ###


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_latency_memory_leak(sync_mode):
    """
    Test to make sure there's no memory leak caused by lowest latency routing

    - make 10 calls -> check memory
    - make 11th call -> no change in memory
    """
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(5)
    end_time = time.time()
    for _ in range(10):
        if sync_mode:
            lowest_latency_logger.log_success_event(
                response_obj=response_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            await lowest_latency_logger.async_log_success_event(
                response_obj=response_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
    latency_key = f"{model_group}_map"
    cache_value = copy.deepcopy(
        test_cache.get_cache(key=latency_key)
    )  # MAKE SURE NO MEMORY LEAK IN CACHING OBJECT

    if sync_mode:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await lowest_latency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    new_cache_value = test_cache.get_cache(key=latency_key)
    # Assert that the size of the cache doesn't grow unreasonably
    assert get_size(new_cache_value) <= get_size(
        cache_value
    ), f"Memory leak detected in function call! new_cache size={get_size(new_cache_value)}, old cache size={get_size(cache_value)}"


def get_size(obj, seen=None):
    # From https://goshippo.com/blog/measure-real-size-any-python-object/
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def test_latency_updated():
    test_cache = DualCache()
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(5)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    latency_key = f"{model_group}_map"
    assert (
        end_time - start_time
        == test_cache.get_cache(key=latency_key)[deployment_id]["latency"][0]
    )


# test_tpm_rpm_updated()


def test_latency_updated_custom_ttl():
    """
    Invalidate the cached request.

    Test that the cache is empty
    """
    test_cache = DualCache()
    model_list = []
    cache_time = 3
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, routing_args={"ttl": cache_time}
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(5)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    latency_key = f"{model_group}_map"
    print(f"cache: {test_cache.get_cache(key=latency_key)}")
    assert isinstance(test_cache.get_cache(key=latency_key), dict)
    time.sleep(cache_time)
    assert test_cache.get_cache(key=latency_key) is None


def test_get_available_deployments():
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "1234"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "5678"},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    ## DEPLOYMENT 1 ##
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(3)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    ## DEPLOYMENT 2 ##
    deployment_id = "5678"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 20}}
    time.sleep(2)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    print(
        lowest_latency_logger.get_available_deployments(
            model_group=model_group, healthy_deployments=model_list
        )
    )
    assert (
        lowest_latency_logger.get_available_deployments(
            model_group=model_group, healthy_deployments=model_list
        )["model_info"]["id"]
        == "5678"
    )


async def _deploy(lowest_latency_logger, deployment_id, tokens_used, duration):
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": tokens_used}}
    await asyncio.sleep(duration)
    end_time = time.time()
    lowest_latency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )


async def _gather_deploy(all_deploys):
    return await asyncio.gather(*[_deploy(*t) for t in all_deploys])


@pytest.mark.parametrize(
    "ans_rpm", [1, 5]
)  # 1 should produce nothing, 10 should select first
@pytest.mark.flaky(retries=3, delay=1)
def test_get_available_endpoints_tpm_rpm_check_async(ans_rpm):
    """
    Pass in list of 2 valid models

    Update cache with 1 model clearly being at tpm/rpm limit

    assert that only the valid model is returned
    """
    test_cache = DualCache()
    ans = "1234"
    non_ans_rpm = 3
    assert ans_rpm != non_ans_rpm, "invalid test"
    if ans_rpm < non_ans_rpm:
        ans = None
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "1234", "rpm": ans_rpm},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "5678", "rpm": non_ans_rpm},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    d1 = [(lowest_latency_logger, "1234", 50, 0.01)] * non_ans_rpm
    d2 = [(lowest_latency_logger, "5678", 50, 0.01)] * non_ans_rpm
    asyncio.run(_gather_deploy([*d1, *d2]))
    time.sleep(3)
    ## CHECK WHAT'S SELECTED ##
    d_ans = lowest_latency_logger.get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print(d_ans)
    assert (d_ans and d_ans["model_info"]["id"]) == ans


# test_get_available_endpoints_tpm_rpm_check_async()


@pytest.mark.parametrize(
    "ans_rpm", [1, 5]
)  # 1 should produce nothing, 10 should select first
@pytest.mark.flaky(retries=3, delay=1)
def test_get_available_endpoints_tpm_rpm_check(ans_rpm):
    """
    Pass in list of 2 valid models

    Update cache with 1 model clearly being at tpm/rpm limit

    assert that only the valid model is returned
    """
    test_cache = DualCache()
    ans = "1234"
    non_ans_rpm = 3
    assert ans_rpm != non_ans_rpm, "invalid test"
    if ans_rpm < non_ans_rpm:
        ans = None
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "1234", "rpm": ans_rpm},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/gpt-4.1-nano"},
            "model_info": {"id": "5678", "rpm": non_ans_rpm},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    ## DEPLOYMENT 1 ##
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    for _ in range(non_ans_rpm):
        start_time = time.time()
        response_obj = {"usage": {"total_tokens": 50}}
        time.sleep(0.01)
        end_time = time.time()
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    ## DEPLOYMENT 2 ##
    deployment_id = "5678"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": deployment_id},
        }
    }
    for _ in range(non_ans_rpm):
        start_time = time.time()
        response_obj = {"usage": {"total_tokens": 20}}
        time.sleep(0.5)
        end_time = time.time()
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    ## CHECK WHAT'S SELECTED ##
    d_ans = lowest_latency_logger.get_available_deployments(
        model_group=model_group, healthy_deployments=model_list
    )
    print(d_ans)
    assert (d_ans and d_ans["model_info"]["id"]) == ans


def test_router_get_available_deployments():
    """
    Test if routers 'get_available_deployments' returns the fastest deployment
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ## DEPLOYMENT 1 ##
    deployment_id = 1
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(3)
    end_time = time.time()
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    ## DEPLOYMENT 2 ##
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 20}}
    time.sleep(2)
    end_time = time.time()
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    # print(router.lowesttpm_logger.get_available_deployments(model_group="azure-model"))
    print(router.get_available_deployment(model="azure-model"))
    assert (
        router.get_available_deployment(model="azure-model")["model_info"]["id"] == "2"
    )


# test_router_get_available_deployments()


@pytest.mark.asyncio
async def test_router_completion_streaming():
    messages = [
        {"role": "user", "content": "Hello, can you generate a 500 words poem?"}
    ]
    model = "azure-model"
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
                "mock_response": "Hello world",
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
                "mock_response": "Hello world",
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ### Make 3 calls, test if 3rd call goes to fastest deployment

    ## CALL 1+2
    tasks = []
    response = None
    final_response = None
    for _ in range(2):
        tasks.append(router.acompletion(model=model, messages=messages))
    response = await asyncio.gather(*tasks)

    if response is not None:
        ## CALL 3
        await asyncio.sleep(1)  # let the cache update happen
        picked_deployment = router.lowestlatency_logger.get_available_deployments(
            model_group=model, healthy_deployments=router.healthy_deployments
        )
        final_response = await router.acompletion(model=model, messages=messages)
        print(f"min deployment id: {picked_deployment}")
        print(f"model id: {final_response._hidden_params['model_id']}")
        assert (
            final_response._hidden_params["model_id"]
            == picked_deployment["model_info"]["id"]
        )


# asyncio.run(test_router_completion_streaming())


@pytest.mark.asyncio
async def test_lowest_latency_routing_with_timeouts():
    """
    PROD Test:
    - Endpoint 1: triggers timeout errors (it takes 10+ seconds to respond)
    - Endpoint 2: Responds in under 1s
    - Run 5 requests to collect data on latency
    - Run Wait till cache is filled with data
    - Run 10 more requests
    - All requests should have been routed to endpoint 2
    """
    import litellm

    litellm.set_verbose = True

    router = Router(
        model_list=[
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/slow-endpoint",
                    "api_base": "https://exampleopenaiendpoint-production-c715.up.railway.app/",  # If you are Krrish, this is OpenAI Endpoint3 on our Railway endpoint :)
                    "api_key": "fake-key",
                },
                "model_info": {"id": "slow-endpoint"},
            },
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint"},
            },
        ],
        routing_strategy="latency-based-routing",
        set_verbose=True,
        debug_level="DEBUG",
        timeout=1,
    )  # type: ignore

    # make 4 requests
    for _ in range(4):
        try:
            response = await router.acompletion(
                model="azure-model", messages=[{"role": "user", "content": "hello"}]
            )
            print(response)
        except Exception as e:
            print("got exception", e)

    await asyncio.sleep(1)
    print("done sending initial requests to collect latency")
    """
    Note: for debugging
    - By this point: slow-endpoint should have timed out 3-4 times and should be heavily penalized :)
    - The next 10 requests should all be routed to the fast-endpoint
    """

    deployments = {}
    # make 10 requests
    for _ in range(10):
        response = await router.acompletion(
            model="azure-model", messages=[{"role": "user", "content": "hello"}]
        )
        print(response)
        _picked_model_id = response._hidden_params["model_id"]
        if _picked_model_id not in deployments:
            deployments[_picked_model_id] = 1
        else:
            deployments[_picked_model_id] += 1
    print("deployments", deployments)

    # ALL the Requests should have been routed to the fast-endpoint
    assert deployments["fast-endpoint"] == 10


@pytest.mark.asyncio
async def test_lowest_latency_routing_first_pick():
    """
    PROD Test:
    - When all deployments are latency=0, it should randomly pick a deployment
    - IT SHOULD NEVER PICK THE Very First deployment everytime all deployment latencies are 0
    - This ensures that after the ttl window resets it randomly picks a deployment
    """
    import litellm

    litellm.set_verbose = True

    router = Router(
        model_list=[
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint"},
            },
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint-2",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint-2"},
            },
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint-2",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint-3"},
            },
            {
                "model_name": "azure-model",
                "litellm_params": {
                    "model": "openai/fast-endpoint-2",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "fast-endpoint-4"},
            },
        ],
        routing_strategy="latency-based-routing",
        routing_strategy_args={"ttl": 0.0000000001},
        set_verbose=True,
        debug_level="DEBUG",
    )  # type: ignore

    deployments = {}
    for _ in range(10):
        response = await router.acompletion(
            model="azure-model", messages=[{"role": "user", "content": "hello"}]
        )
        print(response)
        _picked_model_id = response._hidden_params["model_id"]
        if _picked_model_id not in deployments:
            deployments[_picked_model_id] = 1
        else:
            deployments[_picked_model_id] += 1
        await asyncio.sleep(0.000000000005)

    print("deployments", deployments)

    # assert that len(deployments) >1
    assert len(deployments) > 1


@pytest.mark.parametrize("buffer", [0, 1])
@pytest.mark.asyncio
async def test_lowest_latency_routing_buffer(buffer):
    """
    Allow shuffling calls within a certain latency buffer
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
        routing_strategy_args={"lowest_latency_buffer": buffer},
    )  # type: ignore

    ## DEPLOYMENT 1 ##
    deployment_id = 1
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 50}}
    time.sleep(3)
    end_time = time.time()
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    ## DEPLOYMENT 2 ##
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 20}}
    time.sleep(2)
    end_time = time.time()
    router.lowestlatency_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    # print(router.lowesttpm_logger.get_available_deployments(model_group="azure-model"))
    selected_deployments = {}
    for _ in range(50):
        print(router.get_available_deployment(model="azure-model"))
        selected_deployments[
            router.get_available_deployment(model="azure-model")["model_info"]["id"]
        ] = 1

    if buffer == 0:
        assert len(selected_deployments.keys()) == 1
    else:
        assert len(selected_deployments.keys()) == 2


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_lowest_latency_routing_time_to_first_token(sync_mode):
    """
    If a deployment has
    - a fast time to first token
    - slow latency/output token

    test if:
    - for streaming, the deployment with fastest time to first token is picked
    - for non-streaming, fastest overall deployment is picked
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore
    ## DEPLOYMENT 1 ##
    deployment_id = 1
    start_time = datetime.now()
    one_second_later = start_time + timedelta(seconds=1)

    # Compute 3 seconds after the current time
    three_seconds_later = start_time + timedelta(seconds=3)
    four_seconds_later = start_time + timedelta(seconds=4)

    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        },
        "stream": True,
        "completion_start_time": one_second_later,
    }

    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(completion_tokens=50, total_tokens=50)
    )
    end_time = four_seconds_later

    if sync_mode:
        router.lowestlatency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await router.lowestlatency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    ## DEPLOYMENT 2 ##
    deployment_id = 2
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        },
        "stream": True,
        "completion_start_time": three_seconds_later,
    }
    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(completion_tokens=50, total_tokens=50)
    )
    end_time = three_seconds_later
    if sync_mode:
        router.lowestlatency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        await router.lowestlatency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )

    """
    TESTING

    - expect deployment 1 to be picked for streaming
    - expect deployment 2 to be picked for non-streaming
    """
    # print(router.lowesttpm_logger.get_available_deployments(model_group="azure-model"))
    selected_deployments = {}
    for _ in range(3):
        print(router.get_available_deployment(model="azure-model"))
        ## for non-streaming
        selected_deployments[
            router.get_available_deployment(model="azure-model")["model_info"]["id"]
        ] = 1

    assert len(selected_deployments.keys()) == 1
    assert "2" in list(selected_deployments.keys())

    selected_deployments = {}
    for _ in range(50):
        print(router.get_available_deployment(model="azure-model"))
        ## for non-streaming
        selected_deployments[
            router.get_available_deployment(
                model="azure-model", request_kwargs={"stream": True}
            )["model_info"]["id"]
        ] = 1

    assert len(selected_deployments.keys()) == 1
    assert "1" in list(selected_deployments.keys())
