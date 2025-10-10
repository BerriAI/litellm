#### What this tests ####
#    This tests the router's ability to identify the least busy deployment

import asyncio
import os
import random
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

### UNIT TESTS FOR LEAST BUSY LOGGING ###


def test_model_added():
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/gpt-4.1-nano",
            },
            "model_info": {"id": "1234"},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    request_count_api_key = f"gpt-3.5-turbo_request_count"
    assert test_cache.get_cache(key=request_count_api_key) is not None


def test_get_available_deployments():
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment = "azure/gpt-4.1-nano"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": "1234"},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    request_count_api_key = f"{model_group}_request_count"
    assert test_cache.get_cache(key=request_count_api_key) is not None


# test_get_available_deployments()


@pytest.mark.parametrize("async_test", [True, False])
@pytest.mark.asyncio
async def test_router_get_available_deployments(async_test):
    """
    Tests if 'get_available_deployments' returns the least busy deployment
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
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_CANADA_API_KEY",
                "api_base": "https://my-endpoint-canada-berri992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    router.leastbusy_logger.test_flag = True

    model_group = "azure-model"
    deployment = "azure/gpt-4.1-nano"
    request_count_dict = {1: 10, 2: 54, 3: 100}
    cache_key = f"{model_group}_request_count"
    if async_test is True:
        await router.cache.async_set_cache(key=cache_key, value=request_count_dict)
        deployment = await router.async_get_available_deployment(
            model=model_group, messages=None, request_kwargs={}
        )
    else:
        router.cache.set_cache(key=cache_key, value=request_count_dict)
        deployment = router.get_available_deployment(model=model_group, messages=None)
    print(f"deployment: {deployment}")
    assert deployment["model_info"]["id"] == "1"

    ## run router completion - assert completion event, no change in 'busy'ness once calls are complete

    router.completion(
        model=model_group,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    return_dict = router.cache.get_cache(key=cache_key)

    # wait 2 seconds
    time.sleep(2)

    assert router.leastbusy_logger.logged_success == 1
    assert return_dict[1] == 10
    assert return_dict[2] == 54
    assert return_dict[3] == 100


## Test with Real calls ##


@pytest.mark.asyncio
async def test_router_atext_completion_streaming():
    prompt = "Hello, can you generate a 500 words poem?"
    model = "azure-model"
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
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ### Call the async calls in sequence, so we start 1 call before going to the next.

    ## CALL 1
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    ## CALL 2
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    ## CALL 3
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    cache_key = f"{model}_request_count"
    ## check if calls equally distributed
    cache_dict = router.cache.get_cache(key=cache_key)
    for k, v in cache_dict.items():
        assert v == 1, f"Failed. K={k} called v={v} times, cache_dict={cache_dict}"


# asyncio.run(test_router_atext_completion_streaming())


@pytest.mark.asyncio
async def test_router_completion_streaming():
    litellm.set_verbose = True
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
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ### Call the async calls in sequence, so we start 1 call before going to the next.

    ## CALL 1
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    ## CALL 2
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    ## CALL 3
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    cache_key = f"{model}_request_count"
    ## check if calls equally distributed
    cache_dict = router.cache.get_cache(key=cache_key)
    for k, v in cache_dict.items():
        assert v == 1, f"Failed. K={k} called v={v} times, cache_dict={cache_dict}"
