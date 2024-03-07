#### What this tests ####
#    This tests the router's ability to pick deployment with lowest latency

import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import Router
import litellm
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.caching import DualCache

### UNIT TESTS FOR LATENCY ROUTING ###


def test_latency_updated():
    test_cache = DualCache()
    model_list = []
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
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
        router_cache=test_cache, model_list=model_list, routing_args={"ttl": cache_time}
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
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
            "litellm_params": {"model": "azure/chatgpt-v-2"},
            "model_info": {"id": "1234"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/chatgpt-v-2"},
            "model_info": {"id": "5678"},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    model_group = "gpt-3.5-turbo"
    ## DEPLOYMENT 1 ##
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
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
                "deployment": "azure/chatgpt-v-2",
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


# test_get_available_deployments()


def test_get_available_endpoints_tpm_rpm_check():
    """
    Pass in list of 2 valid models

    Update cache with 1 model clearly being at tpm/rpm limit

    assert that only the valid model is returned
    """
    test_cache = DualCache()
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/chatgpt-v-2"},
            "model_info": {"id": "1234", "rpm": 10},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "azure/chatgpt-v-2"},
            "model_info": {"id": "5678", "rpm": 3},
        },
    ]
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache, model_list=model_list
    )
    model_group = "gpt-3.5-turbo"
    ## DEPLOYMENT 1 ##
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
            },
            "model_info": {"id": deployment_id},
        }
    }
    for _ in range(3):
        start_time = time.time()
        response_obj = {"usage": {"total_tokens": 50}}
        time.sleep(0.05)
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
                "deployment": "azure/chatgpt-v-2",
            },
            "model_info": {"id": deployment_id},
        }
    }
    for _ in range(3):
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
        == "1234"
    )


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
    assert router.get_available_deployment(model="azure-model")["model_info"]["id"] == 2


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
