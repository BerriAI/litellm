#### What this tests ####
#    This tests the router's ability to pick deployment with lowest tpm

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
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.caching import DualCache

### UNIT TESTS FOR TPM/RPM ROUTING ###


def test_tpm_rpm_updated():
    test_cache = DualCache()
    model_list = []
    lowest_tpm_logger = LowestTPMLoggingHandler(
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
    end_time = time.time()
    lowest_tpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    current_minute = datetime.now().strftime("%H-%M")
    tpm_count_api_key = f"{model_group}:tpm:{current_minute}"
    rpm_count_api_key = f"{model_group}:rpm:{current_minute}"
    assert (
        response_obj["usage"]["total_tokens"]
        == test_cache.get_cache(key=tpm_count_api_key)[deployment_id]
    )
    assert 1 == test_cache.get_cache(key=rpm_count_api_key)[deployment_id]


# test_tpm_rpm_updated()


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
    lowest_tpm_logger = LowestTPMLoggingHandler(
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
    end_time = time.time()
    lowest_tpm_logger.log_success_event(
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
    end_time = time.time()
    lowest_tpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    print(
        lowest_tpm_logger.get_available_deployments(
            model_group=model_group,
            healthy_deployments=model_list,
            input=["Hello world"],
        )
    )
    assert (
        lowest_tpm_logger.get_available_deployments(
            model_group=model_group,
            healthy_deployments=model_list,
            input=["Hello world"],
        )["model_info"]["id"]
        == "5678"
    )


# test_get_available_deployments()


def test_router_get_available_deployments():
    """
    Test if routers 'get_available_deployments' returns the least busy deployment
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
        routing_strategy="usage-based-routing",
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
    end_time = time.time()
    router.lowesttpm_logger.log_success_event(
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
    end_time = time.time()
    router.lowesttpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    # print(router.lowesttpm_logger.get_available_deployments(model_group="azure-model"))
    assert router.get_available_deployment(model="azure-model")["model_info"]["id"] == 2


# test_get_available_deployments()
# test_router_get_available_deployments()


def test_router_skip_rate_limited_deployments():
    """
    Test if routers 'get_available_deployments' raises No Models Available error if max tpm would be reached by message
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "tpm": 1440,
            },
            "model_info": {"id": 1},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing",
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
            "model_info": {"id": deployment_id},
        }
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 1439}}
    end_time = time.time()
    router.lowesttpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ## - should skip 2, and pick 1
    # print(router.lowesttpm_logger.get_available_deployments(model_group="azure-model"))
    try:
        router.get_available_deployment(
            model="azure-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        pytest.fail(f"Should have raised No Models Available error")
    except Exception as e:
        pass


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
        routing_strategy="usage-based-routing",
        set_verbose=False,
    )  # type: ignore

    ### Make 3 calls, test if 3rd call goes to lowest tpm deployment

    ## CALL 1+2
    tasks = []
    response = None
    final_response = None
    for _ in range(2):
        tasks.append(router.acompletion(model=model, messages=messages))
    response = await asyncio.gather(*tasks)

    if response is not None:
        ## CALL 3
        await asyncio.sleep(1)  # let the token update happen
        current_minute = datetime.now().strftime("%H-%M")
        picked_deployment = router.lowesttpm_logger.get_available_deployments(
            model_group=model,
            healthy_deployments=router.healthy_deployments,
            messages=messages,
        )
        final_response = await router.acompletion(model=model, messages=messages)
        print(f"min deployment id: {picked_deployment}")
        tpm_key = f"{model}:tpm:{current_minute}"
        rpm_key = f"{model}:rpm:{current_minute}"

        tpm_dict = router.cache.get_cache(key=tpm_key)
        print(f"tpm_dict: {tpm_dict}")
        rpm_dict = router.cache.get_cache(key=rpm_key)
        print(f"rpm_dict: {rpm_dict}")
        print(f"model id: {final_response._hidden_params['model_id']}")
        assert (
            final_response._hidden_params["model_id"]
            == picked_deployment["model_info"]["id"]
        )


# asyncio.run(test_router_completion_streaming())
