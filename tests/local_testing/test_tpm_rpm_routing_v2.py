#### What this tests ####
#    This tests the router's ability to pick deployment with lowest tpm using 'usage-based-routing-v2-v2'

import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.types.utils import StandardLoggingPayload
import pytest
from litellm.types.router import DeploymentTypedDict
import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import (
    LowestTPMLoggingHandler_v2 as LowestTPMLoggingHandler,
)
from litellm.utils import get_utc_datetime
from create_mock_standard_logging_payload import create_standard_logging_payload

### UNIT TESTS FOR TPM/RPM ROUTING ###

"""
- Given 2 deployments, make sure it's shuffling deployments correctly.
"""


def test_tpm_rpm_updated():
    test_cache = DualCache()
    lowest_tpm_logger = LowestTPMLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    deployment = "azure/gpt-4.1-nano"
    total_tokens = 50
    standard_logging_payload: StandardLoggingPayload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = model_group
    standard_logging_payload["model_id"] = deployment_id
    standard_logging_payload["total_tokens"] = total_tokens
    standard_logging_payload["hidden_params"]["litellm_model_name"] = deployment
    kwargs = {
        "litellm_params": {
            "model": deployment,
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }

    litellm_deployment_dict: DeploymentTypedDict = {
        "model_name": model_group,
        "litellm_params": {"model": deployment},
        "model_info": {"id": deployment_id},
    }

    start_time = time.time()
    response_obj = {"usage": {"total_tokens": total_tokens}}
    end_time = time.time()
    lowest_tpm_logger.pre_call_check(deployment=litellm_deployment_dict)
    lowest_tpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    tpm_count_api_key = f"{deployment_id}:{deployment}:tpm:{current_minute}"
    rpm_count_api_key = f"{deployment_id}:{deployment}:rpm:{current_minute}"

    print(f"tpm_count_api_key={tpm_count_api_key}")
    assert response_obj["usage"]["total_tokens"] == test_cache.get_cache(
        key=tpm_count_api_key
    )
    assert 1 == test_cache.get_cache(key=rpm_count_api_key)


# test_tpm_rpm_updated()


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
    lowest_tpm_logger = LowestTPMLoggingHandler(
        router_cache=test_cache
    )
    model_group = "gpt-3.5-turbo"
    ## DEPLOYMENT 1 ##
    total_tokens = 50
    deployment_id = "1234"
    deployment = "azure/gpt-4.1-nano"
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = model_group
    standard_logging_payload["model_id"] = deployment_id
    standard_logging_payload["total_tokens"] = total_tokens
    standard_logging_payload["hidden_params"]["litellm_model_name"] = deployment
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": total_tokens}}
    end_time = time.time()
    lowest_tpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    ## DEPLOYMENT 2 ##
    total_tokens = 20
    deployment_id = "5678"
    standard_logging_payload: StandardLoggingPayload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = model_group
    standard_logging_payload["model_id"] = deployment_id
    standard_logging_payload["total_tokens"] = total_tokens
    standard_logging_payload["hidden_params"]["litellm_model_name"] = deployment
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": total_tokens}}
    end_time = time.time()
    lowest_tpm_logger.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
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
    Test if routers 'get_available_deployments' returns the lowest tpm deployment
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
        routing_strategy="usage-based-routing-v2",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    print(f"router id's: {router.get_model_ids()}")
    ## DEPLOYMENT 1 ##
    deployment_id = 1
    standard_logging_payload: StandardLoggingPayload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = "azure-model"
    standard_logging_payload["model_id"] = str(deployment_id)
    total_tokens = 50
    standard_logging_payload["total_tokens"] = total_tokens
    standard_logging_payload["hidden_params"]["litellm_model_name"] = "azure/gpt-turbo"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 1},
        },
        "standard_logging_object": standard_logging_payload,
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": total_tokens}}
    end_time = time.time()
    router.lowesttpm_logger_v2.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )
    ## DEPLOYMENT 2 ##
    deployment_id = 2
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = "azure-model"
    standard_logging_payload["model_id"] = str(deployment_id)
    standard_logging_payload["hidden_params"][
        "litellm_model_name"
    ] = "azure/gpt-35-turbo"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": 2},
        },
        "standard_logging_object": standard_logging_payload,
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": 20}}
    end_time = time.time()
    router.lowesttpm_logger_v2.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    # print(router.lowesttpm_logger_v2.get_available_deployments(model_group="azure-model"))
    assert (
        router.get_available_deployment(model="azure-model")["model_info"]["id"] == "2"
    )


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
        routing_strategy="usage-based-routing-v2",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ## DEPLOYMENT 1 ##
    deployment_id = 1
    total_tokens = 1439
    standard_logging_payload: StandardLoggingPayload = create_standard_logging_payload()
    standard_logging_payload["model_group"] = "azure-model"
    standard_logging_payload["model_id"] = str(deployment_id)
    standard_logging_payload["total_tokens"] = total_tokens
    standard_logging_payload["hidden_params"]["litellm_model_name"] = "azure/gpt-turbo"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": "azure-model",
            },
            "model_info": {"id": deployment_id},
        },
        "standard_logging_object": standard_logging_payload,
    }
    start_time = time.time()
    response_obj = {"usage": {"total_tokens": total_tokens}}
    end_time = time.time()
    router.lowesttpm_logger_v2.log_success_event(
        response_obj=response_obj,
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
    )

    ## CHECK WHAT'S SELECTED ##
    # print(router.lowesttpm_logger_v2.get_available_deployments(model_group="azure-model"))
    try:
        router.get_available_deployment(
            model="azure-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        pytest.fail(f"Should have raised No Models Available error")
    except Exception as e:
        print(f"An exception occurred! {str(e)}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_multiple_potential_deployments(sync_mode):
    """
    If multiple deployments have the same tpm value

    call 5 times, test if deployments are shuffled.

    -> prevents single deployment from being overloaded in high-concurrency scenario
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
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo-2",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "tpm": 1440,
            },
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    model_ids = set()
    for _ in range(1000):
        if sync_mode:
            deployment = router.get_available_deployment(
                model="azure-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
            )
        else:
            deployment = await router.async_get_available_deployment(
                model="azure-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                request_kwargs={},
            )

        ## get id ##
        id = deployment.get("model_info", {}).get("id")
        model_ids.add(id)

    assert len(model_ids) == 2


def test_single_deployment_tpm_zero():
    import os
    from datetime import datetime

    import litellm

    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "tpm": 0,
            },
        }
    ]

    router = litellm.Router(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
        cache_responses=True,
    )

    model = "gpt-3.5-turbo"
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    try:
        router.get_available_deployment(
            model=model,
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
        pytest.fail(f"Should have raised No Models Available error")
    except Exception as e:
        print(f"it worked - {str(e)}! \n{traceback.format_exc()}")


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
                "tpm": 1440,
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
                "tpm": 6,
                "mock_response": "Hello world",
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
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
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        picked_deployment = router.lowesttpm_logger_v2.get_available_deployments(
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

"""
- Unit test for sync 'pre_call_checks' 
- Unit test for async 'async_pre_call_checks' 
"""


@pytest.mark.asyncio
async def test_router_caching_ttl():
    """
    Confirm caching ttl's work as expected.

    Relevant issue: https://github.com/BerriAI/litellm/issues/5609
    """
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
                "tpm": 1440,
                "mock_response": "Hello world",
            },
            "model_info": {"id": 1},
        }
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
        set_verbose=False,
        redis_host=os.getenv("REDIS_HOST"),
        redis_password=os.getenv("REDIS_PASSWORD"),
        redis_port=os.getenv("REDIS_PORT"),
    )

    assert router.cache.redis_cache is not None

    increment_cache_kwargs = {}
    with patch.object(
        router.cache,
        "async_increment_cache_pipeline",
        new=AsyncMock(),
    ) as mock_client:
        await router.acompletion(model=model, messages=messages)

        # mock_client.assert_called_once()
        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        print(f"mock_client.call_args.args: {mock_client.call_args.args}")

        # Get the increment_list from the first positional argument or the keyword argument
        increment_list = mock_client.call_args.kwargs.get(
            "increment_list",
            mock_client.call_args.args[0] if mock_client.call_args.args else None,
        )
        assert increment_list is not None
        assert len(increment_list) > 0

        # Check that TTL is set to 60 for all operations
        for operation in increment_list:
            assert operation["ttl"] == 60

        # Get the first operation for testing the redis increment
        first_operation = increment_list[0]
        increment_cache_kwargs = {
            "key": first_operation["key"],
            "value": first_operation["increment_value"],
            "ttl": first_operation["ttl"],
        }

    ## call redis async increment and check if ttl correctly set
    await router.cache.redis_cache.async_increment(**increment_cache_kwargs)

    _redis_client = router.cache.redis_cache.init_async_client()

    async with _redis_client as redis_client:
        current_ttl = await redis_client.ttl(increment_cache_kwargs["key"])

        assert current_ttl >= 0

        print(f"current_ttl: {current_ttl}")


def test_router_caching_ttl_sync():
    """
    Confirm caching ttl's work as expected.

    Relevant issue: https://github.com/BerriAI/litellm/issues/5609
    """
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
                "tpm": 1440,
                "mock_response": "Hello world",
            },
            "model_info": {"id": 1},
        }
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
        set_verbose=False,
        redis_host=os.getenv("REDIS_HOST"),
        redis_password=os.getenv("REDIS_PASSWORD"),
        redis_port=os.getenv("REDIS_PORT"),
    )

    assert router.cache.redis_cache is not None

    increment_cache_kwargs = {}
    with patch.object(
        router.cache.redis_cache,
        "increment_cache",
        new=MagicMock(),
    ) as mock_client:
        router.completion(model=model, messages=messages)

        print(mock_client.call_args_list)
        mock_client.assert_called()
        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        print(f"mock_client.call_args.args: {mock_client.call_args.args}")

        increment_cache_kwargs = {
            "key": mock_client.call_args.args[0],
            "value": mock_client.call_args.args[1],
            "ttl": mock_client.call_args.kwargs["ttl"],
        }

        assert mock_client.call_args.kwargs["ttl"] == 60

    ## call redis async increment and check if ttl correctly set
    router.cache.redis_cache.increment_cache(**increment_cache_kwargs)

    _redis_client = router.cache.redis_cache.redis_client

    current_ttl = _redis_client.ttl(increment_cache_kwargs["key"])

    assert current_ttl >= 0

    print(f"current_ttl: {current_ttl}")


def test_return_potential_deployments():
    """
    Assert deployment at limit is filtered out
    """

    test_cache = DualCache()
    lowest_tpm_logger = LowestTPMLoggingHandler(
        router_cache=test_cache
    )

    args: Dict = {
        "healthy_deployments": [
            {
                "model_name": "model-test",
                "litellm_params": {
                    "rpm": 1,
                    "api_key": "sk-1234",
                    "model": "openai/gpt-3.5-turbo",
                    "mock_response": "Hello, world!",
                },
                "model_info": {
                    "id": "dd8e67fce56963bae6a60206b48d3f03faeb43be20cf0fd96a5f39b1a2bbd11d",
                    "db_model": False,
                },
            },
            {
                "model_name": "model-test",
                "litellm_params": {
                    "rpm": 10,
                    "api_key": "sk-1234",
                    "model": "openai/o1-mini",
                    "mock_response": "Hello, world, it's o1!",
                },
                "model_info": {
                    "id": "e13a56981607e1749b1433e6968ffc7df5552540ad3faa44b0b44ba4f3443bfe",
                    "db_model": False,
                },
            },
        ],
        "all_deployments": {
            "dd8e67fce56963bae6a60206b48d3f03faeb43be20cf0fd96a5f39b1a2bbd11d": None,
            "e13a56981607e1749b1433e6968ffc7df5552540ad3faa44b0b44ba4f3443bfe": None,
            "dd8e67fce56963bae6a60206b48d3f03faeb43be20cf0fd96a5f39b1a2bbd11d:tpm:02-17": 0,
            "e13a56981607e1749b1433e6968ffc7df5552540ad3faa44b0b44ba4f3443bfe:tpm:02-17": 0,
        },
        "input_tokens": 98,
        "rpm_dict": {
            "dd8e67fce56963bae6a60206b48d3f03faeb43be20cf0fd96a5f39b1a2bbd11d": 1,
            "e13a56981607e1749b1433e6968ffc7df5552540ad3faa44b0b44ba4f3443bfe": None,
        },
    }

    potential_deployments = lowest_tpm_logger._return_potential_deployments(
        healthy_deployments=args["healthy_deployments"],
        all_deployments=args["all_deployments"],
        input_tokens=args["input_tokens"],
        rpm_dict=args["rpm_dict"],
    )

    assert len(potential_deployments) == 1


@pytest.mark.asyncio
async def test_tpm_rpm_routing_model_name_checks():
    deployment = {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "azure/gpt-4.1-nano",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "mock_response": "Hey, how's it going?",
        },
    }
    router = Router(model_list=[deployment], routing_strategy="usage-based-routing-v2")

    async def side_effect_pre_call_check(*args, **kwargs):
        return args[0]

    with patch.object(
        router.lowesttpm_logger_v2,
        "async_pre_call_check",
        side_effect=side_effect_pre_call_check,
    ) as mock_object, patch.object(
        router.lowesttpm_logger_v2, "async_log_success_event"
    ) as mock_logging_event:
        response = await router.acompletion(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey!"}]
        )

        mock_object.assert_called()
        print(f"mock_object.call_args: {mock_object.call_args[0][0]}")
        assert (
            mock_object.call_args[0][0]["litellm_params"]["model"]
            == deployment["litellm_params"]["model"]
        )

        await asyncio.sleep(1)

        mock_logging_event.assert_called()

        print(f"mock_logging_event: {mock_logging_event.call_args.kwargs}")
        standard_logging_payload: StandardLoggingPayload = (
            mock_logging_event.call_args.kwargs.get("kwargs", {}).get(
                "standard_logging_object"
            )
        )

        assert (
            standard_logging_payload["hidden_params"]["litellm_model_name"]
            == "azure/gpt-4.1-nano"
        )
