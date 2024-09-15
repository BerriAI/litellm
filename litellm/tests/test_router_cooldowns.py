#### What this tests ####
#    This tests calling router with fallback models

import asyncio
import os
import random
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.cooldown_handlers import _async_get_cooldown_deployments
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict


@pytest.mark.asyncio
async def test_cooldown_badrequest_error():
    """
    Test 1. It SHOULD NOT cooldown a deployment on a BadRequestError
    """

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
        debug_level="DEBUG",
        set_verbose=True,
        cooldown_time=300,
        num_retries=0,
        allowed_fails=0,
    )

    # Act & Assert
    try:

        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "gm"}],
            bad_param=200,
        )
    except:
        pass

    await asyncio.sleep(3)  # wait for deployment to get cooled-down

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "gm"}],
        mock_response="hello",
    )

    assert response is not None

    print(response)


@pytest.mark.asyncio
async def test_dynamic_cooldowns():
    """
    Assert kwargs for completion/embedding have 'cooldown_time' as a litellm_param
    """
    # litellm.set_verbose = True
    tmp_mock = MagicMock()

    litellm.failure_callback = [tmp_mock]

    router = Router(
        model_list=[
            {
                "model_name": "my-fake-model",
                "litellm_params": {
                    "model": "openai/gpt-1",
                    "api_key": "my-key",
                    "mock_response": Exception("this is an error"),
                },
            }
        ],
        cooldown_time=60,
    )

    try:
        _ = router.completion(
            model="my-fake-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            cooldown_time=0,
            num_retries=0,
        )
    except Exception:
        pass

    tmp_mock.assert_called_once()

    print(tmp_mock.call_count)

    assert "cooldown_time" in tmp_mock.call_args[0][0]["litellm_params"]
    assert tmp_mock.call_args[0][0]["litellm_params"]["cooldown_time"] == 0


@pytest.mark.parametrize("num_deployments", [1, 2])
def test_single_deployment_no_cooldowns(num_deployments):
    """
    Do not cooldown on single deployment.

    Cooldown on multiple deployments.
    """
    model_list = []
    for i in range(num_deployments):
        model = DeploymentTypedDict(
            model_name="gpt-3.5-turbo",
            litellm_params=LiteLLMParamsTypedDict(
                model="gpt-3.5-turbo",
            ),
        )
        model_list.append(model)

    router = Router(model_list=model_list, allowed_fails=0, num_retries=0)

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        try:
            router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        if num_deployments == 1:
            mock_client.assert_not_called()
        else:
            mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_single_deployment_no_cooldowns_test_prod():
    """
    Do not cooldown on single deployment.

    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
        allowed_fails=0,
        num_retries=0,
    )

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        await asyncio.sleep(2)

        mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_single_deployment_no_cooldowns_test_prod_mock_completion_calls():
    """
    Do not cooldown on single deployment.

    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
    )

    for _ in range(20):
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router
    )
    assert len(cooldown_list) == 0

    healthy_deployments, _ = await router._async_get_healthy_deployments(
        model="gpt-3.5-turbo"
    )

    print("healthy_deployments: ", healthy_deployments)


"""
E2E - Test router cooldowns 

Test 1: 3 deployments, each deployment fails 25% requests. Assert that no deployments get put into cooldown
Test 2: 3 deployments, 1- deployment fails 6/10 requests, assert that bad deployment gets put into cooldown
Test 3: 3 deployments, 1 deployment has a period of 429 errors. Assert it is put into cooldown and other deployments work

"""


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_all_healthy_deployments():
    """
    PROD TEST - 3 deployments, each deployment fails 25% requests. Assert that no deployments get put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    import random
    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})

    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router
    )
    assert len(cooldown_list) == 0


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_one_bad_deployment():
    """
    PROD TEST - 3 deployments, 1- deployment fails 6/10 requests, assert that bad deployment gets put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    import random
    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})
    bad_deployment_id = random.choice(all_deployment_ids)
    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif bad_deployment_id == model_id:
                if num_failures / total_requests <= 0.6:

                    mock_response = "litellm.InternalServerError"

            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router
    )
    assert len(cooldown_list) == 1


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_one_rate_limited_deployment():
    """
    PROD TEST - 3 deployments, 1- deployment fails 6/10 requests, assert that bad deployment gets put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    import random
    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})
    bad_deployment_id = random.choice(all_deployment_ids)
    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif bad_deployment_id == model_id:
                if num_failures / total_requests <= 0.6:

                    mock_response = "litellm.RateLimitError"

            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except litellm.RateLimitError:
            model_stats[bad_deployment_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router
    )
    assert len(cooldown_list) == 1


"""
Unit tests for router set_cooldowns

1. _set_cooldown_deployments() will cooldown a deployment after it fails 50% requests
"""
