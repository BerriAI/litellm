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
)  # Adds the parent directory to the system-path

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.cooldown_handlers import (
    _async_get_cooldown_deployments,
    _should_run_cooldown_logic,
)
from litellm.types.router import (
    AllowedFailsPolicy,
    DeploymentTypedDict,
    LiteLLMParamsTypedDict,
)


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
                    "model": "azure/chatgpt-v-3",
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
    except Exception:
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


@pytest.mark.asyncio
async def test_cooldown_time_zero_uses_zero_not_default():
    """
    Test that when cooldown_time=0 is passed, it uses 0 instead of the default cooldown time
    AND that the early exit logic prevents cooldown entirely
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "cooldown_time": 0,
                },
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            },
        ],
        cooldown_time=300,  # Default cooldown time is 300 seconds
        num_retries=0,
    )

    # Mock the add_deployment_to_cooldown method to verify it's NOT called
    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown"
    ) as mock_add_cooldown:
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        # Verify that add_deployment_to_cooldown was NOT called due to early exit
        mock_add_cooldown.assert_not_called()

    # Also verify the deployment is not in cooldown
    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    assert len(cooldown_list) == 0

    # Verify the deployment is still healthy and available
    healthy_deployments, _ = await router._async_get_healthy_deployments(
        model="gpt-3.5-turbo", parent_otel_span=None
    )
    assert len(healthy_deployments) == 1


def test_should_run_cooldown_logic_early_exit_on_zero_cooldown():
    """
    Unit test for _should_run_cooldown_logic to verify early exit when time_to_cooldown is 0
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
                "model_info": {
                    "id": "test-deployment-id",
                },
            }
        ],
        cooldown_time=300,
    )

    # Test with time_to_cooldown = 0 - should return False (don't run cooldown logic)
    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError(
            "test error", "openai", "gpt-3.5-turbo"
        ),
        time_to_cooldown=0.0,
    )
    assert result is False, "Should not run cooldown logic when time_to_cooldown is 0"

    # Test with very small time_to_cooldown (effectively 0) - should return False
    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError(
            "test error", "openai", "gpt-3.5-turbo"
        ),
        time_to_cooldown=1e-10,
    )
    assert (
        result is False
    ), "Should not run cooldown logic when time_to_cooldown is effectively 0"

    # Test with None time_to_cooldown - should return True (use default cooldown logic)
    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError(
            "test error", "openai", "gpt-3.5-turbo"
        ),
        time_to_cooldown=None,
    )
    assert result is True, "Should run cooldown logic when time_to_cooldown is None"

    # Test with positive time_to_cooldown - should return True
    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError(
            "test error", "openai", "gpt-3.5-turbo"
        ),
        time_to_cooldown=60.0,
    )
    assert result is True, "Should run cooldown logic when time_to_cooldown is positive"


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

    router = Router(model_list=model_list, num_retries=0)

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
async def test_single_deployment_cooldown_with_allowed_fails():
    """
    When `allowed_fails` is set, use the allowed_fails to determine cooldown for 1 deployment
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
        allowed_fails=1,
        num_retries=0,
    )

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        for _ in range(2):
            try:
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    timeout=0.0001,
                )
            except litellm.Timeout:
                pass

        await asyncio.sleep(2)

        mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_single_deployment_cooldown_with_allowed_fail_policy():
    """
    When `allowed_fails_policy` is set, use the allowed_fails_policy to determine cooldown for 1 deployment
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
        allowed_fails_policy=AllowedFailsPolicy(
            TimeoutErrorAllowedFails=1,
        ),
        num_retries=0,
    )

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        for _ in range(2):
            try:
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    timeout=0.0001,
                )
            except litellm.Timeout:
                pass

        await asyncio.sleep(2)

        mock_client.assert_called_once()


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
        litellm_router_instance=router, parent_otel_span=None
    )
    assert len(cooldown_list) == 0

    healthy_deployments, _ = await router._async_get_healthy_deployments(
        model="gpt-3.5-turbo", parent_otel_span=None
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
        litellm_router_instance=router, parent_otel_span=None
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
        litellm_router_instance=router, parent_otel_span=None
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
        litellm_router_instance=router, parent_otel_span=None
    )
    assert len(cooldown_list) == 1


"""
Unit tests for router set_cooldowns

1. _set_cooldown_deployments() will cooldown a deployment after it fails 50% requests
"""


def test_router_fallbacks_with_cooldowns_and_model_id():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "rpm": 1},
                "model_info": {
                    "id": "123",
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        fallbacks=[{"gpt-3.5-turbo": ["123"]}],
    )

    ## trigger ratelimit
    try:
        router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="litellm.RateLimitError",
        )
    except litellm.RateLimitError:
        pass

    router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.asyncio()
async def test_router_fallbacks_with_cooldowns_and_dynamic_credentials():
    """
    Ensure cooldown on credential 1 does not affect credential 2
    """
    from litellm.router_utils.cooldown_handlers import _async_get_cooldown_deployments

    litellm._turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "rpm": 1},
                "model_info": {
                    "id": "123",
                },
            }
        ]
    )

    ## trigger ratelimit
    try:
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            api_key="my-bad-key-1",
            mock_response="litellm.RateLimitError",
        )
        pytest.fail("Expected RateLimitError")
    except litellm.RateLimitError:
        pass

    await asyncio.sleep(1)

    cooldown_list = await _async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    print("cooldown_list: ", cooldown_list)
    assert len(cooldown_list) == 1

    await router.acompletion(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        messages=[{"role": "user", "content": "hi"}],
    )
