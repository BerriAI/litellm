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
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": "azure/gpt-4.1-mini",
            },
            "model_info": {"id": deployment_id},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    # New cache key format uses individual keys per deployment
    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
    assert test_cache.get_cache(key=cache_key) is not None
    assert test_cache.get_cache(key=cache_key) == 1


def test_get_available_deployments():
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment = "azure/gpt-4.1-mini"
    deployment_id = "1234"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": deployment,
            },
            "model_info": {"id": deployment_id},
        }
    }
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    # New cache key format uses individual keys per deployment
    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
    assert test_cache.get_cache(key=cache_key) is not None
    assert test_cache.get_cache(key=cache_key) == 1


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
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
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
    # Set individual cache keys for each deployment (new format)
    request_counts = {"1": 10, "2": 54, "3": 100}
    for deployment_id, count in request_counts.items():
        cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
        if async_test is True:
            await router.cache.async_set_cache(key=cache_key, value=count)
        else:
            router.cache.set_cache(key=cache_key, value=count)

    if async_test is True:
        deployment = await router.async_get_available_deployment(
            model=model_group, messages=None, request_kwargs={}
        )
    else:
        deployment = router.get_available_deployment(model=model_group, messages=None)
    print(f"deployment: {deployment}")
    assert deployment["model_info"]["id"] == "1"

    ## run router completion - assert completion event, no change in 'busy'ness once calls are complete

    router.completion(
        model=model_group,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    # wait 2 seconds for callbacks to complete
    time.sleep(2)

    # Verify the counts are back to what they were (increment then decrement)
    assert router.leastbusy_logger.logged_success == 1

    # With new format, we check individual keys
    for deployment_id, expected_count in request_counts.items():
        cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
        actual_count = router.cache.get_cache(key=cache_key)
        assert actual_count == expected_count, f"Expected {expected_count} for {deployment_id}, got {actual_count}"


## Test with Real calls ##


@pytest.mark.asyncio
async def test_router_atext_completion_streaming():
    prompt = "Hello, can you generate a 500 words poem?"
    model = "azure-model"
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
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

    # With new format, check individual keys for each deployment
    # Each deployment should have been called once (round-robin like behavior when all start at 0)
    for deployment_id in ["1", "2", "3"]:
        cache_key = f"deployment:{model}:{deployment_id}:request_count"
        count = router.cache.get_cache(key=cache_key)
        # After completion, count should be back to 0 (or 1 if still in flight)
        # Since calls complete sequentially, all should be back to 0
        assert count is None or count == 0 or count == 1, f"Failed. deployment_id={deployment_id} has count={count}"


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
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
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

    # With new format, check individual keys for each deployment
    for deployment_id in ["1", "2", "3"]:
        cache_key = f"deployment:{model}:{deployment_id}:request_count"
        count = router.cache.get_cache(key=cache_key)
        # After completion, count should be back to 0 (or 1 if still in flight)
        assert count is None or count == 0 or count == 1, f"Failed. deployment_id={deployment_id} has count={count}"


def test_atomic_increment_decrement():
    """
    Test that atomic increment and decrement operations work correctly
    """
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"
    deployment_id = "test-deployment"

    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
            },
            "model_info": {"id": deployment_id},
        }
    }

    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"

    # Increment multiple times
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    assert test_cache.get_cache(key=cache_key) == 1

    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    assert test_cache.get_cache(key=cache_key) == 2

    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    assert test_cache.get_cache(key=cache_key) == 3

    # Decrement via success callback
    least_busy_logger.log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    assert test_cache.get_cache(key=cache_key) == 2

    # Decrement via failure callback
    least_busy_logger.log_failure_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    assert test_cache.get_cache(key=cache_key) == 1

    # Decrement again
    least_busy_logger.log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    assert test_cache.get_cache(key=cache_key) == 0

    # Decrement past 0 should reset to 0 (not go negative)
    least_busy_logger.log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    count = test_cache.get_cache(key=cache_key)
    assert count == 0, f"Count should be 0, got {count}"


@pytest.mark.asyncio
async def test_async_atomic_increment_decrement():
    """
    Test that async atomic increment and decrement operations work correctly
    """
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"
    deployment_id = "test-deployment"

    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
            },
            "model_info": {"id": deployment_id},
        }
    }

    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"

    # Increment via pre_api_call (sync)
    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    assert test_cache.get_cache(key=cache_key) == 1

    least_busy_logger.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
    assert test_cache.get_cache(key=cache_key) == 2

    # Decrement via async success callback
    await least_busy_logger.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 1

    # Decrement via async failure callback
    await least_busy_logger.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 0

    # Decrement past 0 should reset to 0
    await least_busy_logger.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 0, f"Count should be 0, got {count}"


def test_get_least_busy_deployment():
    """
    Test that the least busy deployment is correctly selected
    """
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"

    # Create healthy deployments
    healthy_deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "model-1"}},
        {"model_info": {"id": "dep-2"}, "litellm_params": {"model": "model-2"}},
        {"model_info": {"id": "dep-3"}, "litellm_params": {"model": "model-3"}},
    ]

    # Set request counts: dep-1=5, dep-2=2, dep-3=10
    test_cache.set_cache(key=f"deployment:{model_group}:dep-1:request_count", value=5)
    test_cache.set_cache(key=f"deployment:{model_group}:dep-2:request_count", value=2)
    test_cache.set_cache(key=f"deployment:{model_group}:dep-3:request_count", value=10)

    # Should select dep-2 (least busy with count=2)
    selected = least_busy_logger.get_available_deployments(
        model_group=model_group,
        healthy_deployments=healthy_deployments,
    )

    assert selected["model_info"]["id"] == "dep-2", f"Expected dep-2, got {selected['model_info']['id']}"


@pytest.mark.asyncio
async def test_async_get_least_busy_deployment():
    """
    Test that the async least busy deployment selection works correctly
    """
    test_cache = DualCache()
    least_busy_logger = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"

    # Create healthy deployments
    healthy_deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "model-1"}},
        {"model_info": {"id": "dep-2"}, "litellm_params": {"model": "model-2"}},
        {"model_info": {"id": "dep-3"}, "litellm_params": {"model": "model-3"}},
    ]

    # Set request counts: dep-1=5, dep-2=10, dep-3=1
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-1:request_count", value=5)
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-2:request_count", value=10)
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-3:request_count", value=1)

    # Should select dep-3 (least busy with count=1)
    selected = await least_busy_logger.async_get_available_deployments(
        model_group=model_group,
        healthy_deployments=healthy_deployments,
    )

    assert selected["model_info"]["id"] == "dep-3", f"Expected dep-3, got {selected['model_info']['id']}"
