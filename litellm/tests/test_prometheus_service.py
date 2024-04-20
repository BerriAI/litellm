# What is this?
## Unit Tests for prometheus service monitoring

import json
import sys
import os
import io, asyncio

sys.path.insert(0, os.path.abspath("../.."))
import pytest
from litellm import acompletion, Cache
from litellm._service_logger import ServiceLogging
from litellm.integrations.prometheus_services import PrometheusServicesLogger
import litellm

"""
- Check if it receives a call when redis is used 
- Check if it fires messages accordingly
"""


@pytest.mark.asyncio
async def test_init_prometheus():
    """
    - Run completion with caching
    - Assert success callback gets called
    """

    pl = PrometheusServicesLogger(mock_testing=True)


@pytest.mark.asyncio
async def test_completion_with_caching():
    """
    - Run completion with caching
    - Assert success callback gets called
    """

    litellm.set_verbose = True
    litellm.cache = Cache(type="redis")
    litellm.service_callback = ["prometheus_system"]

    sl = ServiceLogging(mock_testing=True)
    sl.prometheusServicesLogger.mock_testing = True
    litellm.cache.cache.service_logger_obj = sl

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response1 = await acompletion(
        model="gpt-3.5-turbo", messages=messages, caching=True
    )
    response1 = await acompletion(
        model="gpt-3.5-turbo", messages=messages, caching=True
    )

    assert sl.mock_testing_async_success_hook > 0
    assert sl.prometheusServicesLogger.mock_testing_success_calls > 0
    assert sl.mock_testing_sync_failure_hook == 0
    assert sl.mock_testing_async_failure_hook == 0


@pytest.mark.asyncio
async def test_completion_with_caching_bad_call():
    """
    - Run completion with caching (incorrect credentials)
    - Assert failure callback gets called
    """
    litellm.set_verbose = True

    try:
        from litellm.caching import RedisCache

        litellm.service_callback = ["prometheus_system"]
        sl = ServiceLogging(mock_testing=True)

        RedisCache(host="hello-world", service_logger_obj=sl)
    except Exception as e:
        print(f"Receives exception = {str(e)}")

    await asyncio.sleep(5)
    assert sl.mock_testing_async_failure_hook > 0
    assert sl.mock_testing_async_success_hook == 0
    assert sl.mock_testing_sync_success_hook == 0


@pytest.mark.asyncio
async def test_router_with_caching():
    """
    - Run router with usage-based-routing-v2
    - Assert success callback gets called
    """
    try:

        def get_azure_params(deployment_name: str):
            params = {
                "model": f"azure/{deployment_name}",
                "api_key": os.environ["AZURE_API_KEY"],
                "api_version": os.environ["AZURE_API_VERSION"],
                "api_base": os.environ["AZURE_API_BASE"],
            }
            return params

        model_list = [
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_azure_params("chatgpt-v-2"),
                "tpm": 100,
            },
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_azure_params("chatgpt-v-2"),
                "tpm": 1000,
            },
        ]

        router = litellm.Router(
            model_list=model_list,
            set_verbose=True,
            debug_level="DEBUG",
            routing_strategy="usage-based-routing-v2",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=os.environ["REDIS_PORT"],
            redis_password=os.environ["REDIS_PASSWORD"],
        )

        litellm.service_callback = ["prometheus_system"]

        sl = ServiceLogging(mock_testing=True)
        sl.prometheusServicesLogger.mock_testing = True
        router.cache.redis_cache.service_logger_obj = sl

        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        response1 = await router.acompletion(model="azure/gpt-4", messages=messages)
        response1 = await router.acompletion(model="azure/gpt-4", messages=messages)

        assert sl.mock_testing_async_success_hook > 0
        assert sl.mock_testing_sync_failure_hook == 0
        assert sl.mock_testing_async_failure_hook == 0
        assert sl.prometheusServicesLogger.mock_testing_success_calls > 0

    except Exception as e:
        pytest.fail(f"An exception occured - {str(e)}")
