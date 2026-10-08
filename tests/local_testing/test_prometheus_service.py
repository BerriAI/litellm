# What is this?
## Unit Tests for prometheus service monitoring

import asyncio

import pytest
from litellm import acompletion, Cache
from litellm._service_logger import ServiceLogging
import litellm

"""
- Check if it receives a call when redis is used 
- Check if it fires messages accordingly
"""


@pytest.mark.flaky(retries=3, delay=5)
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
        from litellm.caching.caching import RedisCache

        litellm.service_callback = ["prometheus_system"]
        sl = ServiceLogging(mock_testing=True)

        RedisCache(host="hello-world", service_logger_obj=sl)
    except Exception as e:
        print(f"Receives exception = {str(e)}")

    await asyncio.sleep(5)
    assert sl.mock_testing_async_failure_hook > 0
    assert sl.mock_testing_async_success_hook == 0
    assert sl.mock_testing_sync_success_hook == 0
