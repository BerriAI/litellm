import time

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler


def test_least_busy_pre_api_call_model_info_none():
    """log_pre_api_call should not crash when model_info is None."""
    logger = LeastBusyLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_pre_api_call(model="test-model", messages=[], kwargs=kwargs)


def test_least_busy_sync_success_model_info_none():
    """log_success_event should not crash when model_info is None."""
    logger = LeastBusyLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


def test_least_busy_sync_failure_model_info_none():
    """log_failure_event should not crash when model_info is None."""
    logger = LeastBusyLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_failure_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


@pytest.mark.asyncio
async def test_least_busy_async_success_model_info_none():
    """async_log_success_event should not crash when model_info is None."""
    logger = LeastBusyLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    await logger.async_log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


@pytest.mark.asyncio
async def test_least_busy_async_failure_model_info_none():
    """async_log_failure_event should not crash when model_info is None."""
    logger = LeastBusyLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    await logger.async_log_failure_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )
