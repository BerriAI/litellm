import time

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


def test_lowest_cost_sync_success_model_info_none():
    """log_success_event should not crash when model_info is None."""
    logger = LowestCostLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


def test_lowest_cost_sync_failure_model_info_none():
    """log_failure_event should not crash when model_info is None."""
    logger = LowestCostLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    logger.log_failure_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


def test_lowest_cost_sync_success_model_info_missing():
    """log_success_event should not crash when model_info key is missing."""
    logger = LowestCostLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
        }
    }
    logger.log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )


@pytest.mark.asyncio
async def test_lowest_cost_async_success_model_info_none():
    """async_log_success_event should not crash when model_info is None."""
    logger = LowestCostLoggingHandler(router_cache=DualCache())
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "gpt-4"},
            "model_info": None,
        }
    }
    await logger.async_log_success_event(
        kwargs=kwargs, response_obj={}, start_time=time.time(), end_time=time.time()
    )
