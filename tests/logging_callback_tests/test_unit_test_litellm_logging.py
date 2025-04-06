import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from typing import Literal

import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.cache_control_check import _PROXY_CacheControlCheck
from litellm._service_logger import ServiceLogging
import asyncio


from litellm.litellm_core_utils.litellm_logging import Logging
import litellm

service_logger = ServiceLogging()


def setup_logging():
    return Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="123",
        function_id="456",
    )


def test_get_callback_name():
    """
    Ensure we can get the name of a callback
    """
    logging = setup_logging()

    # Test function with __name__
    def test_func():
        pass

    assert logging._get_callback_name(test_func) == "test_func"

    # Test function with __func__
    class TestClass:
        def method(self):
            pass

    bound_method = TestClass().method
    assert logging._get_callback_name(bound_method) == "method"

    # Test string callback
    assert logging._get_callback_name("callback_string") == "callback_string"


def test_is_internal_litellm_proxy_callback():
    """
    Ensure we can determine if a callback is an internal litellm proxy callback

    eg. `_PROXY_MaxBudgetLimiter`, `_PROXY_CacheControlCheck`
    """
    logging = setup_logging()

    assert logging._is_internal_litellm_proxy_callback(_PROXY_MaxBudgetLimiter) == True

    # Test non-internal callbacks
    def regular_callback():
        pass

    assert logging._is_internal_litellm_proxy_callback(regular_callback) == False

    # Test string callback
    assert logging._is_internal_litellm_proxy_callback("callback_string") == False


def test_should_run_sync_callbacks_for_async_calls():
    """
    Ensure we can determine if we should run sync callbacks for async calls

    Note: We don't want to run sync callbacks for async calls because we don't want to block the event loop
    """
    logging = setup_logging()

    # Test with no callbacks
    logging.dynamic_success_callbacks = None
    litellm.success_callback = []
    assert logging._should_run_sync_callbacks_for_async_calls() == False

    # Test with regular callback
    def regular_callback():
        pass

    litellm.success_callback = [regular_callback]
    assert logging._should_run_sync_callbacks_for_async_calls() == True

    # Test with internal callback only
    litellm.success_callback = [_PROXY_MaxBudgetLimiter]
    assert logging._should_run_sync_callbacks_for_async_calls() == False


def test_remove_internal_litellm_callbacks():
    logging = setup_logging()

    def regular_callback():
        pass

    callbacks = [
        regular_callback,
        _PROXY_MaxBudgetLimiter,
        _PROXY_CacheControlCheck,
        "string_callback",
    ]

    filtered = logging._remove_internal_litellm_callbacks(callbacks)
    assert len(filtered) == 2  # Should only keep regular_callback and string_callback
    assert regular_callback in filtered
    assert "string_callback" in filtered
    assert _PROXY_MaxBudgetLimiter not in filtered
    assert _PROXY_CacheControlCheck not in filtered
