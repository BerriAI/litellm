import pytest
import time
from unittest.mock import MagicMock
from litellm.litellm_core_utils.litellm_logging import Logging

def test_should_run_logging_double_count_reproduction():
    """
    Reproduction for Issue #19929.
    
    Verifies that 'should_run_logging' currently returns True for both 'async_success' 
    and 'sync_success' for the same request, leading to double counting.
    """
    logging_obj = Logging(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="test_id",
        function_id="test_func_id"
    )
    logging_obj.model_call_details = {}
    
    # 1. First call: async_success
    should_run_async = logging_obj.should_run_logging(event_type="async_success", stream=False)
    assert should_run_async is True
    
    # Simulate the handler running and setting the flag
    logging_obj.model_call_details["has_logged_async_success"] = True
    
    # 2. Second call: sync_success (triggered by fallback logic)
    should_run_sync = logging_obj.should_run_logging(event_type="sync_success", stream=False)
    
    # BUG: This currently returns True, causing double logging
    # assert should_run_sync is True 
    
    # Updated to assert correct behavior (Fix for #19929)
    assert should_run_sync is False

def test_should_run_logging_double_count_fix_expectation():
    """
    This test currently FAILS if the bug exists. 
    It represents the Desired Behavior.
    """
    logging_obj = Logging(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="test_id",
        function_id="test_func_id"
    )
    logging_obj.model_call_details = {}
    
    # 1. First call: async_success
    should_run_async = logging_obj.should_run_logging(event_type="async_success", stream=False)
    assert should_run_async is True
    logging_obj.model_call_details["has_logged_async_success"] = True
    
    # 2. Second call: sync_success
    should_run_sync = logging_obj.should_run_logging(event_type="sync_success", stream=False)
    
    # DESIRED: Should be False to prevent double counting
    assert should_run_sync is False
