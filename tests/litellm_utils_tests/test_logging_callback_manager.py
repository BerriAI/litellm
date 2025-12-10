import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.opentelemetry import OpenTelemetry


# Test fixtures
@pytest.fixture
def callback_manager():
    manager = LoggingCallbackManager()
    # Reset callbacks before each test
    manager._reset_all_callbacks()
    return manager


@pytest.fixture
def mock_custom_logger():
    class TestLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    return TestLogger()


# Test cases
def test_add_string_callback():
    """
    Test adding a string callback to litellm.callbacks - only 1 instance of the string callback should be added
    """
    manager = LoggingCallbackManager()
    test_callback = "test_callback"

    # Add string callback
    manager.add_litellm_callback(test_callback)
    assert test_callback in litellm.callbacks

    # Test duplicate prevention
    manager.add_litellm_callback(test_callback)
    assert litellm.callbacks.count(test_callback) == 1


def test_duplicate_langfuse_logger_test():
    manager = LoggingCallbackManager()
    for _ in range(10):
        langfuse_logger = LangfusePromptManagement()
        manager.add_litellm_success_callback(langfuse_logger)
    print("litellm.success_callback: ", litellm.success_callback)
    assert len(litellm.success_callback) == 1


def test_duplicate_multiple_loggers_test():
    manager = LoggingCallbackManager()
    for _ in range(10):
        langfuse_logger = LangfusePromptManagement()
        otel_logger = OpenTelemetry()
        manager.add_litellm_success_callback(langfuse_logger)
        manager.add_litellm_success_callback(otel_logger)
    print("litellm.success_callback: ", litellm.success_callback)
    assert len(litellm.success_callback) == 2

    # Check exactly one instance of each logger type
    langfuse_count = sum(
        1
        for callback in litellm.success_callback
        if isinstance(callback, LangfusePromptManagement)
    )
    otel_count = sum(
        1
        for callback in litellm.success_callback
        if isinstance(callback, OpenTelemetry)
    )

    assert (
        langfuse_count == 1
    ), "Should have exactly one LangfusePromptManagement instance"
    assert otel_count == 1, "Should have exactly one OpenTelemetry instance"


def test_add_function_callback():
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    # Add function callback
    manager.add_litellm_callback(test_func)
    assert test_func in litellm.callbacks

    # Test duplicate prevention
    manager.add_litellm_callback(test_func)
    assert litellm.callbacks.count(test_func) == 1


def test_add_custom_logger(mock_custom_logger):
    manager = LoggingCallbackManager()

    # Add custom logger
    manager.add_litellm_callback(mock_custom_logger)
    assert mock_custom_logger in litellm.callbacks


def test_add_multiple_callback_types(mock_custom_logger):
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    string_callback = "test_callback"

    # Add different types of callbacks
    manager.add_litellm_callback(string_callback)
    manager.add_litellm_callback(test_func)
    manager.add_litellm_callback(mock_custom_logger)

    assert string_callback in litellm.callbacks
    assert test_func in litellm.callbacks
    assert mock_custom_logger in litellm.callbacks
    assert len(litellm.callbacks) == 3


def test_success_failure_callbacks():
    manager = LoggingCallbackManager()

    success_callback = "success_callback"
    failure_callback = "failure_callback"

    # Add callbacks
    manager.add_litellm_success_callback(success_callback)
    manager.add_litellm_failure_callback(failure_callback)

    assert success_callback in litellm.success_callback
    assert failure_callback in litellm.failure_callback


def test_async_callbacks():
    manager = LoggingCallbackManager()

    async_success = "async_success"
    async_failure = "async_failure"

    # Add async callbacks
    manager.add_litellm_async_success_callback(async_success)
    manager.add_litellm_async_failure_callback(async_failure)

    assert async_success in litellm._async_success_callback
    assert async_failure in litellm._async_failure_callback


def test_remove_callback_from_list_by_object():
    manager = LoggingCallbackManager()
    # Reset all callbacks
    manager._reset_all_callbacks()

    def TestObject():
        def __init__(self):
            manager.add_litellm_callback(self.callback)
            manager.add_litellm_success_callback(self.callback)
            manager.add_litellm_failure_callback(self.callback)
            manager.add_litellm_async_success_callback(self.callback)
            manager.add_litellm_async_failure_callback(self.callback)

        def callback(self):
            pass    

    obj = TestObject()

    manager.remove_callback_from_list_by_object(litellm.callbacks, obj)
    manager.remove_callback_from_list_by_object(litellm.success_callback, obj)
    manager.remove_callback_from_list_by_object(litellm.failure_callback, obj)
    manager.remove_callback_from_list_by_object(litellm._async_success_callback, obj)
    manager.remove_callback_from_list_by_object(litellm._async_failure_callback, obj)

    # Verify all callback lists are empty
    assert len(litellm.callbacks) == 0
    assert len(litellm.success_callback) == 0
    assert len(litellm.failure_callback) == 0
    assert len(litellm._async_success_callback) == 0
    assert len(litellm._async_failure_callback) == 0



def test_reset_callbacks(callback_manager):
    # Add various callbacks
    callback_manager.add_litellm_callback("test")
    callback_manager.add_litellm_success_callback("success")
    callback_manager.add_litellm_failure_callback("failure")
    callback_manager.add_litellm_async_success_callback("async_success")
    callback_manager.add_litellm_async_failure_callback("async_failure")

    # Reset all callbacks
    callback_manager._reset_all_callbacks()

    # Verify all callback lists are empty
    assert len(litellm.callbacks) == 0
    assert len(litellm.success_callback) == 0
    assert len(litellm.failure_callback) == 0
    assert len(litellm._async_success_callback) == 0
    assert len(litellm._async_failure_callback) == 0


@pytest.mark.asyncio
async def test_slack_alerting_callback_registration(callback_manager):
    """
    Test that litellm callbacks are correctly registered for slack alerting
    when outage_alerts or region_outage_alerts are enabled
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from unittest.mock import AsyncMock, patch

    # Mock the async HTTP handler
    with patch('litellm.integrations.SlackAlerting.slack_alerting.get_async_httpx_client') as mock_http:
        mock_http.return_value = AsyncMock()
        
        # Create a fresh ProxyLogging instance
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        
        # Test 1: No callbacks should be added when alerting is None
        proxy_logging.update_values(
            alerting=None,
            alert_types=["outage_alerts", "region_outage_alerts"]
        )
        assert len(litellm.callbacks) == 0
        
        # Test 2: Callbacks should be added when slack alerting is enabled with outage alerts
        proxy_logging.update_values(
            alerting=["slack"],
            alert_types=["outage_alerts"]
        )
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)
        
        # Test 3: Callbacks should be added when slack alerting is enabled with region outage alerts
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(
            alerting=["slack"],
            alert_types=["region_outage_alerts"]
        )
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)
        
        # Test 4: No callbacks should be added for other alert types
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(
            alerting=["slack"],
            alert_types=["budget_alerts"]  # Some other alert type
        )
        assert len(litellm.callbacks) == 0

        # Test 5: Both success and regular callbacks should be added
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(
            alerting=["slack"],
            alert_types=["outage_alerts"]
        )
        assert len(litellm.callbacks) == 1  # Regular callback for outage alerts
        assert len(litellm.success_callback) == 1  # Success callback for response_taking_too_long
        assert isinstance(litellm.callbacks[0], SlackAlerting)
        # Get the method reference for comparison
        response_taking_too_long_callback = proxy_logging.slack_alerting_instance.response_taking_too_long_callback
        assert litellm.success_callback[0] == response_taking_too_long_callback

        # Cleanup
        callback_manager._reset_all_callbacks()

@pytest.mark.asyncio
async def test_generic_api_compatible_callbacks_json():
    """
    Test that callbacks defined in generic_api_compatible_callbacks.json
    are properly loaded and initialized by _add_custom_callback_generic_api_str
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    # Mock environment variable for SumoLogic webhook URL
    test_sumologic_url = "https://collectors.sumologic.com/receiver/v1/http/test123"

    with patch.dict(os.environ, {"SUMOLOGIC_WEBHOOK_URL": test_sumologic_url}):
        # Test that sumologic callback is recognized from JSON file
        result = LoggingCallbackManager._add_custom_callback_generic_api_str(
            "sumologic"
        )

        # Verify a GenericAPILogger instance is returned
        assert isinstance(
            result, GenericAPILogger
        ), "Should return GenericAPILogger instance for sumologic callback"

        # Verify the endpoint is correctly loaded from environment variable
        assert (
            result.endpoint == test_sumologic_url
        ), f"Endpoint should be {test_sumologic_url}"

        # Verify headers only contain Content-Type (no Authorization for SumoLogic)
        assert "Content-Type" in result.headers, "Should have Content-Type header"
        assert (
            result.headers["Content-Type"] == "application/json"
        ), "Content-Type should be application/json"
        assert (
            "Authorization" not in result.headers
        ), "Should not have Authorization header for SumoLogic"


@pytest.mark.asyncio
async def test_generic_api_compatible_callbacks_json_rubrik():
    """
    Test the rubrik callback from generic_api_compatible_callbacks.json
    which requires both API key and webhook URL
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    # Mock environment variables for Rubrik
    test_rubrik_url = "https://webhook.site/test-rubrik"
    test_rubrik_api_key = "sk-rubrik-test-key"

    with patch.dict(
        os.environ,
        {"RUBRIK_WEBHOOK_URL": test_rubrik_url, "RUBRIK_API_KEY": test_rubrik_api_key},
    ):
        # Test that rubrik callback is recognized from JSON file
        result = LoggingCallbackManager._add_custom_callback_generic_api_str("rubrik")

        # Verify a GenericAPILogger instance is returned
        assert isinstance(
            result, GenericAPILogger
        ), "Should return GenericAPILogger instance for rubrik callback"

        # Verify the endpoint is correctly loaded
        assert (
            result.endpoint == test_rubrik_url
        ), f"Endpoint should be {test_rubrik_url}"

        # Verify headers include Authorization with Bearer token
        assert "Content-Type" in result.headers, "Should have Content-Type header"
        assert (
            "Authorization" in result.headers
        ), "Should have Authorization header for Rubrik"
        assert (
            result.headers["Authorization"] == f"Bearer {test_rubrik_api_key}"
        ), "Authorization should have correct API key"

        # Verify event_types filter (rubrik only logs success events)
        assert result.event_types == [
            "llm_api_success"
        ], "Rubrik should only log success events"

def test_generic_api_compatible_callbacks_json_unknown_callback():
    """
    Test that unknown callbacks (not in JSON or callback_settings) are returned unchanged
    """
    # Test with a callback that doesn't exist in the JSON file
    result = LoggingCallbackManager._add_custom_callback_generic_api_str(
        "unknown_callback"
    )

    # Should return the string unchanged
    assert result == "unknown_callback", "Unknown callback should be returned as-is"
    assert isinstance(result, str), "Unknown callback should remain a string"

