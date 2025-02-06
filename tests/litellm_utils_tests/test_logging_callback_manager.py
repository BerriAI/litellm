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
