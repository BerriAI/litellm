import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.proxy.utils import log_to_opentelemetry, ServiceTypes
from datetime import datetime


# Test async function to decorate
@log_to_opentelemetry
async def sample_db_function(*args, **kwargs):
    return "success"


@log_to_opentelemetry
async def sample_proxy_function(*args, **kwargs):
    return "success"


@pytest.mark.asyncio
async def test_log_to_opentelemetry_success():
    # Mock the proxy_logging_obj
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        # Setup mock
        mock_proxy_logging.service_logging_obj.async_service_success_hook = AsyncMock()

        # Call the decorated function
        result = await sample_db_function(parent_otel_span="test_span")

        # Assertions
        assert result == "success"
        mock_proxy_logging.service_logging_obj.async_service_success_hook.assert_called_once()
        call_args = (
            mock_proxy_logging.service_logging_obj.async_service_success_hook.call_args[
                1
            ]
        )

        assert call_args["service"] == ServiceTypes.DB
        assert call_args["call_type"] == "sample_db_function"
        assert call_args["parent_otel_span"] == "test_span"
        assert isinstance(call_args["duration"], float)
        assert isinstance(call_args["start_time"], datetime)
        assert isinstance(call_args["end_time"], datetime)
        assert "function_name" in call_args["event_metadata"]


@pytest.mark.asyncio
async def test_log_to_opentelemetry_duration():
    # Mock the proxy_logging_obj
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        # Setup mock
        mock_proxy_logging.service_logging_obj.async_service_success_hook = AsyncMock()

        # Add a delay to the function to test duration
        @log_to_opentelemetry
        async def delayed_function(**kwargs):
            await asyncio.sleep(1)  # 1 second delay
            return "success"

        # Call the decorated function
        start = time.time()
        result = await delayed_function(parent_otel_span="test_span")
        end = time.time()

        # Get the actual duration
        actual_duration = end - start

        # Get the logged duration from the mock call
        call_args = (
            mock_proxy_logging.service_logging_obj.async_service_success_hook.call_args[
                1
            ]
        )
        logged_duration = call_args["duration"]

        # Assert the logged duration is approximately equal to actual duration (within 0.1 seconds)
        assert abs(logged_duration - actual_duration) < 0.1
        assert result == "success"


@pytest.mark.asyncio
async def test_log_to_opentelemetry_failure():
    # Mock the proxy_logging_obj
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        # Setup mock
        mock_proxy_logging.service_logging_obj.async_service_failure_hook = AsyncMock()

        # Create a failing function
        @log_to_opentelemetry
        async def failing_function(**kwargs):
            raise ValueError("Test error")

        # Call the decorated function and expect it to raise
        with pytest.raises(ValueError) as exc_info:
            await failing_function(parent_otel_span="test_span")

        # Assertions
        assert str(exc_info.value) == "Test error"
        mock_proxy_logging.service_logging_obj.async_service_failure_hook.assert_called_once()
        call_args = (
            mock_proxy_logging.service_logging_obj.async_service_failure_hook.call_args[
                1
            ]
        )

        assert call_args["service"] == ServiceTypes.DB
        assert call_args["call_type"] == "failing_function"
        assert call_args["parent_otel_span"] == "test_span"
        assert isinstance(call_args["duration"], float)
        assert isinstance(call_args["start_time"], datetime)
        assert isinstance(call_args["end_time"], datetime)
        assert isinstance(call_args["error"], ValueError)
