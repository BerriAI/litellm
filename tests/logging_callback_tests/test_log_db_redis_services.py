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
from litellm.proxy.utils import log_db_metrics, ServiceTypes
from datetime import datetime
import httpx
from prisma.errors import ClientNotConnectedError


# Test async function to decorate
@log_db_metrics
async def sample_db_function(*args, **kwargs):
    return "success"


@log_db_metrics
async def sample_proxy_function(*args, **kwargs):
    return "success"


@pytest.mark.asyncio
async def test_log_db_metrics_success():
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
async def test_log_db_metrics_duration():
    # Mock the proxy_logging_obj
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        # Setup mock
        mock_proxy_logging.service_logging_obj.async_service_success_hook = AsyncMock()

        # Add a delay to the function to test duration
        @log_db_metrics
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
async def test_log_db_metrics_failure():
    """
    should log a failure if a prisma error is raised
    """
    # Mock the proxy_logging_obj
    from prisma.errors import ClientNotConnectedError

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        # Setup mock
        mock_proxy_logging.service_logging_obj.async_service_failure_hook = AsyncMock()

        # Create a failing function
        @log_db_metrics
        async def failing_function(**kwargs):
            raise ClientNotConnectedError()

        # Call the decorated function and expect it to raise
        with pytest.raises(ClientNotConnectedError) as exc_info:
            await failing_function(parent_otel_span="test_span")

        # Assertions
        assert "Client is not connected to the query engine" in str(exc_info.value)
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
        assert isinstance(call_args["error"], ClientNotConnectedError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception,should_log",
    [
        (ValueError("Generic error"), False),
        (KeyError("Missing key"), False),
        (TypeError("Type error"), False),
        (httpx.ConnectError("Failed to connect"), True),
        (httpx.TimeoutException("Request timed out"), True),
        (ClientNotConnectedError(), True),  # Prisma error
    ],
)
async def test_log_db_metrics_failure_error_types(exception, should_log):
    """
    Why Test?
    Users were seeing that non-DB errors were being logged as DB Service Failures
    Example a failure to read a value from cache was being logged as a DB Service Failure


    Parameterized test to verify:
    - DB-related errors (Prisma, httpx) are logged as service failures
    - Non-DB errors (ValueError, KeyError, etc.) are not logged
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        mock_proxy_logging.service_logging_obj.async_service_failure_hook = AsyncMock()

        @log_db_metrics
        async def failing_function(**kwargs):
            raise exception

        # Call the function and expect it to raise the exception
        with pytest.raises(type(exception)):
            await failing_function(parent_otel_span="test_span")

        if should_log:
            # Assert failure was logged for DB-related errors
            mock_proxy_logging.service_logging_obj.async_service_failure_hook.assert_called_once()
            call_args = mock_proxy_logging.service_logging_obj.async_service_failure_hook.call_args[
                1
            ]
            assert call_args["service"] == ServiceTypes.DB
            assert call_args["call_type"] == "failing_function"
            assert call_args["parent_otel_span"] == "test_span"
            assert isinstance(call_args["duration"], float)
            assert isinstance(call_args["start_time"], datetime)
            assert isinstance(call_args["end_time"], datetime)
            assert isinstance(call_args["error"], type(exception))
        else:
            # Assert failure was NOT logged for non-DB errors
            mock_proxy_logging.service_logging_obj.async_service_failure_hook.assert_not_called()
