"""
Tests for litellm/_service_logger.py

Regression test for KeyError: 'call_type' when async_log_success_event
is called without call_type in kwargs (e.g. from batch polling callbacks).
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from litellm._service_logger import ServiceLogging
from litellm.types.services import ServiceTypes


@pytest.mark.asyncio
async def test_async_log_success_event_should_not_raise_when_call_type_missing():
    """
    When async_log_success_event is called with kwargs that omit 'call_type',
    it should not raise a KeyError. This happens in the batch polling flow
    where check_batch_cost.py creates a Logging object whose model_call_details
    don't include call_type.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = datetime(2026, 2, 13, 22, 35, 0)
    end_time = datetime(2026, 2, 13, 22, 35, 1)
    kwargs_without_call_type = {"model": "gpt-4", "stream": False}

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs=kwargs_without_call_type,
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["call_type"] == "unknown"


@pytest.mark.asyncio
async def test_async_log_success_event_should_pass_call_type_when_present():
    """
    When call_type IS present in kwargs, it should be forwarded correctly.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = datetime(2026, 2, 13, 22, 35, 0)
    end_time = datetime(2026, 2, 13, 22, 35, 1)
    kwargs_with_call_type = {
        "model": "gpt-4",
        "stream": False,
        "call_type": "aretrieve_batch",
    }

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs=kwargs_with_call_type,
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["call_type"] == "aretrieve_batch"


@pytest.mark.asyncio
async def test_async_log_success_event_should_handle_float_duration():
    """
    When start_time and end_time produce a float duration (not timedelta),
    it should still work correctly.
    """
    service_logger = ServiceLogging(mock_testing=True)

    start_time = 1000.0
    end_time = 1001.5

    with patch.object(
        service_logger, "async_service_success_hook", new_callable=AsyncMock
    ) as mock_hook:
        await service_logger.async_log_success_event(
            kwargs={"call_type": "completion"},
            response_obj=None,
            start_time=start_time,
            end_time=end_time,
        )

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["duration"] == 1.5


@pytest.mark.asyncio
async def test_service_failure_hook_should_schedule_async_failure_hook():
    service_logger = ServiceLogging(mock_testing=True)

    with patch.object(
        service_logger, "async_service_failure_hook", new_callable=AsyncMock
    ) as mock_hook:
        service_logger.service_failure_hook(
            service=ServiceTypes.REDIS,
            duration=0.123,
            error=Exception("boom"),
            call_type="health_check",
        )

        # Let the event loop run scheduled tasks
        await asyncio.sleep(0)

        mock_hook.assert_awaited_once()
        call_kwargs = mock_hook.call_args
        assert call_kwargs.kwargs["service"] == ServiceTypes.REDIS
        assert call_kwargs.kwargs["duration"] == 0.123
        assert isinstance(call_kwargs.kwargs["error"], Exception)
        assert str(call_kwargs.kwargs["error"]) == "boom"
        assert call_kwargs.kwargs["call_type"] == "health_check"
