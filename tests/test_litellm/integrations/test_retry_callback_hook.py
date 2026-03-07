"""
Tests for the retry callback hook functionality.

Tests that log_retry_event and async_log_retry_event methods are called
on CustomLogger instances when retries occur.
"""

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.main import _async_fire_retry_callbacks, _fire_retry_callbacks


class RetryCallbackHandler(CustomLogger):
    """Test logger that tracks retry events."""

    def __init__(self):
        super().__init__()
        self.retry_events = []
        self.async_retry_events = []

    def log_retry_event(
        self,
        kwargs: dict,
        exception: Exception,
        retry_count: int,
        max_retries: int,
    ) -> None:
        self.retry_events.append(
            {
                "kwargs": kwargs,
                "exception": exception,
                "retry_count": retry_count,
                "max_retries": max_retries,
            }
        )

    async def async_log_retry_event(
        self,
        kwargs: dict,
        exception: Exception,
        retry_count: int,
        max_retries: int,
    ) -> None:
        self.async_retry_events.append(
            {
                "kwargs": kwargs,
                "exception": exception,
                "retry_count": retry_count,
                "max_retries": max_retries,
            }
        )


class FailingRetryHandler(CustomLogger):
    """Handler that raises an exception in log_retry_event."""

    def log_retry_event(
        self,
        kwargs: dict,
        exception: Exception,
        retry_count: int,
        max_retries: int,
    ) -> None:
        raise RuntimeError("Callback failed")


class TestRetryCallbackHook:
    """Tests for retry callback hook functionality."""

    def test_fire_retry_callbacks_calls_log_retry_event(self):
        """Test that _fire_retry_callbacks calls log_retry_event on CustomLogger."""
        test_handler = RetryCallbackHandler()
        original_callbacks = litellm.callbacks.copy()

        try:
            litellm.callbacks = [test_handler]

            # Create mock retry_state
            mock_retry_state = MagicMock()
            mock_exception = Exception("Test error")
            mock_retry_state.outcome.exception.return_value = mock_exception
            mock_retry_state.attempt_number = 2

            test_kwargs = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
            }

            _fire_retry_callbacks(mock_retry_state, num_retries=3, kwargs=test_kwargs)

            assert len(test_handler.retry_events) == 1
            event = test_handler.retry_events[0]
            assert event["exception"] == mock_exception
            assert event["retry_count"] == 2
            assert event["max_retries"] == 3
            assert event["kwargs"] == test_kwargs
        finally:
            litellm.callbacks = original_callbacks

    @pytest.mark.asyncio
    async def test_async_fire_retry_callbacks_calls_async_log_retry_event(self):
        """Test that _async_fire_retry_callbacks calls async_log_retry_event on CustomLogger."""
        test_handler = RetryCallbackHandler()
        original_callbacks = litellm.callbacks.copy()

        try:
            litellm.callbacks = [test_handler]

            # Create mock retry_state
            mock_retry_state = MagicMock()
            mock_exception = Exception("Test async error")
            mock_retry_state.outcome.exception.return_value = mock_exception
            mock_retry_state.attempt_number = 1

            test_kwargs = {
                "model": "claude-3",
                "messages": [{"role": "user", "content": "async test"}],
            }

            await _async_fire_retry_callbacks(
                mock_retry_state, num_retries=5, kwargs=test_kwargs
            )

            assert len(test_handler.async_retry_events) == 1
            event = test_handler.async_retry_events[0]
            assert event["exception"] == mock_exception
            assert event["retry_count"] == 1
            assert event["max_retries"] == 5
            assert event["kwargs"] == test_kwargs
        finally:
            litellm.callbacks = original_callbacks

    def test_fire_retry_callbacks_skips_callbacks_without_method(self):
        """Test that callbacks without log_retry_event are skipped."""

        # Create a callback without log_retry_event method
        class SimpleCallback:
            pass

        simple_callback = SimpleCallback()
        test_handler = RetryCallbackHandler()
        original_callbacks = litellm.callbacks.copy()

        try:
            litellm.callbacks = [simple_callback, test_handler]

            mock_retry_state = MagicMock()
            mock_retry_state.outcome.exception.return_value = Exception("Test")
            mock_retry_state.attempt_number = 1

            # Should not raise, should skip simple_callback
            _fire_retry_callbacks(mock_retry_state, num_retries=3, kwargs={})

            # Only test_handler should have received the event
            assert len(test_handler.retry_events) == 1
        finally:
            litellm.callbacks = original_callbacks

    def test_fire_retry_callbacks_handles_callback_exceptions(self):
        """Test that exceptions in callbacks are caught and logged."""
        failing_handler = FailingRetryHandler()
        test_handler = RetryCallbackHandler()
        original_callbacks = litellm.callbacks.copy()

        try:
            litellm.callbacks = [failing_handler, test_handler]

            mock_retry_state = MagicMock()
            mock_retry_state.outcome.exception.return_value = Exception("Test")
            mock_retry_state.attempt_number = 1

            # Should not raise, should continue to next callback
            _fire_retry_callbacks(mock_retry_state, num_retries=3, kwargs={})

            # test_handler should still receive the event
            assert len(test_handler.retry_events) == 1
        finally:
            litellm.callbacks = original_callbacks


class TestCustomLoggerBaseRetryMethods:
    """Tests for base CustomLogger retry methods."""

    def test_custom_logger_base_retry_methods_are_no_ops(self):
        """Test that base CustomLogger retry methods are no-ops."""
        logger = CustomLogger()

        # Should not raise
        logger.log_retry_event(
            kwargs={},
            exception=Exception("test"),
            retry_count=1,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_custom_logger_async_base_retry_method_is_no_op(self):
        """Test that base CustomLogger async retry method is a no-op."""
        logger = CustomLogger()

        # Should not raise
        await logger.async_log_retry_event(
            kwargs={},
            exception=Exception("test"),
            retry_count=1,
            max_retries=3,
        )
