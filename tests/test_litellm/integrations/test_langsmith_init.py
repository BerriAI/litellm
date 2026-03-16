import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.langsmith import LangsmithLogger


class TestLangsmithLoggerInit:
    """Test cases for LangSmith logger initialization, particularly sampling rate handling.

    These tests verify that the sampling_rate attribute is set during initialization.
    Note: The current implementation has some edge cases in the sampling rate logic.
    """

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_parameter_respected_with_valid_env(self):
        """Test that langsmith_sampling_rate parameter is properly set when env var condition is met."""
        sampling_rate = 0.5
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_sampling_rate=sampling_rate,
        )

        assert (
            logger.sampling_rate == sampling_rate
        ), f"Expected sampling_rate to be {sampling_rate}, got {logger.sampling_rate}"

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_zero_parameter_falls_back_to_env(self):
        """Test that 0.0 parameter falls back to env var due to falsy value."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_sampling_rate=0.0,
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to fall back to 1.0 from env, got {logger.sampling_rate}"

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_from_integer_env_var(self):
        """Test that sampling rate uses environment variable when parameter not provided and env var is integer."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to be 1.0 from env var, got {logger.sampling_rate}"

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "0.8"}, clear=False)
    def test_langsmith_sampling_rate_decimal_env_var_ignored(self):
        """Test that decimal environment variables are ignored due to isdigit() check."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 (decimal env ignored), got {logger.sampling_rate}"

    @patch.dict(os.environ, {}, clear=True)
    def test_langsmith_sampling_rate_default_value(self):
        """Test that sampling rate defaults to 1.0 when no parameter or env var provided."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected default sampling_rate to be 1.0, got {logger.sampling_rate}"

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "invalid"}, clear=False)
    def test_langsmith_sampling_rate_invalid_env_var_defaults(self):
        """Test that invalid environment variable falls back to default value."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 with invalid env var, got {logger.sampling_rate}"

    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": ""}, clear=False)
    def test_langsmith_sampling_rate_empty_env_var_defaults(self):
        """Test that empty environment variable falls back to default value."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 with empty env var, got {logger.sampling_rate}"

    def test_langsmith_sampling_rate_attribute_exists(self):
        """Test that the sampling_rate attribute is always set on the logger instance."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert hasattr(
            logger, "sampling_rate"
        ), "LangsmithLogger should have sampling_rate attribute"
        assert isinstance(
            logger.sampling_rate, float
        ), f"sampling_rate should be a float, got {type(logger.sampling_rate)}"
        assert (
            logger.sampling_rate >= 0.0
        ), f"sampling_rate should be non-negative, got {logger.sampling_rate}"

    @patch.object(LangsmithLogger, "_start_periodic_flush_task", return_value=None)
    def test_langsmith_init_skips_periodic_flush_without_running_loop(
        self, mock_start_periodic_flush_task
    ):
        """Test that sync initialization leaves the periodic flush task unset."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert logger is not None
        mock_start_periodic_flush_task.assert_called_once()
        assert logger._flush_task is None

    @patch("asyncio.get_running_loop", side_effect=RuntimeError("no running event loop"))
    def test_start_periodic_flush_task_returns_none_without_running_loop(
        self, mock_get_running_loop
    ):
        """Test that helper returns None when no running event loop exists."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            start_periodic_flush=False,
        )

        mock_get_running_loop.reset_mock()

        assert logger._start_periodic_flush_task() is None
        mock_get_running_loop.assert_called_once()

    @patch("asyncio.get_running_loop")
    def test_langsmith_init_starts_periodic_flush_with_running_loop(
        self, mock_get_running_loop
    ):
        """Test that init schedules periodic flush when a running loop exists."""
        mock_loop = MagicMock()
        mock_task = MagicMock()
        mock_loop.create_task.return_value = mock_task
        mock_get_running_loop.return_value = mock_loop

        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert logger._flush_task == mock_task
        mock_loop.create_task.assert_called_once()
        scheduled_coro = mock_loop.create_task.call_args.args[0]
        scheduled_coro.close()

    @pytest.mark.asyncio
    async def test_async_log_success_event_lazily_starts_periodic_flush(self):
        """Test that async logging lazily starts periodic flush after sync init."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            start_periodic_flush=False,
        )
        logger._get_sampling_rate_to_use_for_request = MagicMock(return_value=1.0)
        logger._get_credentials_to_use_for_request = MagicMock(
            return_value=logger.default_credentials
        )
        logger._prepare_log_data = MagicMock(return_value={"id": "run-id"})
        logger._start_periodic_flush_task = MagicMock(return_value=MagicMock())

        await logger.async_log_success_event({}, {}, None, None)

        logger._start_periodic_flush_task.assert_called_once()
        assert len(logger.log_queue) == 1
