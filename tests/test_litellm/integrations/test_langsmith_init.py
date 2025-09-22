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

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_parameter_respected_with_valid_env(
        self, mock_create_task
    ):
        """Test that langsmith_sampling_rate parameter is properly set when env var condition is met."""
        # When there's a valid integer in env var, the parameter should be used due to 'or' logic
        sampling_rate = 0.5
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_sampling_rate=sampling_rate,
        )

        # With the current 'or' logic and valid env var, the parameter should be used
        assert (
            logger.sampling_rate == sampling_rate
        ), f"Expected sampling_rate to be {sampling_rate}, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_zero_parameter_falls_back_to_env(
        self, mock_create_task
    ):
        """Test that 0.0 parameter falls back to env var due to falsy value."""
        # This demonstrates the current behavior where 0.0 is falsy and falls back to env
        logger = LangsmithLogger(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_sampling_rate=0.0,  # This is falsy!
        )

        # Due to current 'or' logic, 0.0 falls back to env var
        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to fall back to 1.0 from env, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "1"}, clear=False)
    def test_langsmith_sampling_rate_from_integer_env_var(self, mock_create_task):
        """Test that sampling rate uses environment variable when parameter not provided and env var is integer."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        # Should use env var since it's a valid integer
        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to be 1.0 from env var, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "0.8"}, clear=False)
    def test_langsmith_sampling_rate_decimal_env_var_ignored(self, mock_create_task):
        """Test that decimal environment variables are ignored due to isdigit() check."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        # Decimal env vars are ignored due to isdigit() check, falls back to 1.0
        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 (decimal env ignored), got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {}, clear=True)
    def test_langsmith_sampling_rate_default_value(self, mock_create_task):
        """Test that sampling rate defaults to 1.0 when no parameter or env var provided."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected default sampling_rate to be 1.0, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": "invalid"}, clear=False)
    def test_langsmith_sampling_rate_invalid_env_var_defaults(self, mock_create_task):
        """Test that invalid environment variable falls back to default value."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 with invalid env var, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    @patch.dict(os.environ, {"LANGSMITH_SAMPLING_RATE": ""}, clear=False)
    def test_langsmith_sampling_rate_empty_env_var_defaults(self, mock_create_task):
        """Test that empty environment variable falls back to default value."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        assert (
            logger.sampling_rate == 1.0
        ), f"Expected sampling_rate to default to 1.0 with empty env var, got {logger.sampling_rate}"

    @patch("asyncio.create_task")
    def test_langsmith_sampling_rate_attribute_exists(self, mock_create_task):
        """Test that the sampling_rate attribute is always set on the logger instance."""
        logger = LangsmithLogger(
            langsmith_api_key="test-key", langsmith_project="test-project"
        )

        # Verify the attribute exists and is a float
        assert hasattr(
            logger, "sampling_rate"
        ), "LangsmithLogger should have sampling_rate attribute"
        assert isinstance(
            logger.sampling_rate, float
        ), f"sampling_rate should be a float, got {type(logger.sampling_rate)}"
        assert (
            logger.sampling_rate >= 0.0
        ), f"sampling_rate should be non-negative, got {logger.sampling_rate}"
