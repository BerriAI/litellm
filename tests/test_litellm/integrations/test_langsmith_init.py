import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.langsmith import LangsmithLogger


class TestLangsmithPrepareLogData:
    """Test cases for LangSmith _prepare_log_data output formatting."""

    def _make_kwargs(self, extra_payload=None):
        payload = {
            "id": "test-id",
            "call_type": "completion",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "model_parameters": {},
            "metadata": {},
            "response": {
                "id": "chatcmpl-test",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "hello", "role": "assistant"},
                    }
                ],
                "usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 3,
                    "total_tokens": 8,
                },
            },
            "startTime": "2024-01-01T00:00:00Z",
            "endTime": "2024-01-01T00:00:01Z",
            "request_tags": [],
            "error_str": None,
            "status": "success",
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "response_cost": 0.00015,
            "cost_breakdown": {"input_cost": 0.00009, "output_cost": 0.00006},
        }
        if extra_payload:
            payload.update(extra_payload)
        return {"litellm_params": {"metadata": {}}, "standard_logging_object": payload}

    @patch("asyncio.create_task")
    def test_prepare_log_data_populates_usage_metadata(self, mock_create_task):
        """outputs["usage_metadata"] must be present so LangSmith can show the Cost column."""
        logger = LangsmithLogger(langsmith_api_key="test-key")
        credentials = logger.get_credentials_from_env(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_base_url="https://api.smith.langchain.com",
        )
        data = logger._prepare_log_data(
            kwargs=self._make_kwargs(),
            response_obj=None,
            start_time=None,
            end_time=None,
            credentials=credentials,
        )
        outputs = data["outputs"]
        assert "usage_metadata" in outputs, (
            "outputs must contain usage_metadata so LangSmith can display the Cost column"
        )
        usage_metadata = outputs["usage_metadata"]
        assert usage_metadata["input_tokens"] == 3
        assert usage_metadata["output_tokens"] == 5
        assert usage_metadata["total_tokens"] == 8
        assert usage_metadata["total_cost"] == pytest.approx(0.00015)
        assert usage_metadata["input_cost"] == pytest.approx(0.00009)
        assert usage_metadata["output_cost"] == pytest.approx(0.00006)

    @patch("asyncio.create_task")
    def test_prepare_log_data_usage_metadata_none_values_default_to_zero(
        self, mock_create_task
    ):
        """None token/cost fields must not be passed to LangSmith; they must fall back to 0."""
        logger = LangsmithLogger(langsmith_api_key="test-key")
        credentials = logger.get_credentials_from_env(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_base_url="https://api.smith.langchain.com",
        )
        data = logger._prepare_log_data(
            kwargs=self._make_kwargs(
                {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "response_cost": None,
                    "cost_breakdown": None,
                }
            ),
            response_obj=None,
            start_time=None,
            end_time=None,
            credentials=credentials,
        )
        usage_metadata = data["outputs"]["usage_metadata"]
        assert usage_metadata["input_tokens"] == 0
        assert usage_metadata["output_tokens"] == 0
        assert usage_metadata["total_tokens"] == 0
        assert usage_metadata["input_cost"] == 0.0
        assert usage_metadata["output_cost"] == 0.0
        assert usage_metadata["total_cost"] == 0.0

    @patch("asyncio.create_task")
    def test_prepare_log_data_does_not_mutate_original_response(
        self, mock_create_task
    ):
        """Injecting usage_metadata must not modify the original payload['response'] dict."""
        logger = LangsmithLogger(langsmith_api_key="test-key")
        credentials = logger.get_credentials_from_env(
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            langsmith_base_url="https://api.smith.langchain.com",
        )
        kwargs = self._make_kwargs()
        original_response = kwargs["standard_logging_object"]["response"]
        logger._prepare_log_data(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
            credentials=credentials,
        )
        assert "usage_metadata" not in original_response, (
            "_prepare_log_data must not mutate the original payload['response'] dict"
        )


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
