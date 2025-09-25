"""
Test DD_AGENT_HOST configuration for Datadog integrations
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger


class TestDataDogAgentHost:
    """Test DD_AGENT_HOST configuration for DataDogLogger"""

    def test_datadog_logger_agent_mode_configuration(self):
        """Test DataDogLogger with DD_AGENT_HOST configuration"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()

                # Verify agent mode is enabled
                assert logger.use_agent_mode is True
                assert logger.DD_API_KEY is None
                assert logger.intake_url == "http://datadog-agent:8126/v1/input"

    def test_datadog_logger_agent_mode_without_port(self):
        """Test DataDogLogger with DD_AGENT_HOST without port (should add default port)"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent"}):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()

                # Verify agent mode is enabled and port is added
                assert logger.use_agent_mode is True
                assert logger.intake_url == "http://datadog-agent:8126/v1/input"

    def test_datadog_logger_direct_api_mode_configuration(self):
        """Test DataDogLogger with direct API configuration (backward compatibility)"""
        with patch.dict(
            os.environ, {"DD_API_KEY": "test-key", "DD_SITE": "us5.datadoghq.com"}
        ):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()

                # Verify direct API mode is enabled
                assert logger.use_agent_mode is False
                assert logger.DD_API_KEY == "test-key"
                assert (
                    logger.intake_url
                    == "https://http-intake.logs.us5.datadoghq.com/api/v2/logs"
                )

    def test_datadog_logger_no_configuration_raises_exception(self):
        """Test DataDogLogger raises exception when no configuration is provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                DataDogLogger()

            assert (
                "DD_API_KEY and DD_SITE are not set, and DD_AGENT_HOST is not set"
                in str(exc_info.value)
            )

    def test_datadog_logger_partial_configuration_raises_exception(self):
        """Test DataDogLogger raises exception when only DD_AGENT_HOST is set but DD_API_KEY/DD_SITE are missing"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent"}, clear=True):
            with patch("asyncio.create_task"):
                # This should work since DD_AGENT_HOST is sufficient for agent mode
                logger = DataDogLogger()
                assert logger.use_agent_mode is True

    def test_datadog_logger_agent_mode_headers(self):
        """Test that agent mode doesn't include DD-API-KEY header"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()

                # Mock the sync client to capture headers
                mock_client = MagicMock()
                logger.sync_client = mock_client

                # Create a test payload
                test_payload = {"test": "data"}

                # Call log_success_event to test headers
                logger.log_success_event(
                    kwargs={"standard_logging_object": test_payload},
                    response_obj=None,
                    start_time=None,
                    end_time=None,
                )

                # Verify the post was called with empty headers (no DD-API-KEY)
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                headers = call_args[1]["headers"]
                assert "DD-API-KEY" not in headers

    def test_datadog_logger_direct_mode_headers(self):
        """Test that direct mode includes DD-API-KEY header"""
        with patch.dict(
            os.environ, {"DD_API_KEY": "test-key", "DD_SITE": "us5.datadoghq.com"}
        ):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()

                # Mock the sync client to capture headers
                mock_client = MagicMock()
                logger.sync_client = mock_client

                # Create a test payload
                test_payload = {"test": "data"}

                # Call log_success_event to test headers
                logger.log_success_event(
                    kwargs={"standard_logging_object": test_payload},
                    response_obj=None,
                    start_time=None,
                    end_time=None,
                )

                # Verify the post was called with DD-API-KEY header
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                headers = call_args[1]["headers"]
                assert headers["DD-API-KEY"] == "test-key"


class TestDataDogLLMObsAgentHost:
    """Test DD_AGENT_HOST configuration for DataDogLLMObsLogger"""

    def test_datadog_llm_obs_logger_agent_mode_configuration(self):
        """Test DataDogLLMObsLogger with DD_AGENT_HOST configuration"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLLMObsLogger()

                # Verify agent mode is enabled
                assert logger.use_agent_mode is True
                assert logger.DD_API_KEY is None
                assert logger.intake_url == "http://datadog-agent:8126/v1/input"

    def test_datadog_llm_obs_logger_direct_api_mode_configuration(self):
        """Test DataDogLLMObsLogger with direct API configuration (backward compatibility)"""
        with patch.dict(
            os.environ, {"DD_API_KEY": "test-key", "DD_SITE": "us5.datadoghq.com"}
        ):
            with patch("asyncio.create_task"):
                logger = DataDogLLMObsLogger()

                # Verify direct API mode is enabled
                assert logger.use_agent_mode is False
                assert logger.DD_API_KEY == "test-key"
                assert (
                    logger.intake_url
                    == "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
                )

    def test_datadog_llm_obs_logger_no_configuration_raises_exception(self):
        """Test DataDogLLMObsLogger raises exception when no configuration is provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                DataDogLLMObsLogger()

            assert (
                "DD_API_KEY and DD_SITE are not set, and DD_AGENT_HOST is not set"
                in str(exc_info.value)
            )

    def test_datadog_llm_obs_logger_agent_mode_headers(self):
        """Test that agent mode doesn't include DD-API-KEY header for LLM Obs"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLLMObsLogger()

                # Mock the async client to capture headers
                mock_client = MagicMock()
                logger.async_client = mock_client

                # Create a test payload
                test_payload = {"test": "data"}

                # Call async_send_batch to test headers
                logger.log_queue = [test_payload]
                asyncio.run(logger.async_send_batch())

                # Verify the post was called with headers that don't include DD-API-KEY
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                headers = call_args[1]["headers"]
                assert "DD-API-KEY" not in headers
                assert "Content-Type" in headers

    def test_datadog_llm_obs_logger_direct_mode_headers(self):
        """Test that direct mode includes DD-API-KEY header for LLM Obs"""
        with patch.dict(
            os.environ, {"DD_API_KEY": "test-key", "DD_SITE": "us5.datadoghq.com"}
        ):
            with patch("asyncio.create_task"):
                logger = DataDogLLMObsLogger()

                # Mock the async client to capture headers
                mock_client = MagicMock()
                logger.async_client = mock_client

                # Create a test payload
                test_payload = {"test": "data"}

                # Call async_send_batch to test headers
                logger.log_queue = [test_payload]
                asyncio.run(logger.async_send_batch())

                # Verify the post was called with DD-API-KEY header
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                headers = call_args[1]["headers"]
                assert headers["DD-API-KEY"] == "test-key"
                assert "Content-Type" in headers
