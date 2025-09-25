"""
Test DD_AGENT_HOST configuration for Datadog integrations
"""

import os
import pytest
from unittest.mock import patch
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger


class TestDataDogAgentHost:
    """Test DD_AGENT_HOST configuration for DataDogLogger"""

    def test_datadog_logger_agent_mode(self):
        """Test DataDogLogger with DD_AGENT_HOST configuration"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLogger()
                
                # Verify agent mode is enabled
                assert logger.use_agent_mode is True
                assert logger.DD_API_KEY is None
                assert logger.intake_url == "http://datadog-agent:8126/v1/input"

    def test_datadog_logger_direct_api_mode(self):
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
            
            assert "DD_API_KEY is not set" in str(exc_info.value)


class TestDataDogLLMObsAgentHost:
    """Test DD_AGENT_HOST configuration for DataDogLLMObsLogger"""

    def test_datadog_llm_obs_logger_agent_mode(self):
        """Test DataDogLLMObsLogger with DD_AGENT_HOST configuration"""
        with patch.dict(os.environ, {"DD_AGENT_HOST": "datadog-agent:8126"}):
            with patch("asyncio.create_task"):
                logger = DataDogLLMObsLogger()
                
                # Verify agent mode is enabled
                assert logger.use_agent_mode is True
                assert logger.DD_API_KEY is None
                assert logger.intake_url == "http://datadog-agent:8126/v1/input"

    def test_datadog_llm_obs_logger_direct_api_mode(self):
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
            
            assert "DD_API_KEY is not set" in str(exc_info.value)
