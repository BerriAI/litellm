import os
from unittest.mock import patch
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger


def test_datadog_llm_obs_agent_configuration():
    """
    Test that DataDog LLM Obs logger correctly configures agent endpoint.
    """
    test_env = {
        "LITELLM_DD_AGENT_HOST": "localhost",
        "LITELLM_DD_LLM_OBS_PORT": "10518",
        "DD_API_KEY": "test-api-key",  # Optional, but checking if it's preserved
    }

    # Ensure DD_SITE is NOT set to verify we don't need it in agent mode

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):  # Prevent periodic flush task from running
            dd_logger = DataDogLLMObsLogger()

        expected_url = "http://localhost:10518/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url
        assert dd_logger.DD_API_KEY == "test-api-key"


def test_datadog_llm_obs_agent_no_api_key_ok():
    """
    Test that agent mode works WITHOUT DD_API_KEY (agent handles auth).
    """
    test_env = {
        "LITELLM_DD_AGENT_HOST": "localhost",
        # No DD_API_KEY
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            # Should NOT raise exception anymore
            dd_logger = DataDogLLMObsLogger()

            assert dd_logger.DD_API_KEY is None
            # Default port is 8126 if not set
            expected_url = "http://localhost:8126/api/intake/llm-obs/v1/trace/spans"
            assert dd_logger.intake_url == expected_url


def test_datadog_llm_obs_direct_api_configuration():
    """
    Test that direct API configuration still works as expected.
    """
    test_env = {
        "DD_API_KEY": "direct-api-key",
        "DD_SITE": "us5.datadoghq.com",
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            dd_logger = DataDogLLMObsLogger()

        expected_url = "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url
        assert dd_logger.DD_API_KEY == "direct-api-key"
