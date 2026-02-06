import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.mark.asyncio
async def test_get_openai_compatible_provider_info():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    config = AzureAIStudioConfig()

    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


def test_azure_ai_validate_environment():
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="azure_ai/gpt-4o-mini",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()
    
    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") == False
    assert config._supports_stop_reason("grok-4") == False
    assert config._supports_stop_reason("grok-3-mini") == False
    assert config._supports_stop_reason("grok-code-fast") == False
    assert config._supports_stop_reason("gpt-4") == True
    
    # Test supported parameters for Grok models
    grok_params = config.get_supported_openai_params("grok-4-fast")
    assert "stop" not in grok_params, "Grok models should not support stop parameter"
    
    # Test supported parameters for non-Grok models
    gpt_params = config.get_supported_openai_params("gpt-4")
    assert "stop" in gpt_params, "GPT models should support stop parameter"
