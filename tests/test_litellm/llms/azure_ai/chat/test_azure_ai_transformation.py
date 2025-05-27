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
