"""
E2E Test suite for Azure Anthropic structured outputs via litellm.anthropic.messages.

Tests that structured outputs work correctly with Azure AI Foundry Anthropic models
by making actual API calls and validating JSON response format.

Requires Azure AI credentials and model deployment.
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath("../../../.."))

from .base_anthropic_messages_structured_output_test import (
    BaseAnthropicMessagesStructuredOutputTest,
)


class TestAzureAnthropicStructuredOutput(BaseAnthropicMessagesStructuredOutputTest):
    """
    E2E tests for structured outputs with Azure AI Foundry Anthropic models.

    Uses the azure_ai/ prefix which routes through Azure AI Foundry
    while maintaining the Anthropic Messages API format.
    """

    def get_model(self) -> str:
        return "azure_ai/claude-opus-4-5"

    def get_api_base(self) -> Optional[str]:
        return "https://krish-mh44t553-eastus2.services.ai.azure.com/"

    def get_api_key(self) -> Optional[str]:
        return os.environ.get("AZURE_ANTHROPIC_API_KEY")