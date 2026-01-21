"""
E2E Test suite for Azure Anthropic structured outputs via litellm.anthropic.messages.

Tests that structured outputs work correctly with Azure AI Foundry Anthropic models
by making actual API calls and validating JSON response format.

Requires Azure AI credentials and model deployment.
"""

import os
import sys

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
        return "azure_ai/claude-3-5-sonnet-20241022"