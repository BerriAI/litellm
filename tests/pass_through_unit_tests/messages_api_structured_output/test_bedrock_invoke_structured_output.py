"""
E2E Test suite for Bedrock Invoke API structured outputs via litellm.anthropic.messages.

Tests that structured outputs work correctly with Bedrock Invoke API (native Anthropic format)
by making actual API calls and validating JSON response format.

Requires AWS credentials and Bedrock model access.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from .base_anthropic_messages_structured_output_test import (
    BaseAnthropicMessagesStructuredOutputTest,
)


@pytest.mark.skip(reason="Skipping Bedrock Invoke structured output tests")
class TestBedrockInvokeStructuredOutput(BaseAnthropicMessagesStructuredOutputTest):
    """
    E2E tests for structured outputs with Bedrock Invoke API.

    Uses the bedrock/invoke/ prefix which routes through the native
    Anthropic Messages API format on Bedrock.
    """

    def get_model(self) -> str:
        return "bedrock/invoke/us.anthropic.claude-3-5-sonnet-20241022-v2:0"