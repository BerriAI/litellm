"""
E2E Test suite for Bedrock Converse API structured outputs via litellm.anthropic.messages.

Tests that structured outputs work correctly with Bedrock Converse API
by making actual API calls and validating JSON response format.

Requires AWS credentials and Bedrock model access.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from .base_anthropic_messages_structured_output_test import (
    BaseAnthropicMessagesStructuredOutputTest,
)


class TestBedrockConverseStructuredOutput(BaseAnthropicMessagesStructuredOutputTest):
    """
    E2E tests for structured outputs with Bedrock Converse API.

    Uses the bedrock/converse/ prefix which routes through litellm.completion()
    and the AmazonConverseConfig transformation.
    """

    def get_model(self) -> str:
        return "bedrock/converse/us.anthropic.claude-3-5-sonnet-20241022-v2:0"