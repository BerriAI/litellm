"""
E2E Test suite for Anthropic API structured outputs via litellm.anthropic.messages.

Tests that structured outputs work correctly with direct Anthropic API calls
by making actual API calls and validating JSON response format.

Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from .base_anthropic_messages_structured_output_test import (
    BaseAnthropicMessagesStructuredOutputTest,
)


class TestAnthropicAPIStructuredOutput(BaseAnthropicMessagesStructuredOutputTest):
    """
    E2E tests for structured outputs with direct Anthropic API.

    Uses Claude Sonnet 4.5 which supports structured outputs with the
    'anthropic-beta: structured-outputs-2025-11-13' header.
    """

    def get_model(self) -> str:
        return "claude-sonnet-4-5-20250929"