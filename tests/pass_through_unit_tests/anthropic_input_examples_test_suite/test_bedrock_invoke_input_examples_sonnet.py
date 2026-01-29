"""
E2E Test for Bedrock Invoke with Claude Sonnet 4.5 and input_examples.

This test validates that LiteLLM correctly handles input_examples for Sonnet 4.5:
- Claude Code (or users) will send tools WITH input_examples
- LiteLLM should automatically REMOVE input_examples for non-Opus 4.5 models
- The request should succeed without errors

This ensures compatibility when Claude Code sends input_examples to all models,
but Bedrock only supports them on Opus 4.5.

Reference: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#providing-tool-use-examples
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest
from tests.pass_through_unit_tests.anthropic_input_examples_test_suite.base_anthropic_messages_input_examples_test import (
    BaseAnthropicMessagesInputExamplesTest,
)


class TestBedrockInvokeSonnetInputExamples(BaseAnthropicMessagesInputExamplesTest):
    """
    E2E tests for Bedrock Invoke with Claude Sonnet 4.5.
    
    This test sends tools WITH input_examples (as Claude Code would),
    and validates that LiteLLM automatically removes them for Sonnet 4.5
    since Bedrock only supports input_examples on Opus 4.5.
    """

    def get_model(self) -> str:
        """
        Use Claude Sonnet 4.5.
        """
        return "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def get_extra_headers(self) -> dict:
        """
        No extra headers needed.
        """
        return {}
