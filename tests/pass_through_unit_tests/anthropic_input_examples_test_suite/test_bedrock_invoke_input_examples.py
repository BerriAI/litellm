"""
E2E Test suite for Anthropic Messages API input_examples with Bedrock Invoke.

Tests that input_examples works correctly via litellm.anthropic.messages interface
by making actual API calls to AWS Bedrock.

Bedrock Invoke:
- Beta header: tool-examples-2025-10-29 (auto-injected by LiteLLM)
- Supported models: Claude Opus 4.5 only

Reference: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#providing-tool-use-examples
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest
from tests.pass_through_unit_tests.anthropic_input_examples_test_suite.base_anthropic_messages_input_examples_test import (
    BaseAnthropicMessagesInputExamplesTest,
)


class TestBedrockInvokeInputExamples(BaseAnthropicMessagesInputExamplesTest):
    """
    E2E tests for input_examples with Bedrock Invoke API.
    
    Uses the bedrock/invoke/ prefix which routes through the native
    Anthropic Messages API format on Bedrock.
    
    Beta header: tool-examples-2025-10-29 (auto-injected by LiteLLM)
    
    Note: Input examples on Bedrock is only supported on Claude Opus 4.5.
    """

    def get_model(self) -> str:
        """
        Use Claude Opus 4.5 which is the only model that supports input_examples on Bedrock.
        """
        return "bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0"

    def get_extra_headers(self) -> dict:
        """
        For Bedrock, we don't need to pass the beta header explicitly.
        LiteLLM will auto-inject the correct beta header (tool-examples-2025-10-29)
        when it detects input_examples in the tools.
        
        However, we can optionally pass it to test that user-provided headers work.
        """
        return {}
