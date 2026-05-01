"""
E2E Test suite for Anthropic Messages API prompt caching across different providers.

Tests that prompt caching works correctly via litellm.anthropic.messages interface
by making actual API calls and validating usage metrics.

Per AWS docs (https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html):
- Converse API uses: cachePoint: { type: "default" }
- InvokeModel API uses: cache_control: { type: "ephemeral" }
- Claude 3.7 Sonnet: GA, 1024 min tokens
- Claude 3.5 Haiku: GA, 2048 min tokens
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from base_anthropic_messages_prompt_caching_test import (
    BaseAnthropicMessagesPromptCachingTest,
)

# $BEDROCK_ANTHROPIC_{CONVERSE,INVOKE}_MODEL hold the full route-prefixed
# model paths, set in CircleCI project env vars. EOL bumps are a CI UI change.


class TestBedrockConversePromptCaching(BaseAnthropicMessagesPromptCachingTest):
    """
    E2E tests for prompt caching with Bedrock Converse API.

    Uses the bedrock/converse/ prefix which routes through litellm.completion()
    and the AmazonConverseConfig transformation.
    """

    def get_model(self) -> str:
        return os.environ["BEDROCK_ANTHROPIC_CONVERSE_MODEL"]


class TestBedrockInvokePromptCaching(BaseAnthropicMessagesPromptCachingTest):
    """
    E2E tests for prompt caching with Bedrock Invoke API.

    Uses the bedrock/invoke/ prefix which routes through the native
    Anthropic Messages API format.
    """

    def get_model(self) -> str:
        return os.environ["BEDROCK_ANTHROPIC_INVOKE_MODEL"]
