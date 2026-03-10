"""
E2E Test suite for Anthropic Messages API tool search across different providers.

Tests that tool search works correctly via litellm.anthropic.messages interface
by making actual API calls.

Supported providers:
- Anthropic API: advanced-tool-use-2025-11-20
- Azure Anthropic: advanced-tool-use-2025-11-20
- Vertex AI: tool-search-tool-2025-10-19
- Bedrock Invoke: tool-search-tool-2025-10-19

Reference: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from base_anthropic_messages_tool_search_test import (
    BaseAnthropicMessagesToolSearchTest,
)


class TestAnthropicAPIToolSearch(BaseAnthropicMessagesToolSearchTest):
    """
    E2E tests for tool search with Anthropic API directly.
    
    Uses the anthropic/ prefix which routes through the native
    Anthropic Messages API.
    
    Beta header: advanced-tool-use-2025-11-20
    
    Note: Tool search is only supported on Claude Opus 4.5 and Claude Sonnet 4.5.
    """

    def get_model(self) -> str:
        return "anthropic/claude-sonnet-4-5-20250929"


# class TestAzureAnthropicToolSearch(BaseAnthropicMessagesToolSearchTest):
#     """
#     E2E tests for tool search with Azure Anthropic (Microsoft Foundry).
    
#     Uses the azure/ prefix which routes through Azure's Anthropic endpoint.
    
#     Beta header: advanced-tool-use-2025-11-20
#     """

#     def get_model(self) -> str:
#         return "azure/claude-sonnet-4-20250514"


# class TestVertexAIToolSearch(BaseAnthropicMessagesToolSearchTest):
#     """
#     E2E tests for tool search with Vertex AI.
    
#     Uses the vertex_ai/ prefix which routes through Google Cloud's
#     Vertex AI Anthropic partner models.
    
#     Beta header: tool-search-tool-2025-10-19
#     """

#     def get_model(self) -> str:
#         return "vertex_ai/claude-sonnet-4@20250514"


# class TestBedrockInvokeToolSearch(BaseAnthropicMessagesToolSearchTest):
#     """
#     E2E tests for tool search with Bedrock Invoke API.
    
#     Uses the bedrock/invoke/ prefix which routes through the native
#     Anthropic Messages API format on Bedrock.
    
#     Beta header: advanced-tool-use-2025-11-20 (passed via extra_headers)
    
#     Note: Tool search on Bedrock is only supported on Claude Opus 4.5.
#     """

#     def get_model(self) -> str:
#         return "bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0"
