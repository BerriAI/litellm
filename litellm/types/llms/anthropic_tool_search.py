"""
Tool Search Beta Header Configuration

Reference: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
"""

from typing import Dict

from litellm.types.utils import LlmProviders

# Tool search beta header values
TOOL_SEARCH_BETA_HEADER_ANTHROPIC = "advanced-tool-use-2025-11-20"
TOOL_SEARCH_BETA_HEADER_VERTEX = "tool-search-tool-2025-10-19"
TOOL_SEARCH_BETA_HEADER_BEDROCK = "tool-search-tool-2025-10-19"


# Mapping of custom_llm_provider -> tool search beta header
TOOL_SEARCH_BETA_HEADER_BY_PROVIDER: Dict[str, str] = {
    LlmProviders.ANTHROPIC.value: TOOL_SEARCH_BETA_HEADER_ANTHROPIC,
    LlmProviders.AZURE.value: TOOL_SEARCH_BETA_HEADER_ANTHROPIC,
    LlmProviders.AZURE_AI.value: TOOL_SEARCH_BETA_HEADER_ANTHROPIC,
    LlmProviders.VERTEX_AI.value: TOOL_SEARCH_BETA_HEADER_VERTEX,
    LlmProviders.VERTEX_AI_BETA.value: TOOL_SEARCH_BETA_HEADER_VERTEX,
    LlmProviders.BEDROCK.value: TOOL_SEARCH_BETA_HEADER_BEDROCK,
}


def get_tool_search_beta_header(custom_llm_provider: str) -> str:
    """
    Get the tool search beta header for a given provider.
    """
    return TOOL_SEARCH_BETA_HEADER_BY_PROVIDER.get(
        custom_llm_provider,
        TOOL_SEARCH_BETA_HEADER_ANTHROPIC
    )

