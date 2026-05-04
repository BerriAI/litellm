"""
Mapping from OpenAI-safe tool names back to client-supplied names.

OpenAI requires tools[].function.name to match ^[a-zA-Z0-9_-]+$. LiteLLM sanitizes
outbound requests and stores sanitized -> original in an in-memory cache (same pattern
as litellm.bedrock_tool_name_mappings / make_valid_bedrock_tool_name).
"""

from __future__ import annotations

from typing import Optional

from litellm.caching.in_memory_cache import InMemoryCache

# Mirrors bedrock_tool_name_mappings in llms/bedrock/chat/invoke_handler.py
openai_tool_name_mappings: InMemoryCache = InMemoryCache(
    max_size_in_memory=50, default_ttl=600
)


def get_openai_tool_name(response_tool_name: str) -> str:
    """
    If LiteLLM sanitized the outbound tool name, map the API response name back to the original.

    Same idea as get_bedrock_tool_name for Bedrock toolSpec names.
    """
    if response_tool_name in openai_tool_name_mappings.cache_dict:
        response_tool_name = openai_tool_name_mappings.cache_dict[response_tool_name]
    return response_tool_name


def restore_openai_tool_name_for_user(sanitized_name: Optional[str]) -> Optional[str]:
    """Nullable wrapper used when normalizing tool_calls from responses."""
    if sanitized_name is None:
        return None
    return get_openai_tool_name(sanitized_name)
