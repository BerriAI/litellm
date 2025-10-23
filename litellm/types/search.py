"""
LiteLLM Search API Types

This module defines types for the unified search API across different providers.
"""

from litellm.types.utils import SearchProviders

# Re-export SearchProviders as SearchProvider for backwards compatibility
SearchProvider = SearchProviders

__all__ = ["SearchProvider", "SearchProviders"]

