"""
LiteLLM Search API Types

This module defines types for the unified search API across different providers.
"""

from typing import Literal

# Supported search providers
SearchProvider = Literal[
    "perplexity",
    "tavily", 
    "parallel_ai",
    "exa_ai",
]

__all__ = ["SearchProvider"]

