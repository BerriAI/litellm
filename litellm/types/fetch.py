"""
LiteLLM Fetch API Types

This module defines types for the unified fetch API across different providers.
"""

from typing import List, Optional

from typing_extensions import Required, TypedDict

from litellm.types.utils import FetchProviders

# Re-export FetchProviders as FetchProvider for backwards compatibility
FetchProvider = FetchProviders

__all__ = ["FetchProvider", "FetchProviders"]


class FetchToolLiteLLMParams(TypedDict, total=False):
    """
    LiteLLM params for fetch tools configuration.
    """

    fetch_provider: Required[str]
    api_key: Optional[str]
    api_base: Optional[str]
    timeout: Optional[float]
    max_retries: Optional[int]


class FetchTool(TypedDict, total=False):
    """
    Fetch tool configuration.

    Example:
        {
            "fetch_tool_id": "123e4567-e89b-12d3-a456-426614174000",
            "fetch_tool_name": "litellm-fetch",
            "litellm_params": {
                "fetch_provider": "firecrawl",
                "api_key": "fc-..."
            },
            "fetch_tool_info": {
                "description": "Firecrawl fetch tool"
            }
        }
    """

    fetch_tool_id: Optional[str]
    fetch_tool_name: Required[str]
    litellm_params: Required[FetchToolLiteLLMParams]
    fetch_tool_info: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]


class FetchToolInfoResponse(TypedDict, total=False):
    """Response model for fetch tool information."""

    fetch_tool_id: Optional[str]
    fetch_tool_name: str
    litellm_params: dict
    fetch_tool_info: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]
    is_from_config: Optional[
        bool
    ]  # True if this tool is defined in config file, False if from DB


class ListFetchToolsResponse(TypedDict):
    """Response model for listing fetch tools."""

    fetch_tools: List[FetchToolInfoResponse]


class AvailableFetchProvider(TypedDict):
    """Information about an available fetch provider."""

    provider_name: str
    ui_friendly_name: str
