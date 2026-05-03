"""
Base Fetch transformation configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class WebFetchResponse(BaseModel):
    """Standard WebFetch response format."""

    url: str
    title: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class BaseFetchConfig(ABC):
    """
    Base configuration for Fetch transformations.
    Handles provider-agnostic Fetch operations.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def ui_friendly_name() -> str:
        """
        UI-friendly name for the fetch provider.
        Override in provider-specific implementations.
        """
        return "Unknown Fetch Provider"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        Override in provider-specific implementations.
        """
        return headers

    @abstractmethod
    async def afetch_url(
        self,
        url: str,
        headers: Dict[str, str],
        optional_params: Dict[str, Any],
    ) -> WebFetchResponse:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch
            headers: HTTP headers (including auth)
            optional_params: Optional parameters for the request

        Returns:
            WebFetchResponse with content
        """
        pass

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
