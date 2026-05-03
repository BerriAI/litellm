"""
Firecrawl Fetch API module.

Calls Firecrawl's /scrape endpoint to fetch and scrape web content.
"""

import json
from typing import Any, Dict, Optional

import httpx

from litellm.llms.base_llm.fetch.transformation import BaseFetchConfig, WebFetchResponse
from litellm.secret_managers.main import get_secret_str


class FirecrawlFetchConfig(BaseFetchConfig):
    """
    Firecrawl fetch configuration.
    Uses the /scrape endpoint to fetch and convert web content to markdown.
    """

    FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Firecrawl"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        """
        api_key = api_key or get_secret_str("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY is not set. Set `FIRECRAWL_API_KEY` environment variable."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(self, api_base: Optional[str] = None) -> str:
        """Get complete URL for Fetch endpoint."""
        api_base = (
            api_base or get_secret_str("FIRECRAWL_API_BASE") or self.FIRECRAWL_API_BASE
        )

        # Append "/scrape" to the api base if it's not already there
        if not api_base.endswith("/scrape"):
            api_base = f"{api_base}/scrape"

        return api_base

    async def afetch_url(
        self,
        url: str,
        headers: Dict[str, str],
        optional_params: Dict[str, Any],
    ) -> WebFetchResponse:
        """
        Fetch content from a URL using Firecrawl's /scrape endpoint.

        Args:
            url: URL to fetch
            headers: HTTP headers (including auth)
            optional_params: Optional parameters for the request
                - formats: List of formats to return (default: ["markdown"])
                - onlyMainContent: bool (default: True)
                - includeTags: List[str] - tags to include
                - excludeTags: List[str] - tags to exclude
                - headers: Dict[str, str] - custom headers
                - waitFor: int - time to wait in ms
                - timeout: int - timeout in ms

        Returns:
            WebFetchResponse with content
        """
        request_data = {
            "url": url,
            "formats": optional_params.get("formats", ["markdown"]),
            "onlyMainContent": optional_params.get("onlyMainContent", True),
        }

        # Add optional parameters
        for key in ["includeTags", "excludeTags", "headers", "waitFor", "timeout"]:
            if key in optional_params:
                request_data[key] = optional_params[key]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.get_complete_url(),
                headers=headers,
                json=request_data,
            )

            if response.status_code != 200:
                raise self.get_error_class(
                    error_message=f"Firecrawl fetch failed: {response.status_code} - {response.text}",
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            data = response.json()

            # Firecrawl response format: {"data": {"markdown": "...", "metadata": {...}}}
            response_data = data.get("data", {})
            content = response_data.get("markdown") or response_data.get("content", "")
            metadata = response_data.get("metadata", {})
            title = metadata.get("title") if isinstance(metadata, dict) else None

            return WebFetchResponse(
                url=url,
                title=title,
                content=content,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
