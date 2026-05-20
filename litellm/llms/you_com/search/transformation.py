"""
Calls You.com's /v1/search endpoint to search the web.

You.com API Reference: https://you.com/docs/api-reference/search/v1-search
OpenAPI spec:          https://you.com/specs/openapi_search_v1.yaml
"""

from typing import Dict, List, Optional, TypedDict, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _YouComSearchRequestRequired(TypedDict):
    """Required fields for You.com Search API request."""

    query: str


class YouComSearchRequest(_YouComSearchRequestRequired, total=False):
    """
    You.com Search API request format.
    Based on: https://you.com/specs/openapi_search_v1.yaml
    """

    count: int
    country: str
    language: str
    freshness: str
    include_domains: List[str]
    exclude_domains: List[str]
    safesearch: str


class YouComSearchConfig(BaseSearchConfig):
    YOU_COM_API_BASE = "https://ydc-index.io"

    @staticmethod
    def ui_friendly_name() -> str:
        return "You.com"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        api_key = api_key or get_secret_str("YOUCOM_API_KEY")
        if not api_key:
            raise ValueError(
                "YOUCOM_API_KEY is not set. Set `YOUCOM_API_KEY` environment variable."
            )
        headers["X-API-Key"] = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        api_base = (
            api_base or get_secret_str("YOUCOM_API_BASE") or self.YOU_COM_API_BASE
        )

        if not api_base.endswith("/v1/search"):
            api_base = f"{api_base.rstrip('/')}/v1/search"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to You.com API format.

        Perplexity unified spec → You.com mappings:
        - query                 → query
        - max_results           → count
        - search_domain_filter  → include_domains
        - country               → country
        - max_tokens_per_page   → (not applicable, ignored)
        """
        if isinstance(query, list):
            query = " ".join(query)

        request_data: YouComSearchRequest = {
            "query": query,
        }

        if "max_results" in optional_params:
            request_data["count"] = optional_params["max_results"]

        if "search_domain_filter" in optional_params:
            request_data["include_domains"] = optional_params["search_domain_filter"]

        if "country" in optional_params:
            request_data["country"] = optional_params["country"]

        result_data = dict(request_data)

        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform You.com API response to LiteLLM unified SearchResponse format.

        You.com → LiteLLM mappings (for both `results.web[]` and `results.news[]`):
        - title       → SearchResult.title
        - url         → SearchResult.url
        - snippets[0] → SearchResult.snippet (falls back to `description`)
        - page_age    → SearchResult.date
        """
        response_json = raw_response.json()
        raw_results = response_json.get("results") or {}

        web_results = raw_results.get("web") or []
        news_results = raw_results.get("news") or []

        results: List[SearchResult] = []
        for item in list(web_results) + list(news_results):
            snippets = item.get("snippets") or []
            snippet = snippets[0] if snippets else item.get("description", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    date=item.get("page_age"),
                    last_updated=None,
                )
            )

        return SearchResponse(
            results=results,
            object="search",
        )
