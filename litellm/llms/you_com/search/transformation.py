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
    # Keyed tier (higher rate limits): authenticate with X-API-Key.
    YOU_COM_API_BASE = "https://ydc-index.io"
    # Keyless free tier: IP-throttled (100 queries/day) and requires no auth.
    # Used automatically when YOUCOM_API_KEY is not set.
    YOU_COM_FREE_API_BASE = "https://api.you.com/v1/agents/search"

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
        """
        Set headers for the You.com Search API.

        If YOUCOM_API_KEY (or an explicit api_key) is present, use the keyed
        endpoint with the `X-API-Key` header. Otherwise fall through to the
        keyless free tier; no auth header is required.
        """
        api_key = api_key or get_secret_str("YOUCOM_API_KEY")
        headers["Content-Type"] = "application/json"
        # Pin Accept-Encoding to identity: the keyless `api.you.com/v1/agents/search`
        # endpoint advertises gzip content-encoding but returns body bytes the
        # decoder rejects, which surfaces as httpx.DecodingError through litellm's
        # http handler. Identity is harmless on the keyed endpoint.
        headers.setdefault("Accept-Encoding", "identity")
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        """
        Pick the endpoint based on whether an API key is configured.

        - api_base explicit override     -> use it as-is (normalized)
        - YOUCOM_API_KEY set             -> keyed endpoint (ydc-index.io/v1/search)
        - no key                         -> keyless free tier (api.you.com/v1/agents/search)
        """
        if api_base is None:
            api_base = get_secret_str("YOUCOM_API_BASE")

        if api_base is None:
            api_key = kwargs.get("api_key") or get_secret_str("YOUCOM_API_KEY")
            if api_key:
                api_base = self.YOU_COM_API_BASE
            else:
                # Keyless free tier already includes the full path.
                return self.YOU_COM_FREE_API_BASE

        api_base = api_base.rstrip("/")

        if not api_base.endswith("/v1/search") and not api_base.endswith(
            "/v1/agents/search"
        ):
            api_base = f"{api_base}/v1/search"

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
            request_data["country"] = optional_params["country"].lower()

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
