"""
Calls Exa AI's /search endpoint to search the web.

Exa AI API Reference: https://docs.exa.ai/reference/search
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


class _ExaAISearchRequestRequired(TypedDict):
    """Required fields for Exa AI Search API request."""

    query: str  # Required - search query


class ExaAISearchRequest(_ExaAISearchRequestRequired, total=False):
    """
    Exa AI Search API request format.
    Based on: https://docs.exa.ai/reference/search
    """

    type: str  # Optional - search type ('keyword', 'neural', 'fast', 'auto'), default 'auto'
    category: str  # Optional - data category ('company', 'research paper', 'news', 'pdf', 'github', 'tweet', 'personal site', 'linkedin profile', 'financial report')
    userLocation: str  # Optional - two-letter ISO country code
    numResults: int  # Optional - number of results (max 100), default 10
    includeDomains: List[str]  # Optional - list of domains to include
    excludeDomains: List[str]  # Optional - list of domains to exclude
    startCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    endCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    startPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    endPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    includeText: List[str]  # Optional - strings that must be present in webpage text
    excludeText: List[
        str
    ]  # Optional - strings that must not be present in webpage text
    context: Union[bool, dict]  # Optional - format results for LLMs
    moderation: bool  # Optional - enable content moderation, default false
    contents: dict  # Optional - content retrieval options


class ExaAISearchConfig(BaseSearchConfig):
    EXA_AI_API_BASE = "https://api.exa.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Exa AI"

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
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("EXA_API_KEY",),
            base_env_var="EXA_API_BASE",
            default_api_base=self.EXA_AI_API_BASE,
        )
        if not api_key:
            raise ValueError(
                "EXA_API_KEY is not set. Set `EXA_API_KEY` environment variable."
            )
        headers["x-api-key"] = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        api_base = api_base or get_secret_str("EXA_API_BASE") or self.EXA_AI_API_BASE

        # Append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            api_base = f"{api_base}/search"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to Exa AI API format.

        Transforms Perplexity unified spec parameters:
        - query → query (same)
        - max_results → numResults
        - search_domain_filter → includeDomains
        - country → userLocation
        - max_tokens_per_page → (not applicable, ignored)

        All other Exa-specific parameters are passed through as-is.

        Args:
            query: Search query (string or list of strings). Exa AI only supports single string queries.
            optional_params: Optional parameters for the request

        Returns:
            Dict with typed request data following ExaAISearchRequest spec
        """
        if isinstance(query, list):
            # Exa AI only supports single string queries, join with spaces
            query = " ".join(query)

        request_data: ExaAISearchRequest = {
            "query": query,
        }

        # Transform Perplexity unified spec parameters to Exa format
        if "max_results" in optional_params:
            request_data["numResults"] = optional_params["max_results"]

        if "search_domain_filter" in optional_params:
            request_data["includeDomains"] = optional_params["search_domain_filter"]

        if "country" in optional_params:
            request_data["userLocation"] = optional_params["country"]

        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)

        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        # By default, request text content if not explicitly specified
        # Exa AI doesn't return content/text unless explicitly requested
        if "contents" not in result_data:
            result_data["contents"] = {"text": True}

        return result_data

    # SearchResult fields populated from dedicated Exa keys. Every *other* key
    # on an Exa result/response is preserved verbatim via ``extra="allow"`` so
    # rich Exa data (highlights, images, and the deep-search ``output`` summary
    # with its grounding citations) is not dropped by the normalization layer.
    _RESULT_RESERVED_FIELDS = {"title", "url", "snippet", "date", "last_updated"}
    _RESPONSE_RESERVED_FIELDS = {"results", "object"}

    @staticmethod
    def _extract_snippet(result: dict) -> str:
        """Resolve the snippet from an Exa result.

        Exa only returns ``text`` when ``contents.text`` is requested; when the
        caller asks for ``contents.highlights`` instead, the content lives under
        ``highlights`` (a list of strings). Fall back to it so the snippet isn't
        empty for highlight-based requests.
        """
        text = result.get("text")
        if text:
            return text
        highlights = result.get("highlights")
        if isinstance(highlights, list) and highlights:
            return "\n".join(str(highlight) for highlight in highlights)
        return ""

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Exa AI API response to LiteLLM unified SearchResponse format.

        Exa AI → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].text (or results[].highlights) → SearchResult.snippet
        - results[].publishedDate → SearchResult.date
        - No last_updated field in Exa AI response (set to None)

        All other Exa fields are preserved as-is: per-result extras (e.g.
        ``highlights``, ``image``, ``author``, ``score``) on each SearchResult,
        and top-level extras (e.g. ``output`` deep-search summary + grounding,
        ``costDollars``, ``requestId``) on the SearchResponse.

        Args:
            raw_response: Raw httpx response from Exa AI API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        raw_results = response_json.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        # Transform results to SearchResult objects, preserving any extra fields.
        results = []
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            result_extras = {
                key: value
                for key, value in result.items()
                if key not in self._RESULT_RESERVED_FIELDS
            }
            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=self._extract_snippet(result),
                date=result.get("publishedDate"),  # ISO 8601 datetime string
                last_updated=None,  # Exa AI doesn't provide last_updated in response
                **result_extras,
            )
            results.append(search_result)

        response_extras = {
            key: value
            for key, value in response_json.items()
            if key not in self._RESPONSE_RESERVED_FIELDS
        }
        return SearchResponse(
            results=results,
            object="search",
            **response_extras,
        )
