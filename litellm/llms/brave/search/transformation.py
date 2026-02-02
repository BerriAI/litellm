"""
Brave Search /web/search endpoint.
Documentation: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
"""

from __future__ import annotations
from datetime import datetime, timezone
from dateutil import parser
from typing import Dict, List, Literal, Optional, TypedDict, Union
import httpx
import re

_ISO_YMD = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$")
_UNIX_TIMESTAMP = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
BRAVE_SECTIONS = ["web", "discussions", "faqs", "faq", "news", "videos"]

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)

from litellm.secret_managers.main import get_secret_str


def to_yyyy_mm_dd(
    s: Union[str, int, float, None],
    *,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> Optional[str]:
    """
    Convert a string/int/float to YYYY-MM-DD; return None if parsing fails.
    """
    if not s:
        return None

    s = str(s).strip()

    # Handle Unix timestamps (seconds or milliseconds).
    if _UNIX_TIMESTAMP.match(s):
        try:
            ts_float = float(s)
            # Treat large values as milliseconds.
            if ts_float > 1e11 or ts_float < -1e11:
                ts_float /= 1000.0
            return datetime.fromtimestamp(ts_float, tz=timezone.utc).date().isoformat()
        except Exception:
            return None

    # If it looks like YYYY-M-D (ISO-ish), force yearfirst to avoid surprises.
    try:
        if _ISO_YMD.match(s):
            dt = parser.parse(s, yearfirst=True, dayfirst=False, fuzzy=True)
        else:
            dt = parser.parse(s, yearfirst=yearfirst, dayfirst=dayfirst, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None


class _BraveSearchRequestRequired(TypedDict):
    """Required fields for Brave Search API request."""

    q: str  # Required - search query


class BraveSearchRequest(_BraveSearchRequestRequired, total=False):
    """
    Brave Search API request format.
    Based on: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
    """

    count: int  # Optional - number of web results to return (Brave max is 20)
    offset: int  # Optional - pagination offset
    country: str  # Optional - two-letter ISO country code
    search_lang: str  # Optional - language to bias results
    ui_lang: str  # Optional - language for UI strings
    freshness: str  # Optional - Brave freshness window (e.g., "pd", "pw", "pm")
    safesearch: str  # Optional - "off" | "moderate" | "strict"
    spellcheck: str  # Optional - "strict" | "moderate" | "off"
    text_decorations: bool  # Optional - enable/disable text decorations
    result_filter: str  # Optional - e.g., "web"
    units: str  # Optional - measurement units
    goggles_id: str  # Optional - Brave Goggles id
    goggles: str  # Optional - Brave Goggles DSL
    extra_snippets: bool  # Optional - request extra snippets
    summary: bool  # Optional - include summary block
    enable_rich_callback: bool  # Optional - structured result blocks
    include_fetch_metadata: bool  # Optional - include fetch metadata
    operators: bool  # Optional - enable advanced operators


class BraveSearchConfig(BaseSearchConfig):
    BRAVE_API_BASE = "https://api.search.brave.com/res/v1/web/search"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Brave Search"

    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        Brave Search API uses GET requests for search.
        """
        return "GET"

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
        api_key = api_key or get_secret_str("BRAVE_API_KEY")

        if not api_key:
            raise ValueError(
                "BRAVE_API_KEY is not set. Set `BRAVE_API_KEY` environment variable."
            )

        headers["X-Subscription-Token"] = api_key
        headers["Accept"] = "application/json"
        headers["Accept-Encoding"] = "gzip"
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
        Get complete URL for Search endpoint with query parameters.

        The Brave Search API uses GET requests and therefore needs the request
        body (data) to construct query parameters in the URL.
        """
        from urllib.parse import urlencode

        api_base = api_base or get_secret_str("BRAVE_API_BASE") or self.BRAVE_API_BASE

        # Build query parameters from the transformed request body
        if data and isinstance(data, dict) and "_brave_params" in data:
            params = data["_brave_params"]
            query_string = urlencode(params, doseq=True)
            return f"{api_base}?{query_string}"

        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Transform Search request to Brave Search API format.

        Transforms Perplexity unified spec parameters:
        - query → q (same)
        - max_results → count
        - search_domain_filter → q (append domain filters)
        - country → country
        - max_tokens_per_page → (not applicable, ignored)

        All other Brave Search API-specific parameters are passed through as-is.

        Args:
            query: Search query (string or list of strings). Brave Search API supports single string queries.
            optional_params: Optional parameters for the request

        Returns:
            Dict with typed request data following Brave Search API spec
        """
        if isinstance(query, list):
            # Brave Search API only supports single string queries
            query = " ".join(query)

        request_data: BraveSearchRequest = {
            "q": query,
        }

        # Only include "include_fetch_metadata" if it is not explicitly set to False
        # This parameter results (more often than not) in a timestamp which we can use for last_updated
        if (
            "include_fetch_metadata" in optional_params
            and optional_params["include_fetch_metadata"] is False
        ):
            request_data["include_fetch_metadata"] = False
        else:
            request_data["include_fetch_metadata"] = True

        # Transform unified spec parameters to Brave Search API format
        if "max_results" in optional_params:
            # Brave Search API supports 1-20 results per /web/search request
            num_results = min(optional_params["max_results"], 20)
            request_data["count"] = num_results

        if "search_domain_filter" in optional_params:
            # Convert to multiple "site:domain" clauses, joined by OR
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                request_data["q"] = self._append_domain_filters(
                    request_data["q"], domains
                )

        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)

        # Pass through all other parameters as-is
        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        # Store params in special key for URL building (Brave Search API uses GET not POST)
        # Return a wrapper dict that stores params for get_complete_url to use
        return {
            "_brave_params": result_data,
        }

    @staticmethod
    def _append_domain_filters(query: str, domains: List[str]) -> str:
        """
        Add site: filters to emulate domain restriction in Brave.
        """
        domain_clauses = [f"site:{domain}" for domain in domains]
        domain_query = " OR ".join(domain_clauses)

        return f"({query}) AND ({domain_query})"

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Optional[LiteLLMLoggingObj],
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Brave Search API response to LiteLLM unified SearchResponse format.
        """
        response_json = raw_response.json()

        # Transform results to SearchResult objects
        results: List[SearchResult] = []

        query_params = raw_response.request.url.params if raw_response.request else {}
        sections_to_process = self._sections_from_params(dict(query_params))
        max_results = max(1, min(int(query_params.get("count", 20)), 20))

        for section in sections_to_process:
            for result in response_json.get(section, {}).get("results", []):
                # Because the `max_results`/`count` parameters do not affect
                # the number of "discussion", "faq", "news", or "videos"
                # results, we need to manually limit the number of results
                # returned when an explicit limit has been provided.
                if len(results) >= max_results:
                    break

                title = result.get("title", "")
                url = result.get("url", "")
                snippet = result.get("description", "")
                date = to_yyyy_mm_dd(result.get("page_age") or result.get("age"))
                last_updated = to_yyyy_mm_dd(
                    result.get("fetched_content_timestamp", "")
                )

                search_result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    date=date,
                    last_updated=last_updated,
                )

                results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )

    @staticmethod
    def _sections_from_params(query_params: dict) -> List[str]:
        """
        Returns a list of sections the user has requested via the Brave Search
        API's `result_filter` parameter. If no `result_filter` parameter is
        provided, returns all sections.
        """
        raw_filter = query_params.get("result_filter")
        requested_filters: List[str] = []

        if raw_filter and isinstance(raw_filter, str):
            requested_filters = [part.strip() for part in raw_filter.split(",")]

        sections = [s.lower() for s in requested_filters if s.lower() in BRAVE_SECTIONS]
        return sections or BRAVE_SECTIONS
