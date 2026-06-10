"""
TinyFish Search API.
Endpoint: GET https://api.search.tinyfish.ai
Docs: https://docs.tinyfish.ai/search-api
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict, Union
from urllib.parse import urlencode

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _TinyfishSearchRequestRequired(TypedDict):
    query: str


class TinyfishSearchRequest(_TinyfishSearchRequestRequired, total=False):
    location: str
    language: str
    page: int
    include_thumbnail: bool
    max_results: int


class TinyfishSearchConfig(BaseSearchConfig):
    TINYFISH_API_BASE = "https://api.search.tinyfish.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "TinyFish"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        api_key = api_key or get_secret_str("TINYFISH_API_KEY")
        if not api_key:
            raise ValueError(
                "TINYFISH_API_KEY is not set. Set `TINYFISH_API_KEY` environment variable."
            )
        headers["X-API-Key"] = api_key
        headers["Accept"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        api_base = (
            api_base or get_secret_str("TINYFISH_API_BASE") or self.TINYFISH_API_BASE
        )
        if data and isinstance(data, dict) and "_tinyfish_params" in data:
            query_string = urlencode(data["_tinyfish_params"], doseq=True)
            return f"{api_base}?{query_string}"
        return api_base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        if isinstance(query, list):
            query = " ".join(query)

        request_data: TinyfishSearchRequest = {"query": query}

        if "country" in optional_params:
            request_data["location"] = optional_params["country"]

        if "max_results" in optional_params:
            request_data["max_results"] = max(
                1, min(int(optional_params["max_results"]), 20)
            )

        if "search_domain_filter" in optional_params:
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                request_data["query"] = self._append_domain_filters(
                    request_data["query"], domains
                )

        result_data = dict(request_data)

        for param, value in optional_params.items():
            if (
                param not in self.get_supported_perplexity_optional_params()
                and param not in result_data
            ):
                result_data[param] = value

        return {"_tinyfish_params": result_data}

    @staticmethod
    def _append_domain_filters(query: str, domains: List[str]) -> str:
        domain_clauses = " OR ".join(f"site:{d}" for d in domains)
        return f"({query}) ({domain_clauses})"

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Optional[LiteLLMLoggingObj],
        **kwargs,
    ) -> SearchResponse:
        response_json = raw_response.json()

        query_params = raw_response.request.url.params if raw_response.request else {}
        max_results = max(1, min(int(query_params.get("max_results", 20)), 20))

        results: List[SearchResult] = []
        for item in response_json.get("results", []):
            if len(results) >= max_results:
                break
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                )
            )

        return SearchResponse(results=results, object="search")
