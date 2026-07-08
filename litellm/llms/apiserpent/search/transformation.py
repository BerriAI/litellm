"""
Calls APISerpent's search endpoints to search Google, Bing, Yahoo, or DuckDuckGo.

Two endpoints under one provider, selected via the ``deep`` boolean param:
- ``deep=False`` (default) -> quick search (/api/search/quick)
- ``deep=True``            -> deep search  (/api/search)

APISerpent API Reference: https://apiserpent.com/docs
"""

from typing import Dict, List, Literal, Optional, Union, cast
from urllib.parse import urlencode

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.apiserpent.search.defaults import (
    DEEP_SEARCH_PATH,
    NUM_MAX,
    NUM_MIN,
    NUM_MIN_DEEP,
    QUICK_SEARCH_PATH,
    APISerpentSearchParams,
)
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

DEEP_SEARCH_PARAM = "deep"
APISERPENT_BASE = "https://apiserpent.com"
APISERPENT_PARAMS_KEY = "_apiserpent_params"


class APISerpentSearchConfig(BaseSearchConfig):
    @staticmethod
    def ui_friendly_name() -> str:
        return "APISerpent"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    @staticmethod
    def _is_deep_search(optional_params: dict) -> bool:
        return bool(optional_params.get(DEEP_SEARCH_PARAM))

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("APISERPENT_API_KEY",),
            base_env_var="APISERPENT_API_BASE",
            default_api_base=APISERPENT_BASE,
        )
        if not api_key:
            raise ValueError("APISERPENT_API_KEY is not set. Set `APISERPENT_API_KEY` environment variable.")
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
        """
        Build the search URL. APISerpent uses GET, so the transformed request is
        serialized into the query string. The endpoint path (quick vs deep) is
        always applied; an ``api_base`` / ``APISERPENT_API_BASE`` override only
        changes the host. The ``endswith`` guard keeps this idempotent, since the
        handler re-invokes this method with the already-resolved URL as api_base.
        """
        base = (api_base or get_secret_str("APISERPENT_API_BASE") or APISERPENT_BASE).rstrip("/")
        path = DEEP_SEARCH_PATH if self._is_deep_search(optional_params) else QUICK_SEARCH_PATH
        if not base.endswith(path):
            base = f"{base}{path}"

        if data and isinstance(data, dict) and APISERPENT_PARAMS_KEY in data:
            query_string = urlencode(data[APISERPENT_PARAMS_KEY], doseq=True)
            return f"{base}?{query_string}"

        return base

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Dict:
        """
        Transform a unified search request into APISerpent query params.

        Unified spec mappings:
        - query               -> q
        - max_results         -> num (clamped to the endpoint's valid range)
        - country             -> country (lowercased)
        - search_domain_filter -> site: clauses appended to q

        All other APISerpent params (engine, language, freshness, safe, pages,
        format, pixel_position) pass through, defaulting via APISerpentSearchParams.
        """
        if isinstance(query, list):
            query = " ".join(query)

        is_deep = self._is_deep_search(optional_params)

        overrides: Dict = {}
        if "max_results" in optional_params:
            num_min = NUM_MIN_DEEP if is_deep else NUM_MIN
            overrides["num"] = max(num_min, min(optional_params["max_results"], NUM_MAX))
        if "country" in optional_params:
            overrides["country"] = cast(str, optional_params["country"]).lower()

        for param, value in optional_params.items():
            if param in APISerpentSearchParams.field_names() and param not in overrides:
                overrides[param] = value

        params = {**APISerpentSearchParams(**overrides).to_request_params(), "q": query}

        if "search_domain_filter" in optional_params:
            domains = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                params["q"] = self._append_domain_filters(str(params["q"]), domains)

        return {APISERPENT_PARAMS_KEY: params}

    @staticmethod
    def _append_domain_filters(query: str, domains: List[str]) -> str:
        domain_clauses = " OR ".join(f"site:{domain}" for domain in domains)
        return f"({query}) ({domain_clauses})"

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Optional[LiteLLMLoggingObj],
        **kwargs,
    ) -> SearchResponse:
        """
        Transform APISerpent response to the unified SearchResponse format.

        Full format nests results under ``results.organic[]``; simple format
        returns a flat ``results[]`` array. Both expose title/url/snippet.
        """
        response_json = raw_response.json()

        raw_results = response_json.get("results") or {}
        organic = raw_results.get("organic", []) if isinstance(raw_results, dict) else raw_results

        results: List[SearchResult] = []
        for result in organic:
            results.append(
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("snippet", ""),
                    date=result.get("date"),
                    last_updated=None,
                )
            )

        return SearchResponse(
            results=results,
            object="search",
        )
