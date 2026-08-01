"""
Calls Aliyun IQS UnifiedSearch endpoint to search the web.

Aliyun IQS (Information Query Service) UnifiedSearch API Reference:
https://help.aliyun.com/zh/document_detail/2883041.html
"""

from types import MappingProxyType
from typing import TypedDict

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str


class _AliyunIQSSearchRequestRequired(TypedDict):
    """Required fields for Aliyun IQS UnifiedSearch API request."""

    query: str  # Required - search query (1-500 chars)


class AliyunIQSSearchRequest(_AliyunIQSSearchRequestRequired, total=False):
    """
    Aliyun IQS UnifiedSearch API request format.
    Based on: https://help.aliyun.com/zh/document_detail/2883041.html
    """

    engineType: str  # Optional - Generic (default) / GenericAdvanced / LiteAdvanced / Deep
    timeRange: str  # Optional - OneDay / OneWeek / OneMonth / OneYear / NoLimit (default)
    category: str  # Optional - industry category (finance, law, medical, ...)
    advancedParams: (
        dict  # mutable-ok: nested API payload object; Optional - numResults (1-50), start/endPublishedDate, ...
    )


class AliyunIQSSearchConfig(BaseSearchConfig):
    ALIYUN_IQS_API_BASE = "https://cloud-iqs.aliyuncs.com"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Aliyun IQS"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        """
        Validate environment and return headers.
        """
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("ALIYUN_IQS_API_KEY",),
            base_env_var="ALIYUN_IQS_API_BASE",
            default_api_base=self.ALIYUN_IQS_API_BASE,
        )
        if not api_key:
            raise ValueError("ALIYUN_IQS_API_KEY is not set. Set `ALIYUN_IQS_API_KEY` environment variable.")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        data: dict | list[dict] = None,  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        api_base = (api_base or get_secret_str("ALIYUN_IQS_API_BASE") or self.ALIYUN_IQS_API_BASE).rstrip("/")

        # Append "/search/unified" to the api base if it's not already there
        if not api_base.endswith("/search/unified"):
            api_base = f"{api_base}/search/unified"

        return api_base

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        optional_params: dict,  # mutable-ok: overrides BaseSearchConfig dict-shaped interface
        **kwargs,
    ) -> dict:  # mutable-ok: returns httpx json payload dict
        """
        Transform Search request to Aliyun IQS UnifiedSearch API format.

        Args:
            query: Search query (string or list of strings). IQS only supports single string queries.
            optional_params: Optional parameters for the request
                - max_results: Maximum number of search results (1-50) -> maps to advancedParams.numResults
                - engineType: Search engine type (default 'Generic')
                - timeRange: Time range filter ('OneDay', 'OneWeek', 'OneMonth', 'OneYear', 'NoLimit')
                - category: Industry category ('finance', 'law', 'medical', ...)

        Returns:
            Dict with typed request data following AliyunIQSSearchRequest spec
        """
        if isinstance(query, list):
            # IQS only supports single string queries
            query = " ".join(query)

        request_data: AliyunIQSSearchRequest = {  # mutable-ok: request payload built once, serialized by httpx json=
            "query": query,
            "engineType": optional_params.get("engineType", "Generic"),
        }

        # Transform Perplexity unified spec parameters to IQS format
        if "max_results" in optional_params:
            # merge with caller-supplied advancedParams instead of replacing,
            # so native filters (date ranges, ...) survive
            request_data["advancedParams"] = {  # mutable-ok: nested payload dict
                **optional_params.get("advancedParams", MappingProxyType({})),
                "numResults": optional_params["max_results"],
            }

        # Convert to dict before dynamic key assignments
        result_data = dict(request_data)  # mutable-ok: payload passed to httpx json= requires a real dict

        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value

        return result_data

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform Aliyun IQS UnifiedSearch API response to LiteLLM unified SearchResponse format.

        IQS → LiteLLM mappings:
        - pageItems[].title → SearchResult.title
        - pageItems[].link → SearchResult.url
        - pageItems[].snippet (fallback: summary) → SearchResult.snippet
        - pageItems[].publishedTime → SearchResult.date
        - No last_updated field in IQS response (set to None)

        Args:
            raw_response: Raw httpx response from IQS
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json = raw_response.json()

        # 200-with-error-body guard: IQS business errors (errorCode/errorMessage)
        # must not masquerade as "zero results" — non-2xx already raises upstream
        if "pageItems" not in response_json and ("errorCode" in response_json or "errorMessage" in response_json):
            raise ValueError(f"Aliyun IQS error response: {response_json}")

        # Transform pageItems to SearchResult objects
        results = []  # mutable-ok: collected into pydantic SearchResponse(results=...)
        for item in response_json.get("pageItems") or ():
            search_result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet") or item.get("summary") or "",
                date=item.get("publishedTime"),
                last_updated=None,  # IQS doesn't provide last_updated in response
            )
            results.append(search_result)

        return SearchResponse(
            results=results,
            object="search",
        )
