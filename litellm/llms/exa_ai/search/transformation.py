"""
Calls Exa AI's /search endpoint to search the web.

Exa AI API Reference: https://docs.exa.ai/reference/search
"""

from collections.abc import Iterable, Mapping
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.hidden_params import set_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.exa_ai.search.cost_calculator import EXA_AI_COST_DOLLARS_PARAM
from litellm.llms.exa_ai.search.types import ExaAISearchRequest
from litellm.secret_managers.main import get_secret_str

__all__ = ["ExaAISearchConfig", "ExaAISearchRequest"]

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))
_STRS: Final = TypeAdapter(Iterable[str], config=ConfigDict(hide_input_in_errors=True))
_RENAMED_RESULT_FIELDS: Final = frozenset({"text", "publishedDate", "highlightScores"})


class ExaAISearchConfig(BaseSearchConfig):
    EXA_AI_API_BASE = "https://api.exa.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Exa AI"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:
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
            raise ValueError("EXA_API_KEY is not set. Set `EXA_API_KEY` environment variable.")
        headers["x-api-key"] = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: dict | list[dict] | None = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        return ensure_api_route(api_base or get_secret_str("EXA_API_BASE") or self.EXA_AI_API_BASE, "search")

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: Mapping[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
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
        mapped: Final = {
            target: optional_params[source]
            for source, target in (
                ("max_results", "numResults"),
                ("search_domain_filter", "includeDomains"),
                ("country", "userLocation"),
            )
            if source in optional_params
        }
        provider_params: Final = {
            key: value
            for key, value in optional_params.items()
            if key not in self.get_supported_perplexity_optional_params() and key not in mapped
        }
        return {
            "query": " ".join(query) if isinstance(query, list) else query,
            **mapped,
            **provider_params,
            **({"contents": {"text": True}} if "contents" not in provider_params else {}),
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,
    ) -> SearchResponse:
        """
        Transform Exa AI API response to LiteLLM unified SearchResponse format.

        Exa AI → LiteLLM mappings:
        - results[].title → SearchResult.title
        - results[].url → SearchResult.url
        - results[].text → SearchResult.snippet
        - results[].highlights → SearchResult.snippet (fallback when "text" is absent)
        - results[].summary → SearchResult.snippet (fallback when "text" and "highlights" are absent)
        - results[].publishedDate → SearchResult.date
        - No last_updated field in Exa AI response (set to None)
        - results[].highlightScores → SearchResult.highlight_scores, set only when Exa returned it
        - every other results[] field Exa returned (id, author, image, summary, highlights, score, ...)
          → the SearchResult as is
        - every top-level key except results (requestId, costDollars, ...) → SearchResponse.provider_metadata
        - costDollars → the response's hidden params, where search cost tracking reads Exa's reported total

        Args:
            raw_response: Raw httpx response from Exa AI API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())
        response: Final = SearchResponse.model_validate(
            {
                "results": [
                    _transform_result(value)
                    for value in _JSON_VALUES.validate_python(response_json.get("results") or [])
                ],
                "object": "search",
                "provider_metadata": {key: value for key, value in response_json.items() if key != "results"},
            }
        )
        params: Final = _JSON_OBJECT.validate_python(kwargs.get("optional_params") or {})
        set_hidden_params(
            response,
            {
                **{
                    key: value
                    for key, value in params.items()
                    if key in ("type", "max_results", "numResults", "contents")
                },
                EXA_AI_COST_DOLLARS_PARAM: response_json.get("costDollars"),
            },
        )
        return response


def _transform_result(raw_result: object) -> SearchResult:
    result: Final = _JSON_OBJECT.validate_python(raw_result)
    return SearchResult.model_validate(
        {
            **{key: value for key, value in result.items() if key not in _RENAMED_RESULT_FIELDS},
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("text")
            or "\n\n".join(_STRS.validate_python(result.get("highlights") or []))
            or result.get("summary")
            or "",
            "date": result.get("publishedDate"),
            "last_updated": None,
            **({"highlight_scores": result["highlightScores"]} if "highlightScores" in result else {}),
        }
    )
