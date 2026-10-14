"""
Calls Exa AI's /search endpoint to search the web.

Exa AI API Reference: https://exa.ai/docs/reference/search
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.exa_ai.search.types import ExaAISearchRequest
from litellm.secret_managers.main import get_secret_str

__all__ = ["ExaAISearchConfig", "ExaAISearchRequest"]

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))
_STRS: Final = TypeAdapter(Iterable[str], config=ConfigDict(hide_input_in_errors=True))


class ExaAISearchConfig(BaseSearchConfig):
    EXA_AI_API_BASE = "https://api.exa.ai"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Exa AI"

    def validate_environment(
        self,
        headers: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """
        Validate environment and return headers.
        """
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("EXA_API_KEY",),
            base_env_var="EXA_API_BASE",
            default_api_base=self.EXA_AI_API_BASE,
        )
        if not resolved_key:
            raise ValueError("EXA_API_KEY is not set. Set `EXA_API_KEY` environment variable.")
        return {**headers, "x-api-key": resolved_key, "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: Mapping[str, object],
        data: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        **kwargs: object,
    ) -> str:
        """
        Get complete URL for Search endpoint.
        """
        resolved_api_base: Final = api_base or get_secret_str("EXA_API_BASE") or self.EXA_AI_API_BASE
        clean_api_base: Final = resolved_api_base.rstrip("/")
        if clean_api_base.endswith("/search"):
            return clean_api_base
        return f"{clean_api_base}/search"

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

    def should_stream_search(self, optional_params: Mapping[str, object]) -> bool:
        return optional_params.get("stream") is True and optional_params.get("outputSchema") is not None

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

        Args:
            raw_response: Raw httpx response from Exa AI API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())

        return SearchResponse.model_validate(
            {
                **response_json,
                "results": [
                    self._transform_result(value)
                    for value in _JSON_VALUES.validate_python(response_json.get("results") or [])
                ],
                "object": "search",
            }
        )

    @staticmethod
    def _transform_result(value: object) -> SearchResult:
        result: Final = _JSON_OBJECT.validate_python(value)
        return SearchResult.model_validate(
            {
                **result,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("text")
                or "\n\n".join(_STRS.validate_python(result.get("highlights") or []))
                or result.get("summary")
                or "",
                "date": result.get("publishedDate"),
                "last_updated": None,
            }
        )
