"""
Calls Exa AI's /search endpoint to search the web.

Exa AI API Reference: https://docs.exa.ai/reference/search
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypedDict

import httpx
from pydantic import BaseModel, ConfigDict

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
    includeDomains: list[str]  # Optional - list of domains to include
    excludeDomains: list[str]  # Optional - list of domains to exclude
    startCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    endCrawlDate: str  # Optional - crawl date filter (ISO 8601 format)
    startPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    endPublishedDate: str  # Optional - published date filter (ISO 8601 format)
    includeText: list[str]  # Optional - strings that must be present in webpage text
    excludeText: list[str]  # Optional - strings that must not be present in webpage text
    context: bool | dict  # Optional - format results for LLMs
    moderation: bool  # Optional - enable content moderation, default false
    contents: dict  # Optional - content retrieval options


_HIGHLIGHT_SEPARATOR: Final[str] = "\n\n"

_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})


def _optional(key: str, value: object) -> Mapping[str, object]:
    """A one-entry mapping to spread into a SearchResult, or nothing when the value is absent."""
    return MappingProxyType({key: value}) if value is not None else _NOTHING


class _ExaHighlightFields(BaseModel):
    """
    Parses just the two content-mode fields whose runtime shape Exa doesn't guarantee,
    typed as `object` (not the documented `list[str]`/`list[float]` shape) so
    `_exa_highlights`/`_exa_highlight_scores` can isinstance-check them meaningfully
    against a real unknown, rather than a shape basedpyright would otherwise trust.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    highlights: object = None
    highlightScores: object = None


def _exa_highlights(result: Mapping[str, object]) -> tuple[object, ...] | None:
    """
    Exa documents `highlights` as a list of strings, but a malformed response (e.g. a bare
    string) must not be silently iterated character-by-character. Items are passed through
    as-is rather than filtered by type, since `highlightScores[i]` is Exa's relevance score
    for `highlights[i]`; independently filtering either array by item type would desync
    that positional pairing.
    """
    raw: Final = _ExaHighlightFields.model_validate(result).highlights
    return tuple(raw) if isinstance(raw, (list, tuple)) else None


def _exa_highlight_scores(result: Mapping[str, object]) -> tuple[object, ...] | None:
    raw: Final = _ExaHighlightFields.model_validate(result).highlightScores
    return tuple(raw) if isinstance(raw, (list, tuple)) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _exa_snippet(result: Mapping[str, object]) -> str:
    """
    Exa returns each requested content mode in its own field, and omits `text`
    entirely when only `highlights` or `summary` were asked for. Non-string highlight
    entries are dropped only for this joined-snippet computation, not from the raw
    `highlights` extra field, and blank entries are dropped too, since joining only
    blanks would otherwise produce a non-empty separator-only string that wrongly wins
    over `summary`.
    """
    highlights_snippet: Final = _HIGHLIGHT_SEPARATOR.join(
        h for h in (_exa_highlights(result) or ()) if isinstance(h, str) and h.strip()
    )
    return _as_str(result.get("text")) or highlights_snippet or _as_str(result.get("summary")) or ""


def _exa_results(response_json: object) -> tuple[Mapping[str, object], ...]:
    """Filters out a malformed `results` entry (e.g. `null`) instead of letting it
    crash `.get()` calls downstream."""
    raw: Final = response_json.get("results") if isinstance(response_json, dict) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(r for r in raw if isinstance(r, dict))


def _to_search_result(result: Mapping[str, object]) -> SearchResult:
    return SearchResult(
        title=_as_str(result.get("title")) or "",
        url=_as_str(result.get("url")) or "",
        snippet=_exa_snippet(result),
        date=_as_str(result.get("publishedDate")),  # ISO 8601 datetime string
        last_updated=None,  # Exa AI doesn't provide last_updated in response
        **_optional("highlights", _exa_highlights(result)),
        **_optional("highlight_scores", _exa_highlight_scores(result)),
        **_optional("summary", result.get("summary")),
        **_optional("score", result.get("score")),
    )


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
        api_base = api_base or get_secret_str("EXA_API_BASE") or self.EXA_AI_API_BASE

        # Append "/search" to the api base if it's not already there
        if not api_base.endswith("/search"):
            api_base = f"{api_base}/search"

        return api_base

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict,
        **kwargs,
    ) -> dict:
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

        request_data: Final[ExaAISearchRequest] = {
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
        result_data: Final = dict(request_data)

        # pass through all other parameters as-is
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value

        # By default, request text content if not explicitly specified
        # Exa AI doesn't return content/text unless explicitly requested
        if "contents" not in result_data:
            result_data["contents"] = {"text": True}

        return result_data

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
        - results[].text, else results[].highlights, else results[].summary → SearchResult.snippet
        - results[].highlights, results[].highlightScores, results[].summary, results[].score → passed through when present
        - results[].publishedDate → SearchResult.date
        - No last_updated field in Exa AI response (set to None)

        Args:
            raw_response: Raw httpx response from Exa AI API
            logging_obj: Logging object for tracking

        Returns:
            SearchResponse with standardized format
        """
        response_json: Final = raw_response.json()

        return SearchResponse(
            results=[  # mutable-ok: SearchResponse.results is declared list[SearchResult]
                _to_search_result(result) for result in _exa_results(response_json)
            ],
            object="search",
        )
