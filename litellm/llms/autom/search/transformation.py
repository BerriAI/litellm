"""
Calls Autom's Google Search endpoint.

Autom API Reference: https://docs.autom.dev
"""

from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypedDict
from urllib.parse import urlencode

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))

# Key under which the transformed request carries the query string for the GET URL.
AUTOM_PARAMS_KEY: Final = "_autom_params"
AUTOM_SEARCH_ROUTE: Final = "v1/google/search"


class _AutomSearchRequestRequired(TypedDict):
    """Required fields for an Autom Google Search request."""

    query: str  # Required - search query


class AutomSearchRequest(_AutomSearchRequestRequired, total=False):
    """
    Autom Google Search request format (query string of a GET request).
    Based on: https://docs.autom.dev
    """

    gl: str  # Optional - country code (e.g., "us", "de")
    hl: str  # Optional - interface language (e.g., "en", "fr")
    location: str  # Optional - human-readable location (e.g., "Paris, France")
    google_domain: str  # Optional - Google domain (default "google.com")
    tbs: str  # Optional - advanced filters, e.g. time ranges ("qdr:d", "qdr:w")
    safe: str  # Optional - SafeSearch ("active" or "off")
    page: int  # Optional - results page (1 for the first)


class AutomSearchConfig(BaseSearchConfig):
    AUTOM_API_BASE = "https://api.autom.dev"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Autom"

    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        Autom search endpoints are GET requests with the parameters in the query string.
        """
        return "GET"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Validate environment and return headers. The key travels in the x-api-key header, never in the URL.
        """
        api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("AUTOM_API_KEY",),
            base_env_var="AUTOM_API_BASE",
            default_api_base=self.AUTOM_API_BASE,
        )
        if not api_key:
            raise ValueError("AUTOM_API_KEY is not set. Set `AUTOM_API_KEY` environment variable.")
        headers["x-api-key"] = api_key
        headers["Accept"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: dict | list[dict] | None = None,
        **kwargs,
    ) -> str:
        """
        Get the complete URL for the Google Search endpoint, with the query string.
        """
        api_base = api_base or get_secret_str("AUTOM_API_BASE") or self.AUTOM_API_BASE
        url: Final = ensure_api_route(api_base, AUTOM_SEARCH_ROUTE)
        if data and isinstance(data, dict) and AUTOM_PARAMS_KEY in data:
            return f"{url}?{urlencode(data[AUTOM_PARAMS_KEY], doseq=True)}"
        return url

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        """
        Transform a Search request to Autom's Google Search parameters.

        Args:
            query: Search query (string or list of strings). Autom takes a single string query.
            optional_params: Optional parameters for the request
                - max_results: not sent; Google returns one page of about 10 results
                - search_domain_filter: List of domains -> `site:` clauses in `query`;
                  a domain with a leading `-` becomes a `-site:` exclusion
                - country: Country code filter (e.g., 'US', 'GB') -> maps to `gl` (lowercased)

        Returns:
            Dict holding the query-string parameters under `_autom_params`
        """
        if isinstance(query, list):
            query = " ".join(query)

        request_data: Final[AutomSearchRequest] = {
            "query": query,
        }

        if "search_domain_filter" in optional_params:
            domains: Final = optional_params["search_domain_filter"]
            if isinstance(domains, list) and len(domains) > 0:
                request_data["query"] = self._append_domain_filters(request_data["query"], domains)

        if "country" in optional_params:
            request_data["gl"] = str(optional_params["country"]).lower()

        result_data: Final = dict(request_data)

        # Pass through Autom-specific parameters (hl, location, tbs, page, ...) as-is.
        for param, value in optional_params.items():
            if param not in self.get_supported_perplexity_optional_params() and param not in result_data:
                result_data[param] = value

        # Store params in a special key for URL building (GET request)
        return {
            AUTOM_PARAMS_KEY: result_data,
        }

    @staticmethod
    def _append_domain_filters(query: str, domains: list[str]) -> str:
        """
        Restrict the query to the included domains (OR) and remove the excluded ones (`-domain`, AND).
        """
        cleaned: Final = [str(d).strip() for d in domains if str(d).strip()]
        include: Final = [d for d in cleaned if not d.startswith("-")]
        exclude: Final = [d[1:].strip() for d in cleaned if d.startswith("-") and d[1:].strip()]
        parts: Final = [f"({query})" if include or exclude else query]
        if include:
            parts.append("(" + " OR ".join(f"site:{d}" for d in include) + ")")
        parts.extend(f"-site:{d}" for d in exclude)
        return " ".join(parts)

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform an Autom response to the LiteLLM unified SearchResponse format.

        Autom -> LiteLLM mappings:
        - organic_results[].title -> SearchResult.title
        - organic_results[].link -> SearchResult.url
        - organic_results[].snippet -> SearchResult.snippet
        - organic_results[].date -> SearchResult.date (when present)

        Rows without an http(s) link are skipped.
        """
        response_json: Final = _JSON_OBJECT.validate_python(raw_response.json())

        results: Final[list[SearchResult]] = []
        for raw_result in _JSON_VALUES.validate_python(response_json.get("organic_results", [])):
            result = _JSON_OBJECT.validate_python(raw_result)
            url = result.get("link")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            title = result.get("title")
            snippet = result.get("snippet")
            date = result.get("date")
            results.append(
                SearchResult.model_validate(
                    {
                        "title": title if isinstance(title, str) else "",
                        "url": url,
                        "snippet": snippet if isinstance(snippet, str) else "",
                        "date": date if isinstance(date, str) else None,
                        "last_updated": None,
                    }
                )
            )

        return SearchResponse(
            results=results,
            object="search",
        )
