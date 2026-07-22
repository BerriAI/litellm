"""
TinyFish Search API.
Endpoint: GET https://api.search.tinyfish.ai
Docs: https://docs.tinyfish.ai/search-api
"""

from __future__ import annotations

import json
from typing import Literal
from urllib.parse import urlencode

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
)
from litellm.secret_managers.main import get_secret_str

_UrlEncodableParams = TypeAdapter(dict[str, str | int | float | bool])
_StrList = TypeAdapter(list[str])
_StrFrozenSet = TypeAdapter(frozenset[str])

_TINYFISH_PARAMS_KEY = "_tinyfish_params"
_TINYFISH_DOCS_URL = "https://docs.tinyfish.ai/search-api"
_TINYFISH_RESULT_CAP = 10  # Client-side truncation cap for max_results


class TinyfishSearchConfig(BaseSearchConfig):
    TINYFISH_API_BASE = "https://api.search.tinyfish.ai"

    def __init__(self) -> None:
        super().__init__()
        # Threaded from transform_search_request → transform_search_response so the
        # response slice honors the caller's max_results without re-sending it on
        # the wire (TinyFish doesn't honor it server-side). Safe because the
        # config is instantiated per-call via ProviderConfigManager.
        self._caller_max_results: int | None = None

    @staticmethod
    def ui_friendly_name() -> str:
        return "TinyFish"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    def validate_environment(
        self,
        headers: dict[str, str],
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,
    ) -> dict[str, str]:
        resolved_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("TINYFISH_API_KEY",),
            base_env_var="TINYFISH_API_BASE",
            default_api_base=self.TINYFISH_API_BASE,
        )
        if not resolved_key:
            raise ValueError("TINYFISH_API_KEY is not set. Set `TINYFISH_API_KEY` environment variable.")
        return {**headers, "X-API-Key": resolved_key, "Accept": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],
        data: dict[str, object] | list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> str:
        resolved_base = api_base or get_secret_str("TINYFISH_API_BASE") or self.TINYFISH_API_BASE
        if isinstance(data, dict) and _TINYFISH_PARAMS_KEY in data:
            validated_params = _UrlEncodableParams.validate_python(data[_TINYFISH_PARAMS_KEY])
            return f"{resolved_base}?{urlencode(validated_params, doseq=True)}"
        return resolved_base

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        """
        Transform a LiteLLM search request to TinyFish's querystring format.

        Maps LiteLLM's unified-spec params (see
        ``BaseSearchConfig.get_supported_perplexity_optional_params``) to
        TinyFish equivalents:
        - ``query`` (str or list[str]) → ``query`` (list joined by spaces)
        - ``country`` → ``location``
        - ``search_domain_filter`` (list[str]) → folded into the query using
          search operators
        - ``max_results`` → not sent on the wire; stashed on
          ``self._caller_max_results`` for client-side response truncation
          (TinyFish doesn't honor it server-side)
        - ``max_tokens_per_page`` → silently dropped (no TinyFish equivalent)

        Any other ``optional_params`` keys are forwarded to TinyFish as-is.
        dict and list values are JSON-encoded so structured payloads survive
        ``urlencode``.

        Returns:
            ``{_TINYFISH_PARAMS_KEY: <dict of querystring entries>}``.
            ``get_complete_url`` reads this back to build the final URL.
        """
        resolved_query = " ".join(query) if isinstance(query, list) else query

        try:
            domains = _StrList.validate_python(optional_params.get("search_domain_filter"))
        except (ValidationError, TypeError):
            domains = []
        if domains:
            resolved_query = _append_domain_filters(resolved_query, domains)

        request_data: dict[str, object] = {"query": resolved_query}

        country = optional_params.get("country")
        if isinstance(country, str):
            request_data["location"] = country

        # max_results is enforced client-side on the response (TinyFish ignores
        # the param and always returns ~10). Clamp to [1, 10] and stash on self
        # so transform_search_response can slice without re-reading the URL.
        raw_max = optional_params.get("max_results")
        if isinstance(raw_max, (int, float, str)):
            try:
                self._caller_max_results = max(1, min(int(raw_max), _TINYFISH_RESULT_CAP))
            except (ValueError, TypeError, OverflowError):
                # OverflowError covers int(float('inf')) and similar non-finite floats.
                verbose_logger.warning(
                    "TinyFish Search: max_results=%r is not a valid integer; ignoring.",
                    raw_max,
                )

        raw_supported: object = (
            self.get_supported_perplexity_optional_params()  # any-ok: base class returns bare set
        )
        supported_perplexity = _StrFrozenSet.validate_python(raw_supported)
        for param, value in optional_params.items():
            if param not in supported_perplexity and param not in request_data:
                # Serialize dicts/lists as JSON so structured params survive urlencode.
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(",", ":"))
                # `urlencode` would render Python bool as "True"/"False"
                # (capitalized). TinyFish Search's bool params require lowercase
                # "true"/"false" strings on the wire; normalize here.
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                request_data[param] = value

        return {_TINYFISH_PARAMS_KEY: request_data}

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None,
        **kwargs: object,
    ) -> SearchResponse:
        """
        Transform a TinyFish response to LiteLLM's unified ``SearchResponse``.

        Per-result field handling:
        - ``title``, ``url``, ``snippet`` are declared on ``SearchResult`` and
          populated by ``SearchResponse.model_validate`` when present. Missing
          or ``None`` values are defaulted to ``""`` beforehand by
          ``_default_missing_result_fields`` so a degraded result flows through
          instead of failing the whole call.
        - All undeclared per-result fields (``position``, ``site_name``, and
          any others TinyFish returns) ride through as extras via
          ``SearchResult``'s ``extra="allow"`` config — accessible as
          attributes on the result object or enumerable via
          ``result.model_extra``.

        Top-level ``parameter_warnings`` is read when present and each entry
        is re-fired via ``verbose_logger.warning``. Absent or malformed
        entries are silently skipped — never throws.

        Top-level extras (``query``, ``total_results``, ``page``, and any
        future TinyFish additions) ride through via
        ``SearchResponse.extra="allow"``. The validated response is returned
        in place after truncating ``results`` to the caller's ``max_results``,
        so every field pydantic populated survives regardless of which
        storage bucket (declared attribute or ``__pydantic_extra__``) holds it.

        TinyFish response headers (e.g. ``x-request-id``, ``retry-after``,
        ``x-ratelimit-limit`` — httpx normalizes header names to lowercase)
        are stashed on ``response._hidden_params["headers"]`` (raw) and
        ``response._hidden_params["additional_headers"]`` (sanitized via
        ``process_response_headers``) so callers can correlate a search with
        server-side logs.

        Error paths routed through ``self._wrap_error`` for uniform
        ``"TinyFish Search: <msg>. See <docs> for details."`` wrapping:
        - non-2xx HTTP status (caught here because ``AsyncHTTPHandler.get``
          does not call ``raise_for_status``)
        - 200 with non-JSON body
        - 200 with valid JSON whose shape doesn't satisfy ``SearchResponse``

        Returns:
            ``SearchResponse`` truncated to ``self._caller_max_results`` (or
            ``_TINYFISH_RESULT_CAP`` when the caller didn't set ``max_results``).
        """
        # AsyncHTTPHandler.get does not call raise_for_status, so non-2xx
        # responses arrive here looking successful. Dispatch through
        # get_error_class so callers see a uniform attributed error.
        if not (200 <= raw_response.status_code < 300):
            raise self._wrap_error(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        try:
            raw_json: object = raw_response.json()  # any-ok: httpx Response.json() -> Any
        except json.JSONDecodeError:
            raise self._wrap_error(
                error_message=f"Expected JSON response, got: {raw_response.text[:200]}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        _default_missing_result_fields(raw_json)

        try:
            parsed = SearchResponse.model_validate(raw_json)
        except ValidationError as e:
            raise self._wrap_error(
                error_message=(f"Response shape does not match LiteLLM's SearchResponse schema: {e}"),
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        _emit_parameter_warnings(parsed)

        max_results = self._caller_max_results or _TINYFISH_RESULT_CAP
        # Truncate in place so all pydantic-populated fields survive — declared and extras.
        parsed.results = list(parsed.results[:max_results])
        raw_headers = dict(raw_response.headers)
        parsed._hidden_params["headers"] = raw_headers
        parsed._hidden_params["additional_headers"] = process_response_headers(raw_headers)
        return parsed

    def _wrap_error(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str],
    ) -> Exception:
        """
        Build an attributed ``BaseLLMException`` from a TinyFish error body.

        Used only at the call sites we control inside
        ``transform_search_response`` (non-2xx, JSONDecodeError, ValidationError).
        Not an override of ``BaseSearchConfig.get_error_class``: that path is
        left to inherit from the base so it auto-picks-up any future LiteLLM
        improvements. Trade-off: network failures (routed through LiteLLM
        core's ``_handle_error`` → ``BaseSearchConfig.get_error_class``) won't
        carry the ``TinyFish Search:`` prefix — the bare error already names
        the host in the URL, so attribution is implicit there.
        """
        # TinyFish Search wraps every error body as {"error": {"code", "message", "details"?}}.
        # Best-effort unwrap to surface the inner message; fall back to the raw body
        # for other envelope shapes (CDN HTML pages, other JSON envelopes, plain text).
        inner_message = error_message
        try:
            body: object = json.loads(error_message)  # any-ok: json.loads -> Any
            if isinstance(body, dict):
                error_obj: object = body.get("error")  # any-ok: untyped dict
                if isinstance(error_obj, dict):
                    candidate: object = error_obj.get("message")  # any-ok: untyped dict
                    if isinstance(candidate, str) and candidate:
                        inner_message = candidate
        except (json.JSONDecodeError, TypeError):
            pass

        return BaseLLMException(
            status_code=status_code,
            message=f"TinyFish Search: {inner_message}. See {_TINYFISH_DOCS_URL} for details.",
            headers=headers,
        )


def _append_domain_filters(query: str, domains: list[str]) -> str:
    domain_clauses = " OR ".join(f"site:{d}" for d in domains)
    return f"({query}) ({domain_clauses})"


def _default_missing_result_fields(raw_json: object) -> None:
    """Default missing/null title/url/snippet to "" on each result item in place.

    SearchResult requires these three fields; a degraded TinyFish result flows
    through with empty strings instead of failing the whole call.
    """
    if not isinstance(raw_json, dict):
        return
    results_in = raw_json.get("results")
    if not isinstance(results_in, list):
        return
    for item in results_in:
        if not isinstance(item, dict):
            continue
        for field in ("title", "url", "snippet"):
            if not isinstance(item.get(field), str):
                item[field] = ""


def _emit_parameter_warnings(parsed: SearchResponse) -> None:
    """Re-fire TinyFish-side ``parameter_warnings`` as warnings.

    Defensive: skip silently on any shape we don't recognize so a malformed
    entry (or an early/partial rollout of the field) never throws.
    Schema per entry: ``{type, parameter, message, docs_url?}``.
    """
    warnings_field: object = (
        getattr(parsed, "parameter_warnings", None)  # any-ok: extras=allow field
    )
    if not isinstance(warnings_field, list):
        return
    for entry in warnings_field:
        if not isinstance(entry, dict):
            continue
        warning_type: object = entry.get("type")  # any-ok: untyped dict
        parameter: object = entry.get("parameter")  # any-ok: untyped dict
        message: object = entry.get("message")  # any-ok: untyped dict
        if not isinstance(warning_type, str) or not warning_type:
            continue
        if not isinstance(parameter, str) or not parameter:
            continue
        if not isinstance(message, str) or not message:
            continue
        verbose_logger.warning(
            "TinyFish Search parameter_warning (%s) `%s`: %s",
            warning_type,
            parameter,
            message,
        )
