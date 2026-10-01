"""
Calls the Microsoft Foundry Responses API with the `bing_grounding` or `web_search`
tool to search the web (Grounding with Bing Search).

Microsoft docs: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/bing-grounding

Setup:
    1. Set BING_GROUNDING_PROJECT_ENDPOINT to the Foundry project endpoint, e.g.
       https://<account>.services.ai.azure.com/api/projects/<project>
    2. Set BING_GROUNDING_MODEL to a model deployment in that project (e.g. gpt-4.1);
       it runs the grounded search and its tokens are billed on that deployment
    3. Optional: set BING_GROUNDING_CONNECTION_ID to a Grounding with Bing Search
       project connection id to use the `bing_grounding` tool; without it the
       project's built-in `web_search` tool is used
    4. Auth: pass api_key (an Azure API key, sent in the api-key header), or set
       BING_GROUNDING_TOKEN to an Entra bearer token for scope
       https://ai.azure.com/.default, or configure azure-identity (AZURE_CLIENT_ID /
       AZURE_CLIENT_SECRET / AZURE_TENANT_ID, managed identity, or any
       DefaultAzureCredential source) and the token is minted automatically

Usage:
    response = litellm.search(
        query="latest AI developments",
        search_provider="bing_grounding",
        max_results=5,
    )
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_DOCS_URL: Final = "https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/bing-grounding"

PROJECT_ENDPOINT_ENV: Final = "BING_GROUNDING_PROJECT_ENDPOINT"
MODEL_ENV: Final = "BING_GROUNDING_MODEL"
CONNECTION_ID_ENV: Final = "BING_GROUNDING_CONNECTION_ID"
TOKEN_ENV: Final = "BING_GROUNDING_TOKEN"

ENTRA_SCOPE: Final = "https://ai.azure.com/.default"

_RESPONSES_PATH: Final = "/openai/v1/responses"
_SNIPPET_FALLBACK_LENGTH: Final = 300
_UPSTREAM_ERROR_STATUS: Final = 502
_RESPONSE_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"


class _Annotation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    url: str | None = None
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None


class _ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    text: str = ""
    annotations: tuple[_Annotation, ...] = ()


class _OutputItem(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    content: tuple[_ContentPart, ...] = ()


class _ErrorBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: str | None = None


class _IncompleteDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    reason: str | None = None


class _ResponsesEnvelope(BaseModel):
    """A Foundry Responses API body. `output` is required: a body without it is not a
    Responses API response and must not be reported as a successful empty search.

    A 200 body can still carry `status` `failed` or `incomplete`; those are surfaced as
    errors rather than reported as a successful empty search."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    output: tuple[_OutputItem, ...]
    status: str | None = None
    error: _ErrorBody | None = None
    incomplete_details: _IncompleteDetails | None = None


class _ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error: _ErrorBody | None = None


def _unwrap_error_detail(error_message: str) -> str:
    """
    Surface the human-readable message inside Foundry's error envelope.

    Tool failures nest a second JSON document as a string inside `error.message`
    (observed live for `bing_grounding` connection errors), so the unwrap runs twice.
    Falls back to the raw body for anything else.
    """
    try:
        envelope: Final = _ErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    message: Final = envelope.error.message if envelope.error else None
    if message is None:
        return error_message
    try:
        nested: Final = _ErrorBody.model_validate_json(message)
    except ValidationError:
        return message
    return nested.message or message


def _snippet(text: str, annotation: _Annotation) -> str:
    """
    The text a citation supports, not the citation marker itself.

    A url_citation's start/end indices span the inline marker ("([host](url))"),
    which follows the claim it backs, so the snippet is the marker's own line up
    to where the marker starts.
    """
    start: Final = annotation.start_index
    marker_start: Final = start if start is not None and 0 <= start <= len(text) else len(text)
    claim: Final = text[:marker_start].rsplit("\n", 1)[-1].strip()
    if claim:
        return claim[-_SNIPPET_FALLBACK_LENGTH:]
    return text[:_SNIPPET_FALLBACK_LENGTH]


def _citation_results(envelope: _ResponsesEnvelope) -> tuple[SearchResult, ...]:
    """One result per cited URL: first occurrence wins, order preserved as answered."""
    cited: Final = tuple(
        SearchResult(
            title=annotation.title or "",
            url=annotation.url or "",
            snippet=_snippet(part.text, annotation),
            date=None,
            last_updated=None,
        )
        for item in envelope.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for annotation in part.annotations
        if annotation.type == "url_citation" and annotation.url
    )
    first_by_url: Final = MappingProxyType({result.url: result for result in reversed(cited)})
    return tuple(first_by_url[url] for url in dict.fromkeys(result.url for result in cited))


def _valid_max_results(max_results: object) -> int | None:
    """A positive-int `max_results`, else None. Rejects bools, an `int` subclass, and
    non-positive values so neither the request-side `count` nor the response-side cap
    forwards a value the other would silently ignore.
    """
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        return None
    return max_results if max_results > 0 else None


def _requested_max_results(response_kwargs: Mapping[str, object]) -> int | None:
    """The unified `max_results` cap the caller asked for, if any.

    The built-in web_search tool has no server-side result-count knob, so the cap is
    enforced here after the fact; connection mode also honors it as a hard ceiling on
    top of the tool's `count` hint.
    """
    optional_params: Final = response_kwargs.get("optional_params")
    if not isinstance(optional_params, Mapping):
        return None
    return _valid_max_results(optional_params.get("max_results"))


def _capped(results: tuple[SearchResult, ...], max_results: int | None) -> tuple[SearchResult, ...]:
    return results[:max_results] if max_results is not None else results


class _SearchConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_connection_id: str
    count: int | None = None


class _BingGroundingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_configurations: tuple[_SearchConfiguration, ...]


class _BingGroundingTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["bing_grounding"] = "bing_grounding"
    bing_grounding: _BingGroundingParams


class _UserLocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["approximate"] = "approximate"
    country: str


class _WebSearchTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["web_search"] = "web_search"
    user_location: _UserLocation | None = None


class _ResponsesRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    input: str
    tools: tuple[_BingGroundingTool | _WebSearchTool, ...]


def _search_tool(optional_params: Mapping[str, object]) -> _BingGroundingTool | _WebSearchTool:
    connection_id: Final = get_secret_str(CONNECTION_ID_ENV)
    max_results: Final = optional_params.get("max_results")
    country: Final = optional_params.get("country")
    if connection_id:
        configuration: Final = _SearchConfiguration(
            project_connection_id=connection_id,
            count=_valid_max_results(max_results),
        )
        return _BingGroundingTool(bing_grounding=_BingGroundingParams(search_configurations=(configuration,)))
    location: Final = _UserLocation(country=country.upper()) if isinstance(country, str) else None
    return _WebSearchTool(user_location=location)


def _default_entra_token_minter() -> str:
    from litellm.secret_managers.get_azure_ad_token_provider import get_azure_ad_token_provider

    return get_azure_ad_token_provider(azure_scope=ENTRA_SCOPE)()


class BingGroundingSearchConfig(BaseSearchConfig):
    def __init__(self, entra_token_minter: Callable[[], str] | None = None) -> None:
        super().__init__()
        self._entra_token_minter = entra_token_minter

    @staticmethod
    def ui_friendly_name() -> str:
        return "Grounding with Bing Search"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        """
        Validate environment and return headers.

        Returns a new dict rather than mutating ``headers``: the http handler calls this
        a second time after ``litellm/search/main.py`` already did, so it has to be idempotent.
        """
        return {  # mutable-ok: httpx requires a plain dict of headers
            **headers,
            **self._auth_header(api_key, api_base),
            "Content-Type": "application/json",
        }

    def _auth_header(self, api_key: str | None, api_base: str | None) -> Mapping[str, str]:
        """
        A caller-supplied ``api_key`` is an Azure API key and rides the ``api-key`` header;
        an Entra bearer token (``BING_GROUNDING_TOKEN`` or one minted via azure-identity)
        rides ``Authorization: Bearer``. Foundry rejects the wrong scheme for each.
        """
        if api_key:
            return MappingProxyType({"api-key": api_key})
        token: Final = self.resolve_server_api_key(
            caller_api_key=None,
            caller_api_base=api_base,
            key_env_vars=(TOKEN_ENV,),
            base_env_var=PROJECT_ENDPOINT_ENV,
            default_api_base=None,
        ) or self._mint_entra_token(api_base)
        return MappingProxyType({"Authorization": f"Bearer {token}"})

    def _mint_entra_token(self, caller_api_base: str | None) -> str:
        self._assert_trusted_api_base_for_server_credential(
            caller_api_base, None, PROJECT_ENDPOINT_ENV, "Azure AD token"
        )
        minter: Final = self._entra_token_minter or _default_entra_token_minter
        try:
            return minter()
        except Exception as e:
            raise ValueError(
                f"Grounding with Bing Search: no credential available. Pass api_key, set {TOKEN_ENV} "
                f"to an Entra bearer token, or configure azure-identity (AZURE_CLIENT_ID / "
                f"AZURE_CLIENT_SECRET / AZURE_TENANT_ID or any DefaultAzureCredential source) "
                f"for scope {ENTRA_SCOPE}. Underlying error: {e}"
            ) from e

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str(PROJECT_ENDPOINT_ENV)
        if not resolved_base:
            raise ValueError(
                f"{PROJECT_ENDPOINT_ENV} is not set. Set it to your Microsoft Foundry project "
                f"endpoint, e.g. https://<account>.services.ai.azure.com/api/projects/<project>."
            )
        trimmed: Final = resolved_base.rstrip("/")
        if trimmed.endswith(_RESPONSES_PATH):
            return trimmed
        return f"{trimmed}{_RESPONSES_PATH}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as the JSON body
        """
        Transform Search request to the Foundry Responses API format.

        The unified params map as far as the API allows:
        - max_results -> the bing_grounding search configuration's `count`; the built-in
          web_search tool has no result-count knob, so that mode instead caps the returned
          results after the fact (see transform_search_response)
        - country -> web_search's approximate `user_location` (bing_grounding's `market`
          wants a full locale like en-US, which a bare country code cannot fill)
        - search_domain_filter, max_tokens_per_page -> no API equivalent, dropped
        """
        model: Final = get_secret_str(MODEL_ENV)
        if not model:
            raise ValueError(
                f"{MODEL_ENV} is not set. Set it to a model deployment in the Foundry project "
                f"that runs the grounded search, e.g. gpt-4.1."
            )
        request: Final = _ResponsesRequest(
            model=model,
            input=" ".join(query) if isinstance(query, list) else query,
            tools=(_search_tool(optional_params),),
        )
        return request.model_dump(mode="json", exclude_none=True)

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        try:
            parsed: Final = _ResponsesEnvelope.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise self.get_error_class(
                error_message=f"response does not match the Foundry Responses API schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )
        if parsed.status == "failed":
            detail: Final = (
                parsed.error.message if parsed.error and parsed.error.message else "the grounded search failed"
            )
            raise self._upstream_error(detail, raw_response)
        results: Final = _capped(_citation_results(parsed), _requested_max_results(kwargs))
        if not results and parsed.status == "incomplete":
            reason: Final = (
                parsed.incomplete_details.reason
                if parsed.incomplete_details and parsed.incomplete_details.reason
                else "unknown reason"
            )
            raise self._upstream_error(f"the grounded search was incomplete: {reason}", raw_response)
        return self._priced(results)

    def _upstream_error(self, detail: str, raw_response: httpx.Response) -> Exception:
        return self.get_error_class(
            error_message=detail,
            status_code=_UPSTREAM_ERROR_STATUS,
            headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
        )

    def _priced(self, results: tuple[SearchResult, ...]) -> SearchResponse:
        """web_search mode runs no paid Grounding with Bing transaction, so it must not
        inherit the connection-mode ``bing_grounding/search`` price; zero its per-query
        cost while leaving connection mode to the cost map."""
        response: Final = SearchResponse(
            results=list(results),  # mutable-ok: SearchResponse.results is list[SearchResult]
            object="search",
        )
        if get_secret_str(CONNECTION_ID_ENV):
            return response
        response._hidden_params[
            "additional_headers"
        ] = {  # mutable-ok: response_cost_calculator writes into _hidden_params
            _RESPONSE_COST_HEADER: 0.0
        }
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.get_error_class signature
    ) -> Exception:
        detail: Final = _unwrap_error_detail(error_message).rstrip(". ")
        return BaseLLMException(
            status_code=status_code,
            message=f"Grounding with Bing Search: {detail}. See {_DOCS_URL} for details.",
            headers=headers,
        )
