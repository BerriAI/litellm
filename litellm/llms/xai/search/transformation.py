"""
Calls xAI's Responses API with the `x_search` tool (xAI's Live Search over X/Twitter).

xAI docs: https://docs.x.ai/docs/guides/live-search

Setup:
    Set XAI_API_KEY (or litellm.xai_key), the same credential xAI chat/responses calls use.
    Optional: pass model=... in optional_params to override the default (grok-4-fast).

Usage:
    response = litellm.search(
        query="what happened at the last SpaceX launch",
        search_provider="xai",
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.constants import XAI_API_BASE
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.xai.common_utils import xai_reported_cost_in_usd
from litellm.secret_managers.main import get_secret_str
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelInfo

_RESPONSES_PATH: Final = "/responses"
_DEFAULT_MODEL: Final = "grok-4-fast"
_UPSTREAM_ERROR_STATUS: Final = 502
_RESPONSE_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"
_API_KEY_ENV_VAR: Final = "XAI_API_KEY"
_API_BASE_ENV_VAR: Final = "XAI_API_BASE"
# https://docs.x.ai/developers/pricing#tools-pricing — proxy for "sources used" only when
# xAI's own cost_in_usd_ticks isn't present on the response (see _resolved_cost).
_PER_CITATION_SURCHARGE_USD: Final = 0.025


class _Annotation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str = ""
    url: str | None = None
    title: str | None = None


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


class _OutputTokensDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    reasoning_tokens: int | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_details: _OutputTokensDetails | None = None
    cost_in_usd_ticks: int | None = None


class _ResponsesEnvelope(BaseModel):
    """An xAI Responses API body. `output` is required: a body without it is not a
    Responses API response and must not be reported as a successful empty search.

    A 200 body can still carry `status` `failed` or `incomplete`; those are surfaced as
    errors rather than reported as a successful empty search."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    output: tuple[_OutputItem, ...]
    status: str | None = None
    error: _ErrorBody | None = None
    incomplete_details: _IncompleteDetails | None = None
    usage: _Usage | None = None


class _XSearchTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["x_search"] = "x_search"
    allowed_x_handles: tuple[str, ...] | None = None
    excluded_x_handles: tuple[str, ...] | None = None
    from_date: str | None = None
    to_date: str | None = None
    enable_image_understanding: bool | None = None
    enable_video_understanding: bool | None = None


class _ResponsesRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    input: str
    tools: tuple[_XSearchTool, ...]


def _typed_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _typed_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _typed_str_tuple(value: object) -> tuple[str, ...] | None:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _model(optional_params: Mapping[str, object]) -> str:
    model: Final = optional_params.get("model")
    return model if isinstance(model, str) and model else _DEFAULT_MODEL


def _requested_model(response_kwargs: Mapping[str, object]) -> str:
    optional_params: Final = response_kwargs.get("optional_params")
    if not isinstance(optional_params, Mapping):
        return _DEFAULT_MODEL
    return _model(optional_params)


def _valid_max_results(max_results: object) -> int | None:
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        return None
    return max_results if max_results > 0 else None


def _requested_max_results(response_kwargs: Mapping[str, object]) -> int | None:
    optional_params: Final = response_kwargs.get("optional_params")
    if not isinstance(optional_params, Mapping):
        return None
    return _valid_max_results(optional_params.get("max_results"))


def _x_search_tool(optional_params: Mapping[str, object]) -> _XSearchTool:
    return _XSearchTool(
        allowed_x_handles=_typed_str_tuple(optional_params.get("allowed_x_handles")),
        excluded_x_handles=_typed_str_tuple(optional_params.get("excluded_x_handles")),
        from_date=_typed_str(optional_params.get("from_date")),
        to_date=_typed_str(optional_params.get("to_date")),
        enable_image_understanding=_typed_bool(optional_params.get("enable_image_understanding")),
        enable_video_understanding=_typed_bool(optional_params.get("enable_video_understanding")),
    )


def _citation_results(envelope: _ResponsesEnvelope) -> tuple[SearchResult, ...]:
    """One result per distinct cited URL, in first-appearance order.

    xAI's url_citation annotations carry start_index/end_index, but they are reportedly
    always 0 in practice, so there is no reliable span to slice a per-citation excerpt
    from; the full synthesized answer is shared as `snippet` across every citation
    instead of attempting a per-citation slice.
    """
    full_text: Final = "\n\n".join(
        part.text
        for item in envelope.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )
    citations: Final = tuple(
        (annotation.url, annotation.title or "")
        for item in envelope.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for annotation in part.annotations
        if annotation.type == "url_citation" and annotation.url
    )
    first_title_by_url: Final = MappingProxyType({url: title for url, title in reversed(citations)})
    return tuple(
        SearchResult(title=first_title_by_url[url], url=url, snippet=full_text, date=None, last_updated=None)
        for url in dict.fromkeys(url for url, _ in citations)
    )


def _model_info(model: str) -> ModelInfo | None:
    try:
        return get_model_info(model=f"xai/{model}", custom_llm_provider="xai")
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return None


def _token_and_surcharge_cost(usage: _Usage, model: str, distinct_citation_count: int) -> float | None:
    """
    Fallback cost when xAI doesn't report cost_in_usd_ticks: real per-token cost from
    the model cost map plus a per-source surcharge proxied by distinct citation count.

    Tries the bare model name first, then the -reasoning/-non-reasoning suffix picked
    by whether any reasoning tokens were billed, since only the suffixed variants carry
    a price entry for grok-4-fast-family models.
    """
    reasoning_tokens: Final = usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else None
    suffix: Final = "-reasoning" if (reasoning_tokens or 0) > 0 else "-non-reasoning"
    info: Final = _model_info(model) or _model_info(f"{model}{suffix}")
    if info is None:
        return None
    input_cost: Final = (usage.input_tokens or 0) * float(info.get("input_cost_per_token") or 0.0)
    output_cost: Final = (usage.output_tokens or 0) * float(info.get("output_cost_per_token") or 0.0)
    return input_cost + output_cost + _PER_CITATION_SURCHARGE_USD * distinct_citation_count


def _resolved_cost(usage: _Usage | None, model: str, distinct_citation_count: int) -> float | None:
    if usage is None:
        return None
    reported: Final = xai_reported_cost_in_usd(usage.cost_in_usd_ticks)
    if reported is not None:
        return reported
    return _token_and_surcharge_cost(usage, model, distinct_citation_count)


class XAISearchConfig(BaseSearchConfig):
    """
    x_search only exists as a tool inside an xAI Responses API turn: the model plans
    its own queries, calls x_search itself, and synthesizes a prose answer with inline
    url_citation annotations. There is no discrete results list the way every other
    Search API provider returns one, so this config reverse-engineers a SearchResponse
    out of that Responses API turn instead of calling a dedicated search endpoint.
    """

    @staticmethod
    def ui_friendly_name() -> str:
        return "xAI Live Search (x_search)"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: httpx requires a plain dict of headers
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=(_API_KEY_ENV_VAR,),
            base_env_var=_API_BASE_ENV_VAR,
            default_api_base=XAI_API_BASE,
        )
        if not resolved_key:
            raise ValueError(f"{_API_KEY_ENV_VAR} is required. Set it, or pass api_key explicitly.")
        return {  # mutable-ok: httpx requires a plain dict of headers
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = (api_base or get_secret_str(_API_BASE_ENV_VAR) or XAI_API_BASE).rstrip("/")
        if resolved_base.endswith(_RESPONSES_PATH):
            return resolved_base
        return f"{resolved_base}{_RESPONSES_PATH}"

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as the JSON body
        request: Final = _ResponsesRequest(
            model=_model(optional_params),
            input=" ".join(query) if isinstance(query, list) else query,
            tools=(_x_search_tool(optional_params),),
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
                error_message=f"response does not match the xAI Responses API schema: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
            )
        if parsed.status == "failed":
            detail: Final = parsed.error.message if parsed.error and parsed.error.message else "the search failed"
            raise self._upstream_error(detail, raw_response)
        results: Final = _citation_results(parsed)
        if not results and parsed.status == "incomplete":
            reason: Final = (
                parsed.incomplete_details.reason
                if parsed.incomplete_details and parsed.incomplete_details.reason
                else "unknown reason"
            )
            raise self._upstream_error(f"the search was incomplete: {reason}", raw_response)
        max_results: Final = _requested_max_results(kwargs)
        capped_results: Final = results[:max_results] if max_results is not None else results
        return self._priced(capped_results, parsed.usage, _requested_model(kwargs), len(results))

    def _priced(
        self,
        results: tuple[SearchResult, ...],
        usage: _Usage | None,
        model: str,
        distinct_citation_count: int,
    ) -> SearchResponse:
        response: Final = SearchResponse(
            results=list(results),  # mutable-ok: SearchResponse.results is list[SearchResult]
            object="search",
        )
        cost: Final = _resolved_cost(usage, model, distinct_citation_count)
        if cost is not None:
            response._hidden_params[  # pyright: ignore[reportPrivateUsage]  # response_cost_calculator's own contract
                "additional_headers"
            ] = {  # mutable-ok: response_cost_calculator writes into _hidden_params
                _RESPONSE_COST_HEADER: cost
            }
        return response

    def _upstream_error(self, detail: str, raw_response: httpx.Response) -> Exception:
        return self.get_error_class(
            error_message=detail,
            status_code=_UPSTREAM_ERROR_STATUS,
            headers=dict(raw_response.headers),  # mutable-ok: BaseSearchConfig.get_error_class signature
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.get_error_class signature
    ) -> Exception:
        return BaseLLMException(
            status_code=status_code,
            message=f"xAI x_search: {error_message}",
            headers=headers,
        )
