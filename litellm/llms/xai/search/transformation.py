from collections.abc import Mapping
from typing import Annotated, Final

import httpx
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import XAI_API_BASE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.responses import (
    ResponsesSearchEnvelope,
    responses_search_citations,
    responses_search_text_parts,
)
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)
from litellm.llms.xai.common_utils import xai_reported_cost_in_usd
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.xai import XAIXSearchTool

_X_SEARCH_TOOL: Final = TypeAdapter(XAIXSearchTool)


class _SearchOptions(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    search_model: str | None = None
    max_results: Annotated[int, Field(gt=0, strict=True)] | None = None


class _XAIResponsesSearchEnvelope(ResponsesSearchEnvelope):
    usage: Mapping[str, object] | None = None


class XAISearchConfig(BaseSearchConfig):
    """Search X using the Responses API. ``search_model`` selects the Grok model.

    Each citation becomes a result with the answer as its snippet. ``max_results``
    caps returned citations, not upstream tool usage or the cost of the request.
    """

    @staticmethod
    def ui_friendly_name() -> str:
        return "xAI"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment contract
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment contract
    ) -> dict[str, str]:  # mutable-ok: HTTP handler consumes the headers
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("XAI_API_KEY",),
            base_env_var="XAI_API_BASE",
            default_api_base=XAI_API_BASE,
        )
        if not resolved_key:
            raise ValueError("Set XAI_API_KEY or pass api_key to use xAI search.")
        return {
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url contract
        data: (dict[str, object] | list[dict[str, object]] | None) = None,  # mutable-ok: base contract
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url contract
    ) -> str:
        return ensure_api_route(api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE, "responses")

    def transform_search_request(
        self,
        query: (str | list[str]),  # mutable-ok: BaseSearchConfig.transform_search_request contract
        optional_params: dict[str, object],  # mutable-ok: base contract
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request contract
    ) -> dict[str, object]:  # mutable-ok: HTTP handler sends the JSON body
        options: Final = _SearchOptions.model_validate(optional_params)
        tool: Final = _X_SEARCH_TOOL.validate_python({**optional_params, "type": "x_search"})
        return {
            "model": options.search_model or "grok-4.7",
            "input": "\n".join(query) if isinstance(query, list) else query,
            "tools": [tool],
            "tool_choice": "required",
            "store": False,
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response contract
    ) -> SearchResponse:
        try:
            parsed: Final = _XAIResponsesSearchEnvelope.model_validate_json(raw_response.content)
        except ValidationError:
            raise BaseLLMException(
                message="xAI returned an invalid Responses API search response.",
                status_code=502,
                headers=dict(raw_response.headers),
            )
        if parsed.error or parsed.status in ("failed", "incomplete"):
            detail: Final = parsed.error.message if parsed.error else f"Search response status was {parsed.status}"
            raise BaseLLMException(
                message=detail or "xAI search failed",
                status_code=502,
                headers=dict(raw_response.headers),
            )
        cost: Final = xai_reported_cost_in_usd(parsed.usage.get("cost_in_usd_ticks") if parsed.usage else None)
        if cost is None:
            raise BaseLLMException(
                message="xAI search response is missing a valid usage.cost_in_usd_ticks for billing.",
                status_code=502,
                headers=dict(raw_response.headers),
            )
        options: Final = _SearchOptions.model_validate(kwargs.get("optional_params") or {})
        answer: Final = "\n".join(part.text for part in responses_search_text_parts(parsed.output))
        results: Final = tuple(
            SearchResult(
                title=annotation.title or annotation.url or "",
                url=annotation.url or "",
                snippet=answer,
            )
            for _, annotation in responses_search_citations(parsed.output)
        )
        response: Final = SearchResponse(results=list(results[: options.max_results]))
        response.hidden_params = {"response_cost": cost}
        return response
