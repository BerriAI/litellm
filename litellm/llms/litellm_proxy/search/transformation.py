"""
Calls a search tool registered on a LiteLLM proxy.
Endpoint: POST <proxy>/search/{search_tool_name}
"""

from __future__ import annotations

from typing import Final

import httpx

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segment, ensure_api_route
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse
from litellm.secret_managers.main import get_secret_str

_SEARCH_TOOL_NAME: Final = "search_tool_name"
_FORWARDED_LITELLM_PARAMS: Final = frozenset((_SEARCH_TOOL_NAME,))
_UNROUTABLE_TOOL_NAMES: Final = frozenset(("", ".", ".."))


class LiteLLMProxySearchConfig(BaseSearchConfig):
    @staticmethod
    def ui_friendly_name() -> str:
        return "LiteLLM Proxy"

    def forwarded_litellm_params(self) -> frozenset[str]:
        return _FORWARDED_LITELLM_PARAMS

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment signature
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: the http handler passes this straight to httpx as headers
        resolved_key: Final = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("LITELLM_PROXY_API_KEY",),
            base_env_var="LITELLM_PROXY_API_BASE",
            default_api_base=None,
        )
        auth: Final = {"Authorization": f"Bearer {resolved_key}"} if resolved_key else {}
        return {**headers, "Content-Type": "application/json", **auth}

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url signature
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str("LITELLM_PROXY_API_BASE")
        if not resolved_base:
            raise _bad_request(
                "Set api_base (or the LITELLM_PROXY_API_BASE environment variable) to the LiteLLM proxy URL."
            )
        tool_name: Final = optional_params.get(_SEARCH_TOOL_NAME)
        if not isinstance(tool_name, str) or tool_name in _UNROUTABLE_TOOL_NAMES:
            raise _bad_request("Pass search_tool_name, the name of a search tool registered on the LiteLLM proxy.")
        return ensure_api_route(
            resolved_base, f"search/{encode_url_path_segment(tool_name, field_name=_SEARCH_TOOL_NAME)}"
        )

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig.transform_search_request signature
        optional_params: dict[str, object],  # mutable-ok: base signature
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_request signature
    ) -> dict[str, object]:  # mutable-ok: the http handler sends this as the JSON body
        return {
            "query": query,
            **{key: value for key, value in optional_params.items() if key != _SEARCH_TOOL_NAME},
        }

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig.transform_search_response signature
    ) -> SearchResponse:
        return SearchResponse.model_validate_json(raw_response.content)


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(message=message, model="litellm_proxy/search", llm_provider="litellm_proxy")
