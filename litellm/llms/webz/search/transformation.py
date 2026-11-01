"""
Webz.io Search transformation configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
)
from litellm.search.webz import (
    WEBZ_API_URL,
    _build_webz_payload,
    _parse_webz_response,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class WebzSearchConfig(BaseSearchConfig):
    WEBZ_API_BASE: Final = WEBZ_API_URL

    @staticmethod
    def ui_friendly_name() -> str:
        return "Webz.io"

    def validate_environment(
        self,
        headers: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> dict:
        resolved_api_key = self.resolve_server_api_key(
            caller_api_key=api_key,
            caller_api_base=api_base,
            key_env_vars=("WEBZ_API_TOKEN", "WEBZ_API_KEY"),
            base_env_var="WEBZ_API_BASE",
            default_api_base=self.WEBZ_API_BASE,
        )
        if not resolved_api_key:
            raise ValueError("WEBZ_API_TOKEN is not set. Set `WEBZ_API_TOKEN` environment variable.")
        headers["Authorization"] = f"Bearer {resolved_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict,
        data: dict | list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        return api_base or get_secret_str("WEBZ_API_BASE") or self.WEBZ_API_BASE

    def transform_search_request(
        self,
        query: str | list[str],
        optional_params: dict,
        **kwargs: Any,
    ) -> dict:
        params = {**kwargs, **optional_params}
        return _build_webz_payload(
            query=query,
            **params,
        )

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None = None,
        **kwargs: Any,
    ) -> SearchResponse:
        return _parse_webz_response(raw_response.json())
