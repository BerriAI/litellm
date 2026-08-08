from typing import Final

import httpx

import litellm
from litellm.llms.volcengine.responses.transformation import VolcEngineResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    BytePlusError,
    get_byteplus_base_url,
    get_byteplus_headers,
)


class BytePlusResponsesAPIConfig(VolcEngineResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BYTEPLUS

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BytePlusError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return BytePlusError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        api_key_from_params: str | None = None
        if litellm_params is not None:
            if isinstance(litellm_params, dict):
                api_key_from_params = litellm_params.get("api_key")
            elif hasattr(litellm_params, "api_key"):
                api_key_from_params = getattr(litellm_params, "api_key", None)

        api_key: Final = (
            api_key_from_params
            or litellm.api_key
            or get_secret_str("BYTEPLUS_API_KEY")
            or get_secret_str("ARK_API_KEY")
        )

        if api_key is None:
            raise ValueError("BytePlus API key is required. Set BYTEPLUS_API_KEY or ARK_API_KEY or pass api_key.")

        return get_byteplus_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_byteplus_base_url()
        )

        base_url = base_url.rstrip("/")

        if base_url.endswith("/responses"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/responses"
        return f"{base_url}/api/v3/responses"
