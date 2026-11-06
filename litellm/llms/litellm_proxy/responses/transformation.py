"""
Responses API transformation for LiteLLM Proxy provider.

LiteLLM Proxy supports the OpenAI Responses API natively when the underlying model supports it.
This config enables pass-through behavior to the proxy's /v1/responses endpoint.
"""

from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.responses.transformation import ANY_REASONING_FIELD_VALUES, ReasoningFieldValues
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

_ERROR_BODY: Final = TypeAdapter(dict[str, object])


def _openai_sdk_error_body(
    error_message: str,
) -> dict[str, object] | None:  # mutable-ok: BaseLLMException.body is a dict
    try:
        decoded: Final = _ERROR_BODY.validate_json(error_message)
    except ValidationError:
        return None
    nested: Final = decoded.get("error")
    return _ERROR_BODY.validate_python(nested) if isinstance(nested, dict) else decoded


class LiteLLMProxyResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for LiteLLM Proxy Responses API support.

    Extends OpenAI's config since the proxy follows OpenAI's API spec,
    but uses LITELLM_PROXY_API_BASE for the base URL.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.LITELLM_PROXY

    def reasoning_field_values(self) -> ReasoningFieldValues:
        return ANY_REASONING_FIELD_VALUES

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the endpoint for LiteLLM Proxy responses API.

        Uses LITELLM_PROXY_API_BASE environment variable if api_base is not provided.
        """
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")

        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy responses API. "
                "Set via api_base parameter or LITELLM_PROXY_API_BASE environment variable"
            )

        return join_api_base(api_base, "responses")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
            body=_openai_sdk_error_body(error_message),
        )

    def supports_native_websocket(self) -> bool:
        """LiteLLM Proxy does not support native WebSocket for Responses API"""
        return False
