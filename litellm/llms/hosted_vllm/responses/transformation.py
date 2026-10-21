"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.hosted_vllm.common_utils import (
    CACHE_SALT_FIELD,
    HOSTED_VLLM_CACHE_SALT_PARAM,
    derive_cache_salt,
    get_cache_salt_caller,
    with_caller_cache_salt,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


def _cache_salt_caller(litellm_params: GenericLiteLLMParams) -> str | None:
    # On /v1/responses `metadata` is the caller's own OpenAI field, so the key hash is read from
    # `litellm_metadata` only.
    extras: Final[Mapping[str, object]] = litellm_params.model_extra or MappingProxyType({})
    return get_cache_salt_caller(
        MappingProxyType(
            {
                HOSTED_VLLM_CACHE_SALT_PARAM: litellm_params.hosted_vllm_cache_salt,
                "litellm_metadata": extras.get("litellm_metadata"),
            }
        ),
        metadata_keys=("litellm_metadata",),
    )


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Hosted VLLM Responses API support.

    Extends OpenAI's config since vLLM follows OpenAI's API spec,
    but uses HOSTED_VLLM_API_BASE for the base URL and defaults
    to "fake-api-key" when no API key is provided (vLLM does not
    require authentication by default).
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.HOSTED_VLLM

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key: Final = (
            litellm_params.api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        )  # vllm does not require an api key
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: overrides the base class signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: overrides the base class signature
    ) -> dict[str, object]:  # mutable-ok: overrides the base class signature
        request: Final = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        caller: Final = _cache_salt_caller(litellm_params)
        if caller is None:
            return request
        return {**request, CACHE_SALT_FIELD: derive_cache_salt(caller, request.get(CACHE_SALT_FIELD))}

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        return with_caller_cache_salt(extra_body, _cache_salt_caller(litellm_params))

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")

        if api_base is None:
            raise ValueError(
                "api_base not set for Hosted VLLM responses API. "
                "Set via api_base parameter or HOSTED_VLLM_API_BASE environment variable"
            )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # If api_base already ends with /v1, append /responses
        # Otherwise append /v1/responses
        if api_base.endswith("/v1"):
            return f"{api_base}/responses"

        return f"{api_base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """Hosted vLLM does not support native WebSocket for Responses API"""
        return False
