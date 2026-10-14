"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

from collections.abc import Mapping
from typing import Final, cast  # noqa: TID251  # a role-swapped input item has no typed rebuild across item shapes

from openai.types.responses import ResponseInputItemParam

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


def _developer_item_as_system(item: ResponseInputItemParam) -> Mapping[str, object] | ResponseInputItemParam:
    if "role" not in item or item["role"] != "developer":
        return item
    return {**item, "role": "system"}


def _developer_items_as_system(input: str | ResponseInputParam) -> str | ResponseInputParam:
    """vLLM hands input roles straight to the served model's chat template, and templates such as
    Kimi K3's only know system/user/assistant/tool, so `developer` goes out as `system`, the same
    mapping the chat path applies through `translate_developer_role_to_system_role`.
    """
    if isinstance(input, str):
        return input
    return cast(  # cast-ok: each item keeps its shape; only a developer role becomes system
        "ResponseInputParam", [_developer_item_as_system(item) for item in input]
    )


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Hosted VLLM Responses API support.

    Extends OpenAI's config since vLLM follows OpenAI's API spec,
    but uses HOSTED_VLLM_API_BASE for the base URL and defaults
    to "fake-api-key" when no API key is provided (vLLM does not
    require authentication by default).

    Input `developer` items are sent as `system`, because vLLM passes roles to the
    model's chat template and many open-weight templates don't know `developer`.
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

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: overrides the base class signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: overrides the base class signature
    ) -> dict:  # mutable-ok: overrides the base class signature
        return super().transform_responses_api_request(
            model=model,
            input=_developer_items_as_system(self._validate_input_param(input)),
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def supports_native_websocket(self) -> bool:
        """Hosted vLLM does not support native WebSocket for Responses API"""
        return False
