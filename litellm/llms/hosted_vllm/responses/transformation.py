"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.openai.responses.instruction_items import instruction_items_folded
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import declared_mid_conversation_system_support, deployment_model_info

HOSTED_VLLM_RESPONSES_SUPPORTED_PARAMS: Final = (
    "chat_template_kwargs",
    "top_k",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
    "best_of",
    "stop_token_ids",
    "spaces_between_special_tokens",
    "prompt_logprobs",
    "skip_special_tokens",
    "include_stop_str_in_output",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
    "guided_whitespace_pattern",
    "priority",
)


def _vllm_param_value(
    param: str, optional_params: Mapping[str, object], litellm_params: GenericLiteLLMParams
) -> object:
    from_request: Final = optional_params.get(param)
    return from_request if from_request is not None else getattr(litellm_params, param, None)


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

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors base signature
        params: Final[list[str]] = list(  # mutable-ok: accumulator list
            super().get_supported_openai_params(model)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # base returns untyped list
        )
        params.extend(HOSTED_VLLM_RESPONSES_SUPPORTED_PARAMS)
        return params

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:
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
        litellm_params: dict[str, object],
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
            return join_api_base(api_base, "responses")

        return join_api_base(api_base, "v1/responses")

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: overrides the base class signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: overrides the base class signature
    ) -> dict[str, object]:  # mutable-ok: overrides the base class signature
        declared: Final = declared_mid_conversation_system_support(
            model, self.custom_llm_provider.value, deployment_model_info(litellm_params)
        )
        folded: Final = (
            instruction_items_folded(self._validate_input_param(input), response_api_optional_request_params, "user")
            if declared is False
            else None
        )
        placed_input, placed_params = folded or (input, response_api_optional_request_params)
        request: Final = super().transform_responses_api_request(
            model=model,
            input=placed_input,
            response_api_optional_request_params=dict(placed_params),
            litellm_params=litellm_params,
            headers=headers,
        )
        vllm_params: Final = {
            param: value
            for param in HOSTED_VLLM_RESPONSES_SUPPORTED_PARAMS
            if (value := _vllm_param_value(param, placed_params, litellm_params)) is not None
        }
        return {**request, **vllm_params}

    def supports_native_websocket(self) -> bool:
        """Hosted vLLM does not support native WebSocket for Responses API"""
        return False
