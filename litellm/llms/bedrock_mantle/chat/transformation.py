"""
Amazon Bedrock Mantle - OpenAI-compatible inference engine in Amazon Bedrock.

API docs: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

Base URL: https://bedrock-mantle.{region}.api.aws/v1
Auth: Bearer token (litellm_params.api_key, BEDROCK_MANTLE_API_KEY, or the
      standard AWS_BEARER_TOKEN_BEDROCK) when present; otherwise AWS SigV4
      (service "bedrock") over the standard credential chain. See
      BedrockMantleAuthMixin in common_utils.
"""

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock_mantle.common_utils import (
    BEDROCK_MANTLE_DEFAULT_REGION,
    BedrockMantleAuthMixin,
)
from litellm.llms.openai.chat.gpt_5_transformation import get_effort_level, is_gpt_reasoning_series_name
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams

from ...base_llm.chat.transformation import BaseLLMException
from ...bedrock.common_utils import BedrockError
from ...openai_like.chat.transformation import OpenAILikeChatConfig
from ..common_utils import mantle_base_segment, split_mantle_region_prefix


def _with_chat_reasoning_effort(non_default_params: Mapping[str, object]) -> Mapping[str, object]:
    """Chat Completions takes ``reasoning_effort`` as a level, so a Responses-shaped ``{"effort", "summary"}``
    passes only its effort, the way the OpenAI GPT-5 chat config reads it."""
    reasoning_effort: Final = non_default_params.get("reasoning_effort")
    if not isinstance(reasoning_effort, dict):
        return non_default_params
    effort: Final = get_effort_level(reasoning_effort)
    rest: Final = {key: value for key, value in non_default_params.items() if key != "reasoning_effort"}
    return rest if effort is None else {**rest, "reasoning_effort": effort}


class BedrockMantleChatConfig(BedrockMantleAuthMixin, OpenAILikeChatConfig):
    """
    Transformation config for Amazon Bedrock Mantle OpenAI-compatible API.
    """

    def __init__(self, aws_signer: BaseAWSLLM | None = None):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock_mantle"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
        litellm_params: GenericLiteLLMParams | None = None,
        model: str | None = None,
    ) -> tuple[str | None, str | None]:
        prefix_region, base_model = split_mantle_region_prefix(model) if model else (None, None)
        region: Final = (
            (litellm_params.aws_region_name if litellm_params else None)
            or prefix_region
            or get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
        BaseAWSLLM.validate_aws_region_name(region)
        # The base path segment is data-driven per model (use_openai_responses_path
        # flag): gemma-4-* and gpt-5.x are served on /openai/v1, everything else on
        # /v1. An explicit api_base still wins over the derived default.
        api_base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws/{mantle_base_segment(base_model, litellm.model_cost)}"
        )
        dynamic_api_key: Final = self._resolve_bearer_token(api_key)
        return api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
        litellm_params: GenericLiteLLMParams | None = None,
        model: str | None = None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key, litellm_params, model)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        project_id: Final = litellm_params.get("aws_bedrock_project_id")
        if project_id:
            headers["OpenAI-Project"] = project_id
        return headers

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = super().get_supported_openai_params(model)
        extra_params: Final = tuple(
            param
            for param, supported in (
                ("verbosity", is_gpt_reasoning_series_name(model)),
                ("reasoning_effort", self._supports_reasoning(model)),
            )
            if supported and param not in base_params
        )
        return [*base_params, *extra_params]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: OpenAILikeChatConfig.map_openai_params signature
        optional_params: dict[str, object],  # mutable-ok: OpenAILikeChatConfig.map_openai_params signature
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict[str, object]:  # mutable-ok: OpenAILikeChatConfig.map_openai_params signature
        gpt_reasoning_model: Final = is_gpt_reasoning_series_name(model)
        mapped_params: Final[dict[str, object]] = super().map_openai_params(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # OpenAILikeChatConfig types its params and result as bare dicts  # mutable-ok: parent returns a dict
            dict(_with_chat_reasoning_effort(non_default_params)),
            optional_params,
            model,
            drop_params,
            replace_max_completion_tokens_with_max_tokens and not gpt_reasoning_model,
        )
        if not gpt_reasoning_model or "max_tokens" not in mapped_params:
            return mapped_params
        return {
            param: value
            for param, value in (("max_completion_tokens", mapped_params["max_tokens"]), *mapped_params.items())
            if param != "max_tokens"
        }

    def _supports_reasoning(self, model: str) -> bool:
        try:
            return litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider)
        except Exception as e:
            verbose_logger.debug("BedrockMantleChatConfig: error checking reasoning support: %s", e)
            return False

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
        from litellm.llms.openai.chat.gpt_transformation import (
            OpenAIChatCompletionStreamingHandler,
        )

        return OpenAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
