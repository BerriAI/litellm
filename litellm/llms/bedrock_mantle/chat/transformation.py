"""
Amazon Bedrock Mantle - OpenAI-compatible inference engine in Amazon Bedrock.

API docs: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

Base URL: https://bedrock-mantle.{region}.api.aws/v1
Auth: AWS Bedrock API key as Bearer token (set via BEDROCK_MANTLE_API_KEY env var)
      or region-aware key via BEDROCK_MANTLE_{REGION}_API_KEY.
"""

from typing import Iterator, AsyncIterator, Any, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams

from ..common_utils import mantle_base_segment
from ...openai_like.chat.transformation import OpenAILikeChatConfig

BEDROCK_MANTLE_DEFAULT_REGION = "us-east-1"


class BedrockMantleChatConfig(OpenAILikeChatConfig):
    """
    Transformation config for Amazon Bedrock Mantle OpenAI-compatible API.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock_mantle"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        litellm_params: Optional[GenericLiteLLMParams] = None,
        model: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        region = (
            (litellm_params.aws_region_name if litellm_params else None)
            or get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
        BaseAWSLLM._validate_aws_region_name(region)
        # The base path segment is data-driven per model (use_openai_responses_path
        # flag): gemma-4-* and gpt-5.x are served on /openai/v1, everything else on
        # /v1. An explicit api_base still wins over the derived default.
        api_base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws/{mantle_base_segment(model, litellm.model_cost)}"
        )
        dynamic_api_key = api_key or get_secret_str("BEDROCK_MANTLE_API_KEY")
        return api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
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
        project_id = litellm_params.get("aws_bedrock_project_id")
        if project_id:
            headers["OpenAI-Project"] = project_id
        return headers

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                if "reasoning_effort" not in base_params:
                    base_params.append("reasoning_effort")
        except Exception as e:
            verbose_logger.debug(
                f"BedrockMantleChatConfig: error checking reasoning support: {e}"
            )
        return base_params

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        from litellm.llms.openai.chat.gpt_transformation import (
            OpenAIChatCompletionStreamingHandler,
        )

        return OpenAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
