"""
Amazon Bedrock Mantle - OpenAI-compatible inference engine in Amazon Bedrock.

API docs: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

Base URL: https://bedrock-mantle.{region}.api.aws/v1
Auth: AWS Bedrock API key as Bearer token (set via BEDROCK_MANTLE_API_KEY env var)
      or region-aware key via BEDROCK_MANTLE_{REGION}_API_KEY.
"""

from typing import Iterator, AsyncIterator, Any, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str

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
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        region = (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
        api_base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws/v1"
        )
        dynamic_api_key = api_key or get_secret_str("BEDROCK_MANTLE_API_KEY")
        return api_base, dynamic_api_key

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
