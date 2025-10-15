"""Support for Azure OpenAI gpt-5 model family."""

from typing import List

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.types.llms.openai import AllMessageValues

from .gpt_transformation import AzureOpenAIConfig


class AzureOpenAIGPT5Config(AzureOpenAIConfig, OpenAIGPT5Config):
    """Azure specific handling for gpt-5 models."""

    GPT5_SERIES_ROUTE = "gpt5_series/"

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        """Check if the Azure model string refers to a gpt-5 variant.

        Accepts both explicit gpt-5 model names and the ``gpt5_series/`` prefix
        used for manual routing.
        """
        return "gpt-5" in model or "gpt5_series" in model

    def get_supported_openai_params(self, model: str) -> List[str]:
        return OpenAIGPT5Config.get_supported_openai_params(self, model=model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        api_version: str = "",
    ) -> dict:
        return OpenAIGPT5Config.map_openai_params(
            self,
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        model = model.replace(self.GPT5_SERIES_ROUTE, "")
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
