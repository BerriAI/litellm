"""Support for Azure OpenAI gpt-5 model family."""

from typing import List

import litellm
from litellm.exceptions import UnsupportedParamsError
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
        reasoning_effort_value = (
            non_default_params.get("reasoning_effort")
            or optional_params.get("reasoning_effort")
        )

        # gpt-5.1 supports reasoning_effort='none', but other gpt-5 models don't
        # See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning
        is_gpt_5_1 = self.is_model_gpt_5_1_model(model)

        if reasoning_effort_value == "none" and not is_gpt_5_1:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                non_default_params = non_default_params.copy()
                optional_params = optional_params.copy()
                if non_default_params.get("reasoning_effort") == "none":
                    non_default_params.pop("reasoning_effort")
                if optional_params.get("reasoning_effort") == "none":
                    optional_params.pop("reasoning_effort")
            else:
                raise UnsupportedParamsError(
                    status_code=400,
                    message=(
                        "Azure OpenAI does not support reasoning_effort='none' for this model. "
                        "Supported values are: 'low', 'medium', and 'high'. "
                        "To drop this parameter, set `litellm.drop_params=True` or for proxy:\n\n"
                        "`litellm_settings:\n drop_params: true`\n"
                        "Issue: https://github.com/BerriAI/litellm/issues/16704"
                    ),
                )

        result = OpenAIGPT5Config.map_openai_params(
            self,
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Only drop reasoning_effort='none' for non-gpt-5.1 models
        if result.get("reasoning_effort") == "none" and not is_gpt_5_1:
            result.pop("reasoning_effort")

        return result

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
