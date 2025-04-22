"""
Support for o1 and o3 model families

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param)
- Temperature => drop param (if user opts in to dropping param)
"""

from typing import List, Optional

import litellm
from litellm import verbose_logger
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_model_info

from ...openai.chat.o_series_transformation import OpenAIOSeriesConfig


class AzureOpenAIO1Config(OpenAIOSeriesConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the Azure O-Series models
        """
        all_openai_params = litellm.OpenAIGPTConfig().get_supported_openai_params(
            model=model
        )
        non_supported_params = [
            "logprobs",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
        ]

        o_series_only_param = ["reasoning_effort"]
        all_openai_params.extend(o_series_only_param)
        return [
            param for param in all_openai_params if param not in non_supported_params
        ]

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Currently no Azure O Series models support native streaming.
        """

        if stream is not True:
            return False

        if (
            model and "o3" in model
        ):  # o3 models support streaming - https://github.com/BerriAI/litellm/issues/8274
            return False

        if model is not None:
            try:
                model_info = get_model_info(
                    model=model, custom_llm_provider=custom_llm_provider
                )  # allow user to override default with model_info={"supports_native_streaming": true}

                if (
                    model_info.get("supports_native_streaming") is True
                ):  # allow user to override default with model_info={"supports_native_streaming": true}
                    return False
            except Exception as e:
                verbose_logger.debug(
                    f"Error getting model info in AzureOpenAIO1Config: {e}"
                )
        return True

    def is_o_series_model(self, model: str) -> bool:
        return "o1" in model or "o3" in model or "o4" in model or "o_series/" in model

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        model = model.replace(
            "o_series/", ""
        )  # handle o_series/my-random-deployment-name
        return super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
