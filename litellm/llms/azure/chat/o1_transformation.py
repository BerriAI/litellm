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

from litellm import verbose_logger
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_model_info

from ...openai.chat.o_series_transformation import OpenAIOSeriesConfig


class AzureOpenAIO1Config(OpenAIOSeriesConfig):
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

        if model is not None:
            try:
                model_info = get_model_info(
                    model=model, custom_llm_provider=custom_llm_provider
                )

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
        return "o1" in model or "o3" in model or "o_series/" in model

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
