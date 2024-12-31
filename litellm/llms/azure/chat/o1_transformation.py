"""
Support for o1 model family 

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param)
- Temperature => drop param (if user opts in to dropping param)
"""

from typing import Optional

from litellm import verbose_logger
from litellm.utils import get_model_info

from ...openai.chat.o1_transformation import OpenAIO1Config


class AzureOpenAIO1Config(OpenAIO1Config):
    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Currently no Azure OpenAI models support native streaming.
        """
        if stream is not True:
            return False

        if model is not None:
            try:
                model_info = get_model_info(
                    model=model, custom_llm_provider=custom_llm_provider
                )
                if model_info.get("supports_native_streaming") is True:
                    return False
            except Exception as e:
                verbose_logger.debug(
                    f"Error getting model info in AzureOpenAIO1Config: {e}"
                )

        return True

    def is_o1_model(self, model: str) -> bool:
        o1_models = ["o1-mini", "o1-preview"]
        for m in o1_models:
            if m in model:
                return True
        return False
