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

from ...openai.chat.o1_transformation import OpenAIO1Config


class AzureOpenAIO1Config(OpenAIO1Config):
    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        if stream is not True:
            return False

        return True

    def is_o1_model(self, model: str) -> bool:
        o1_models = ["o1-mini", "o1-preview"]
        for m in o1_models:
            if m in model:
                return True
        return False
