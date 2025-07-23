"""
Heroku Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""
import os

from typing import Optional, List, Union
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

# Base error class for Heroku
class HerokuError(Exception):
    pass

class HerokuChatConfig(OpenAIGPTConfig):
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    tool_choice: Optional[str] = None
    tools: Optional[list] = None
    top_p: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
        top_p: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "max_tokens",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "top_p",
            "tools",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
    
    def get_complete_url(self, api_base: Optional[str], api_key: Optional[str], model: str, optional_params: dict, litellm_params: dict, stream: Optional[bool] = None) -> str:
        api_base = api_base or os.getenv("HEROKU_API_BASE")
        if not api_base:
            raise HerokuError("No api base was set. Please provide an api_base, or set the HEROKU_API_BASE environment variable.")
        
        if not api_base.endswith("/v1/chat/completions"):
            api_base = f"{api_base}/v1/chat/completions"    
            
        return api_base