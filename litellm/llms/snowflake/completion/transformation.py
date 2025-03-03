'''
Support for Snowflake REST API 
'''
import httpx
from typing import List, Optional, Union, Any

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, TextCompletionResponse
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from ...openai_like.chat.transformation import OpenAILikeChatConfig


class SnowflakeConfig(OpenAILikeChatConfig):
    """
    source: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex

    The class `SnowflakeConfig` provides configuration for Snowflake's REST API interface. Below are the parameters:

        - `temperature` (float, optional): A value between 0 and 1 that controls randomness. Lower temperatures mean lower randomness. Default: 0

        - `top_p` (float, optional):  Limits generation at each step to top `k` most likely tokens. Default: 0

        - `max_tokens `(int, optional): The maximum number of tokens in the response. Default: 4096. Maximum allowed: 8192.

        - `guardrails` (bool, optional): Whether to enable Cortex Guard to filter potentially unsafe responses. Default: False.
        
        - `response_format` (str, optional): A JSON schema that the response should follow 
     """
    temperature: Optional[float]
    top_p: Optional[float]        
    max_tokens: Optional[int]   
    guardrails: Optional[bool]  
    response_format: Optional[str]
    
    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guardrails: Optional[bool] = None,
        response_format: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()
    
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "response_format"
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        If any supported_openai_params are in non_default_params, add them to optional_params, so they are used in API call

        Args:
            non_default_params (dict): Non-default parameters to filter.
            optional_params (dict): Optional parameters to update.
            model (str): Model name for parameter support check.

        Returns:
            dict: Updated optional_params with supported non-default parameters.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params