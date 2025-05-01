"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as DataRobot is openai-compatible.
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from ..common_utils import DataRobotException
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DataRobotConfig(OpenAIGPTConfig):
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return DataRobotException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Ensure that the url references a deployment or the DataRobot LLM Gateway

        """
        if api_base is None:
            api_base = "https://app.datarobot.com"

        # If the api_base is a deployment URL, we do not append the chat completions path
        if "api/v2/deployments" not in api_base:
            # If the api_base is not a deployment URL, we need to append the chat completions path
            if "api/v2/genai/llmgw/chat/completions" not in api_base:
                api_base += "/api/v2/genai/llmgw/chat/completions"

        # Ensure the url ends with a trailing slash
        if not api_base.endswith("/"):
            api_base += "/"

        return str(api_base)
