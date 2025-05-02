"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as DataRobot is openai-compatible.
"""

from typing import Optional, Union, Tuple

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from ..common_utils import DataRobotException
from ...openai_like.chat.transformation import OpenAILikeChatConfig


class DataRobotConfig(OpenAILikeChatConfig):
    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("DATAROBOT_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("DATAROBOT_API_KEY") or ""
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

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
