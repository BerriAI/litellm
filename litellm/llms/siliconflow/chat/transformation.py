from typing import Callable, Optional, cast

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import SiliconFlowException


class SiliconFlowChatConfig(OpenAIGPTConfig):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    CHAT_COMPLETIONS_ENDPOINT = "chat/completions"

    def get_supported_openai_params(self, model: str) -> list[str]:
        get_supported_openai_params = cast(
            Callable[[str], list[str]],
            super().get_supported_openai_params,
        )
        return [*get_supported_openai_params(model), "reasoning_effort"]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: Optional[bool] = None,
    ) -> str:
        complete_url = (
            api_base
            or get_secret_str("SILICONFLOW_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        if self.CHAT_COMPLETIONS_ENDPOINT in complete_url:
            return complete_url
        if complete_url.endswith("/v1"):
            return "{}/{}".format(complete_url, self.CHAT_COMPLETIONS_ENDPOINT)
        if complete_url == "https://api.siliconflow.cn":
            return "{}/v1/{}".format(complete_url, self.CHAT_COMPLETIONS_ENDPOINT)
        if "/v1" not in complete_url.split("//", 1)[-1]:
            return "{}/v1/{}".format(complete_url, self.CHAT_COMPLETIONS_ENDPOINT)
        return "{}/{}".format(complete_url, self.CHAT_COMPLETIONS_ENDPOINT)

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict[str, str]:
        final_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        if final_api_key is None:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        return {
            **headers,
            "Authorization": "Bearer {}".format(final_api_key),
            "Content-Type": "application/json",
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> SiliconFlowException:
        return SiliconFlowException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
