"""
Support for OpenAI's `/v1/chat/completions` endpoint.
"""

from typing import Optional, List

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class BurnCloudChatConfig(OpenAIGPTConfig):

    def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("BURNCLOUD_API_BASE")

        # Remove trailing slashes and ensure clean base URL
        api_base = api_base.rstrip("/")
        # if endswith "/v1"
        if api_base and api_base.endswith("/v1"):
            api_base = f"{api_base}/chat/completions"
        else:
            api_base = f"{api_base}/v1/chat/completions"

        return api_base

    def validate_environment(
            self,
            headers: dict,
            model: str,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("BURNCLOUD_API_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}