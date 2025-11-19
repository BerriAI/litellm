"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as BurnCloud AI is openai-compatible.

Docs: https://docs.burncloud.com/books/api
"""

from typing import List, Optional

from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class BurnCloudConfig(OpenAIGPTConfig):
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
        if api_base is None:
            raise ValueError(
                "Missing BurnCloud API Base - A call is being made to burncloud but no api_base is set either in the environment variables or via params"
            )

        if api_key is None:
            raise ValueError(
                "Missing BurnCloud API Key - A call is being made to burncloud but no api_key is set either in the environment variables or via params"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        headers["X-BurnCloud-Source"] = "litellm"
        return headers
