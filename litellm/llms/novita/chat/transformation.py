"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as Novita AI is openai-compatible.

Docs: https://novita.ai/docs/guides/llm-api
"""

from typing import List, Optional

from ....types.llms.openai import AllMessageValues
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class NovitaConfig(OpenAIGPTConfig):
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
            raise ValueError(
                "Missing Novita AI API Key - A call is being made to novita but no key is set either in the environment variables or via params"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        headers["X-Novita-Source"] = "litellm"
        return headers
