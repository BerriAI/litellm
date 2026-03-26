"""
HPC-AI Chat Completions API — OpenAI-compatible endpoint.

Reference: https://api.hpc-ai.com/inference/v1
"""

from typing import Optional, Tuple

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class HpcAiConfig(OpenAIGPTConfig):
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map max_completion_tokens to max_tokens for OpenAI-compatible API."""
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("HPC_AI_API_BASE")
            or "https://api.hpc-ai.com/inference/v1"
        )
        dynamic_api_key = api_key or get_secret_str("HPC_AI_API_KEY")
        return api_base, dynamic_api_key
