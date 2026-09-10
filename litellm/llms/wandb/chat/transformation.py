"""
Wandb Chat Completions API - Transformation

This is OpenAI compatible - no translation needed / occurs
"""

from typing import Final

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class WandbConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        supported_params: Final = super().get_supported_openai_params(model)
        if litellm.supports_reasoning(model=model, custom_llm_provider="wandb"):
            return supported_params + ["reasoning_effort"]  # mutable-ok: inherited contract
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        map max_completion_tokens param to max_tokens
        """
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
