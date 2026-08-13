"""Nebius Token Factory Chat Completions API transformation."""

from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class NebiusConfig(OpenAIGPTConfig):
    API_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

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
