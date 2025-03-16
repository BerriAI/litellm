"""
Translates from OpenAI's `/v1/chat/completions` to BitdeerAI's `/v1/chat/completions`
"""
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class BitdeerAIChatConfig(OpenAIGPTConfig):

    def get_supported_openai_params(self, model: str) -> list:
        optional_params = super().get_supported_openai_params(model)
        return optional_params

    def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        if (
                "response_format" in mapped_openai_params
                and mapped_openai_params["response_format"] == {"type": "text"}
        ):
            mapped_openai_params.pop("response_format")
        return mapped_openai_params