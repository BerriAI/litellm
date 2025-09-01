"""
Sambanova Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Optional, Union

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class SambanovaConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.sambanova.ai/cloud/api-reference/

    Below are the parameters:
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    tool_choice: Optional[str] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        stop: Optional[str] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """
        from litellm.utils import supports_function_calling

        params = [
            "max_completion_tokens",
            "max_tokens",
            "response_format",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "top_k",
        ]

        if supports_function_calling(model, custom_llm_provider="sambanova"):
            params.append("tools")
            params.append("tool_choice")
            params.append("parallel_tool_calls")

        return params

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
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
