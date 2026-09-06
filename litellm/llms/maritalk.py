from typing import Final

from httpx._models import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class MaritalkError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class MaritalkConfig(OpenAIGPTConfig):
    def __init__(
        self,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        stop: list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "frequency_penalty",
            "presence_penalty",
            "top_p",
            "top_k",
            "temperature",
            "max_tokens",
            "n",
            "stop",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
        ]

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return MaritalkError(status_code=status_code, message=error_message, headers=headers)
