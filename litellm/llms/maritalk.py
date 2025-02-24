from typing import List, Optional, Union

from httpx._models import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class MaritalkError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class MaritalkConfig(OpenAIGPTConfig):

    def __init__(
        self,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List:
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return MaritalkError(
            status_code=status_code, message=error_message, headers=headers
        )
