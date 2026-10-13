from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class V1MError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: object = None,
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


class V1MConfig(OpenAIGPTConfig):
    """Configuration for v1m (System One) calibrated decision engine.

    Website: https://v1m.ir
    """

    def __init__(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: inherited contract
        return super().get_supported_openai_params(model)
