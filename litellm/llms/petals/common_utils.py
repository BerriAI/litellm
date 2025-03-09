from typing import Union

from httpx import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class PetalsError(BaseLLMException):
    def __init__(self, status_code: int, message: str, headers: Union[dict, Headers]):
        super().__init__(status_code=status_code, message=message, headers=headers)
