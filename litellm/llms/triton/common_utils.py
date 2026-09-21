import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class TritonError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)
