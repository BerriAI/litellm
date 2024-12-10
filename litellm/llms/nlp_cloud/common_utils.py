from typing import Optional

from litellm.llms.base_llm.transformation import BaseLLMException


class NLPCloudError(BaseLLMException):
    def __init__(self, status_code: int, message: str, headers: Optional[dict] = None):
        super().__init__(status_code=status_code, message=message, headers=headers)
