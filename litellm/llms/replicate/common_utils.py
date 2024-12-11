from typing import Optional, Union

import httpx

from litellm.llms.base_llm.transformation import BaseLLMException


class ReplicateError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]],
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)
