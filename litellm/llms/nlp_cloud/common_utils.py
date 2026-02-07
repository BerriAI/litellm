from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class NLPCloudError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)
