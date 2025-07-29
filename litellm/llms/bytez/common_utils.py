from typing import Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

API_BASE = "https://api.bytez.com/models/v2"


class BytezError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url=API_BASE)
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )