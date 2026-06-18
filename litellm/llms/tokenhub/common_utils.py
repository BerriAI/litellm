"""
Common utilities for TokenHub LLM provider
"""

from typing import Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class TokenHubError(BaseLLMException):
    """
    Custom exception class for TokenHub provider errors.
    """

    def __init__(
        self, status_code: int, message: str, headers: Optional[httpx.Headers] = None
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(
            status_code=status_code, message=message, headers=dict(self.headers)
        )
