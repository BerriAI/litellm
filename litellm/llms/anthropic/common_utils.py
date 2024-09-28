"""
This file contains common utils for anthropic calls.
"""

from typing import Optional

import httpx


class AnthropicError(Exception):
    def __init__(
        self,
        status_code: int,
        message,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message: str = message
        self.headers = headers
        self.request = httpx.Request(
            method="POST", url="https://api.anthropic.com/v1/messages"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs
