"""
Common utilities for Volcengine LLM provider
"""

from typing import Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class VolcEngineError(BaseLLMException):
    """
    Custom exception class for Volcengine provider errors.
    """

    def __init__(self, status_code: int, message: str, headers: httpx.Headers | None = None):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(status_code=status_code, message=message, headers=dict(self.headers))


def get_volcengine_base_url(api_base: str | None = None) -> str:
    """
    Get the base URL for Volcengine API calls.

    Args:
        api_base: Optional custom API base URL

    Returns:
        The base URL to use for API calls
    """
    if api_base:
        return api_base
    return "https://ark.cn-beijing.volces.com"


def get_volcengine_headers(api_key: str, extra_headers: dict | None = None) -> dict:
    """
    Get headers for Volcengine API calls.

    Args:
        api_key: The API key for authentication
        extra_headers: Optional additional headers

    Returns:
        Dictionary of headers
    """
    headers: Final = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if extra_headers:
        headers.update(extra_headers)

    return headers
