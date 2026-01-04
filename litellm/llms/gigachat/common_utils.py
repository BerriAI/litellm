"""
GigaChat Common Utilities

Constants, exceptions, and helper functions for GigaChat provider.
Based on official GigaChat SDK settings.
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# API Endpoints (from gigachat SDK settings.py)
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Default scope for personal API access
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Token expiry buffer in milliseconds (refresh token 60s before expiry)
TOKEN_EXPIRY_BUFFER_MS = 60000

# Attachment limits (from GigaChat API documentation)
# Note: attachments is an array but only 1 image per message is supported
MAX_IMAGES_PER_MESSAGE = 1
MAX_TOTAL_IMAGES = 10


class GigaChatError(BaseLLMException):
    """Base exception for GigaChat errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
            request=request,
            response=response,
        )


class GigaChatAuthError(GigaChatError):
    """Authentication error."""

    pass
