"""
GigaChat Common Utils

Constants and exceptions for GigaChat provider.
"""

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# GigaChat API endpoints
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Default scope for personal API access
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Token expiry buffer in milliseconds (refresh token 60s before expiry)
TOKEN_EXPIRY_BUFFER_MS = 60000

# User-Agent for GigaChat requests
USER_AGENT = "GigaChat-python-lib"

def build_url(base: str, path: str) -> str:
    """
    Build URL from base and path.

    Works correctly with both trailing slash and without:
    - build_url("url/v1", "chat") → "url/v1/chat"
    - build_url("url/v1/", "chat") → "url/v1/chat"
    - build_url("url/v1", "/chat") → "url/v1/chat"
    """
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


class GigaChatError(BaseLLMException):
    """GigaChat API error."""

    pass

class GigaChatAuthError(BaseLLMException):
    """GigaChat authentication error."""

    pass

class GigaChatEmbeddingError(GigaChatError):
    """GigaChat Embedding API error."""

    pass