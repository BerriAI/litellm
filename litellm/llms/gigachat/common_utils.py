"""
GigaChat Common Utils

Constants and exceptions for GigaChat provider.
"""

from typing import Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str, str_to_bool

# GigaChat API endpoints
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Default scope for personal API access
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Token expiry buffer in milliseconds (refresh token 60s before expiry)
TOKEN_EXPIRY_BUFFER_MS = 60000

# User-Agent for GigaChat requests
USER_AGENT = "GigaChat-python-lib"


def get_gigachat_ssl_verify(ssl_verify: Optional[bool] = None) -> bool:
    """
    Determine the ssl_verify value for GigaChat httpx clients.

    Priority:
    1) Explicit function argument
    3) GIGACHAT_VERIFY_SSL_CERTS (bool-like string) [backwards-compatible alias]
    4) Default: False (GigaChat commonly uses self-signed certificates)
    """

    if ssl_verify is not None:
        return ssl_verify
    env_verify_certs = get_secret_str("GIGACHAT_VERIFY_SSL_CERTS")
    if env_verify_certs is not None:
        parsed = str_to_bool(env_verify_certs)
        if parsed is not None:
            return parsed

    return False


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