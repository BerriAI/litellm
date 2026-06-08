"""API-key principal extraction.

Wraps the existing ``get_api_key`` (header extraction) and
``UserAPIKeyAuth._safe_hash_litellm_api_key`` (hashing, including the
"hashed-jwt-..." prefix for JWT-shaped values).
"""

from typing import Optional

from litellm.identity.principal import ApiKeyPrincipal


def hash_principal_token(api_key: str) -> str:
    """Hash an API key the same way the legacy auth path does.

    Centralized so future principal types (e.g. SSO-issued ephemeral
    keys) can reuse the same hashing without re-importing the Pydantic
    model.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth._safe_hash_litellm_api_key(api_key)


def extract_api_key_principal(api_key: Optional[str]) -> Optional[ApiKeyPrincipal]:
    """Build an ``ApiKeyPrincipal`` from a raw API key string.

    Returns ``None`` when no key is supplied. Callers that need to pull the
    key out of a request should call ``get_api_key`` first.
    """
    if not api_key:
        return None
    return ApiKeyPrincipal(token_hash=hash_principal_token(api_key))
