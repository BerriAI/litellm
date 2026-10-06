from __future__ import annotations

from typing import Final

from fastapi import status

from litellm.constants import INVALID_API_KEY_ERROR_MESSAGE
from litellm.proxy._types import ProxyErrorTypes, ProxyException


class IdentityResolutionError(Exception):
    """Base for every failure raised while resolving a caller's identity."""


class NoDatabaseConnectionError(IdentityResolutionError):
    def __init__(self) -> None:
        super().__init__("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")


class KeyNotInCacheError(IdentityResolutionError):
    def __init__(self, hashed_token: str) -> None:
        super().__init__(f"Key doesn't exist in cache + check_cache_only=True. key={hashed_token}.")


class KeyNotFoundError(IdentityResolutionError, ProxyException):
    """The token matched nothing in the cache or the verification token table.

    Also a ``ProxyException`` so the auth flow keeps mapping a missing key to the
    OpenAI 401 contract unchanged while callers migrate onto the typed hierarchy.

    Unauthenticated callers see the message verbatim, so the hash lives on
    ``hashed_token`` rather than in it.
    """

    def __init__(self, hashed_token: str) -> None:
        self.hashed_token: Final = hashed_token
        ProxyException.__init__(
            self,
            message=INVALID_API_KEY_ERROR_MESSAGE,
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=status.HTTP_401_UNAUTHORIZED,
        )


class PrincipalMissingSourceKeyError(IdentityResolutionError):
    def __init__(self) -> None:
        super().__init__("Principal carries no source key; it was not produced by IdentityStore.resolve")
