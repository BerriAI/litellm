from __future__ import annotations

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.resolvers.exceptions import (
    IdentityResolutionError,
    KeyNotFoundError,
    KeyNotInCacheError,
    NoDatabaseConnectionError,
    PrincipalMissingSourceKeyError,
)


def test_all_resolution_errors_share_one_base():
    errors = [
        NoDatabaseConnectionError(),
        KeyNotInCacheError("hashed"),
        KeyNotFoundError("hashed"),
        PrincipalMissingSourceKeyError(),
    ]
    assert all(isinstance(e, IdentityResolutionError) for e in errors)


def test_key_not_found_preserves_the_public_401_contract():
    # The auth seam catches ProxyException and rewrites the 401 message, so a
    # missing key must keep mapping to that exact contract.
    error = KeyNotFoundError("hashed-token")

    assert isinstance(error, ProxyException)
    assert error.code == "401"
    assert error.type == ProxyErrorTypes.token_not_found_in_db.value
    assert error.param == "key"


def test_key_not_in_cache_names_the_token():
    assert "hashed-token" in str(KeyNotInCacheError("hashed-token"))
