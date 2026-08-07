"""Native OIDC login for the LiteLLM CLI.

The CLI authenticates directly against the identity provider the proxy
advertises, as a public native client using PKCE S256. It never handles a
client secret, and the proxy never brokers the identity-provider exchange.
"""

from .credentials import (
    AUTH_TYPE_NATIVE_OIDC,
    build_native_credential,
    is_native_credential,
    needs_refresh,
    refresh_native_credential,
    save_credential,
    verify_token_with_litellm,
)
from .errors import NativeOIDCAuthRejected, NativeOIDCError, NativeOIDCUnavailable
from .metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
    fetch_native_oidc_metadata,
    fetch_provider_metadata,
)
from .tokens import TokenResponse

__all__ = [  # mutable-ok: __all__ must be a list of names
    "AUTH_TYPE_NATIVE_OIDC",
    "NativeOIDCAuthRejected",
    "NativeOIDCError",
    "NativeOIDCMetadata",
    "NativeOIDCUnavailable",
    "ProviderMetadata",
    "TokenResponse",
    "build_native_credential",
    "fetch_native_oidc_metadata",
    "fetch_provider_metadata",
    "is_native_credential",
    "needs_refresh",
    "refresh_native_credential",
    "save_credential",
    "verify_token_with_litellm",
]
