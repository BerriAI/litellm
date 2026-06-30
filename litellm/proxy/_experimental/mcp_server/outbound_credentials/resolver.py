"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

`none` and `api_key` (shared-key source) are live, as is `authorization_code`, which reads the
user's token from the injected `OAuthTokenStore`. The remaining arms are `not_implemented` stubs
that each land in a follow-up PR with their seam. Pure v2: no imports from v1.
"""

from __future__ import annotations

import httpx
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
    OAuthTokenStore,
    TokenStoreUnavailable,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)


class _NullOAuthTokenStore:
    """Fail-closed default: with no token store wired, every user reads as not authorized."""

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        return None


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode.

    Collaborators (the per-mode credential stores and token fetchers) are injected as each arm is
    built; the live `none` and `api_key`-shared arms read from the config and need none, while
    `authorization_code` reads the user's token from the injected `OAuthTokenStore`.
    """

    def __init__(self, oauth_token_store: OAuthTokenStore | None = None) -> None:
        self._oauth_token_store: OAuthTokenStore = oauth_token_store or _NullOAuthTokenStore()

    async def resolve_credentials(self, subject: Subject, server: ServerSpec) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return Ok(NoOpAuth())
            case ApiKeyConfig() as config:
                return self._api_key(config)
            case PassthroughConfig():
                return _not_implemented(AuthSpecKind.passthrough)
            case ClientCredentialsConfig():
                return _not_implemented(AuthSpecKind.client_credentials)
            case TokenExchangeConfig():
                return _not_implemented(AuthSpecKind.token_exchange)
            case AuthorizationCodeConfig():
                return await self._authorization_code(subject, server)
            case AwsSigV4Config():
                return _not_implemented(AuthSpecKind.aws_sigv4)
        assert_never(server.config)

    async def has_user_token(self, subject: Subject, server: ServerSpec) -> bool:
        """Whether a usable per-user token exists for this server (the preemptive 401's check).

        Reads from the same per-user store as the ``authorization_code`` arm, so the discovery
        challenge and the egress agree on whether the user is authorized. Returns a typed ``bool``
        (no ``httpx.Auth``), unlike ``resolve_credentials``. A non-per-user mode has no token in the
        store, so it reads as False without a per-mode branch here.
        """
        return await self._authz_token(subject, server) is not None

    def _api_key(self, config: ApiKeyConfig) -> Result[httpx.Auth, CredError]:
        match config.key_source:
            case SharedKey() as source:
                header_name, header_value = config.header(source.value.get_secret_value())
                return Ok(StaticHeaderAuth(header_value, header_name=header_name))
            case Byok():
                # Per-user key pulled from the credential store; lands with that seam.
                return Error(CredError.of_not_implemented("api_key BYOK source not implemented yet"))
        assert_never(config.key_source)

    async def _authorization_code(self, subject: Subject, server: ServerSpec) -> Result[StaticHeaderAuth, CredError]:
        token = await self._authz_token(subject, server)
        if token is None:
            return Error(CredError.of_unauthorized("Authorization required: complete the OAuth flow for this server."))
        return Ok(StaticHeaderAuth(f"Bearer {token.access_token}", header_name="Authorization"))

    async def _authz_token(self, subject: Subject, server: ServerSpec) -> OAuthToken | None:
        """The user's authorization_code token, or None when absent or the store is unreachable.

        A store outage is mapped to None (the OAuth challenge), not raised, so a transient outage
        does not 500; it is the store, not this resolver, that declines to cache the failure.
        """
        try:
            return await self._oauth_token_store.fetch(subject.subject_id, server.server_id)
        except TokenStoreUnavailable:
            return None


def _not_implemented(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet"))
