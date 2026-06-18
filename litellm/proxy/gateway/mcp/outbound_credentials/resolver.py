"""The ONE credential resolver: `(subject, server) -> Result[httpx.Auth, CredError]`.

`resolve()` selects exactly one arm off the server's declared per-mode `config` and either
produces an `httpx.Auth` or fails closed with a typed `CredError`. There is no precedence
cascade and no silent downgrade; a header an attacker strips cannot change the mode.

The match is over the `AuthConfig` variant (not a separate enum), so each arm receives its
own fully-typed config with every field guaranteed present — no `None`-checks. The match is
wildcard-free with an `assert_never` tail, so adding a mode without an arm fails the type
gate, and a bypassed gate fails loudly at runtime instead of returning `None`.

Implemented: `none`, `passthrough`, `api_key` (shared from config, or per-user / BYOK pulled
from the injected `CredentialStore`), and `authorization_code` (per-user token read from the
injected `TokenStore`, refreshed proactively via the `TokenRefresher`). The `client_credentials`,
`token_exchange`, and `aws_sigv4` modes are typed stubs that fail closed until their
collaborators are injected.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
from typing_extensions import assert_never

from ..result import Error, Ok, Result
from .clock import Clock
from .credential_store import CredentialKey, CredentialStore
from .httpx_auth import NoOpAuth, StaticHeaderAuth
from .token_refresher import TokenRefresher
from .token_store import StoredToken, TokenKey, TokenStore
from .types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    PerUserEnvVar,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)

# Refresh proactively once the token is within this window of expiry.
_REFRESH_BUFFER = timedelta(seconds=60)


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode."""

    def __init__(
        self,
        credential_store: CredentialStore,
        token_store: TokenStore,
        token_refresher: TokenRefresher,
        clock: Clock,
    ) -> None:
        self._credential_store = credential_store
        self._token_store = token_store
        self._token_refresher = token_refresher
        self._clock = clock

    async def resolve(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match server.config:
            case AuthorizationCodeConfig() as config:
                return await self._authorization_code(subject, server, config)
            case ClientCredentialsConfig() as config:
                return await self._client_credentials(subject, server, config)
            case TokenExchangeConfig() as config:
                return await self._token_exchange(subject, server, config)
            case ApiKeyConfig() as config:
                return await self._api_key(subject, server, config)
            case PassthroughConfig() as config:
                return await self._passthrough(subject, server, config)
            case NoneConfig() as config:
                return await self._none(subject, server, config)
            case AwsSigV4Config() as config:
                return await self._aws_sigv4(subject, server, config)
        assert_never(server.config)

    # --- implemented arms -----------------------------------------------------------------
    async def _none(
        self, subject: Subject, server: ServerSpec, config: NoneConfig
    ) -> Result[httpx.Auth, CredError]:
        return Ok(NoOpAuth())

    async def _api_key(
        self, subject: Subject, server: ServerSpec, config: ApiKeyConfig
    ) -> Result[httpx.Auth, CredError]:
        match config.key_source:
            case SharedKey() as source:
                # The shared key is read straight from ServerSpec.config, not the per-user
                # store; it is the same credential for every caller.
                return Ok(
                    StaticHeaderAuth(config.header_for(source.value.get_secret_value()))
                )
            case PerUserEnvVar():
                value = await self._per_user_value(subject, server)
                if value is None:
                    return Error(
                        CredError.of_precondition_required(
                            "api_key: per-user env var not set for this subject"
                        )
                    )
                return Ok(StaticHeaderAuth(config.header_for(value)))
            case Byok():
                value = await self._per_user_value(subject, server)
                if value is None:
                    return Error(
                        CredError.of_unauthorized(
                            "api_key: no BYOK credential for this subject"
                        )
                    )
                return Ok(StaticHeaderAuth(config.header_for(value)))
        assert_never(config.key_source)

    async def _per_user_value(self, subject: Subject, server: ServerSpec) -> str | None:
        return await self._credential_store.get(
            CredentialKey(
                tenant_id=subject.tenant_id,
                subject_id=subject.subject_id,
                server_id=server.server_id,
            )
        )

    async def _passthrough(
        self, subject: Subject, server: ServerSpec, config: PassthroughConfig
    ) -> Result[httpx.Auth, CredError]:
        # The one arm that forwards a caller-supplied token, and only one the client obtained
        # for the upstream's audience — never a gateway-bound credential.
        if subject.inbound_token is None:
            return Error(
                CredError.of_unauthorized("passthrough: no inbound token to forward")
            )
        return Ok(
            StaticHeaderAuth(f"Bearer {subject.inbound_token.get_secret_value()}")
        )

    async def _authorization_code(
        self, subject: Subject, server: ServerSpec, config: AuthorizationCodeConfig
    ) -> Result[httpx.Auth, CredError]:
        # Per-user 3LO: read the stored token, refresh proactively near expiry, or fail closed
        # so the edge returns a 401 that starts the OAuth dance (the AS surface writes the token
        # this reads). The inbound caller bearer is never sent upstream.
        key = TokenKey(
            tenant_id=subject.tenant_id,
            subject_id=subject.subject_id,
            server_id=server.server_id,
            resource=server.resource,
        )
        token = await self._token_store.get(key)
        if token is None:
            return Error(
                CredError.of_unauthorized(
                    "authorization_code: no stored token; start the OAuth flow"
                )
            )
        if not self._is_near_expiry(token):
            return Ok(_bearer(token))
        if token.refresh_token is None:
            return Error(
                CredError.of_unauthorized(
                    "authorization_code: token expired with no refresh token; re-authenticate"
                )
            )
        refreshed = await self._token_refresher.refresh(config, token.refresh_token)
        match refreshed:
            case Ok(new_token):
                await self._token_store.put(key, new_token)
                return Ok(_bearer(new_token))
            case Error(err):
                return Error(err)
        assert_never(refreshed)

    def _is_near_expiry(self, token: StoredToken) -> bool:
        return self._clock.now() >= token.expires_at - _REFRESH_BUFFER

    # --- arms awaiting their collaborators (typed stubs, fail closed) ----------------------
    async def _client_credentials(
        self, subject: Subject, server: ServerSpec, config: ClientCredentialsConfig
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.client_credentials)

    async def _token_exchange(
        self, subject: Subject, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.token_exchange)

    async def _aws_sigv4(
        self, subject: Subject, server: ServerSpec, config: AwsSigV4Config
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.aws_sigv4)


def _bearer(token: StoredToken) -> StaticHeaderAuth:
    return StaticHeaderAuth(f"Bearer {token.access_token.get_secret_value()}")


def _todo(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet")
    )
