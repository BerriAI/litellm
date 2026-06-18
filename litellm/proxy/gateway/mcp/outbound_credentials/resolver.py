"""The ONE credential resolver: `(subject, server) -> Result[httpx.Auth, CredError]`.

`resolve()` selects exactly one arm off the server's declared per-mode `config` and either
produces an `httpx.Auth` or fails closed with a typed `CredError`. There is no precedence
cascade and no silent downgrade; a header an attacker strips cannot change the mode.

The match is over the `AuthConfig` variant (not a separate enum), so each arm receives its
own fully-typed config with every field guaranteed present — no `None`-checks. The match is
wildcard-free with an `assert_never` tail, so adding a mode without an arm fails the type
gate, and a bypassed gate fails loudly at runtime instead of returning `None`.

Implemented: `none`, `passthrough`, `api_key` (shared from config, or per-user / BYOK pulled
from the injected `CredentialStore`), `authorization_code` (per-user token read from the
injected `TokenStore`, refreshed proactively via the `TokenRefresher`), and `client_credentials`
(shared service-account token cached in the `ServiceTokenStore`, minted by the
`ClientCredentialsFetcher`), and `token_exchange` (RFC 8693 OBO: the caller's inbound token is
swapped for an upstream-audience token via the `TokenExchanger`, cached per-user). The
`aws_sigv4` mode is a typed stub that fails closed until its signer is injected.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
from typing_extensions import assert_never

from ..result import Error, Ok, Result
from .client_credentials_fetcher import ClientCredentialsFetcher
from .clock import Clock
from .credential_store import CredentialKey, CredentialStore
from .httpx_auth import NoOpAuth, StaticHeaderAuth
from .service_token_store import ServiceTokenKey, ServiceTokenStore
from .token_exchanger import TokenExchanger
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
        service_token_store: ServiceTokenStore,
        client_credentials_fetcher: ClientCredentialsFetcher,
        token_exchanger: TokenExchanger,
    ) -> None:
        self._credential_store = credential_store
        self._token_store = token_store
        self._token_refresher = token_refresher
        self._clock = clock
        self._service_token_store = service_token_store
        self._client_credentials_fetcher = client_credentials_fetcher
        self._token_exchanger = token_exchanger

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
                fetched = await self._per_user_value(subject, server)
                if isinstance(fetched, Error):
                    return Error(fetched.error)  # store/DB down -> 503
                value = fetched.ok
                if value is None:
                    return Error(
                        CredError.of_precondition_required(
                            "api_key: per-user env var not set for this subject"
                        )
                    )
                return Ok(StaticHeaderAuth(config.header_for(value)))
            case Byok():
                fetched = await self._per_user_value(subject, server)
                if isinstance(fetched, Error):
                    return Error(fetched.error)  # store/DB down -> 503
                value = fetched.ok
                if value is None:
                    return Error(
                        CredError.of_unauthorized(
                            "api_key: no BYOK credential for this subject"
                        )
                    )
                return Ok(StaticHeaderAuth(config.header_for(value)))
        assert_never(config.key_source)

    async def _per_user_value(
        self, subject: Subject, server: ServerSpec
    ) -> Result[str | None, CredError]:
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
        fetched = await self._token_store.get(key)
        if isinstance(fetched, Error):
            return Error(fetched.error)  # store/DB down -> 503
        token = fetched.ok
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
        if isinstance(refreshed, Error):
            return Error(
                refreshed.error
            )  # refresh rejected (401) or endpoint down (503)
        new_token = refreshed.ok
        persisted = await self._token_store.put(key, new_token)
        if isinstance(persisted, Error):
            return Error(persisted.error)  # store/DB down on write -> 503
        return Ok(_bearer(new_token))

    def _is_near_expiry(self, token: StoredToken) -> bool:
        return self._clock.now() >= token.expires_at - _REFRESH_BUFFER

    async def _client_credentials(
        self, subject: Subject, server: ServerSpec, config: ClientCredentialsConfig
    ) -> Result[httpx.Auth, CredError]:
        # M2M: one shared service-account token per (server, resource), cached without a
        # subject; the caller bearer is never read. The cache is best-effort - re-mint on a
        # read failure, ignore a write failure - because the token is always re-mintable.
        key = ServiceTokenKey(server_id=server.server_id, resource=server.resource)
        cached = await self._service_token_store.get(key)
        if isinstance(cached, Ok):
            token = cached.ok
            if token is not None and not self._is_near_expiry(token):
                return Ok(_bearer(token))
        fetched = await self._client_credentials_fetcher.fetch(config)
        if isinstance(fetched, Error):
            return Error(fetched.error)  # rejected creds -> 500, endpoint down -> 503
        minted = fetched.ok
        await self._service_token_store.put(
            key, minted
        )  # best-effort; token is valid anyway
        return Ok(_bearer(minted))

    async def _token_exchange(
        self, subject: Subject, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[httpx.Auth, CredError]:
        # OBO: swap the caller's live inbound token for a token bound to server.resource at the
        # IdP's exchange endpoint. The inbound token is sent ONLY to the exchanger, never to the
        # upstream; the upstream gets the exchanged token. Per-user cache, best-effort like M2M
        # (read failure re-exchanges, write failure is ignored) since it is re-exchangeable.
        if subject.inbound_token is None:
            return Error(
                CredError.of_unauthorized(
                    "token_exchange: no inbound token to exchange; authenticate to the gateway"
                )
            )
        key = TokenKey(
            tenant_id=subject.tenant_id,
            subject_id=subject.subject_id,
            server_id=server.server_id,
            resource=server.resource,
        )
        cached = await self._token_store.get(key)
        if isinstance(cached, Ok):
            token = cached.ok
            if token is not None and not self._is_near_expiry(token):
                return Ok(_bearer(token))
        exchanged = await self._token_exchanger.exchange(
            config, subject.inbound_token, server.resource
        )
        if isinstance(exchanged, Error):
            return Error(
                exchanged.error
            )  # subject_token bad -> 401, config -> 500, down -> 503
        minted = exchanged.ok
        await self._token_store.put(
            key, minted
        )  # best-effort; token is re-exchangeable
        return Ok(_bearer(minted))

    # --- arms awaiting their collaborators (typed stubs, fail closed) ----------------------
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
