"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

`none`, `api_key` (shared-key source), and `passthrough` (forwards the caller's own inbound token)
are live, as is `authorization_code`, which reads the user's token from the injected
`OAuthTokenStore`, and `token_exchange`, which swaps the caller's inbound token through the
injected `TokenExchanger`. The remaining arms are `not_implemented` stubs that each land in a
follow-up PR with their seam. Pure v2: no imports from v1.
"""

from __future__ import annotations

import httpx
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_source import (
    IdpSubjectTokenSource,
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
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    TokenExchanger,
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


class _NullTokenExchanger:
    """Fail-closed default: with no exchanger wired, token_exchange cannot produce a credential."""

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> Result[OAuthToken, CredError]:
        return Error(CredError.of_misconfigured("token exchange collaborator not wired"))

    async def invalidate(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> None:
        return None


class _NullIdpSubjectTokenSource:
    """Fail-closed default: with no source wired, a delegated OBO request has no subject token, so
    the token_exchange arm challenges rather than falling back to the agent's admission credential."""

    async def subject_token(self, user_id: str, config: TokenExchangeConfig) -> str | None:
        return None


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode.

    Collaborators (the per-mode credential stores and token fetchers) are injected as each arm is
    built; the live `none` and `api_key`-shared arms read from the config and need none, while
    `authorization_code` reads the user's token from the injected `OAuthTokenStore` and
    `token_exchange` swaps the caller's token through the injected `TokenExchanger`.
    """

    def __init__(
        self,
        oauth_token_store: OAuthTokenStore | None = None,
        token_exchanger: TokenExchanger | None = None,
        idp_subject_source: IdpSubjectTokenSource | None = None,
    ) -> None:
        self._oauth_token_store: OAuthTokenStore = oauth_token_store or _NullOAuthTokenStore()
        self._token_exchanger: TokenExchanger = token_exchanger or _NullTokenExchanger()
        self._idp_subject_source: IdpSubjectTokenSource = idp_subject_source or _NullIdpSubjectTokenSource()

    async def resolve_credentials(self, subject: Subject, server: ServerSpec) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return Ok(NoOpAuth())
            case ApiKeyConfig() as config:
                return self._api_key(config)
            case PassthroughConfig():
                return self._passthrough(subject)
            case ClientCredentialsConfig():
                return _not_implemented(AuthSpecKind.client_credentials)
            case TokenExchangeConfig() as config:
                return await self._token_exchange(subject, server, config)
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

    def _passthrough(self, subject: Subject) -> Result[httpx.Auth, CredError]:
        """Forward the caller's own upstream credential verbatim; the gateway mints nothing.

        The inbound token is the caller's already-disambiguated ``Authorization`` (never the LiteLLM
        admission credential; the edge adapter drops that before building the ``Subject``). When it is
        absent the request is sent unauthenticated so the upstream's own 401 surfaces, rather than the
        gateway challenging on the upstream's behalf.
        """
        if subject.inbound_token is None:
            return Ok(NoOpAuth())
        return Ok(StaticHeaderAuth(subject.inbound_token.get_secret_value(), header_name="Authorization"))

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

    async def _token_exchange(
        self, subject: Subject, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[StaticHeaderAuth, CredError]:
        """RFC 8693 OBO: exchange the subject's token for an upstream-bound bearer.

        The subject token is the caller's own inbound token for a direct request, or the delegated
        user's IdP token (Path B) for an agent-delegated one; it is never the agent's admission
        credential (see ``_obo_subject_token``). A missing subject token fails closed with a 401 rather
        than falling through to a weaker source (§1.5); the exchanger handles the IdP round-trip and
        caching and returns the upstream token or a typed error.
        """
        subject_token = await self._obo_subject_token(subject, config)
        if isinstance(subject_token, Error):
            return Error(subject_token.error)
        match await self._token_exchanger.exchange(subject_token.ok, server, config, tenant_id=subject.tenant_id):
            case Ok(token):
                return Ok(StaticHeaderAuth(f"Bearer {token.access_token}", header_name="Authorization"))
            case Error(err):
                return Error(err)

    async def _obo_subject_token(self, subject: Subject, config: TokenExchangeConfig) -> Result[str, CredError]:
        """The subject_token to exchange (OBO): the caller's own token, or the delegated user's IdP
        token, never the agent's admission credential.

        For a direct (non-delegated) request the subject presents its own token inline, so the inbound
        token IS the subject's proof and is exchanged as before; a missing one is the arm's 401. For an
        agent-delegated request (``delegated_user_id`` set) the inbound token is the agent's admission
        credential and must not be exchanged (exchanging it would mint agent-scoped access, the exact
        escalation this design prevents); the delegated user's IdP grant is sourced instead (Path B),
        and its absence fails closed with a 401 telling the client to authenticate the user with the
        IdP. The agent's token is never a fallback.
        """
        if subject.delegated_user_id is None:
            inbound = subject.inbound_token
            if inbound is None:
                return Error(
                    CredError.of_unauthorized(
                        "Token exchange requires a caller token to exchange (OBO).",
                        www_authenticate='Bearer error="invalid_request"',
                    )
                )
            return Ok(inbound.get_secret_value())
        subject_token = await self._idp_subject_source.subject_token(subject.delegated_user_id, config)
        if subject_token is None:
            return Error(
                CredError.of_unauthorized(
                    "Delegated user has not authorized the IdP for on-behalf-of access; complete consent and retry.",
                    www_authenticate='Bearer error="invalid_token"',
                )
            )
        return Ok(subject_token)

    async def invalidate_credentials(self, subject: Subject, server: ServerSpec) -> None:
        """Drop any cached credential the resolver owns for this `(subject, server)`.

        Used after an upstream rejects the injected credential, so the next resolve re-mints rather
        than serving the same rejected token until TTL. Only `token_exchange` holds a re-mintable
        cached credential here; other modes are a no-op. The exchange cache is keyed on the
        subject_token that minted it, so invalidation recomputes the key from the same source
        `_token_exchange` uses (the delegated user's IdP token, or the caller's own inbound token),
        never the agent's admission credential. This is best-effort: if the delegated user's IdP token
        has since rotated the recomputed key no longer matches the rejected entry, which then lapses on
        its own TTL rather than being dropped early.
        """
        if not isinstance(server.config, TokenExchangeConfig):
            return
        match await self._obo_subject_token(subject, server.config):
            case Error(_):
                return
            case Ok(subject_token):
                await self._token_exchanger.invalidate(
                    subject_token, server, server.config, tenant_id=subject.tenant_id
                )

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
