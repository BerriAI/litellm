"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

`none`, `api_key` (shared-key source), and `passthrough` (forwards the caller's own inbound token)
are live, as is `authorization_code`, which reads the user's token from the injected
`OAuthTokenStore`, `token_exchange`, which swaps the caller's inbound token through the injected
`TokenExchanger`, `client_credentials`, which mints and caches the gateway's M2M token through the
injected `ClientCredentialsTokenSource`, and `id_jag`, which runs the two-leg identity-assertion
grant against a subject token taken from the request or from the injected `SSOAssertionStore`. The
remaining arms are `not_implemented` stubs that each land in a follow-up PR with their seam. Pure
v2: no imports from v1.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import partial
from typing import Final

import httpx
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.client_credentials import (
    ClientCredentialsBearerAuth,
    ClientCredentialsTokenSource,
)
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
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    AssertionStoreUnavailable,
    DbSSOAssertionStore,
    SSOAssertionStore,
    SSOIdentityAssertion,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (
    ExchangedToken,
    ExchangedTokenCache,
    TokenEndpointClient,
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
    ClientAuth,
    ClientCredentialsConfig,
    ClientSecretAuth,
    CredError,
    IdJagConfig,
    NoneConfig,
    PassthroughConfig,
    PrivateKeyJwtAuth,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)

_TOKEN_EXCHANGE_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_BEARER_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_ID_JAG_REQUESTED_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:id-jag"


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
        token_endpoint: TokenEndpointClient | None = None,
        exchanged_tokens: ExchangedTokenCache | None = None,
        client_credentials_source: ClientCredentialsTokenSource | None = None,
        sso_assertion_store: SSOAssertionStore | None = None,
    ) -> None:
        self._oauth_token_store: OAuthTokenStore = oauth_token_store or _NullOAuthTokenStore()
        self._token_exchanger: TokenExchanger = token_exchanger or _NullTokenExchanger()
        self._token_endpoint: TokenEndpointClient = token_endpoint or TokenEndpointClient()
        self._exchanged_tokens: ExchangedTokenCache = exchanged_tokens or ExchangedTokenCache()
        self._client_credentials_source = client_credentials_source or ClientCredentialsTokenSource()
        self._sso_assertion_store: SSOAssertionStore = sso_assertion_store or DbSSOAssertionStore()

    async def resolve_credentials(self, subject: Subject, server: ServerSpec) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return Ok(NoOpAuth())
            case ApiKeyConfig() as config:
                return self._api_key(config)
            case PassthroughConfig():
                return self._passthrough(subject)
            case ClientCredentialsConfig() as config:
                return await self._client_credentials(server.server_id, config)
            case TokenExchangeConfig() as config:
                return await self._token_exchange(subject, server, config)
            case IdJagConfig() as config:
                return await self._id_jag(subject, server, config)
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

    async def _id_jag(self, subject: Subject, server: ServerSpec, config: IdJagConfig) -> Result[httpx.Auth, CredError]:
        match await self._id_jag_subject_token(subject):
            case Error(err):
                return Error(err)
            case Ok(subject_token):
                return await self._id_jag_exchange(subject, subject_token, server, config)

    async def _id_jag_subject_token(self, subject: Subject) -> Result[str, CredError]:
        """The identity token ID-JAG leg 1 asserts, from the request or from the SSO login it was captured at.

        A caller that presents its own IdP identity token wins: that is the strongest available
        assertion of who is calling. Otherwise the subject is the assertion captured for this user
        at LiteLLM SSO login, which is what lets an agent holding a brokered LiteLLM credential
        reach an upstream as the user it was issued for. The user is always taken from the
        authenticated principal, never from a caller-supplied field, so no caller can select whose
        identity is asserted upstream.

        Every miss is ``precondition_required`` (412) rather than a fall-through to a weaker
        credential: ID-JAG exists to assert a specific user, so a missing subject has no safe
        substitute. A store outage is the one exception: it is ``upstream_unavailable`` (503), not
        412, because the user has nothing to fix by signing in again, and it is a value rather than
        a raised error so a DB blip cannot 500 the egress or the upstream-401 retry.
        """
        if subject.inbound_token is not None:
            return Ok(subject.inbound_token.get_secret_value())
        if not subject.subject_id:
            return Error(
                CredError.of_precondition_required(
                    "ID-JAG requires an identified caller; this request carries neither an "
                    "identity token nor a resolved LiteLLM user."
                )
            )
        try:
            assertion: Final = await self._sso_assertion_store.fetch(subject.subject_id)
        except AssertionStoreUnavailable as exc:
            # The driver's message can name hosts, schemas or connection details, and this summary
            # is returned to the caller verbatim as a 503 body. Operators get it from the log.
            verbose_proxy_logger.warning(
                "ID-JAG: the IdP identity assertion store is unreachable for user_id=%s: %s",
                subject.subject_id,
                exc,
            )
            return Error(
                CredError.of_upstream_unavailable(
                    "The IdP identity assertion store is unreachable, so ID-JAG cannot resolve a subject."
                )
            )
        if assertion is None:
            return Error(
                CredError.of_precondition_required(
                    "ID-JAG requires an IdP identity assertion for this user and none is stored. "
                    "Sign in through LiteLLM SSO so the gateway captures one."
                )
            )
        if _assertion_expired(assertion, datetime.now(timezone.utc)):
            return Error(
                CredError.of_precondition_required(
                    "The stored IdP identity assertion for this user has expired. Sign in through "
                    "LiteLLM SSO again to capture a current one."
                )
            )
        return Ok(assertion.id_token.get_secret_value())

    async def _id_jag_exchange(
        self, subject: Subject, token: str, server: ServerSpec, config: IdJagConfig
    ) -> Result[httpx.Auth, CredError]:
        slot: Final = _id_jag_slot_key(subject, server)
        fingerprint: Final = _id_jag_fingerprint(token, server.server_id, config)

        async def _exchange() -> Result[ExchangedToken, CredError]:
            leg1_params: Final = {
                "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
                "requested_token_type": _ID_JAG_REQUESTED_TOKEN_TYPE,
                "subject_token": token,
                "subject_token_type": config.subject_token_type,
                **({"audience": config.audience} if config.audience else {}),
                **({"resource": config.resource} if config.resource else {}),
                **({"scope": " ".join(config.scopes)} if config.scopes else {}),
            }
            match await self._token_endpoint.fetch(
                config.org_token_endpoint,
                config.client_id,
                leg1_params,
                config.client_auth,
            ):
                case Error(err):
                    return Error(err)
                case Ok(id_jag):
                    leg2_params: Final = {
                        "grant_type": _JWT_BEARER_GRANT_TYPE,
                        "assertion": id_jag.access_token,
                    }
                    return await self._token_endpoint.fetch(
                        config.resource_token_endpoint,
                        config.client_id,
                        leg2_params,
                        config.client_auth,
                    )

        match await self._exchanged_tokens.get_or_compute(slot, _exchange, fingerprint=fingerprint):
            case Ok(access_token):
                return Ok(StaticHeaderAuth(f"Bearer {access_token}"))
            case Error(err):
                return Error(err)

    async def _authorization_code(self, subject: Subject, server: ServerSpec) -> Result[StaticHeaderAuth, CredError]:
        token: Final = await self._authz_token(subject, server)
        if token is None:
            return Error(CredError.of_unauthorized("Authorization required: complete the OAuth flow for this server."))
        return Ok(StaticHeaderAuth(f"Bearer {token.access_token}", header_name="Authorization"))

    async def _client_credentials(
        self, server_id: str, config: ClientCredentialsConfig
    ) -> Result[httpx.Auth, CredError]:
        """The M2M arm: resolve a cached (or freshly minted) gateway token; no user context.

        The token is resolved here, before any upstream request, so a misconfigured grant or an
        unreachable IdP surfaces as a typed ``CredError``. The returned auth carries the source's
        ``refetch``, so an upstream 401 is retried exactly once with a freshly minted token (the
        contract's invalid-token recovery); a second 401 surfaces the upstream's own error.
        """
        match await self._client_credentials_source.get(server_id, config):
            case Ok(token):
                refetch: Final = partial(self._client_credentials_source.refetch, server_id, config)
                return Ok(ClientCredentialsBearerAuth(token.access_token, refetch))
            case Error(err):
                return Error(err)

    async def _token_exchange(
        self, subject: Subject, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[StaticHeaderAuth, CredError]:
        """RFC 8693 OBO: exchange the caller's inbound token for an upstream-bound bearer.

        No inbound token means there is nothing to exchange, so the arm fails closed with a 401 rather
        than falling through to a weaker source (§1.5); the exchanger handles the IdP round-trip and
        caching and returns the upstream token or a typed error.
        """
        inbound: Final = subject.inbound_token
        if inbound is None:
            return Error(
                CredError.of_unauthorized(
                    "Token exchange requires a caller token to exchange (OBO).",
                    www_authenticate='Bearer error="invalid_request"',
                )
            )
        match await self._token_exchanger.exchange(
            inbound.get_secret_value(), server, config, tenant_id=subject.tenant_id
        ):
            case Ok(token):
                return Ok(StaticHeaderAuth(f"Bearer {token.access_token}", header_name="Authorization"))
            case Error(err):
                return Error(err)

    async def invalidate_credentials(self, subject: Subject, server: ServerSpec) -> None:
        """Drop any cached credential the resolver owns for this `(subject, server)`.

        Used after an upstream rejects the injected credential, so the next resolve re-mints rather
        than serving the same rejected token until TTL. `token_exchange` and `id_jag` hold a
        re-mintable cached credential here; `client_credentials` recovers inside its own auth flow
        (`ClientCredentialsBearerAuth` retries the 401'd request once with a fresh token), and
        other modes are a no-op.

        `id_jag` evicts by a slot key derived from the principal, so it needs no lookup against the
        assertion store on this path; the fingerprint stored beside the entry is what keeps a slot
        shared between callers safe.
        """
        if isinstance(server.config, IdJagConfig):
            self._invalidate_id_jag(subject, server)
        elif isinstance(server.config, TokenExchangeConfig) and subject.inbound_token is not None:
            await self._token_exchanger.invalidate(
                subject.inbound_token.get_secret_value(), server, server.config, tenant_id=subject.tenant_id
            )

    def _invalidate_id_jag(self, subject: Subject, server: ServerSpec) -> None:
        """Evict the bearer this `(subject, server)` last resolved, without depending on the store.

        The slot is addressed by the principal (plus the caller's own token when it presented one),
        never by the credential material, so it stays computable when the assertion store is down.
        The fingerprint stored with the entry is what keeps that safe: an entry minted for different
        inputs reads as a miss rather than being served.
        """
        self._exchanged_tokens.invalidate(_id_jag_slot_key(subject, server))

    async def _authz_token(self, subject: Subject, server: ServerSpec) -> OAuthToken | None:
        """The user's authorization_code token, or None when absent or the store is unreachable.

        A store outage is mapped to None (the OAuth challenge), not raised, so a transient outage
        does not 500; it is the store, not this resolver, that declines to cache the failure.
        """
        try:
            return await self._oauth_token_store.fetch(subject.subject_id, server.server_id)
        except TokenStoreUnavailable:
            return None


def _id_jag_slot_key(subject: Subject, server: ServerSpec) -> str:
    """Which cache slot this caller's bearer for this upstream lives in.

    Addressed by the principal, plus the caller's own token when it presented one so two callers
    sharing an empty principal do not contend for one slot. Deliberately free of the stored
    assertion, which is what lets invalidation compute this while the assertion store is down. The
    entry's fingerprint, not this key, is what guarantees a cached bearer matches current inputs.
    """
    inbound: Final = subject.inbound_token.get_secret_value() if subject.inbound_token is not None else ""
    material: Final = "\x00".join((subject.tenant_id, subject.subject_id, server.server_id, inbound))
    return hashlib.sha256(material.encode()).hexdigest()


def _assertion_expired(assertion: SSOIdentityAssertion, now: datetime) -> bool:
    """Whether the stored assertion's ``exp`` has passed. An assertion carrying no expiry is
    treated as usable and left for the IdP to reject, since the store records what the id_token
    claimed rather than imposing a lifetime of its own. A naive ``expires_at`` is read as UTC so a
    stored value that lost its offset compares instead of raising.
    """
    expires_at: Final = assertion.expires_at
    if expires_at is None:
        return False
    normalized: Final = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=timezone.utc)
    return normalized <= now


def _id_jag_fingerprint(subject_token: str, server_id: str, config: IdJagConfig) -> str:
    """What the cached leg-2 bearer was minted from: the subject token, the server, and the config.

    Stored beside the bearer and compared on every read, so a rotated assertion or an edited server
    config reads as a miss and re-mints instead of serving a bearer authorized under the old policy.

    Every exchange parameter derives from the config (endpoints, audience, resource, scopes, client
    auth), so a server update that changes any of them must change the key; otherwise the old bearer,
    authorized under the old policy, keeps being served until its TTL. Everything is hashed, so no
    secret is held in the key.
    """
    material: Final = "\x00".join(
        (
            subject_token,
            server_id,
            config.org_token_endpoint,
            config.resource_token_endpoint,
            config.client_id,
            _client_auth_fingerprint(config.client_auth),
            config.subject_token_type,
            config.audience or "",
            config.resource or "",
            " ".join(config.scopes),
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _client_auth_fingerprint(client_auth: ClientAuth) -> str:
    match client_auth:
        case PrivateKeyJwtAuth() as auth:
            return "\x00".join(
                ("private_key_jwt", auth.private_key.get_secret_value(), auth.key_id or "", auth.signing_alg)
            )
        case ClientSecretAuth() as auth:
            return "\x00".join(("client_secret", auth.client_secret.get_secret_value()))
    assert_never(client_auth)


def _not_implemented(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet"))
