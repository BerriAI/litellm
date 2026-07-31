"""Renew the stored SSO identity assertion so an ID-JAG agent outlives one id_token.

The ``oauth2_id_jag`` arm asserts the id_token captured at the user's last interactive sign-in, so
without renewal an agent holding a brokered LiteLLM key can act for that user only until that token's
``exp``, typically an hour, and the sole recovery is another interactive login. The assertion already
carries the IdP refresh token beside it; this module is what redeems it.

``RefreshingSSOAssertionStore`` wraps any ``SSOAssertionStore`` and satisfies the same protocol, so
the egress arm is unchanged: it still reads one assertion and still judges expiry itself. Renewal is
lazy (only a read that finds a near-expiry assertion triggers one, so IdP traffic tracks actual use,
not the size of the user table) and single-flighted per user through the same ``RefreshCoordinator``
the ``authorization_code`` arm uses, because an IdP that rotates refresh tokens treats two concurrent
redemptions of one token as replay and can revoke the whole grant chain.

The refresh is redeemed against the generic-OIDC client the login itself used
(``GENERIC_TOKEN_ENDPOINT`` / ``GENERIC_CLIENT_ID`` / ``GENERIC_CLIENT_SECRET``, which the proxy
reconciles from the stored SSO row into the process environment at startup), authenticated the way
that login authenticated: the non-PKCE path always sends HTTP Basic, while the PKCE path sends the
credentials in the body when ``GENERIC_INCLUDE_CLIENT_ID`` is set, and an IdP application may accept
only one of the two. An assertion can only exist if that client minted it, so no other client could
redeem its refresh token, and no other method is known to be accepted. A deployment whose
``GENERIC_SCOPE`` omits ``offline_access`` captures no refresh token at all, which is why that miss
logs the scope by name rather than failing silently.

Failures are values internally (``Result[_, RefreshFailure]``). At the store boundary they collapse
onto the protocol's existing two-outcome contract: a refusal returns the expired assertion unchanged
so the reader's own guard challenges the user to sign in again, while a transient IdP failure raises
``AssertionStoreUnavailable`` so the reader answers 503 instead of blaming the user for an outage.

One gap in that mapping is left open deliberately. Under the conjunction of Redis-coordinated
renewal across replicas, concurrent requests for one user landing on different pods, and a storage
write that fails while reads still succeed, the elected refresher answers 503 but a lock loser on
another pod re-reads the unchanged row and gets the sign-in challenge, which will not help it: it is
blocked until storage recovers either way. In-process contention is unaffected and already answers
503, since those callers share the winner's outcome. It stays open because a rejected refresh and a
failed write leave an identical row, so a loser cannot tell them apart; closing it means sharing the
winner's outcome across replicas, which is new distributed state and belongs in its own change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import Timeout
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # litellm http handler is untyped
)
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    build_token_endpoint_client_auth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InProcessRefreshCoordinator,
    RefreshCoordinator,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.runtime_refresh_coordinator import (
    runtime_refresh_coordinator,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    AssertionStoreUnavailable,
    DbSSOAssertionStore,
    SSOAssertionStore,
    SSOIdentityAssertion,
    assertion_expired,
    assertion_from_sso_login,
    fetch_sso_identity_assertion,
    persist_sso_identity_assertion,
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import MCPTokenEndpointAuthMethod

_BODY_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])

_REFRESH_GRANT_TYPE = "refresh_token"
# The lock namespace for the one assertion row a user has; the sibling arm keys the same lock by
# server_id, and no server_id can collide with this literal.
_SINGLE_FLIGHT_KEY = "sso_identity_assertion"
# Renew this far ahead of ``exp`` so a token that would die between resolution and the second leg of
# the exchange is replaced first. Matches the sibling per-user token store's skew.
_DEFAULT_EXPIRY_SKEW_SECONDS = 60.0


class AssertionRead(Protocol):
    """Reads the user's stored assertion row."""

    async def __call__(self, user_id: str) -> SSOIdentityAssertion | None: ...


class AssertionWrite(Protocol):
    """Replaces the user's stored assertion row."""

    async def __call__(self, user_id: str, assertion: SSOIdentityAssertion) -> None: ...


class CoordinatorFactory(Protocol):
    """Builds the cross-replica coordinator, or ``None`` when there is no shared lock to build on."""

    def __call__(self) -> RefreshCoordinator | None: ...


class FormPost(Protocol):
    """POSTs an OAuth form and hands back the raw response."""

    async def __call__(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> httpx.Response | None: ...


@dataclass(frozen=True, slots=True)
class SSOClientConfig:
    """The generic-OIDC client credentials a refresh_token grant has to authenticate as, and how."""

    token_endpoint: str
    client_id: str
    client_secret: SecretStr
    auth_method: MCPTokenEndpointAuthMethod


def sso_client_config(env: Mapping[str, str]) -> SSOClientConfig | None:
    """The configured generic-OIDC client, or ``None`` when the deployment has none.

    Read from the process environment because that is where the login path reads it
    (``_setup_generic_sso_env_vars``) and where the proxy materializes the stored ``sso_config`` row
    at startup, so this resolves to the same client that minted the assertion. ``None`` is an
    ordinary state, not an error: a deployment signing in through a provider that captures no
    assertion has nothing here to renew, and a client with no secret is not a confidential client
    that could redeem one.

    ``auth_method`` is derived from the same ``GENERIC_INCLUDE_CLIENT_ID`` the login reads, because
    the two login paths do not agree: the non-PKCE path always authenticates with HTTP Basic, while
    the PKCE path puts the credentials in the body when that flag is set. Both capture assertions, so
    a constant here would authenticate the renewal differently from the sign-in that produced the
    refresh token and 401 against an IdP application registered for only one of the two.
    """
    token_endpoint = env.get("GENERIC_TOKEN_ENDPOINT")
    client_id = env.get("GENERIC_CLIENT_ID")
    client_secret = env.get("GENERIC_CLIENT_SECRET")
    if not token_endpoint or not client_id or not client_secret:
        return None
    includes_client_id = env.get("GENERIC_INCLUDE_CLIENT_ID", "false").lower() == "true"
    return SSOClientConfig(
        token_endpoint=token_endpoint,
        client_id=client_id,
        client_secret=SecretStr(client_secret),
        auth_method="client_secret_post" if includes_client_id else "client_secret_basic",
    )


@dataclass(frozen=True, slots=True)
class RefreshFailure:
    """Why a renewal produced nothing, split by what the caller can do about it.

    ``rejected`` is settled: this refresh token will never work again, so the user has to sign in.
    ``unavailable`` is transient: the same attempt may succeed in a minute, so telling the user to
    sign in again would be a lie about whose problem it is. Both arms carry the same payload, so
    this is a ``Literal`` discriminant rather than a ``tagged_union``; consumers still ``match`` on
    ``kind`` with an ``assert_never`` tail.
    """

    kind: Literal["rejected", "unavailable"]
    detail: str

    @staticmethod
    def of_rejected(detail: str) -> RefreshFailure:
        return RefreshFailure(kind="rejected", detail=detail)

    @staticmethod
    def of_unavailable(detail: str) -> RefreshFailure:
        return RefreshFailure(kind="unavailable", detail=detail)


class TokenEndpointTransport(Protocol):
    """One form POST to the IdP token endpoint, with the refusal/outage split preserved.

    That split is the whole reason this is not the resolver's ``TokenEndpointClient``: that
    collaborator maps every non-2xx to ``upstream_unavailable``, which is right for an exchange leg
    and wrong here, where a 400 ``invalid_grant`` means the stored refresh token is dead and the user
    must act.
    """

    async def post(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> Result[Mapping[str, object], RefreshFailure]: ...


async def post_form(url: str, form: Mapping[str, str], headers: Mapping[str, str]) -> httpx.Response | None:
    # litellm's httpx handler is only partially typed; nothing but the response object crosses back,
    # and the transport below validates its body, so the untyped boundary is contained here.
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)  # pyright: ignore[reportUnknownVariableType]  # litellm http handler is untyped
    return await client.post(url, data=form, headers=headers)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportReturnType,reportArgumentType]  # litellm http handler is untyped and its stub narrows data=/headers= to dict, which httpx itself does not require


class HttpxTokenEndpointTransport:
    """The live transport. 4xx is the IdP refusing this grant; anything else is an outage.

    The POST itself is injected so that split, which decides whether the user is challenged or told
    to wait, is testable without a live IdP.
    """

    def __init__(self, post: FormPost = post_form) -> None:
        self._post = post

    async def post(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> Result[Mapping[str, object], RefreshFailure]:
        try:
            response = await self._post(url, form, headers)
            if response is None:
                return Error(RefreshFailure.of_unavailable("the IdP token endpoint returned no response"))
            response.raise_for_status()
            body = _BODY_ADAPTER.validate_python(response.json())  # pyright: ignore[reportAny]  # untyped JSON; the adapter is the type gate
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 400 <= status < 500:
                return Error(RefreshFailure.of_rejected(f"the IdP refused the refresh with status {status}"))
            return Error(RefreshFailure.of_unavailable(f"the IdP token endpoint answered with status {status}"))
        except (httpx.RequestError, Timeout) as exc:
            return Error(RefreshFailure.of_unavailable(f"the IdP token endpoint is unreachable ({type(exc).__name__})"))
        except json.JSONDecodeError:
            return Error(RefreshFailure.of_unavailable("the IdP token endpoint returned a non-JSON response"))
        except ValidationError:
            return Error(RefreshFailure.of_unavailable("the IdP token endpoint returned a non-object response"))
        return Ok(body)


class SSOAssertionRefresher:
    """Redeems the stored refresh token for a current id_token and writes the rotation back.

    Collaborators are injected so the orchestration, the untyped response parsing and the
    write-back race are all testable without an IdP or a database.
    """

    def __init__(
        self,
        transport: TokenEndpointTransport,
        *,
        client_config: Callable[[], SSOClientConfig | None] = lambda: sso_client_config(os.environ),
        read: AssertionRead = fetch_sso_identity_assertion,
        write: AssertionWrite = persist_sso_identity_assertion,
    ) -> None:
        self._transport = transport
        self._client_config = client_config
        self._read = read
        self._write = write

    async def refresh(
        self, user_id: str, assertion: SSOIdentityAssertion
    ) -> Result[SSOIdentityAssertion, RefreshFailure]:
        if assertion.refresh_token is None:
            verbose_proxy_logger.warning(
                "ID-JAG: the stored IdP identity assertion for user_id=%s has expired and no refresh token was "
                "captured with it, so it cannot be renewed without another interactive sign-in. Add "
                "'offline_access' to GENERIC_SCOPE so the SSO login captures one.",
                user_id,
            )
            return Error(RefreshFailure.of_rejected("no refresh token was captured at sign-in"))
        config = self._client_config()
        if config is None:
            verbose_proxy_logger.warning(
                "ID-JAG: the stored IdP identity assertion for user_id=%s has expired and cannot be renewed "
                "because the generic SSO client is not configured (GENERIC_TOKEN_ENDPOINT, GENERIC_CLIENT_ID, "
                "GENERIC_CLIENT_SECRET).",
                user_id,
            )
            return Error(RefreshFailure.of_rejected("the generic SSO client is not configured"))

        carried_refresh_token = assertion.refresh_token.get_secret_value()
        # Whichever method the SSO login used for this client, since that is the one the IdP
        # application is known to accept: an assertion only exists to renew because a sign-in already
        # authenticated this client that way.
        client_auth = build_token_endpoint_client_auth(
            auth_method=config.auth_method,
            client_id=config.client_id,
            client_secret=config.client_secret.get_secret_value(),
        )
        form = {  # mutable-ok: the RFC 6749 form body is a wire format the HTTP client takes as a mapping
            "grant_type": _REFRESH_GRANT_TYPE,
            "refresh_token": carried_refresh_token,
            **client_auth.body,
        }
        match await self._transport.post(config.token_endpoint, form, client_auth.headers):
            case Error(failure):
                return Error(failure)
            case Ok(body):
                return await self._renewed_from(user_id, assertion, body, carried_refresh_token)

    async def _renewed_from(
        self,
        user_id: str,
        previous: SSOIdentityAssertion,
        body: Mapping[str, object],
        carried_refresh_token: str,
    ) -> Result[SSOIdentityAssertion, RefreshFailure]:
        """The renewed assertion, built by the same validator the login path uses.

        A rotated refresh token replaces the stored one; an omitted one carries forward, since an
        IdP that does not rotate expects the original to keep working.
        """
        rotated = body.get("refresh_token")
        renewed = assertion_from_sso_login(
            body.get("id_token"),
            rotated if isinstance(rotated, str) and rotated else carried_refresh_token,
        )
        if renewed is None:
            verbose_proxy_logger.warning(
                "ID-JAG: the IdP accepted the refresh for user_id=%s but returned no usable id_token, so there "
                "is nothing to assert upstream. The SSO client's grant needs the 'openid' scope for the token "
                "endpoint to return one on a refresh.",
                user_id,
            )
            return Error(RefreshFailure.of_rejected("the IdP's refresh response carried no usable id_token"))
        failure = await self._store_renewal(user_id, previous, renewed)
        if failure is not None:
            return Error(failure)
        return Ok(renewed)

    async def _store_renewal(
        self, user_id: str, previous: SSOIdentityAssertion, renewed: SSOIdentityAssertion
    ) -> RefreshFailure | None:
        """Write the renewal back, unless the row moved on while this renewal was in flight.

        The row is one per user and last-write-wins, so an interactive sign-in landing mid-renewal
        would otherwise be overwritten with a refresh token the IdP has already rotated away, costing
        that user a sign-in later. Comparing against the id_token this renewal started from is what
        detects that; skipping is safe because the newer row is the one the reader wants anyway.

        A failed write is transient, not settled. The store, not this return value, is what every
        caller reads, so a renewal that could not be recorded is a renewal nobody will see; saying so
        keeps a database problem answering 503 rather than telling the user to sign in again over it.
        """
        try:
            current = await self._read(user_id)
            if current is not None and current.id_token.get_secret_value() != previous.id_token.get_secret_value():
                verbose_proxy_logger.info(
                    "ID-JAG: a newer IdP identity assertion for user_id=%s was stored while this renewal was in "
                    "flight; keeping the stored one.",
                    user_id,
                )
                return None
            await self._write(user_id, renewed)
        except Exception as exc:  # noqa: BLE001  # any storage failure is transient here, never the user's fault
            verbose_proxy_logger.warning(
                "ID-JAG: could not persist the renewed IdP identity assertion for user_id=%s, so the rotated "
                "refresh token is lost and this user will have to sign in again once the renewed token expires: %s",
                user_id,
                exc,
            )
            return RefreshFailure.of_unavailable("the renewed IdP identity assertion could not be persisted")
        return None


class RefreshingSSOAssertionStore:
    """An ``SSOAssertionStore`` that renews a near-expiry assertion before handing it back.

    Reads the inner store; an assertion still comfortably inside its lifetime is returned untouched,
    so the common path costs exactly what it did before. Otherwise one renewal runs per user through
    the injected ``RefreshCoordinator`` and every caller then re-reads the inner store, which is the
    authority: the winner's write is what they all observe, and a renewal the write-back guard
    skipped yields the newer assertion that displaced it rather than a private copy.

    A refusal leaves the expired assertion in place for the reader's own guard to reject, so the user
    sees the same sign-in-again challenge as before this store existed. A transient IdP failure
    raises ``AssertionStoreUnavailable``, the protocol's existing signal for "this is not the user's
    fault"; concurrent in-process callers share that outcome, while a cross-replica loser falls back
    to re-reading and challenges instead.
    """

    def __init__(
        self,
        inner: SSOAssertionStore,
        refresher: SSOAssertionRefresher,
        *,
        coordinator_factory: CoordinatorFactory = runtime_refresh_coordinator,
        expiry_skew_seconds: float = _DEFAULT_EXPIRY_SKEW_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._inner = inner
        self._refresher = refresher
        self._coordinator_factory = coordinator_factory
        self._in_process_coordinator = InProcessRefreshCoordinator()
        self._distributed_coordinator: RefreshCoordinator | None = None
        self._skew = timedelta(seconds=expiry_skew_seconds)
        self._clock = clock

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        assertion = await self._inner.fetch(user_id)
        if assertion is None or not assertion_expired(assertion, self._clock() + self._skew):
            return assertion
        await self._coordinator().run(
            user_id,
            _SINGLE_FLIGHT_KEY,
            refresh=lambda: self._renew(user_id, assertion),
            reread=_nothing_to_reread,
        )
        return await self._inner.fetch(user_id)

    def _coordinator(self) -> RefreshCoordinator:
        """The cross-replica coordinator once Redis is reachable, else the in-process one.

        Built on first use and kept, because the proxy's Redis client is not wired at import time;
        retried while it is absent so a proxy that gains Redis later stops electing per-worker.
        """
        if self._distributed_coordinator is None:
            self._distributed_coordinator = self._coordinator_factory()
        return self._distributed_coordinator or self._in_process_coordinator

    async def _renew(self, user_id: str, expiring: SSOIdentityAssertion) -> None:
        """The elected renewal. Returns nothing: the inner store, not this return value, is what
        every caller reads afterwards, so the winner and the losers cannot disagree."""
        match await self._refresher.refresh(user_id, expiring):
            case Ok(_):
                return
            case Error(failure):
                match failure.kind:
                    case "rejected":
                        return
                    case "unavailable":
                        raise AssertionStoreUnavailable(failure.detail)
                assert_never(failure.kind)


async def _nothing_to_reread() -> None:
    """The coordinator's re-read channel, unused: every caller re-reads the store after ``run``."""


def default_sso_assertion_store() -> SSOAssertionStore:
    """The live read seam for the ``id_jag`` arm: the stored assertion, renewed when it is stale."""
    return RefreshingSSOAssertionStore(DbSSOAssertionStore(), SSOAssertionRefresher(HttpxTokenEndpointTransport()))
