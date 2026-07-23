"""Store for the enterprise IdP identity assertion captured at SSO login (EMA).

The ``oauth2_id_jag`` egress arm needs the user's IdP ``id_token`` as its RFC 8693
``subject_token``. A front-door client holds an identity-only ``llm_session_`` bearer, not an
IdP assertion, so the assertion captured at the one SSO login is the only usable subject
source for it. This module owns both sides of that state: the SSO callback persists here
(write-through to the DB so a login on one pod is visible to every pod) and the resolver
seam reads back by ``user_id``. Retention is gated on an ``oauth2_id_jag`` server actually
being registered, so a gateway with no EMA upstream never stores bearer material.

The row is one encrypted payload per user, latest login wins. ``expires_at`` mirrors the
id_token ``exp`` claim and is judged by the reader, never enforced by deletion here: an
expired assertion with a refresh token is still renewable, and the DB row is the source of
truth, the same contract as the per-user OAuth credential store.
"""

from __future__ import annotations

import json
import os
import time
import weakref
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import jwt
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    RefreshingTokenStore,
    TokenStoreUnavailable,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_ASSERTION_DECRYPT_LOG_KEY = "sso_identity_assertion"
_STR_ADAPTER: TypeAdapter[str] = TypeAdapter(str)
_MAYBE_STR_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)

# An assertion this close to expiry is treated as expired, so a subject token is never handed
# to the exchange with less lifetime than the two token-endpoint legs need to complete.
_ASSERTION_EXPIRY_BUFFER_SECONDS = 30

# How long an assertion carrying no expiry is cached before the DB row is re-read; one that
# declares an expiry is cached until that expiry (minus the buffer above) by the shared cache.
_ASSERTION_CACHE_TTL_SECONDS = 60.0

# The assertion is the user's SSO identity rather than a per-upstream credential, so it occupies a
# single per-user slot in the shared (user_id, server_id) keyed stack.
_ASSERTION_SERVER_KEY = ""


class SSOIdentityAssertion(BaseModel):
    """The IdP material an EMA exchange needs: ``id_token`` is the RFC 8693 subject token,
    ``expires_at`` bounds its usefulness, and the refresh token renews it without re-login."""

    model_config = ConfigDict(frozen=True)

    id_token: SecretStr
    refresh_token: SecretStr | None = None
    issuer: str | None = None
    expires_at: datetime | None = None


class _IdTokenClaims(BaseModel):
    exp: float | None = None
    iss: str | None = None


class _StoredAssertionPayload(BaseModel):
    id_token: str
    refresh_token: str | None = None
    issuer: str | None = None
    expires_at: datetime | None = None


def assertion_from_sso_login(id_token: object, refresh_token: object) -> SSOIdentityAssertion | None:
    """The typed carrier built where the raw token response exists; ``None`` when the provider
    sent no id_token or sent one that is not a decodable JWT, since neither is exchangeable
    under EMA. Inputs are ``object`` because they come straight from the provider's untyped
    token response; this is the one boundary that validates them. The token arrived over TLS
    from the IdP's own token endpoint, so claims are read without signature verification,
    matching how the SSO callback already decodes it for identity."""
    raw_id_token = id_token if isinstance(id_token, str) and id_token else None
    if raw_id_token is None:
        return None
    raw_refresh_token = refresh_token if isinstance(refresh_token, str) and refresh_token else None
    try:
        claims = _IdTokenClaims.model_validate(jwt.decode(raw_id_token, options={"verify_signature": False}))
        expires_at = datetime.fromtimestamp(claims.exp, tz=timezone.utc) if claims.exp is not None else None
    except Exception:  # noqa: BLE001  # decode failure = not retainable; never raise into login
        verbose_proxy_logger.warning(
            "SSO id_token could not be decoded or its claims were unusable; not retaining it for EMA egress."
        )
        return None
    return SSOIdentityAssertion(
        id_token=SecretStr(raw_id_token),
        refresh_token=SecretStr(raw_refresh_token) if raw_refresh_token else None,
        issuer=claims.iss,
        expires_at=expires_at,
    )


async def ema_assertion_retention_enabled() -> bool:
    """Whether any MCP server uses ``oauth2_id_jag``, evaluated per login so the gateway only
    retains bearer material while an EMA upstream exists to spend it on. Judged against the two
    configuration authorities: the pod-local config declaration and the shared DB row. The
    in-memory registry is deliberately not consulted in either direction; it is a per-process
    snapshot of the DB state that can be stale both ways (a server added on another pod would
    silently drop the write, one removed on another pod would keep retaining bearer material),
    and a gate guarding a shared-DB write must judge against that storage's authority."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # avoids import cycle
        global_mcp_server_manager,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global
    from litellm.types.mcp import MCPAuth  # noqa: PLC0415  # runtime global

    config_servers = global_mcp_server_manager.config_mcp_servers.values()
    if any(server.auth_type == MCPAuth.oauth2_id_jag for server in config_servers):
        return True
    if prisma_client is None:
        return False
    row = await prisma_client.db.litellm_mcpservertable.find_first(where={"auth_type": MCPAuth.oauth2_id_jag.value})
    return row is not None


async def persist_sso_identity_assertion(user_id: str, assertion: SSOIdentityAssertion) -> None:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper  # noqa: PLC0415  # runtime global
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    if prisma_client is None:
        return
    payload: dict[str, str] = {
        "id_token": assertion.id_token.get_secret_value(),
        **({"refresh_token": assertion.refresh_token.get_secret_value()} if assertion.refresh_token else {}),
        **({"issuer": assertion.issuer} if assertion.issuer else {}),
        **({"expires_at": assertion.expires_at.isoformat()} if assertion.expires_at else {}),
    }
    encoded = _STR_ADAPTER.validate_python(encrypt_value_helper(json.dumps(payload)))
    await prisma_client.db.litellm_ssoidentityassertion.upsert(
        where={"user_id": user_id},
        data={
            "create": {"user_id": user_id, "assertion_b64": encoded},
            "update": {"assertion_b64": encoded},
        },
    )
    # This is the one place the row is replaced, so it is the one place that must drop a cached
    # predecessor: otherwise a re-login (notably one that reduces the user's IdP claims) would keep
    # serving the superseded assertion until the old id_token expired.
    await _drop_cached_assertion(user_id)


async def fetch_sso_identity_assertion(user_id: str) -> SSOIdentityAssertion | None:
    """The stored assertion for ``user_id``, or ``None`` only when no row was ever written.

    ``None`` is reserved for that one determinate fact, so a fault can never be reported as "this
    user has not signed in". A store that cannot be reached raises ``TokenStoreUnavailable``, which
    is the contract ``OAuthTokenStore`` already defines for exactly this, and a row that cannot be
    decrypted (salt-key rotation) or parsed raises ``SsoAssertionUnrenewable`` because the material
    is unusable and only a fresh sign-in replaces it. Expiry is not judged here; the reader owns
    that policy.
    """
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper  # noqa: PLC0415  # runtime global
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    if prisma_client is None:
        raise TokenStoreUnavailable("the assertion store is not connected")
    try:
        row = await prisma_client.db.litellm_ssoidentityassertion.find_unique(where={"user_id": user_id})
    except Exception as exc:  # noqa: BLE001  # an unreadable store is indeterminate, never an absence
        raise TokenStoreUnavailable("the assertion store could not be read") from exc
    if row is None:
        return None
    raw = _MAYBE_STR_ADAPTER.validate_python(
        decrypt_value_helper(row.assertion_b64, _ASSERTION_DECRYPT_LOG_KEY, exception_type="debug")
    )
    if raw is None:
        raise SsoAssertionUnrenewable("the stored identity assertion could not be decrypted")
    try:
        payload = _StoredAssertionPayload.model_validate_json(raw)
    except ValidationError as exc:
        raise SsoAssertionUnrenewable("the stored identity assertion could not be parsed") from exc
    return SSOIdentityAssertion(
        id_token=SecretStr(payload.id_token),
        refresh_token=SecretStr(payload.refresh_token) if payload.refresh_token else None,
        issuer=payload.issuer,
        expires_at=payload.expires_at,
    )


async def rotate_sso_identity_assertions_master_key(prisma_client: PrismaClient, new_master_key: str) -> None:
    """Re-encrypt every stored assertion under ``new_master_key`` during a salt-key rotation,
    mirroring the sibling per-user credential tables; an unreadable row is skipped so one
    corrupt row does not abort the rotation. Rows are decrypted one at a time inside the loop
    so the whole table's plaintext is never held in memory at once."""
    from prisma.models import LiteLLM_SSOIdentityAssertion as AssertionRow  # noqa: PLC0415  # generated at runtime

    from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: PLC0415  # runtime global
        decrypt_value_helper,
        encrypt_value_helper,
    )

    async def _rotate_row(row: AssertionRow) -> bool:
        plaintext = _MAYBE_STR_ADAPTER.validate_python(
            decrypt_value_helper(row.assertion_b64, _ASSERTION_DECRYPT_LOG_KEY, exception_type="debug")
        )
        if plaintext is None:
            verbose_proxy_logger.warning(
                "rotate_sso_identity_assertions_master_key: could not decrypt assertion for user_id=%s, skipping",
                row.user_id,
            )
            return False
        re_encrypted = _STR_ADAPTER.validate_python(encrypt_value_helper(plaintext, new_encryption_key=new_master_key))
        await prisma_client.db.litellm_ssoidentityassertion.update(
            where={"user_id": row.user_id},
            data={"assertion_b64": re_encrypted},
        )
        return True

    rows = await prisma_client.db.litellm_ssoidentityassertion.find_many()
    outcomes = [await _rotate_row(row) for row in rows]
    verbose_proxy_logger.info(
        "rotate_sso_identity_assertions_master_key: rotated %d row(s), skipped %d",
        sum(outcomes),
        len(outcomes) - sum(outcomes),
    )


class SsoAssertionUnrenewable(Exception):
    """The stored assertion is expired and its grant is definitively dead; only a fresh sign-in
    produces a new one. Raised rather than returned so the verdict survives the shared refresh
    stack, whose ``OAuthToken | None`` would otherwise collapse "the grant is dead" (re-login)
    into "this user never connected" (a different remedy, and a different status)."""


# The renewal POST answers with the token body or raises the verdict its failure proves, so the
# outcome needs no intermediate representation: an unrenewable grant and an unreachable IdP are
# exactly the two terminal states the shared stack already propagates.
SsoTokenEndpointPost = Callable[[str, dict[str, str]], Awaitable[dict[str, object]]]


class _SsoRefreshClient(BaseModel):
    """The gateway's own SSO client registration at the IdP. The stored refresh token was
    issued to THIS client (the generic SSO app), never to an ``id_jag`` server's client, so
    renewing the assertion must authenticate as it or the IdP answers ``invalid_grant``."""

    model_config = ConfigDict(frozen=True)
    client_id: str
    client_secret: SecretStr | None
    token_endpoint: str


def generic_sso_scopes(getenv: Callable[[str], str | None] = os.getenv) -> list[str]:
    """The scopes the generic SSO client requests, shared by the login authorize request and the
    EMA assertion refresh so the two can never diverge. ``openid`` must be among them or the IdP
    returns no ``id_token`` on refresh, which would strand a renewable assertion as expired and
    force a needless re-login; requesting exactly what login was granted keeps the refresh within
    RFC 6749 scope while guaranteeing the id_token comes back. The refresh passes the source's
    injected ``getenv`` so it reads the same environment as the rest of its client config."""
    raw = getenv("GENERIC_SCOPE")
    return (raw if raw is not None else "openid email profile").split(" ")


def _sso_refresh_client_from_env(getenv: Callable[[str], str | None]) -> _SsoRefreshClient | None:
    client_id = getenv("GENERIC_CLIENT_ID")
    token_endpoint = getenv("GENERIC_TOKEN_ENDPOINT")
    if not client_id or not token_endpoint:
        return None
    secret = getenv("GENERIC_CLIENT_SECRET")
    return _SsoRefreshClient(
        client_id=client_id,
        client_secret=SecretStr(secret) if secret else None,
        token_endpoint=token_endpoint,
    )


def _oauth_error_code(response: object) -> str | None:
    """The RFC 6749 section 5.2 ``error`` code from a token-endpoint response body, or None
    when the body is not a JSON object carrying one."""
    try:
        body = response.json()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]  # httpx response is partially typed
    except Exception:  # noqa: BLE001  # an unparseable body simply carries no code
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # narrowed below
    return error if isinstance(error, str) and error else None


async def _post_sso_token_endpoint(url: str, form: dict[str, str]) -> dict[str, object]:
    """The renewal grant, answering with the token body or raising what its failure proves.

    Unrenewable requires PROOF of a grant verdict: a non-429 4xx carrying a parseable RFC 6749
    error code. A 429, a 4xx without the error object (an intermediary answering, not the token
    endpoint), a 5xx, a transport failure, or a 2xx whose body is not a JSON object all prove
    nothing about the grant and read unavailable, so the caller retries instead of being told to
    sign in again.
    """
    import httpx  # noqa: PLC0415

    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # http_handler is untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)  # pyright: ignore[reportUnknownVariableType]  # http_handler is untyped
        response = await client.post(url, headers={"Accept": "application/json"}, data=form)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # httpx handler partially typed
        response.raise_for_status()  # pyright: ignore[reportUnknownMemberType]  # httpx handler partially typed
        body: object = response.json()  # pyright: ignore[reportUnknownMemberType]  # shape-validated below
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if 400 <= status_code < 500 and status_code != 429 and _oauth_error_code(exc.response) is not None:
            raise SsoAssertionUnrenewable(f"the identity provider rejected the refresh grant ({status_code})") from exc
        raise TokenStoreUnavailable(f"the identity provider answered {status_code}") from exc
    except (SsoAssertionUnrenewable, TokenStoreUnavailable):
        raise
    except Exception as exc:  # noqa: BLE001  # transport/parse failure carries no verdict on the grant
        raise TokenStoreUnavailable(f"the identity provider could not be reached ({type(exc).__name__})") from exc
    if not isinstance(body, dict):
        raise TokenStoreUnavailable("the identity provider returned a non-object JSON body")
    return body


def _to_oauth_token(assertion: SSOIdentityAssertion) -> OAuthToken:
    """The assertion as the shared stack's credential: the id_token is the value EMA spends (the
    RFC 8693 subject token), so it rides in ``access_token``. ``issuer`` has no reader."""
    return OAuthToken(
        access_token=assertion.id_token.get_secret_value(),
        expires_at=assertion.expires_at.timestamp() if assertion.expires_at is not None else None,
        refresh_token=assertion.refresh_token.get_secret_value() if assertion.refresh_token else None,
    )


class _SsoAssertionDbStore:
    """``OAuthTokenStore`` over the EMA assertion row. The assertion is the user's SSO identity,
    not a per-upstream credential, so ``server_id`` is ignored: every id_jag server spends it."""

    def __init__(self, fetch: Callable[[str], Awaitable[SSOIdentityAssertion | None]]) -> None:
        self._fetch = fetch

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        assertion = await self._fetch(user_id)
        return None if assertion is None else _to_oauth_token(assertion)


class _SsoAssertionRefresher:
    """``TokenRefresher`` that renews the assertion as the gateway's OWN generic SSO client, whose
    registration issued the stored refresh token (an id_jag server's client would get
    ``invalid_grant``). A renewed token is returned and persisted best-effort, so a rotated refresh
    token is not lost while an in-hand one is never discarded; a dead grant raises
    ``SsoAssertionUnrenewable`` (re-login) and anything proving nothing about the grant raises
    ``TokenStoreUnavailable`` (retry), so a down IdP never tells a user to sign in again.

    A PROVEN dead grant is recorded where the assertion lives, by persisting the row without the
    refresh token the IdP rejected, never in a cache beside it. Later reads then answer from the
    row with no token-endpoint call, on every pod and for as long as the row stands, and a re-login
    lifts it by replacing that row. A verdict in a side cache guarantees neither: it is per-pod, it
    expires on its own clock, and it can refuse a caller who has just signed in again.
    """

    def __init__(
        self,
        persist: Callable[[str, SSOIdentityAssertion], Awaitable[None]],
        post: SsoTokenEndpointPost,
        getenv: Callable[[str], str | None],
        now: Callable[[], float] = time.time,
    ) -> None:
        self._persist = persist
        self._post = post
        self._getenv = getenv
        self._now = now

    async def _record_dead_grant(self, user_id: str, token: OAuthToken) -> None:
        """Persist the row without the rejected refresh token. Best effort: failing to record the
        verdict costs one more POST next time, never a wrong answer."""
        try:
            await self._persist(
                user_id,
                SSOIdentityAssertion(
                    id_token=SecretStr(token.access_token),
                    refresh_token=None,
                    expires_at=(
                        datetime.fromtimestamp(token.expires_at, tz=timezone.utc)
                        if token.expires_at is not None
                        else None
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001  # recording the verdict must not mask it
            verbose_proxy_logger.warning(
                "Could not record the rejected SSO refresh grant for user_id=%s: %s", user_id, exc
            )

    async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> OAuthToken | None:
        if token.refresh_token is None:
            # No grant to spend, because the login never returned one or a proven rejection stripped
            # it. Either way only a re-login helps, and answering costs no token-endpoint call.
            raise SsoAssertionUnrenewable("the stored assertion carries no refresh token")
        client = _sso_refresh_client_from_env(self._getenv)
        if client is None:
            verbose_proxy_logger.warning(
                "Stored SSO assertion for user_id=%s is expired and the SSO client env "
                "(GENERIC_CLIENT_ID/GENERIC_TOKEN_ENDPOINT) is not set; renewal is unavailable "
                "until the deployment restores it.",
                user_id,
            )
            raise TokenStoreUnavailable("the SSO client environment is not configured")
        scope = " ".join(generic_sso_scopes(self._getenv))
        form = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": client.client_id,
            **({"client_secret": client.client_secret.get_secret_value()} if client.client_secret else {}),
            **({"scope": scope} if scope else {}),
        }
        try:
            body = await self._post(client.token_endpoint, form)
        except SsoAssertionUnrenewable:
            await self._record_dead_grant(user_id, token)
            raise
        rotated = body.get("refresh_token")
        carried_refresh = rotated if isinstance(rotated, str) and rotated else token.refresh_token
        refreshed = assertion_from_sso_login(body.get("id_token"), carried_refresh)
        if refreshed is None:
            verbose_proxy_logger.warning(
                "SSO assertion refresh for user_id=%s returned no usable id_token; treating as expired.", user_id
            )
            raise SsoAssertionUnrenewable("the refresh response carried no usable id_token")
        try:
            await self._persist(user_id, refreshed)
        except Exception as exc:  # noqa: BLE001  # a write-back failure must not discard the in-hand token
            verbose_proxy_logger.warning(
                "SSO assertion refresh for user_id=%s succeeded but the write-back failed; serving the "
                "in-hand assertion (a possibly rotated refresh token was not stored, so the next "
                "refresh may require a re-login): %s",
                user_id,
                exc,
            )
        refreshed_token = _to_oauth_token(refreshed)
        if refreshed_token.expires_at is not None and self._now() >= (
            refreshed_token.expires_at - _ASSERTION_EXPIRY_BUFFER_SECONDS
        ):
            # The IdP answered 2xx with a token that is already dead. It is judged by the same
            # expiry predicate a stored assertion is, so it is never handed to an exchange; the
            # rotated refresh token above is persisted first so the rotation is not lost.
            verbose_proxy_logger.warning(
                "SSO assertion refresh for user_id=%s returned an already-expired id_token; treating as expired.",
                user_id,
            )
            raise SsoAssertionUnrenewable("the refreshed id_token was already expired")
        return refreshed_token


# Live sources, so the one write chokepoint can drop a superseded assertion from their caches the
# moment the row is replaced. Weak, so a discarded source never keeps an instance alive.
_LIVE_ASSERTION_SOURCES: weakref.WeakSet[LiveSsoAssertionSource] = weakref.WeakSet()


async def _drop_cached_assertion(user_id: str) -> None:
    for source in tuple(_LIVE_ASSERTION_SOURCES):
        try:
            await source.invalidate(user_id)
        except Exception as exc:  # noqa: BLE001  # a cache drop must never fail the login write
            verbose_proxy_logger.warning(
                "Could not drop the cached SSO assertion for user_id=%s after its row was replaced: %s", user_id, exc
            )


class LiveSsoAssertionSource:
    """The resolver's view of the assertion store, built on the shared per-user credential stack.

    ``Cached(Refreshing(db))`` is the composition ``authorization_code`` uses, so the read-through
    cache (positive-only, never caching an absence, so a fresh login is seen at once), the
    expiry-skewed renewal and the per-user single-flight all come from ``oauth_token_store``
    instead of being rebuilt here; passing a Redis cache and coordinator would extend that
    single-flight across replicas without touching this class.

    That stack answers ``OAuthToken | None`` while EMA needs four outcomes, each with its own
    remedy, so the refresher raises its two terminal verdicts and this adapter maps them with no
    second read: a token is usable, ``SsoAssertionUnrenewable`` is re-login (401),
    ``TokenStoreUnavailable`` is retry (503), and ``None`` then means no row was ever stored (412).
    """

    def __init__(
        self,
        fetch: Callable[[str], Awaitable[SSOIdentityAssertion | None]] = fetch_sso_identity_assertion,
        persist: Callable[[str, SSOIdentityAssertion], Awaitable[None]] = persist_sso_identity_assertion,
        post: SsoTokenEndpointPost = _post_sso_token_endpoint,
        getenv: Callable[[str], str | None] = os.getenv,
        cache_ttl_seconds: float = _ASSERTION_CACHE_TTL_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._store = CachedOAuthTokenStore(
            RefreshingTokenStore(
                _SsoAssertionDbStore(fetch),
                _SsoAssertionRefresher(persist, post, getenv, now=now),
                expiry_skew_seconds=_ASSERTION_EXPIRY_BUFFER_SECONDS,
                clock=now,
            ),
            default_ttl_seconds=cache_ttl_seconds,
            expiry_skew_seconds=_ASSERTION_EXPIRY_BUFFER_SECONDS,
            max_ttl_seconds=cache_ttl_seconds,
            clock=now,
        )
        _LIVE_ASSERTION_SOURCES.add(self)

    async def invalidate(self, user_id: str) -> None:
        """Drop this pod's cached assertion for ``user_id`` so the next read sees the replaced row.

        Called from the write chokepoint, so a re-login (in particular one that reduces the user's
        claims) is honored immediately rather than when the superseded id_token finally expires.
        """
        await self._store.invalidate(user_id, _ASSERTION_SERVER_KEY)

    async def fetch_usable(self, user_id: str) -> OAuthToken | None:
        """The caller's usable assertion, ``None`` only when no row was ever stored.

        ``SsoAssertionUnrenewable`` (re-login) and ``TokenStoreUnavailable`` (retry) propagate, so
        the arm maps three remedies without a parallel result union. ``None`` carries exactly one
        meaning: anything indeterminate reads as unavailable, so a store outage is a retry rather
        than a false "this user has never signed in", and never a 500.
        """
        if not user_id:
            return None
        try:
            return await self._store.fetch(user_id, _ASSERTION_SERVER_KEY)
        except (SsoAssertionUnrenewable, TokenStoreUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001  # indeterminate, so never absence and never a 500
            raise TokenStoreUnavailable("the stored identity assertion could not be read") from exc


async def retain_sso_identity_assertion_for_ema(user_id: str, assertion: SSOIdentityAssertion | None) -> None:
    """The SSO-callback hook: a no-op unless there is material AND an EMA server is registered.
    A store failure is logged and swallowed because the login itself must not fail on an
    egress-side write; the cost of a miss is a 401 challenge at the EMA upstream, not a lockout."""
    if assertion is None:
        return
    try:
        if not await ema_assertion_retention_enabled():
            return
        await persist_sso_identity_assertion(user_id, assertion)
    except Exception as exc:  # noqa: BLE001  # the login itself must not fail on an egress-side write
        verbose_proxy_logger.warning(
            "Failed to persist the SSO identity assertion for EMA egress (user_id=%s): %s", user_id, exc
        )
