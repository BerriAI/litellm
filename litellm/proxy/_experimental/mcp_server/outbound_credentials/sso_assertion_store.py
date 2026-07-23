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

import asyncio
import json
import os
import weakref
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

import jwt
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_ASSERTION_DECRYPT_LOG_KEY = "sso_identity_assertion"
_STR_ADAPTER: TypeAdapter[str] = TypeAdapter(str)
_MAYBE_STR_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)

# An assertion this close to expiry is treated as expired, so a subject token is never handed
# to the exchange with less lifetime than the two token-endpoint legs need to complete.
_ASSERTION_EXPIRY_BUFFER_SECONDS = 30

# How long a usable assertion is served from the in-process memo before the DB row is re-read;
# bounds how long a re-login on another pod goes unseen (the superseded assertion stays valid).
_ASSERTION_MEMO_TTL_SECONDS = 60.0

# How long a definitive-negative or unavailable renewal verdict is served from the memo: bounds
# a dead grant at one token-endpoint POST per pod per window (instead of one per egress call)
# while keeping the post-re-login recovery latency on a warm pod within this window.
_ASSERTION_NEGATIVE_MEMO_TTL_SECONDS = 30.0


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


async def fetch_sso_identity_assertion(user_id: str) -> SSOIdentityAssertion | None:
    """The stored assertion for ``user_id``, or ``None`` when absent, undecryptable (salt-key
    rotation), or unparseable. Expiry is not judged here; the reader owns that policy."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper  # noqa: PLC0415  # runtime global
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    if prisma_client is None:
        return None
    row = await prisma_client.db.litellm_ssoidentityassertion.find_unique(where={"user_id": user_id})
    if row is None:
        return None
    raw = _MAYBE_STR_ADAPTER.validate_python(
        decrypt_value_helper(row.assertion_b64, _ASSERTION_DECRYPT_LOG_KEY, exception_type="debug")
    )
    if raw is None:
        return None
    try:
        payload = _StoredAssertionPayload.model_validate_json(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Stored SSO identity assertion for user_id=%s could not be parsed; treating as absent.", user_id
        )
        return None
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


class UsableSsoAssertion(BaseModel):
    """A stored assertion that is currently exchangeable (unexpired, possibly just refreshed)."""

    model_config = ConfigDict(frozen=True)
    state: Literal["usable"] = "usable"
    assertion: SSOIdentityAssertion


class NoSsoAssertion(BaseModel):
    """No stored assertion exists for the user (or the caller has no user identity)."""

    model_config = ConfigDict(frozen=True)
    state: Literal["absent"] = "absent"


class ExpiredSsoAssertion(BaseModel):
    """A stored assertion exists but is definitively unrenewable (no refresh token, the IdP
    rejected the grant, or the renewed token was already dead); only a fresh interactive
    sign-in can produce a new one, so the caller owes the user a re-login challenge."""

    model_config = ConfigDict(frozen=True)
    state: Literal["expired"] = "expired"


class UnavailableSsoAssertion(BaseModel):
    """Renewal could not be ATTEMPTED to completion: the IdP was unreachable or errored, or
    the SSO client env is not set. Nothing is known about the assertion itself, so telling
    the user to re-login would be the wrong remediation; the caller surfaces a retryable
    unavailability instead."""

    model_config = ConfigDict(frozen=True)
    state: Literal["unavailable"] = "unavailable"


SsoAssertionLookup = UsableSsoAssertion | NoSsoAssertion | ExpiredSsoAssertion | UnavailableSsoAssertion


class SsoRefreshGranted(BaseModel):
    """The token endpoint answered 2xx; ``body`` still needs field-level validation."""

    model_config = ConfigDict(frozen=True)
    outcome: Literal["granted"] = "granted"
    body: dict[str, object]


class SsoRefreshRejected(BaseModel):
    """The token endpoint answered 4xx: a definitive verdict on the grant (spent, revoked,
    mismatched), never a transient fault."""

    model_config = ConfigDict(frozen=True)
    outcome: Literal["rejected"] = "rejected"
    status_code: int


class SsoRefreshUnreachable(BaseModel):
    """Transport failure, 5xx, or a non-JSON answer: no verdict on the grant."""

    model_config = ConfigDict(frozen=True)
    outcome: Literal["unreachable"] = "unreachable"


SsoRefreshOutcome = SsoRefreshGranted | SsoRefreshRejected | SsoRefreshUnreachable

SsoTokenEndpointPost = Callable[[str, dict[str, str]], Awaitable[SsoRefreshOutcome]]


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


def _assertion_is_expired(assertion: SSOIdentityAssertion, now: datetime) -> bool:
    if assertion.expires_at is None:
        return False
    return assertion.expires_at <= now + timedelta(seconds=_ASSERTION_EXPIRY_BUFFER_SECONDS)


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


async def _post_sso_token_endpoint(url: str, form: dict[str, str]) -> SsoRefreshOutcome:
    import httpx  # noqa: PLC0415

    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # http_handler is untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    # Rejected requires PROOF of a grant verdict: a non-429 4xx carrying a parseable RFC 6749
    # error code. A 429, a 4xx without the error object (an intermediary answering, not the
    # token endpoint), a 5xx, a transport failure, or a 2xx whose body is not a JSON object all
    # prove nothing about the grant and read Unreachable, whose short negative cache doubles as
    # the backoff.
    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)  # pyright: ignore[reportUnknownVariableType]  # http_handler is untyped
        response = await client.post(url, headers={"Accept": "application/json"}, data=form)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # httpx handler partially typed
        response.raise_for_status()  # pyright: ignore[reportUnknownMemberType]  # httpx handler partially typed
        body: object = response.json()  # pyright: ignore[reportUnknownMemberType]  # shape-validated below
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if 400 <= status_code < 500 and status_code != 429 and _oauth_error_code(exc.response) is not None:
            verbose_proxy_logger.warning("SSO assertion refresh was rejected with status %s", status_code)
            return SsoRefreshRejected(status_code=status_code)
        verbose_proxy_logger.warning("SSO assertion refresh failed upstream with status %s", status_code)
        return SsoRefreshUnreachable()
    except Exception as exc:  # noqa: BLE001  # transport/parse failure carries no verdict on the grant
        verbose_proxy_logger.warning("SSO assertion refresh request failed: %s", exc)
        return SsoRefreshUnreachable()
    if not isinstance(body, dict):
        verbose_proxy_logger.warning("SSO assertion refresh returned a non-object JSON body")
        return SsoRefreshUnreachable()
    return SsoRefreshGranted(body=body)


class LiveSsoAssertionSource:
    """The resolver's view of the assertion store: one stored assertion made usable, or its
    typed absence. Renewal is single-flighted per user (in-process; concurrent pods may both
    refresh, which is last-write-wins on the row and harmless unless the IdP one-time-uses
    refresh tokens, in which case the loser reads Expired and the next login heals it).

    Usable results are memoized in-process for a short TTL (capped by the assertion's remaining
    life), the same read-through contract the per-user OAuth store keeps in front of the DB, so
    an egress call served by the exchange cache does not pay a DB read and a warm entry keeps
    serving through a store outage. The memo is also the single-flight hand-off channel: the
    refresh winner memoizes before releasing the lock and waiters check it under the lock before
    re-reading the store, so a failed write-back cannot make a waiter re-spend the refresh grant
    against a stale row. Only Usable is memoized: an absent row must be re-checked so
    a fresh login is visible immediately, and the staleness bound is one TTL on other pods (the
    superseded assertion remains valid, so the cached exchange it keys stays correct). A store
    read failure reads as absent, loudly logged; the store, not this source, declines to cache
    the failure."""

    def __init__(
        self,
        fetch: Callable[[str], Awaitable[SSOIdentityAssertion | None]] = fetch_sso_identity_assertion,
        persist: Callable[[str, SSOIdentityAssertion], Awaitable[None]] = persist_sso_identity_assertion,
        post: SsoTokenEndpointPost = _post_sso_token_endpoint,
        getenv: Callable[[str], str | None] = os.getenv,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        memo_ttl_seconds: float = _ASSERTION_MEMO_TTL_SECONDS,
    ) -> None:
        self._fetch = fetch
        self._persist = persist
        self._post = post
        self._getenv = getenv
        self._now = now
        self._memo_ttl_seconds = memo_ttl_seconds
        self._memo: dict[str, tuple[SsoAssertionLookup, datetime]] = {}
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def fetch_usable(self, user_id: str) -> SsoAssertionLookup:
        if not user_id:
            return NoSsoAssertion()
        # Only a positive (usable) memo short-circuits the authoritative store read. A negative
        # verdict must never mask the store: the 401 challenge tells the user to re-login, which
        # writes a fresh row, and that remedy has to be honored the moment it lands. So an
        # Expired/Unavailable memo falls through to the store read below, where it still bounds
        # re-POST thrash for a row that is genuinely still expired.
        memoized = self._memoized(user_id)
        if isinstance(memoized, UsableSsoAssertion):
            return memoized
        try:
            return await self._fetch_usable_uncached(user_id)
        except Exception as exc:  # noqa: BLE001  # a store outage fails closed as absent, never a 500
            verbose_proxy_logger.warning(
                "SSO assertion store read failed for user_id=%s; treating as absent: %s", user_id, exc
            )
            return NoSsoAssertion()

    async def _fetch_usable_uncached(self, user_id: str) -> SsoAssertionLookup:
        stored = await self._fetch(user_id)
        if stored is None:
            return NoSsoAssertion()
        if not _assertion_is_expired(stored, self._now()):
            return self._memoize(user_id, UsableSsoAssertion(assertion=stored))
        # The stored row is still expired (a re-login would have made it usable above), so a recent
        # memo may stand in without another token-endpoint round trip: a negative verdict bounds
        # re-POST thrash, and a usable one is a concurrent refresh winner's token handed off before
        # its best-effort write-back landed. Either way the authoritative read has already run, so a
        # re-login is never blocked.
        memoized = self._memoized(user_id)
        if memoized is not None:
            return memoized
        async with self._lock(user_id):
            handed_off = self._memoized(user_id)
            if handed_off is not None:
                return handed_off
            rechecked = await self._fetch(user_id)
            if rechecked is None:
                return NoSsoAssertion()
            if not _assertion_is_expired(rechecked, self._now()):
                return self._memoize(user_id, UsableSsoAssertion(assertion=rechecked))
            return self._memoize(user_id, await self._refresh(user_id, rechecked))

    def _memoized(self, user_id: str) -> SsoAssertionLookup | None:
        entry = self._memo.get(user_id)
        if entry is None:
            return None
        lookup, valid_until = entry
        if self._now() >= valid_until:
            self._memo.pop(user_id, None)
            return None
        return lookup

    def _memoize(
        self, user_id: str, lookup: UsableSsoAssertion | ExpiredSsoAssertion | UnavailableSsoAssertion
    ) -> SsoAssertionLookup:
        """Record a terminal lookup in the one in-process result channel and return it.

        Every terminal state is memoized: Usable so egress calls stop paying a DB read and the
        single-flight winner can hand its token to waiters independent of the write-back;
        Expired so a dead refresh grant costs one token-endpoint POST per pod per negative TTL
        instead of one per egress call; Unavailable so a down IdP is not hammered. Absent is
        NEVER memoized, so a fresh login (a new row) is visible immediately. Only a Usable memo
        short-circuits ``fetch_usable``; a negative memo is consulted after the authoritative read
        confirms the row is still expired, so a re-login is honored the moment its row lands, on
        every pod, no matter that a stale negative entry is still cached. The Usable horizon is
        capped by the assertion's remaining life; the negative horizon only bounds re-POST thrash.
        """
        if isinstance(lookup, UsableSsoAssertion):
            horizon = self._now() + timedelta(seconds=self._memo_ttl_seconds)
            expires_at = lookup.assertion.expires_at
            if expires_at is not None:
                horizon = min(horizon, expires_at - timedelta(seconds=_ASSERTION_EXPIRY_BUFFER_SECONDS))
        else:
            horizon = self._now() + timedelta(seconds=_ASSERTION_NEGATIVE_MEMO_TTL_SECONDS)
        if horizon > self._now():
            self._memo[user_id] = (lookup, horizon)
        return lookup

    async def _refresh(
        self, user_id: str, stored: SSOIdentityAssertion
    ) -> UsableSsoAssertion | ExpiredSsoAssertion | UnavailableSsoAssertion:
        """One renewal attempt, its result classified by what the failure actually proves.

        A definitive verdict on the grant (no refresh token, a 4xx rejection, or a renewed
        token that is itself dead) reads Expired: only a re-login fixes it. A failure that
        proves nothing about the grant (IdP unreachable or 5xx, SSO client env missing) reads
        Unavailable: retrying or fixing the deployment fixes it, and a re-login instruction
        would be wrong. A rotated refresh token is persisted before the expiry verdict so a
        one-time rotation is never lost, and the persist is best-effort so a write-back
        failure cannot discard the renewed in-hand token.
        """
        if stored.refresh_token is None:
            return ExpiredSsoAssertion()
        client = _sso_refresh_client_from_env(self._getenv)
        if client is None:
            verbose_proxy_logger.warning(
                "Stored SSO assertion for user_id=%s is expired and the SSO client env "
                "(GENERIC_CLIENT_ID/GENERIC_TOKEN_ENDPOINT) is not set; renewal is unavailable "
                "until the deployment restores it.",
                user_id,
            )
            return UnavailableSsoAssertion()
        scope = " ".join(generic_sso_scopes(self._getenv))
        form = {
            "grant_type": "refresh_token",
            "refresh_token": stored.refresh_token.get_secret_value(),
            "client_id": client.client_id,
            **({"client_secret": client.client_secret.get_secret_value()} if client.client_secret else {}),
            **({"scope": scope} if scope else {}),
        }
        outcome = await self._post(client.token_endpoint, form)
        match outcome:
            case SsoRefreshRejected():
                return ExpiredSsoAssertion()
            case SsoRefreshUnreachable():
                return UnavailableSsoAssertion()
            case SsoRefreshGranted(body=body):
                pass
        rotated = body.get("refresh_token")
        carried_refresh = rotated if isinstance(rotated, str) and rotated else stored.refresh_token.get_secret_value()
        refreshed = assertion_from_sso_login(body.get("id_token"), carried_refresh)
        if refreshed is None:
            verbose_proxy_logger.warning(
                "SSO assertion refresh for user_id=%s returned no usable id_token; treating as expired.", user_id
            )
            return ExpiredSsoAssertion()
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
        if _assertion_is_expired(refreshed, self._now()):
            return ExpiredSsoAssertion()
        return UsableSsoAssertion(assertion=refreshed)

    def _lock(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock


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
