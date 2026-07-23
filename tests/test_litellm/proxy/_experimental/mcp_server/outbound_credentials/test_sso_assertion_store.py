"""Tests for the SSO identity assertion store (EMA subject-token capture).

Pins the contract of the store that PR 2's ``_id_jag`` subject-sourcing seam will read:
the carrier validates untyped IdP token-response values at the boundary, retention is
gated on an ``oauth2_id_jag`` server being registered, the row is encrypted at rest and
round-trips exactly, a store failure never escapes into the login path, and a salt-key
rotation re-encrypts stored rows like the sibling per-user credential tables.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    assertion_from_sso_login,
    ema_assertion_retention_enabled,
    fetch_sso_identity_assertion,
    persist_sso_identity_assertion,
    retain_sso_identity_assertion_for_ema,
    rotate_sso_identity_assertions_master_key,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.types.mcp import MCPAuth

SALT_KEY = "test-salt-key-for-sso-assertion-tests-1234"
SIGNING_KEY = "test-idp-signing-key-32-bytes-long-xxxx"
ISSUER = "https://idp.example.com"


@pytest.fixture(autouse=True)
def _set_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", SALT_KEY)


def _make_id_token(exp_offset: int = 3600, iss: str = ISSUER) -> str:
    return pyjwt.encode(
        {"iss": iss, "sub": "u1", "exp": int(time.time()) + exp_offset},
        SIGNING_KEY,
        algorithm="HS256",
    )


def _make_prisma(stored: dict, db_has_id_jag_server: bool = False):
    """A fake prisma client whose sso-assertion table reads and writes ``stored``
    (user_id -> assertion_b64), covering upsert, find_unique, find_many, and update.
    ``db_has_id_jag_server`` drives the retention gate's authoritative DB fallback;
    it is wired explicitly so the gate never reads a truthy bare MagicMock."""
    prisma = MagicMock()
    prisma.db.litellm_mcpservertable.find_first = AsyncMock(return_value=MagicMock() if db_has_id_jag_server else None)

    async def _upsert(where, data):
        stored[where["user_id"]] = data["update"]["assertion_b64"]

    async def _find_unique(where):
        blob = stored.get(where["user_id"])
        if blob is None:
            return None
        row = MagicMock()
        row.user_id = where["user_id"]
        row.assertion_b64 = blob
        return row

    async def _find_many():
        rows = []
        for user_id, blob in stored.items():
            row = MagicMock()
            row.user_id = user_id
            row.assertion_b64 = blob
            rows.append(row)
        return rows

    async def _update(where, data):
        stored[where["user_id"]] = data["assertion_b64"]

    prisma.db.litellm_ssoidentityassertion.upsert = AsyncMock(side_effect=_upsert)
    prisma.db.litellm_ssoidentityassertion.find_unique = AsyncMock(side_effect=_find_unique)
    prisma.db.litellm_ssoidentityassertion.find_many = AsyncMock(side_effect=_find_many)
    prisma.db.litellm_ssoidentityassertion.update = AsyncMock(side_effect=_update)
    return prisma


def _server_with_auth(auth_type):
    server = MagicMock()
    server.auth_type = auth_type
    return server


def test_assertion_from_sso_login_happy_path():
    token = _make_id_token()
    assertion = assertion_from_sso_login(token, "rt_1")
    assert assertion is not None
    assert assertion.id_token.get_secret_value() == token
    assert assertion.refresh_token is not None
    assert assertion.refresh_token.get_secret_value() == "rt_1"
    assert assertion.issuer == ISSUER
    assert assertion.expires_at is not None
    assert assertion.expires_at.timestamp() == pytest.approx(time.time() + 3600, abs=5)


def test_assertion_repr_never_leaks_token_material():
    token = _make_id_token()
    assertion = assertion_from_sso_login(token, "rt_secret_value")
    rendered = repr(assertion) + str(assertion)
    assert token not in rendered
    assert "rt_secret_value" not in rendered


@pytest.mark.parametrize("id_token", [None, "", "not-a-jwt", 12345, ["x"], {"a": 1}])
def test_assertion_from_sso_login_rejects_unusable_id_token(id_token):
    assert assertion_from_sso_login(id_token, "rt") is None


@pytest.mark.parametrize("refresh_token", [None, "", 123, ["rt"], {"rt": 1}])
def test_assertion_from_sso_login_drops_malformed_refresh_token(refresh_token):
    assertion = assertion_from_sso_login(_make_id_token(), refresh_token)
    assert assertion is not None
    assert assertion.refresh_token is None


def test_assertion_without_exp_or_iss_still_retained():
    token = pyjwt.encode({"sub": "u1"}, SIGNING_KEY, algorithm="HS256")
    assertion = assertion_from_sso_login(token, None)
    assert assertion is not None
    assert assertion.expires_at is None
    assert assertion.issuer is None


@pytest.mark.asyncio
async def test_retention_gate_requires_an_id_jag_server():
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", _make_prisma({}, db_has_id_jag_server=False)),
    ):
        manager.config_mcp_servers = {
            "s1": _server_with_auth(MCPAuth.oauth2),
            "s2": _server_with_auth(None),
        }
        assert await ema_assertion_retention_enabled() is False
        manager.config_mcp_servers = {
            "s1": _server_with_auth(MCPAuth.oauth2),
            "s2": _server_with_auth(MCPAuth.oauth2_id_jag),
        }
        assert await ema_assertion_retention_enabled() is True


@pytest.mark.asyncio
async def test_retention_gate_reads_the_db_when_config_declares_no_id_jag_server():
    """A DB-backed server added on another pod (or before this pod's DB load) must still enable
    retention off the authoritative DB row; False only when neither authority knows one."""
    with patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager:
        manager.config_mcp_servers = {"s1": _server_with_auth(MCPAuth.oauth2)}
        db_backed = _make_prisma({}, db_has_id_jag_server=True)
        with patch("litellm.proxy.proxy_server.prisma_client", db_backed):
            assert await ema_assertion_retention_enabled() is True
        db_backed.db.litellm_mcpservertable.find_first.assert_awaited_once_with(
            where={"auth_type": MCPAuth.oauth2_id_jag.value}
        )
        with patch("litellm.proxy.proxy_server.prisma_client", None):
            assert await ema_assertion_retention_enabled() is False


@pytest.mark.asyncio
async def test_retention_gate_never_consults_the_registry_snapshot():
    """The registry is a per-process snapshot of DB state, stale in either direction: trusting
    it positively would keep retaining bearer material after the last EMA server was removed on
    another pod, trusting it negatively would drop writes for one added elsewhere. The gate must
    judge only the config declaration and the DB row, so a stale snapshot listing an id_jag
    server changes nothing."""
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", _make_prisma({}, db_has_id_jag_server=False)),
    ):
        manager.config_mcp_servers = {}
        manager.get_registry.return_value = {"stale": _server_with_auth(MCPAuth.oauth2_id_jag)}
        assert await ema_assertion_retention_enabled() is False
        manager.get_registry.assert_not_called()


@pytest.mark.asyncio
async def test_retain_persists_when_only_the_db_knows_the_id_jag_server():
    stored = {}
    prisma = _make_prisma(stored, db_has_id_jag_server=True)
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
    ):
        manager.config_mcp_servers = {}
        await retain_sso_identity_assertion_for_ema(
            user_id="user-a", assertion=assertion_from_sso_login(_make_id_token(), None)
        )
    assert "user-a" in stored


@pytest.mark.asyncio
async def test_persist_and_fetch_round_trip_encrypted_at_rest():
    stored = {}
    prisma = _make_prisma(stored)
    token = _make_id_token()
    assertion = assertion_from_sso_login(token, "rt_1")
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await persist_sso_identity_assertion("user-a", assertion)
        fetched = await fetch_sso_identity_assertion("user-a")
    assert fetched is not None
    assert fetched.id_token.get_secret_value() == token
    assert fetched.refresh_token is not None
    assert fetched.refresh_token.get_secret_value() == "rt_1"
    assert fetched.issuer == assertion.issuer
    assert fetched.expires_at == assertion.expires_at
    assert token not in stored["user-a"]
    assert "rt_1" not in stored["user-a"]
    decrypted = decrypt_value_helper(stored["user-a"], "test", exception_type="debug")
    assert json.loads(decrypted)["id_token"] == token


@pytest.mark.asyncio
async def test_persist_overwrites_previous_login():
    stored = {}
    prisma = _make_prisma(stored)
    first = _make_id_token(exp_offset=100)
    second = _make_id_token(exp_offset=7200)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await persist_sso_identity_assertion("user-a", assertion_from_sso_login(first, None))
        await persist_sso_identity_assertion("user-a", assertion_from_sso_login(second, "rt_new"))
        fetched = await fetch_sso_identity_assertion("user-a")
    assert fetched is not None
    assert fetched.id_token.get_secret_value() == second
    assert fetched.refresh_token is not None


@pytest.mark.asyncio
async def test_fetch_missing_row_returns_none():
    prisma = _make_prisma({})
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        assert await fetch_sso_identity_assertion("nobody") is None


@pytest.mark.asyncio
async def test_fetch_undecryptable_row_returns_none():
    prisma = _make_prisma({"user-a": "not-an-encrypted-blob"})
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        assert await fetch_sso_identity_assertion("user-a") is None


@pytest.mark.asyncio
async def test_fetch_unparseable_payload_returns_none():
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    prisma = _make_prisma({"user-a": encrypt_value_helper("]]not json")})
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        assert await fetch_sso_identity_assertion("user-a") is None


@pytest.mark.asyncio
async def test_retain_noop_when_no_id_jag_server():
    stored = {}
    prisma = _make_prisma(stored)
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
    ):
        manager.config_mcp_servers = {"s1": _server_with_auth(MCPAuth.oauth2)}
        await retain_sso_identity_assertion_for_ema(
            user_id="user-a", assertion=assertion_from_sso_login(_make_id_token(), None)
        )
    prisma.db.litellm_ssoidentityassertion.upsert.assert_not_called()
    assert stored == {}


@pytest.mark.asyncio
async def test_retain_persists_when_id_jag_server_registered():
    stored = {}
    prisma = _make_prisma(stored)
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
    ):
        manager.config_mcp_servers = {"s1": _server_with_auth(MCPAuth.oauth2_id_jag)}
        await retain_sso_identity_assertion_for_ema(
            user_id="user-a", assertion=assertion_from_sso_login(_make_id_token(), None)
        )
    assert "user-a" in stored


@pytest.mark.asyncio
async def test_retain_none_assertion_never_consults_gate_or_store():
    gate = MagicMock()
    with patch(
        "litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store.ema_assertion_retention_enabled",
        gate,
    ):
        await retain_sso_identity_assertion_for_ema(user_id="user-a", assertion=None)
    gate.assert_not_called()


@pytest.mark.asyncio
async def test_retain_swallows_store_failure():
    prisma = MagicMock()
    prisma.db.litellm_ssoidentityassertion.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    with (
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as manager,
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
    ):
        manager.config_mcp_servers = {"s1": _server_with_auth(MCPAuth.oauth2_id_jag)}
        await retain_sso_identity_assertion_for_ema(
            user_id="user-a", assertion=assertion_from_sso_login(_make_id_token(), None)
        )


@pytest.mark.asyncio
async def test_rotation_reencrypts_under_new_key(monkeypatch):
    stored = {}
    prisma = _make_prisma(stored)
    token = _make_id_token()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await persist_sso_identity_assertion("user-a", assertion_from_sso_login(token, None))
    original_blob = stored["user-a"]

    new_key = "rotated-sso-assertion-salt-key-5678"
    await rotate_sso_identity_assertions_master_key(prisma_client=prisma, new_master_key=new_key)
    assert stored["user-a"] != original_blob

    monkeypatch.setenv("LITELLM_SALT_KEY", new_key)
    decrypted = decrypt_value_helper(stored["user-a"], "test", exception_type="debug")
    assert decrypted is not None
    assert json.loads(decrypted)["id_token"] == token


@pytest.mark.asyncio
async def test_rotation_skips_unreadable_rows_but_rotates_readable_ones():
    stored = {"good": None, "bad": "garbage-blob"}
    prisma = _make_prisma(stored)
    token = _make_id_token()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        await persist_sso_identity_assertion("good", assertion_from_sso_login(token, None))
    good_blob_before = stored["good"]
    await rotate_sso_identity_assertions_master_key(prisma_client=prisma, new_master_key="another-new-salt-key-0000")
    assert stored["bad"] == "garbage-blob"
    assert stored["good"] != good_blob_before


# -- LiveSsoAssertionSource: the resolver-facing usable-assertion lookup + refresh -------------


from datetime import datetime, timedelta, timezone  # noqa: E402

from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (  # noqa: E402
    ExpiredSsoAssertion,
    LiveSsoAssertionSource,
    NoSsoAssertion,
    SsoRefreshGranted,
    SsoRefreshRejected,
    SsoRefreshUnreachable,
    SSOIdentityAssertion,
    UnavailableSsoAssertion,
    UsableSsoAssertion,
)
from pydantic import SecretStr  # noqa: E402

_SSO_ENV = {
    "GENERIC_CLIENT_ID": "sso-client",
    "GENERIC_CLIENT_SECRET": "sso-secret",
    "GENERIC_TOKEN_ENDPOINT": "https://idp.example.com/token",
}


def _stored(expires_in_seconds: int | None, refresh_token: str | None = "rt_stored") -> SSOIdentityAssertion:
    return SSOIdentityAssertion(
        id_token=SecretStr(_make_id_token()),
        refresh_token=SecretStr(refresh_token) if refresh_token else None,
        expires_at=(
            datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
            if expires_in_seconds is not None
            else None
        ),
    )


def _source(stored: SSOIdentityAssertion | None, post_bodies: list, env: dict | None = None):
    fetches: list[str] = []
    persisted: list[tuple[str, SSOIdentityAssertion]] = []
    posts: list[tuple[str, dict]] = []
    state = {"stored": stored}

    async def fetch(user_id: str):
        fetches.append(user_id)
        return state["stored"]

    async def persist(user_id: str, assertion: SSOIdentityAssertion):
        persisted.append((user_id, assertion))
        state["stored"] = assertion

    async def post(url: str, form: dict):
        posts.append((url, dict(form)))
        body = post_bodies.pop(0) if post_bodies else None
        return SsoRefreshGranted(body=body) if body is not None else SsoRefreshUnreachable()

    environment = _SSO_ENV if env is None else env
    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=environment.get)
    return source, fetches, persisted, posts


@pytest.mark.asyncio
async def test_source_empty_user_id_is_absent_without_a_db_read():
    source, fetches, _, _ = _source(_stored(3600), [])
    assert isinstance(await source.fetch_usable(""), NoSsoAssertion)
    assert fetches == []


@pytest.mark.asyncio
async def test_source_missing_row_is_absent():
    source, _, _, posts = _source(None, [])
    assert isinstance(await source.fetch_usable("alice"), NoSsoAssertion)
    assert posts == []


@pytest.mark.asyncio
async def test_source_unexpired_assertion_is_usable_without_refresh():
    stored = _stored(3600)
    source, _, _, posts = _source(stored, [])
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert lookup.assertion.id_token.get_secret_value() == stored.id_token.get_secret_value()
    assert posts == []


@pytest.mark.asyncio
async def test_source_near_expiry_assertion_counts_as_expired():
    """A token inside the buffer would die mid-exchange; it must refresh, not be served."""
    source, _, _, posts = _source(_stored(10, refresh_token=None), [])
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert posts == []


@pytest.mark.asyncio
async def test_source_expired_with_refresh_renews_persists_and_returns_usable():
    new_id_token = _make_id_token(exp_offset=7200)
    source, _, persisted, posts = _source(
        _stored(-100), [{"id_token": new_id_token, "refresh_token": "rt_rotated", "access_token": "at"}]
    )
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert lookup.assertion.id_token.get_secret_value() == new_id_token
    assert lookup.assertion.refresh_token is not None
    assert lookup.assertion.refresh_token.get_secret_value() == "rt_rotated"
    assert len(persisted) == 1 and persisted[0][0] == "alice"
    url, form = posts[0]
    assert url == "https://idp.example.com/token"
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt_stored"
    assert form["client_id"] == "sso-client"
    assert form["client_secret"] == "sso-secret"


@pytest.mark.asyncio
async def test_source_refresh_requests_openid_scope_so_the_idp_returns_an_id_token():
    """The refresh grant must carry a scope containing ``openid`` or the IdP returns no id_token,
    stranding a renewable assertion as expired and forcing a needless re-login (the round-9
    finding). With no GENERIC_SCOPE configured it falls to the same default the login uses, which
    contains ``openid``. Dropping the scope from the refresh form makes this fail."""
    source, _, _, posts = _source(_stored(-100), [{"id_token": _make_id_token(exp_offset=7200)}])
    await source.fetch_usable("alice")
    _, form = posts[0]
    assert "openid" in form["scope"].split(" ")


@pytest.mark.asyncio
async def test_source_refresh_scope_matches_the_configured_login_scope():
    """Refresh requests exactly the scopes the login was granted, read through the same shared
    owner off the injected env, so the two can never diverge and the refresh stays within RFC 6749
    granted scope. Sourcing the scope from the global environment instead of the injected getenv
    makes this fail."""
    env = {**_SSO_ENV, "GENERIC_SCOPE": "openid email offline_access"}
    source, _, _, posts = _source(_stored(-100), [{"id_token": _make_id_token(exp_offset=7200)}], env=env)
    await source.fetch_usable("alice")
    _, form = posts[0]
    assert form["scope"] == "openid email offline_access"


@pytest.mark.asyncio
async def test_source_refresh_without_rotated_token_carries_the_stored_one_forward():
    source, _, _, _ = _source(_stored(-100), [{"id_token": _make_id_token(exp_offset=7200)}])
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert lookup.assertion.refresh_token is not None
    assert lookup.assertion.refresh_token.get_secret_value() == "rt_stored"


@pytest.mark.asyncio
async def test_source_expired_without_refresh_token_is_expired():
    source, _, _, posts = _source(_stored(-100, refresh_token=None), [])
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"id_token": ""}, {"id_token": 123}, {"access_token": "at-only"}])
async def test_source_refresh_granting_no_usable_id_token_is_expired(body):
    source, _, persisted, _ = _source(_stored(-100), [body])
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert persisted == []


@pytest.mark.asyncio
async def test_source_refresh_rejection_is_expired_and_unreachable_is_unavailable():
    """A 4xx is a verdict on the grant (re-login fixes it); an unreachable IdP proves nothing
    about the grant, so the user must NOT be told to re-login for a transient outage."""

    async def fetch(user_id: str):
        return _stored(-100)

    async def persist(user_id, assertion):
        return None

    async def rejected_post(url, form):
        return SsoRefreshRejected(status_code=400)

    async def unreachable_post(url, form):
        return SsoRefreshUnreachable()

    rejected_source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=rejected_post, getenv=_SSO_ENV.get)
    assert isinstance(await rejected_source.fetch_usable("alice"), ExpiredSsoAssertion)

    unreachable_source = LiveSsoAssertionSource(
        fetch=fetch, persist=persist, post=unreachable_post, getenv=_SSO_ENV.get
    )
    assert isinstance(await unreachable_source.fetch_usable("alice"), UnavailableSsoAssertion)


@pytest.mark.asyncio
async def test_source_missing_sso_env_is_unavailable_not_expired():
    """Env absence is a deployment problem; a re-login cannot fix it (SSO needs the same env),
    so the state must be Unavailable, not Expired."""
    source, _, _, posts = _source(_stored(-100), [], env={})
    assert isinstance(await source.fetch_usable("alice"), UnavailableSsoAssertion)
    assert posts == []


@pytest.mark.asyncio
async def test_source_dead_grant_is_negative_cached_one_post_per_window():
    """A dead refresh grant costs one token-endpoint POST per negative-TTL window, not one per
    egress call, and the negative entry expires so a re-login is honored within the window."""
    clock = {"now": time.time()}
    posts: list[dict] = []
    state = {"stored": _stored(-100)}

    async def fetch(user_id: str):
        return state["stored"]

    async def persist(user_id, assertion):
        state["stored"] = assertion

    async def post(url, form):
        posts.append(dict(form))
        return SsoRefreshRejected(status_code=400)

    source = LiveSsoAssertionSource(
        fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get, now=lambda: clock["now"]
    )
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert len(posts) == 1

    clock["now"] = clock["now"] + 31.0
    state["stored"] = _stored(3600)
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_source_relogin_within_the_negative_window_is_honored_immediately():
    """The 401 challenge tells the user to re-login; the fresh row must be served the moment it
    lands, even while a negative verdict is still memoized (no clock advance, same pod). The
    negative memo may only bound re-POST thrash for a row that is genuinely still expired; it must
    never mask the authoritative read, or the recovery the challenge asks for silently fails."""
    posts: list[dict] = []
    state = {"stored": _stored(-100)}

    async def fetch(user_id: str):
        return state["stored"]

    async def persist(user_id, assertion):
        state["stored"] = assertion

    async def post(url, form):
        posts.append(dict(form))
        return SsoRefreshRejected(status_code=400)

    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get)
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert len(posts) == 1

    state["stored"] = _stored(3600)  # re-login lands in the same negative window, no clock advance
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert len(posts) == 1  # the fresh usable row short-circuits before any refresh POST


@pytest.mark.asyncio
async def test_source_waiter_of_an_expired_winner_does_not_respend_the_grant():
    """The hand-off channel carries negative verdicts too: a waiter behind a winner whose
    refresh was rejected reads the memoized Expired instead of re-POSTing the spent token."""
    import asyncio

    posts: list[dict] = []

    async def fetch(user_id: str):
        await asyncio.sleep(0)
        return _stored(-100)

    async def persist(user_id, assertion):
        return None

    async def post(url, form):
        posts.append(dict(form))
        await asyncio.sleep(0)
        return SsoRefreshRejected(status_code=400)

    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get)
    results = await asyncio.gather(source.fetch_usable("alice"), source.fetch_usable("alice"))
    assert all(isinstance(r, ExpiredSsoAssertion) for r in results)
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_source_concurrent_expired_fetches_refresh_once():
    """Two flights hit the same expired assertion; the single-flight lock makes the second
    re-read the row the first refreshed instead of spending the refresh token again. The fake
    post yields to the loop so the flights genuinely interleave."""
    import asyncio

    fetches: list[str] = []
    posts: list[tuple[str, dict]] = []
    state = {"stored": _stored(-100)}
    bodies = [{"id_token": _make_id_token(exp_offset=7200)}, {"id_token": _make_id_token(exp_offset=7200)}]

    async def fetch(user_id: str):
        await asyncio.sleep(0)
        fetches.append(user_id)
        return state["stored"]

    async def persist(user_id: str, assertion: SSOIdentityAssertion):
        await asyncio.sleep(0)
        state["stored"] = assertion

    async def post(url: str, form: dict):
        posts.append((url, dict(form)))
        await asyncio.sleep(0)
        return SsoRefreshGranted(body=bodies.pop(0))

    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get)
    results = await asyncio.gather(source.fetch_usable("alice"), source.fetch_usable("alice"))
    assert all(isinstance(r, UsableSsoAssertion) for r in results)
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_source_memoizes_usable_lookups_within_ttl():
    """A usable assertion is served from the in-process memo, so repeated egress calls do not
    pay a DB read per call; the memo expires with the TTL and the row is re-read."""
    clock = {"now": time.time()}
    fetches: list[str] = []
    # No declared expiry, so the cache horizon is the default TTL rather than the token's own
    # expiry (an assertion that declares one is cached exactly until it becomes refresh-eligible).
    stored = _stored(None)

    async def fetch(user_id: str):
        fetches.append(user_id)
        return stored

    async def persist(user_id, assertion):
        return None

    async def post(url, form):
        return None

    source = LiveSsoAssertionSource(
        fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get, now=lambda: clock["now"], cache_ttl_seconds=60.0
    )
    first = await source.fetch_usable("alice")
    second = await source.fetch_usable("alice")
    assert isinstance(first, UsableSsoAssertion) and isinstance(second, UsableSsoAssertion)
    assert fetches == ["alice"]
    clock["now"] = clock["now"] + 61.0
    third = await source.fetch_usable("alice")
    assert isinstance(third, UsableSsoAssertion)
    assert fetches == ["alice", "alice"]


@pytest.mark.asyncio
async def test_source_never_memoizes_absence_so_a_fresh_login_is_seen_immediately():
    fetches: list[str] = []
    state = {"stored": None}

    async def fetch(user_id: str):
        fetches.append(user_id)
        return state["stored"]

    async def persist(user_id, assertion):
        return None

    async def post(url, form):
        return None

    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get)
    assert isinstance(await source.fetch_usable("alice"), NoSsoAssertion)
    state["stored"] = _stored(3600)
    assert isinstance(await source.fetch_usable("alice"), UsableSsoAssertion)
    assert fetches == ["alice", "alice"]


@pytest.mark.asyncio
async def test_source_store_outage_reads_as_absent_and_warm_memo_survives_it():
    """A store read failure must fail closed as absent (never raise into egress), and a warm
    memo entry keeps serving through the outage."""
    calls = {"n": 0}
    stored = _stored(None)

    async def flaky_fetch(user_id: str):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("db down")
        return stored

    async def persist(user_id, assertion):
        return None

    async def post(url, form):
        return None

    clock = {"now": time.time()}
    source = LiveSsoAssertionSource(
        fetch=flaky_fetch,
        persist=persist,
        post=post,
        getenv=_SSO_ENV.get,
        now=lambda: clock["now"],
        cache_ttl_seconds=60.0,
    )
    assert isinstance(await source.fetch_usable("alice"), UsableSsoAssertion)
    assert isinstance(await source.fetch_usable("alice"), UsableSsoAssertion)
    clock["now"] = clock["now"] + 61.0
    assert isinstance(await source.fetch_usable("alice"), NoSsoAssertion)


@pytest.mark.asyncio
async def test_source_persist_failure_after_refresh_still_serves_the_in_hand_token():
    """A write-back failure is not a read failure: the freshly refreshed assertion is valid
    and in hand, and discarding it would also strand a one-time rotated refresh token."""
    new_id_token = _make_id_token(exp_offset=7200)
    posts: list[tuple[str, dict]] = []

    async def fetch(user_id: str):
        return _stored(-100)

    async def failing_persist(user_id, assertion):
        raise RuntimeError("db write down")

    async def post(url, form):
        posts.append((url, dict(form)))
        return SsoRefreshGranted(body={"id_token": new_id_token, "refresh_token": "rt_rotated"})

    source = LiveSsoAssertionSource(fetch=fetch, persist=failing_persist, post=post, getenv=_SSO_ENV.get)
    lookup = await source.fetch_usable("alice")
    assert isinstance(lookup, UsableSsoAssertion)
    assert lookup.assertion.id_token.get_secret_value() == new_id_token
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_source_refresh_returning_expired_token_reads_expired_but_persists_rotation():
    """Usable has ONE construction gate: a refreshed assertion passes the same expiry
    predicate as a stored one, so an IdP handing back a dead token reads Expired instead of
    being sent to an exchange; the rotated refresh token is still persisted."""
    expired_new = _make_id_token(exp_offset=-10)
    persisted: list[tuple[str, SSOIdentityAssertion]] = []

    async def fetch(user_id: str):
        return _stored(-100)

    async def persist(user_id, assertion):
        persisted.append((user_id, assertion))

    async def post(url, form):
        return SsoRefreshGranted(body={"id_token": expired_new, "refresh_token": "rt_rotated"})

    source = LiveSsoAssertionSource(fetch=fetch, persist=persist, post=post, getenv=_SSO_ENV.get)
    assert isinstance(await source.fetch_usable("alice"), ExpiredSsoAssertion)
    assert len(persisted) == 1
    stored_assertion = persisted[0][1]
    assert stored_assertion.refresh_token is not None
    assert stored_assertion.refresh_token.get_secret_value() == "rt_rotated"


@pytest.mark.asyncio
async def test_source_waiter_gets_the_winners_token_when_persist_failed():
    """The memo is the single-flight hand-off: with the write-back failing and a one-time
    refresh token, the waiter must receive the winner's in-hand assertion from the memo under
    the lock instead of re-reading the stale row and re-spending the refresh grant."""
    import asyncio

    new_id_token = _make_id_token(exp_offset=7200)
    posts: list[dict] = []

    async def fetch(user_id: str):
        await asyncio.sleep(0)
        return _stored(-100)

    async def failing_persist(user_id, assertion):
        raise RuntimeError("db write down")

    async def post(url, form):
        posts.append(dict(form))
        await asyncio.sleep(0)
        if form["refresh_token"] != "rt_stored" or len(posts) > 1:
            return SsoRefreshRejected(status_code=400)
        return SsoRefreshGranted(body={"id_token": new_id_token, "refresh_token": "rt_rotated"})

    source = LiveSsoAssertionSource(fetch=fetch, persist=failing_persist, post=post, getenv=_SSO_ENV.get)
    results = await asyncio.gather(source.fetch_usable("alice"), source.fetch_usable("alice"))
    assert all(isinstance(r, UsableSsoAssertion) for r in results)
    assert len(posts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,body,expected_type",
    [
        (400, {"error": "invalid_grant"}, SsoRefreshRejected),
        (401, {"error": "invalid_client"}, SsoRefreshRejected),
        (429, {"error": "rate_limited"}, SsoRefreshUnreachable),
        (400, "not-json-object", SsoRefreshUnreachable),
        (400, None, SsoRefreshUnreachable),
        (502, {"error": "invalid_grant"}, SsoRefreshUnreachable),
    ],
)
async def test_post_helper_rejection_requires_a_proven_oauth_verdict(status_code, body, expected_type, monkeypatch):
    """Rejected needs proof: a non-429 4xx carrying an RFC 6749 error code. A rate limit, a 4xx
    without the error object (an intermediary, not the token endpoint), or any 5xx proves
    nothing about the grant and must not send the user to re-login."""
    import json as _json

    import httpx

    from litellm.proxy._experimental.mcp_server.outbound_credentials import sso_assertion_store

    response = MagicMock()
    response.status_code = status_code
    if body is None:
        response.json.side_effect = _json.JSONDecodeError("x", "y", 0)
    else:
        response.json.return_value = body
    response.raise_for_status.side_effect = httpx.HTTPStatusError("err", request=MagicMock(), response=response)
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", lambda llm_provider: client)
    outcome = await sso_assertion_store._post_sso_token_endpoint("https://idp.example.com/token", {"a": "b"})
    assert isinstance(outcome, expected_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, [], "string", 42])
async def test_post_helper_2xx_non_object_body_is_unreachable_not_a_crash(body, monkeypatch):
    """A 2xx whose body is not a JSON object must read Unreachable; letting it construct a
    Granted outcome detonates downstream validation into the broad catch, which misreads the
    state as Absent and misdirects the user to a 412 instead of a retryable failure."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = body
    client.post = AsyncMock(return_value=response)
    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", lambda llm_provider: client)
    from litellm.proxy._experimental.mcp_server.outbound_credentials import sso_assertion_store

    outcome = await sso_assertion_store._post_sso_token_endpoint("https://idp.example.com/token", {"a": "b"})
    assert isinstance(outcome, SsoRefreshUnreachable)


@pytest.mark.asyncio
async def test_post_helper_transport_failure_is_unreachable(monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(side_effect=ConnectionError("down"))
    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", lambda llm_provider: client)
    from litellm.proxy._experimental.mcp_server.outbound_credentials import sso_assertion_store

    outcome = await sso_assertion_store._post_sso_token_endpoint("https://idp.example.com/token", {"a": "b"})
    assert isinstance(outcome, SsoRefreshUnreachable)
