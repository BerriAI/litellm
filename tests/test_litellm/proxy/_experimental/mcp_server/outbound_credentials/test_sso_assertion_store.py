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
    prisma.db.litellm_mcpservertable.find_first = AsyncMock(
        return_value=MagicMock() if db_has_id_jag_server else None
    )

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
