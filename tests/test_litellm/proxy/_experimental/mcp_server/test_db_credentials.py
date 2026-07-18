"""
Tests for the encrypted-at-rest persistence of MCP user credentials.

The ``LiteLLM_MCPUserCredentials.credential_b64`` column previously stored
both BYOK API keys and OAuth2 access tokens as plain ``urlsafe_b64encode``
of the raw value, leaving credentials readable from any DB read. The fix
runs every write through ``encrypt_value_helper`` (nacl SecretBox) and
keeps a plain-base64 fallback on read so existing rows continue to work.
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    _decode_user_credential,
    _prepare_mcp_server_data,
    get_user_credential,
    get_user_idp_grant,
    get_user_oauth_credential,
    is_oauth_credential_expired,
    list_user_oauth_credentials,
    resolve_valid_user_oauth_token,
    rotate_mcp_user_credentials_master_key,
    rotate_mcp_user_env_vars_master_key,
    store_user_credential,
    store_user_idp_grant,
    store_user_oauth_credential,
)
from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.mcp import MCPAuth, MCPTransport

SALT_KEY = "test-salt-key-for-byok-credential-tests-1234"


@pytest.fixture(autouse=True)
def _set_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", SALT_KEY)


def _make_prisma_with_existing(row):
    """Build a MagicMock prisma_client whose user-credentials table returns ``row``
    for find_unique and behaves async-correctly for upsert/find_many."""
    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_mcpusercredentials.upsert = AsyncMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[])
    return prisma


def _legacy_row(payload: str):
    """A row exactly as the pre-fix code would have written it: plain
    ``urlsafe_b64encode`` of the raw payload, no encryption."""
    row = MagicMock()
    row.credential_b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    row.user_id = "alice"
    row.server_id = "srv-1"
    return row


def _identity_server(**overrides):
    base = dict(
        url="https://up.example.com/mcp",
        auth_type="oauth2",
        oauth2_flow="authorization_code",
        authorization_url="https://idp.example.com/authorize",
        token_url="https://idp.example.com/token",
        registration_url="https://idp.example.com/register",
        credentials={"client_id": "cid", "client_secret": "csec", "scopes": ["a"]},
        server_name="srv",
        description="d",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": "https://other.example.com/mcp"},
        {"spec_path": "https://up.example.com/openapi.json"},
        {"auth_type": "oauth_delegate"},
        {"oauth2_flow": "client_credentials"},
        {"authorization_url": "https://other.example.com/authorize"},
        {"token_url": "https://other.example.com/token"},
        {"registration_url": "https://other.example.com/register"},
        {"credentials": {"client_id": "new", "client_secret": "csec", "scopes": ["a"]}},
        {"credentials": {"client_id": "cid", "client_secret": "rotated", "scopes": ["a"]}},
        {"credentials": {"client_id": "cid", "client_secret": "csec", "scopes": ["b"]}},
    ],
)
def test_mcp_oauth_token_identity_changes_on_mint_relevant_fields(overrides):
    from litellm.proxy._experimental.mcp_server.db import mcp_oauth_token_identity

    assert mcp_oauth_token_identity(_identity_server()) != mcp_oauth_token_identity(_identity_server(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"server_name": "renamed"},
        {"description": "changed"},
    ],
)
def test_mcp_oauth_token_identity_stable_on_non_mint_fields(overrides):
    from litellm.proxy._experimental.mcp_server.db import mcp_oauth_token_identity

    assert mcp_oauth_token_identity(_identity_server()) == mcp_oauth_token_identity(_identity_server(**overrides))


def _encrypted_creds_json(client_id: str = "cid", client_secret: str = "csec") -> str:
    from litellm.proxy._experimental.mcp_server.db import encrypt_credentials

    encrypted = encrypt_credentials(
        credentials={"client_id": client_id, "client_secret": client_secret, "scopes": ["a"]},
        encryption_key=None,
    )
    return json.dumps(encrypted)


def test_mcp_oauth_token_identity_stable_across_reencryption():
    """Stored client_id/client_secret are NaCl-encrypted with a fresh nonce on every write, so two
    saves of the SAME plaintext produce different ciphertext. The identity must compare decrypted
    values; comparing ciphertext would flag every routine save as a mint-relevant change and purge
    per-user tokens that are still valid."""
    from litellm.proxy._experimental.mcp_server.db import mcp_oauth_token_identity

    first = _encrypted_creds_json()
    second = _encrypted_creds_json()
    assert first != second

    assert mcp_oauth_token_identity(_identity_server(credentials=first)) == mcp_oauth_token_identity(
        _identity_server(credentials=second)
    )


def test_mcp_oauth_token_identity_detects_change_under_encryption():
    from litellm.proxy._experimental.mcp_server.db import mcp_oauth_token_identity

    unchanged = _identity_server(credentials=_encrypted_creds_json())
    changed = _identity_server(credentials=_encrypted_creds_json(client_id="other"))
    assert mcp_oauth_token_identity(unchanged) != mcp_oauth_token_identity(changed)


def _oauth_row(user_id: str, server_id: str = "srv-1"):
    """A stored per-user OAuth token row (payload tagged type=oauth2, legacy plain-base64 encoding)."""
    row = _legacy_row(json.dumps({"type": "oauth2", "access_token": "tok-" + user_id}))
    row.user_id = user_id
    row.server_id = server_id
    return row


def _byok_row(user_id: str, server_id: str = "srv-1"):
    """A stored BYOK API key row: the same column, but the payload is a plain string, not OAuth JSON."""
    row = _legacy_row("sk-byok-" + user_id)
    row.user_id = user_id
    row.server_id = server_id
    return row


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_invalidates_each_user():
    """The purge must route each (user, server) row through the invalidator exactly once."""
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_oauth_row("alice"), _oauth_row("bob")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(return_value=2)

    invalidations = []

    async def record_invalidation(user_id: str, server_id: str) -> None:
        invalidations.append((user_id, server_id))

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1", invalidate_token_cache=record_invalidation)

    assert purged == 2
    prisma.db.litellm_mcpusercredentials.delete_many.assert_awaited_once_with(
        where={"server_id": "srv-1", "user_id": {"in": ["alice", "bob"]}}
    )
    assert set(invalidations) == {("alice", "srv-1"), ("bob", "srv-1")}


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_spares_byok_rows():
    """Regression: the purge used to delete_many on server_id alone, wiping BYOK API keys that share
    the LiteLLM_MCPUserCredentials table. Only rows holding an OAuth2 payload may be deleted (one
    batched query filtered to their user_ids), and only their users' token caches invalidated."""
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_byok_row("carol"), _oauth_row("alice")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(return_value=1)

    invalidations = []

    async def record_invalidation(user_id: str, server_id: str) -> None:
        invalidations.append((user_id, server_id))

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1", invalidate_token_cache=record_invalidation)

    assert purged == 1
    prisma.db.litellm_mcpusercredentials.delete_many.assert_awaited_once_with(
        where={"server_id": "srv-1", "user_id": {"in": ["alice"]}}
    )
    assert invalidations == [("alice", "srv-1")]


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_all_byok_is_noop():
    """An api_key (BYOK-only) server whose identity tuple changes (e.g. its url) must purge nothing."""
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_byok_row("carol"), _byok_row("dave")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock()

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1")

    assert purged == 0
    prisma.db.litellm_mcpusercredentials.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_defaults_to_manager_invalidator(monkeypatch):
    """When no invalidator is injected, the purge must resolve to the manager's shared
    invalidate_user_oauth_token_cache, the single point covering both the legacy per-user token cache
    and the v2 per-user OAuth token store; a wrong or no-op default silently leaves every cache
    serving tokens minted for the superseded config."""
    from litellm.proxy._experimental.mcp_server import mcp_server_manager
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_oauth_row("alice")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(return_value=1)

    shared_invalidator = AsyncMock()
    monkeypatch.setattr(
        mcp_server_manager.global_mcp_server_manager,
        "invalidate_user_oauth_token_cache",
        shared_invalidator,
    )

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1")

    assert purged == 1
    shared_invalidator.assert_awaited_once_with("alice", "srv-1")


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_logs_raced_rows(monkeypatch):
    from litellm.proxy._experimental.mcp_server import db as db_module
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_oauth_row("alice")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(return_value=0)
    warning = MagicMock()
    monkeypatch.setattr(db_module.verbose_proxy_logger, "warning", warning)

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1", invalidate_token_cache=AsyncMock())

    assert purged == 0
    warning.assert_called_once()


@pytest.mark.asyncio
async def test_delete_mcp_server_invalidates_cached_tokens_for_enumerated_users():
    """Deleting a server must invalidate each enumerated user's cached per-user token: the caches are
    keyed by (user_id, server_id), so a re-created server reusing the same server_id would otherwise
    serve tokens minted for the deleted server until TTL."""
    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = MagicMock()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=MagicMock(server_id="srv-1"))
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[_oauth_row("alice"), _byok_row("bob")])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(return_value=2)
    prisma.db.litellm_mcpuserenvvars.delete_many = AsyncMock(return_value=0)

    invalidations = []

    async def record_invalidation(user_id: str, server_id: str) -> None:
        invalidations.append((user_id, server_id))

    deleted = await delete_mcp_server(prisma, "srv-1", invalidate_token_cache=record_invalidation)

    assert deleted is not None
    assert set(invalidations) == {("alice", "srv-1"), ("bob", "srv-1")}


@pytest.mark.asyncio
async def test_delete_mcp_server_returns_none_without_cleanup_when_server_missing():
    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = MagicMock()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=None)
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock()

    deleted = await delete_mcp_server(prisma, "srv-1", invalidate_token_cache=AsyncMock())

    assert deleted is None
    prisma.db.litellm_mcpusercredentials.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_purge_user_oauth_credentials_for_server_noop_when_empty():
    from litellm.proxy._experimental.mcp_server.db import purge_user_oauth_credentials_for_server

    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock()

    purged = await purge_user_oauth_credentials_for_server(prisma, "srv-1")

    assert purged == 0
    prisma.db.litellm_mcpusercredentials.delete_many.assert_not_awaited()


def _stored_value(prisma) -> str:
    """Pull the credential_b64 value passed to the most recent upsert call."""
    call = prisma.db.litellm_mcpusercredentials.upsert.call_args
    data = call.kwargs["data"]
    create_value = data["create"]["credential_b64"]
    update_value = data["update"]["credential_b64"]
    assert create_value == update_value, "create/update must agree"
    return create_value


# ── BYOK round-trip ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_user_credential_does_not_persist_plaintext():
    # Stored bytes must not just be base64 of the secret — that's the regression.
    secret = "sk-proj-very-secret-byok-key"
    prisma = _make_prisma_with_existing(row=None)

    await store_user_credential(prisma, "alice", "srv-1", secret)

    stored = _stored_value(prisma)
    plain_b64 = base64.urlsafe_b64encode(secret.encode()).decode()
    assert stored != plain_b64
    # And the secret must not appear anywhere in a plain-b64 decode of the column.
    try:
        decoded_bytes = base64.urlsafe_b64decode(stored)
    except Exception:
        decoded_bytes = b""
    assert secret.encode() not in decoded_bytes


@pytest.mark.asyncio
async def test_byok_round_trip_returns_plaintext():
    secret = "sk-proj-very-secret-byok-key"
    prisma = _make_prisma_with_existing(row=None)
    await store_user_credential(prisma, "alice", "srv-1", secret)

    stored = _stored_value(prisma)
    row = MagicMock()
    row.credential_b64 = stored
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=row)

    result = await get_user_credential(prisma, "alice", "srv-1")
    assert result == secret


@pytest.mark.asyncio
async def test_byok_get_returns_plaintext_for_legacy_row():
    # Backward-compat: rows persisted by the pre-fix code (plain base64) must
    # still decrypt-or-decode cleanly.
    legacy_secret = "legacy-byok-key"
    prisma = _make_prisma_with_existing(row=_legacy_row(legacy_secret))

    result = await get_user_credential(prisma, "alice", "srv-1")
    assert result == legacy_secret


@pytest.mark.asyncio
async def test_byok_get_returns_none_for_missing_row():
    prisma = _make_prisma_with_existing(row=None)
    result = await get_user_credential(prisma, "alice", "srv-1")
    assert result is None


# ── OAuth2 round-trip ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_user_oauth_credential_does_not_persist_plaintext():
    access_token = "ya29.a0AfH6SMBverysecretaccesstoken"
    prisma = _make_prisma_with_existing(row=None)

    await store_user_oauth_credential(prisma, "alice", "srv-1", access_token, refresh_token="rfr-xyz")

    stored = _stored_value(prisma)
    try:
        decoded_bytes = base64.urlsafe_b64decode(stored)
    except Exception:
        decoded_bytes = b""
    assert access_token.encode() not in decoded_bytes
    assert b"rfr-xyz" not in decoded_bytes


@pytest.mark.asyncio
async def test_oauth_round_trip_returns_payload():
    access_token = "ya29.a0AfH6SMBverysecretaccesstoken"
    prisma = _make_prisma_with_existing(row=None)
    await store_user_oauth_credential(
        prisma,
        "alice",
        "srv-1",
        access_token,
        refresh_token="rfr-xyz",
        scopes=["a", "b"],
    )

    stored = _stored_value(prisma)
    row = MagicMock()
    row.credential_b64 = stored
    row.server_id = "srv-1"
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=row)

    result = await get_user_oauth_credential(prisma, "alice", "srv-1")
    assert result is not None
    assert result["type"] == "oauth2"
    assert result["access_token"] == access_token
    assert result["refresh_token"] == "rfr-xyz"
    assert result["scopes"] == ["a", "b"]


@pytest.mark.asyncio
async def test_oauth_get_returns_payload_for_legacy_row():
    payload = {
        "type": "oauth2",
        "access_token": "legacy-token",
        "connected_at": "2024-01-01T00:00:00Z",
    }
    legacy_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    row = MagicMock()
    row.credential_b64 = legacy_b64
    row.server_id = "srv-1"
    prisma = _make_prisma_with_existing(row=row)

    result = await get_user_oauth_credential(prisma, "alice", "srv-1")
    assert result is not None
    assert result["access_token"] == "legacy-token"


@pytest.mark.asyncio
async def test_oauth_get_returns_none_for_byok_row():
    # A row that holds a BYOK string must not leak as an OAuth payload.
    prisma = _make_prisma_with_existing(row=_legacy_row("plain-byok-not-json"))
    result = await get_user_oauth_credential(prisma, "alice", "srv-1")
    assert result is None


# ── BYOK guard inside store_user_oauth_credential ─────────────────────────────


@pytest.mark.asyncio
async def test_byok_guard_rejects_overwriting_legacy_byok():
    prisma = _make_prisma_with_existing(row=_legacy_row("plain-byok-key"))
    with pytest.raises(ValueError, match="could not be verified as an OAuth2"):
        await store_user_oauth_credential(prisma, "alice", "srv-1", "tok")


@pytest.mark.asyncio
async def test_byok_guard_rejects_overwriting_encrypted_byok():
    # Simulate a row written by the new (encrypted) code path: write a BYOK,
    # then attempt to overwrite with an OAuth token.
    prisma = _make_prisma_with_existing(row=None)
    await store_user_credential(prisma, "alice", "srv-1", "sk-secret-byok")

    encrypted_row = MagicMock()
    encrypted_row.credential_b64 = _stored_value(prisma)
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=encrypted_row)

    with pytest.raises(ValueError, match="could not be verified as an OAuth2"):
        await store_user_oauth_credential(prisma, "alice", "srv-1", "tok")


@pytest.mark.asyncio
async def test_byok_guard_allows_overwriting_existing_oauth():
    # Refresh path: row already holds an OAuth payload, write must succeed.
    prisma = _make_prisma_with_existing(row=None)
    await store_user_oauth_credential(prisma, "alice", "srv-1", "tok-1")

    oauth_row = MagicMock()
    oauth_row.credential_b64 = _stored_value(prisma)
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=oauth_row)

    await store_user_oauth_credential(prisma, "alice", "srv-1", "tok-2")
    # Final upsert wrote a new payload (different from the first)
    assert _stored_value(prisma) != oauth_row.credential_b64


# ── list_user_oauth_credentials ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_oauth_credentials_filters_byok_and_returns_payloads():
    # Three rows: one encrypted OAuth, one legacy-plaintext OAuth, one BYOK.
    # Only the two OAuth rows should come back.
    prisma = _make_prisma_with_existing(row=None)

    await store_user_oauth_credential(prisma, "alice", "srv-encrypted", "tok-enc")
    encrypted_b64 = _stored_value(prisma)
    encrypted_row = MagicMock()
    encrypted_row.credential_b64 = encrypted_b64
    encrypted_row.server_id = "srv-encrypted"

    legacy_payload = {
        "type": "oauth2",
        "access_token": "tok-legacy",
        "connected_at": "2024-01-01T00:00:00Z",
    }
    legacy_row = MagicMock()
    legacy_row.credential_b64 = base64.urlsafe_b64encode(json.dumps(legacy_payload).encode()).decode()
    legacy_row.server_id = "srv-legacy"

    byok_row = MagicMock()
    byok_row.credential_b64 = base64.urlsafe_b64encode(b"plain-byok-key").decode()
    byok_row.server_id = "srv-byok"

    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[encrypted_row, legacy_row, byok_row])

    results = await list_user_oauth_credentials(prisma, "alice")

    server_ids = {r["server_id"] for r in results}
    assert server_ids == {"srv-encrypted", "srv-legacy"}
    tokens = {r["access_token"] for r in results}
    assert tokens == {"tok-enc", "tok-legacy"}


# ── IdP grant (delegated OBO subject material) ────────────────────────────────


@pytest.mark.asyncio
async def test_idp_grant_round_trips():
    prisma = _make_prisma_with_existing(row=None)
    await store_user_idp_grant(
        prisma, "alice", "idp::https://idp.example.com/token", "idp-at", refresh_token="idp-rt", scopes=["read"]
    )

    stored = _stored_value(prisma)
    row = MagicMock()
    row.credential_b64 = stored
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=row)

    result = await get_user_idp_grant(prisma, "alice", "idp::https://idp.example.com/token")
    assert result is not None
    assert result["type"] == "idp_grant"
    assert result["access_token"] == "idp-at"
    assert result["refresh_token"] == "idp-rt"
    assert result["scopes"] == ["read"]


@pytest.mark.asyncio
async def test_idp_grant_does_not_persist_plaintext():
    prisma = _make_prisma_with_existing(row=None)
    await store_user_idp_grant(prisma, "alice", "idp::https://idp/token", "idp-secret-at", refresh_token="idp-secret-rt")
    stored = _stored_value(prisma)
    try:
        decoded_bytes = base64.urlsafe_b64decode(stored)
    except Exception:
        decoded_bytes = b""
    assert b"idp-secret-at" not in decoded_bytes
    assert b"idp-secret-rt" not in decoded_bytes


@pytest.mark.asyncio
async def test_idp_grant_is_not_surfaced_as_an_oauth2_credential():
    # An IdP grant is a THIRD credential kind sharing this table; the oauth2-only readers gate on
    # type == "oauth2", so an idp_grant row must be invisible to get_user_oauth_credential and to the
    # connected-servers listing -- otherwise it would show up as a phantom server the user "connected".
    prisma = _make_prisma_with_existing(row=None)
    await store_user_idp_grant(prisma, "alice", "idp::https://idp/token", "idp-at")
    stored = _stored_value(prisma)
    idp_row = MagicMock()
    idp_row.credential_b64 = stored
    idp_row.server_id = "idp::https://idp/token"

    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=idp_row)
    assert await get_user_oauth_credential(prisma, "alice", "idp::https://idp/token") is None
    assert await get_user_idp_grant(prisma, "alice", "idp::https://idp/token") is not None

    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[idp_row])
    assert await list_user_oauth_credentials(prisma, "alice") == []


# ── _decode_user_credential helper ────────────────────────────────────────────


def test_decode_user_credential_handles_garbage():
    # Malformed input must return None, not raise.
    assert _decode_user_credential("not-base64-and-not-encrypted!!!") is None


def test_decode_user_credential_handles_none():
    # Defensive: a null DB value must return None, not propagate TypeError.
    assert _decode_user_credential(None) is None


def test_decode_user_credential_legacy_path():
    plain = "legacy-secret"
    stored = base64.urlsafe_b64encode(plain.encode()).decode()
    assert _decode_user_credential(stored) == plain


# ── master-key rotation ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rotate_re_encrypts_byok_with_new_key(monkeypatch):
    # Encrypt a row under the current salt, then rotate to a new key, then
    # confirm the stored ciphertext decrypts under the NEW key — and not under
    # the old one.
    prisma = _make_prisma_with_existing(row=None)
    secret = "sk-original-byok-key"
    await store_user_credential(prisma, "alice", "srv-1", secret)
    encrypted_old = _stored_value(prisma)

    row = MagicMock()
    row.user_id = "alice"
    row.server_id = "srv-1"
    row.credential_b64 = encrypted_old
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[row])
    prisma.db.litellm_mcpusercredentials.update = AsyncMock()

    new_master_key = "rotated-salt-key-9999-9999-9999-9999"
    await rotate_mcp_user_credentials_master_key(prisma_client=prisma, new_master_key=new_master_key)

    update_call = prisma.db.litellm_mcpusercredentials.update.call_args
    new_stored = update_call.kwargs["data"]["credential_b64"]
    assert new_stored != encrypted_old, "rotation must produce different ciphertext"

    # Decrypt the rotated value under the NEW salt key — round-trips to plaintext.
    monkeypatch.setenv("LITELLM_SALT_KEY", new_master_key)
    assert (
        decrypt_value_helper(
            value=new_stored,
            key="mcp_user_credential",
            exception_type="debug",
            return_original_value=False,
        )
        == secret
    )


@pytest.mark.asyncio
async def test_rotate_migrates_legacy_plaintext_rows(monkeypatch):
    # A legacy plain-base64 row must also get re-encrypted under the new key.
    prisma = _make_prisma_with_existing(row=None)

    legacy_row = MagicMock()
    legacy_row.user_id = "alice"
    legacy_row.server_id = "srv-legacy"
    legacy_row.credential_b64 = base64.urlsafe_b64encode(b"legacy-plain").decode()
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[legacy_row])
    prisma.db.litellm_mcpusercredentials.update = AsyncMock()

    new_key = "another-rotation-key-aaaa-bbbb-cccc-dddd"
    await rotate_mcp_user_credentials_master_key(prisma_client=prisma, new_master_key=new_key)

    new_stored = prisma.db.litellm_mcpusercredentials.update.call_args.kwargs["data"]["credential_b64"]
    monkeypatch.setenv("LITELLM_SALT_KEY", new_key)
    assert (
        decrypt_value_helper(
            value=new_stored,
            key="mcp_user_credential",
            exception_type="debug",
            return_original_value=False,
        )
        == "legacy-plain"
    )


@pytest.mark.asyncio
async def test_rotate_skips_undecodable_rows():
    # One bad row must not abort the rotation for the rest.
    prisma = _make_prisma_with_existing(row=None)

    bad_row = MagicMock()
    bad_row.user_id = "alice"
    bad_row.server_id = "srv-corrupt"
    bad_row.credential_b64 = "!!! not base64 and not encrypted !!!"

    good_row = MagicMock()
    good_row.user_id = "bob"
    good_row.server_id = "srv-ok"
    good_row.credential_b64 = base64.urlsafe_b64encode(b"good-byok").decode()

    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[bad_row, good_row])
    prisma.db.litellm_mcpusercredentials.update = AsyncMock()

    await rotate_mcp_user_credentials_master_key(prisma_client=prisma, new_master_key="new-key-xxxx")

    # Only one update call — the good row.
    assert prisma.db.litellm_mcpusercredentials.update.call_count == 1
    where = prisma.db.litellm_mcpusercredentials.update.call_args.kwargs["where"]
    assert where["user_id_server_id"]["server_id"] == "srv-ok"


# ── Expiry buffer + refresh-on-expiry (OBO list-refresh regression) ───────────


def _oauth_cred(access_token="at-live", refresh_token=None, expires_in_seconds=None):
    cred = {"type": "oauth2", "access_token": access_token}
    if refresh_token is not None:
        cred["refresh_token"] = refresh_token
    if expires_in_seconds is not None:
        cred["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)).isoformat()
    return cred


def test_expiry_no_buffer_treats_soon_to_expire_as_valid():
    # Without a buffer, a token with 30s of life left is still valid.
    cred = _oauth_cred(expires_in_seconds=30)
    assert is_oauth_credential_expired(cred) is False
    assert is_oauth_credential_expired(cred, buffer_seconds=0) is False


def test_expiry_buffer_treats_soon_to_expire_as_expired():
    # With a 60s buffer, the same 30s-of-life token must be treated as expired
    # so callers refresh before it lapses mid-request.
    cred = _oauth_cred(expires_in_seconds=30)
    assert is_oauth_credential_expired(cred, buffer_seconds=60) is True
    # A token comfortably beyond the buffer stays valid.
    assert is_oauth_credential_expired(_oauth_cred(expires_in_seconds=600), buffer_seconds=60) is False


def test_expiry_past_is_expired_regardless_of_buffer():
    cred = _oauth_cred(expires_in_seconds=-10)
    assert is_oauth_credential_expired(cred) is True
    assert is_oauth_credential_expired(cred, buffer_seconds=60) is True


def test_expiry_missing_expires_at_is_never_expired():
    assert is_oauth_credential_expired(_oauth_cred()) is False
    assert is_oauth_credential_expired(_oauth_cred(), buffer_seconds=60) is False


@pytest.mark.asyncio
async def test_resolve_returns_valid_token_without_refreshing(monkeypatch):
    # A token good for 10 minutes must be returned as-is, with no refresh call.
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refresh = AsyncMock()
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    cred = _oauth_cred(access_token="at-live", refresh_token="rt-1", expires_in_seconds=600)
    result = await resolve_valid_user_oauth_token(
        user_id="alice", server=MagicMock(), cred=cred, prisma_client=MagicMock()
    )

    assert result is cred
    assert result["access_token"] == "at-live"
    refresh.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_refreshes_expired_token_with_refresh_token(monkeypatch):
    # The core regression: an expired OBO cred with a refresh_token must mint a
    # new token rather than returning None (which left the UI tool list empty).
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refreshed = _oauth_cred(access_token="at-fresh", refresh_token="rt-2", expires_in_seconds=3600)
    refresh = AsyncMock(return_value=refreshed)
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    expired = _oauth_cred(access_token="at-dead", refresh_token="rt-1", expires_in_seconds=-5)
    result = await resolve_valid_user_oauth_token(
        user_id="alice", server=MagicMock(), cred=expired, prisma_client=MagicMock()
    )

    refresh.assert_awaited_once()
    assert result["access_token"] == "at-fresh"


@pytest.mark.asyncio
async def test_resolve_refreshes_token_expiring_within_buffer(monkeypatch):
    # A token still technically valid (30s left) but inside the 60s buffer must
    # be proactively refreshed, not handed back.
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refreshed = _oauth_cred(access_token="at-fresh", expires_in_seconds=3600)
    refresh = AsyncMock(return_value=refreshed)
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    soon = _oauth_cred(access_token="at-soon", refresh_token="rt-1", expires_in_seconds=30)
    result = await resolve_valid_user_oauth_token(
        user_id="alice", server=MagicMock(), cred=soon, prisma_client=MagicMock()
    )

    refresh.assert_awaited_once()
    assert result["access_token"] == "at-fresh"


@pytest.mark.asyncio
async def test_resolve_returns_none_when_expired_without_refresh_token(monkeypatch):
    # No refresh_token means nothing to refresh with — return None, never call refresh.
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refresh = AsyncMock()
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    expired = _oauth_cred(access_token="at-dead", expires_in_seconds=-5)
    result = await resolve_valid_user_oauth_token(
        user_id="alice", server=MagicMock(), cred=expired, prisma_client=MagicMock()
    )

    assert result is None
    refresh.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_returns_none_when_refresh_fails(monkeypatch):
    # A failed refresh (provider returns nothing usable) must surface as None.
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refresh = AsyncMock(return_value=None)
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    expired = _oauth_cred(access_token="at-dead", refresh_token="rt-1", expires_in_seconds=-5)
    result = await resolve_valid_user_oauth_token(
        user_id="alice", server=MagicMock(), cred=expired, prisma_client=MagicMock()
    )

    refresh.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_resolve_returns_none_for_missing_credential(monkeypatch):
    import litellm.proxy._experimental.mcp_server.db as db_mod

    refresh = AsyncMock()
    monkeypatch.setattr(db_mod, "refresh_user_oauth_token", refresh)

    assert (
        await resolve_valid_user_oauth_token(user_id="alice", server=MagicMock(), cred=None, prisma_client=MagicMock())
        is None
    )
    assert (
        await resolve_valid_user_oauth_token(
            user_id="alice",
            server=MagicMock(),
            cred={"type": "oauth2"},
            prisma_client=MagicMock(),
        )
        is None
    )
    refresh.assert_not_called()


# ── per-user env-var rotation ─────────────────────────────────────────────────


def _env_var_row(values_b64: str, user_id="alice", server_id="srv-1"):
    row = MagicMock()
    row.values_b64 = values_b64
    row.user_id = user_id
    row.server_id = server_id
    return row


@pytest.mark.asyncio
async def test_rotate_user_env_vars_re_encrypts_with_new_key(monkeypatch):
    # Encrypt env vars under the current salt, rotate to a new key, then confirm
    # the stored ciphertext round-trips under the NEW key.
    values = {"API_KEY": "sk-secret", "REGION": "us-east-1"}
    encrypted_old = encrypt_value_helper(json.dumps(values))

    prisma = MagicMock()
    prisma.db.litellm_mcpuserenvvars.find_many = AsyncMock(return_value=[_env_var_row(encrypted_old)])
    prisma.db.litellm_mcpuserenvvars.update = AsyncMock()

    new_master_key = "rotated-env-key-1111-2222-3333-4444"
    await rotate_mcp_user_env_vars_master_key(prisma_client=prisma, new_master_key=new_master_key)

    new_stored = prisma.db.litellm_mcpuserenvvars.update.call_args.kwargs["data"]["values_b64"]
    assert new_stored != encrypted_old, "rotation must produce different ciphertext"

    monkeypatch.setenv("LITELLM_SALT_KEY", new_master_key)
    decrypted = decrypt_value_helper(
        value=new_stored,
        key="mcp_user_env_vars",
        exception_type="debug",
        return_original_value=False,
    )
    assert json.loads(decrypted) == values


@pytest.mark.asyncio
async def test_rotate_user_env_vars_skips_undecryptable_rows():
    # A corrupt row must be skipped (not overwritten) so recoverable data is
    # preserved and one bad row does not abort the rest of the rotation.
    good = _env_var_row(encrypt_value_helper(json.dumps({"A": "1"})), server_id="srv-ok")
    bad = _env_var_row("!!! not encrypted !!!", server_id="srv-corrupt")

    prisma = MagicMock()
    prisma.db.litellm_mcpuserenvvars.find_many = AsyncMock(return_value=[bad, good])
    prisma.db.litellm_mcpuserenvvars.update = AsyncMock()

    await rotate_mcp_user_env_vars_master_key(prisma_client=prisma, new_master_key="new-key-xxxx")

    assert prisma.db.litellm_mcpuserenvvars.update.call_count == 1
    where = prisma.db.litellm_mcpuserenvvars.update.call_args.kwargs["where"]
    assert where["user_id_server_id"]["server_id"] == "srv-ok"


@pytest.mark.asyncio
async def test_refresh_user_oauth_token_uses_client_secret_basic(monkeypatch):
    """LIT-4091: a per-user refresh against a server with token_endpoint_auth_method=client_secret_basic
    sends HTTP Basic and keeps the secret out of the body."""
    import litellm.proxy._experimental.mcp_server.db as db_mod

    server = MagicMock()
    server.token_url = "https://idp.example.com/oauth2/token"
    server.server_id = "srv"
    server.client_id = "cid"
    server.client_secret = "sec"
    server.token_endpoint_auth_method = "client_secret_basic"

    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "new-at", "expires_in": 3600}
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    monkeypatch.setattr(db_mod, "get_async_httpx_client", lambda **kwargs: mock_client)
    monkeypatch.setattr(db_mod, "store_user_oauth_credential", AsyncMock())
    monkeypatch.setattr(db_mod, "get_user_oauth_credential", AsyncMock(return_value={"access_token": "new-at"}))

    result = await db_mod.refresh_user_oauth_token(
        prisma_client=MagicMock(),
        user_id="alice",
        server=server,
        cred={"refresh_token": "rt"},
    )

    assert result is not None
    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Basic " + base64.b64encode(b"cid:sec").decode()
    assert "client_secret" not in kwargs["data"]
    assert "client_id" not in kwargs["data"]
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "rt"


@pytest.mark.asyncio
async def test_refresh_user_oauth_token_defaults_to_client_secret_post(monkeypatch):
    """Backward compatibility: with no token_endpoint_auth_method the refresh keeps credentials in
    the body (client_secret_post) and sends no Authorization header."""
    import litellm.proxy._experimental.mcp_server.db as db_mod

    server = MagicMock()
    server.token_url = "https://idp.example.com/oauth2/token"
    server.server_id = "srv"
    server.client_id = "cid"
    server.client_secret = "sec"
    server.token_endpoint_auth_method = None

    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "new-at", "expires_in": 3600}
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    monkeypatch.setattr(db_mod, "get_async_httpx_client", lambda **kwargs: mock_client)
    monkeypatch.setattr(db_mod, "store_user_oauth_credential", AsyncMock())
    monkeypatch.setattr(db_mod, "get_user_oauth_credential", AsyncMock(return_value={"access_token": "new-at"}))

    await db_mod.refresh_user_oauth_token(
        prisma_client=MagicMock(),
        user_id="alice",
        server=server,
        cred={"refresh_token": "rt"},
    )

    _, kwargs = mock_client.post.call_args
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["data"]["client_id"] == "cid"
    assert kwargs["data"]["client_secret"] == "sec"


def test_prepare_mcp_server_data_create_carries_token_exchange_columns():
    """The create path (POST /v1/mcp/server) must emit token_exchange_endpoint/audience/
    subject_token_type as top-level column values so an auth_type=oauth2_token_exchange server
    persists via the REST API, not only via config.yaml. Dropping the fields from the request
    model would leave them out of the prepared column data."""
    request = NewMCPServerRequest(
        server_name="te_write",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        token_exchange_endpoint="https://idp.example.com/oauth2/token",
        audience="https://upstream.example.com",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_exchange_profile="entra_obo",
        credentials={"client_id": "te-client", "client_secret": "te-secret"},
    )

    data = _prepare_mcp_server_data(request)

    assert data["token_exchange_endpoint"] == "https://idp.example.com/oauth2/token"
    assert data["audience"] == "https://upstream.example.com"
    assert data["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
    assert data["token_exchange_profile"] == "entra_obo"


def test_prepare_mcp_server_data_update_carries_token_exchange_columns():
    """The partial-update path (PUT /v1/mcp/server, exclude_unset) must carry the three
    token-exchange columns when the caller provides them."""
    request = UpdateMCPServerRequest(
        server_id="te-update",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        token_exchange_endpoint="https://idp.example.com/oauth2/token",
        audience="https://upstream.example.com",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_exchange_profile="entra_obo",
    )

    data = _prepare_mcp_server_data(request, exclude_unset=True)

    assert data["token_exchange_endpoint"] == "https://idp.example.com/oauth2/token"
    assert data["audience"] == "https://upstream.example.com"
    assert data["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
    assert data["token_exchange_profile"] == "entra_obo"
