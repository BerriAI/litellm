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
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    _decode_user_credential,
    get_user_credential,
    get_user_oauth_credential,
    list_user_oauth_credentials,
    rotate_mcp_user_credentials_master_key,
    store_user_credential,
    store_user_oauth_credential,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper


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

    await store_user_oauth_credential(
        prisma, "alice", "srv-1", access_token, refresh_token="rfr-xyz"
    )

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
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(
        return_value=encrypted_row
    )

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
    legacy_row.credential_b64 = base64.urlsafe_b64encode(
        json.dumps(legacy_payload).encode()
    ).decode()
    legacy_row.server_id = "srv-legacy"

    byok_row = MagicMock()
    byok_row.credential_b64 = base64.urlsafe_b64encode(b"plain-byok-key").decode()
    byok_row.server_id = "srv-byok"

    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(
        return_value=[encrypted_row, legacy_row, byok_row]
    )

    results = await list_user_oauth_credentials(prisma, "alice")

    server_ids = {r["server_id"] for r in results}
    assert server_ids == {"srv-encrypted", "srv-legacy"}
    tokens = {r["access_token"] for r in results}
    assert tokens == {"tok-enc", "tok-legacy"}


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
    await rotate_mcp_user_credentials_master_key(
        prisma_client=prisma, new_master_key=new_master_key
    )

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
    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(
        return_value=[legacy_row]
    )
    prisma.db.litellm_mcpusercredentials.update = AsyncMock()

    new_key = "another-rotation-key-aaaa-bbbb-cccc-dddd"
    await rotate_mcp_user_credentials_master_key(
        prisma_client=prisma, new_master_key=new_key
    )

    new_stored = prisma.db.litellm_mcpusercredentials.update.call_args.kwargs["data"][
        "credential_b64"
    ]
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

    prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(
        return_value=[bad_row, good_row]
    )
    prisma.db.litellm_mcpusercredentials.update = AsyncMock()

    await rotate_mcp_user_credentials_master_key(
        prisma_client=prisma, new_master_key="new-key-xxxx"
    )

    # Only one update call — the good row.
    assert prisma.db.litellm_mcpusercredentials.update.call_count == 1
    where = prisma.db.litellm_mcpusercredentials.update.call_args.kwargs["where"]
    assert where["user_id_server_id"]["server_id"] == "srv-ok"
