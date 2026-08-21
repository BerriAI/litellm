"""
Tests for the startup oauth2_flow backfill.

Legacy oauth2 rows with a null oauth2_flow are classified once, at rest, using
signals read-time inference never had (per-user token rows first), and the
result is persisted so the read path never infers again. The signal order is
the spec, and so is the refusal to stamp client_credentials: the M2M credential
shape is shared by DCR-registered interactive servers whose authorization
endpoint lives only in discovery, so ambiguous rows are left unstamped for a
human to assert rather than being permanently mislabeled M2M.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.oauth2_flow_backfill import (
    backfill_null_oauth2_flows,
    classify_null_flow_row,
)


def test_classify_per_user_tokens_beat_m2m_shape():
    """The DCR trap row: creds + token_url, no authorization_url, but a user has
    signed in. Tokens are definitive; the M2M shape must not win."""
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=True,
        authorization_url=None,
        registration_url=None,
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    assert flow == "authorization_code"
    assert rule == "per_user_tokens"


def test_classify_authorization_url_beats_m2m_shape():
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=False,
        authorization_url="https://idp.example.com/authorize",
        registration_url=None,
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    assert flow == "authorization_code"
    assert rule == "authorization_url"


def test_classify_registration_url_beats_m2m_shape():
    """A registration endpoint means DCR, and DCR exists to mint interactive
    clients; an abandoned-DCR row (no sign-in yet) must not be stamped M2M."""
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=False,
        authorization_url=None,
        registration_url="https://idp.example.com/register",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    assert flow == "authorization_code"
    assert rule == "registration_url"


def test_classify_m2m_shape_is_ambiguous_and_unstamped():
    """The M2M shape alone must never stamp client_credentials: a DCR-registered
    interactive server that nobody signed into yet has the identical shape, and a
    wrong M2M stamp would permanently route its per-user traffic through the
    proxy's stored client credential."""
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=False,
        authorization_url=None,
        registration_url=None,
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    assert flow is None
    assert rule == "ambiguous_m2m_shape"


def test_classify_partial_credentials_default_interactive():
    """token_url without a full credential pair is not the M2M shape."""
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=False,
        authorization_url=None,
        registration_url=None,
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid"},
    )
    assert flow == "authorization_code"
    assert rule == "interactive_default"


def test_classify_bare_row_default_interactive():
    flow, rule = classify_null_flow_row(
        has_per_user_tokens=False,
        authorization_url=None,
        registration_url=None,
        token_url=None,
        credentials=None,
    )
    assert flow == "authorization_code"
    assert rule == "interactive_default"


def _row(server_id, *, authorization_url=None, registration_url=None, token_url=None, credentials=None):
    return SimpleNamespace(
        server_id=server_id,
        authorization_url=authorization_url,
        registration_url=registration_url,
        token_url=token_url,
        credentials=credentials,
    )


def _oauth_token_row(server_id):
    payload = json.dumps({"type": "oauth2", "access_token": "tok", "connected_at": "2026-07-01T00:00:00Z"})
    return SimpleNamespace(
        server_id=server_id,
        user_id="u1",
        credential_b64=base64.urlsafe_b64encode(payload.encode()).decode(),
    )


def _byok_key_row(server_id):
    return SimpleNamespace(
        server_id=server_id,
        user_id="u1",
        credential_b64=base64.urlsafe_b64encode(b"sk-user-supplied-upstream-key").decode(),
    )


def _mock_prisma(null_rows, token_rows):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=null_rows)
    mock_prisma.db.litellm_mcpservertable.update_many = AsyncMock(return_value=MagicMock())
    mock_prisma.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=token_rows)
    return mock_prisma


@pytest.mark.asyncio
async def test_backfill_only_targets_null_flow_oauth2_rows():
    """The where clause is the guard that explicit and non-oauth2 rows are never touched."""
    mock_prisma = _mock_prisma([], [])

    counts = await backfill_null_oauth2_flows(mock_prisma)

    assert counts == {}
    mock_prisma.db.litellm_mcpservertable.find_many.assert_awaited_once_with(
        where={"auth_type": "oauth2", "oauth2_flow": None},
    )
    mock_prisma.db.litellm_mcpusercredentials.find_many.assert_not_awaited()
    mock_prisma.db.litellm_mcpservertable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_stamps_rows_and_reports_rule_counts():
    dcr_trap_row = _row(
        "signed_in_dcr",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    m2m_row = _row(
        "legacy_m2m",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    interactive_row = _row("legacy_interactive", authorization_url="https://idp.example.com/authorize")

    mock_prisma = _mock_prisma(
        [dcr_trap_row, m2m_row, interactive_row],
        [_oauth_token_row("signed_in_dcr")],
    )

    counts = await backfill_null_oauth2_flows(mock_prisma)

    assert counts == {"per_user_tokens": 1, "ambiguous_m2m_shape": 1, "authorization_url": 1}

    mock_prisma.db.litellm_mcpservertable.update_many.assert_awaited_once()
    call = mock_prisma.db.litellm_mcpservertable.update_many.await_args
    assert sorted(call.kwargs["where"]["server_id"]["in"]) == ["legacy_interactive", "signed_in_dcr"]
    assert "oauth2_flow" in call.kwargs["where"] and call.kwargs["where"]["oauth2_flow"] is None
    assert call.kwargs["data"] == {"oauth2_flow": "authorization_code", "updated_by": "oauth2_flow_backfill"}


@pytest.mark.asyncio
async def test_backfill_handles_json_string_credentials():
    """JSON-string credential blobs must decode: the M2M shape is recognized (and
    therefore deliberately left unstamped) rather than misread as credential-less."""
    m2m_row = _row(
        "json_creds_m2m",
        token_url="https://idp.example.com/token",
        credentials='{"client_id": "cid", "client_secret": "csecret"}',
    )
    mock_prisma = _mock_prisma([m2m_row], [])

    counts = await backfill_null_oauth2_flows(mock_prisma)

    assert counts == {"ambiguous_m2m_shape": 1}
    mock_prisma.db.litellm_mcpservertable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_treats_undecodable_credentials_as_absent():
    row = _row(
        "corrupt_creds",
        token_url="https://idp.example.com/token",
        credentials="not-json",
    )
    mock_prisma = _mock_prisma([row], [])

    counts = await backfill_null_oauth2_flows(mock_prisma)

    assert counts == {"interactive_default": 1}


@pytest.mark.asyncio
async def test_backfill_byok_key_rows_are_not_sign_in_proof():
    """BYOK API keys live in the same table as per-user OAuth tokens; a bare key row
    must not satisfy the per_user_tokens rule, or a BYOK-flavored M2M-shaped server
    would be permanently stamped authorization_code. Only rows whose payload decodes
    as a type oauth2 token count."""
    byok_shaped_row = _row(
        "byok_m2m_shape",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    mock_prisma = _mock_prisma([byok_shaped_row], [_byok_key_row("byok_m2m_shape")])

    counts = await backfill_null_oauth2_flows(mock_prisma)

    assert counts == {"ambiguous_m2m_shape": 1}
    mock_prisma.db.litellm_mcpservertable.update_many.assert_not_awaited()
