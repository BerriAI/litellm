"""
Tests for the `failed_tokens` field returned by delete_verification_tokens().

Related PR: https://github.com/BerriAI/litellm/pull/12577

Verifies that delete_verification_tokens() includes a `failed_tokens` key in
its result dict in all scenarios, populated with any token hashes that could
not be deleted.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import (
    LiteLLM_VerificationToken,
    LitellmUserRoles,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_verification_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(token: str, user_id: str = "user-123") -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken(
        token=token,
        user_id=user_id,
        team_id=None,
        key_alias=None,
        spend=0.0,
        max_budget=None,
        models=[],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )


def _admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="admin-user",
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )


def _regular_user(user_id: str = "user-123") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id=user_id,
        api_key="sk-regular",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )


def _mock_prisma(keys, deleted_tokens):
    """Return a minimal mock prisma_client for a given set of found keys and deleted tokens."""
    mock = AsyncMock()
    mock.db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)
    mock.delete_data = AsyncMock(return_value=deleted_tokens)
    mock.db.litellm_deletedverificationtoken.create_many = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Test 1 – admin deletes all tokens successfully → failed_tokens is []
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_all_tokens_admin_returns_empty_failed_tokens(monkeypatch):
    """
    PROXY_ADMIN deletes two tokens; both are removed from the DB.
    The response must include `failed_tokens: []`.
    """
    key1 = _make_token("hashed-token-1")
    key2 = _make_token("hashed-token-2")
    mock_prisma = _mock_prisma(
        keys=[key1, key2],
        deleted_tokens=["hashed-token-1", "hashed-token-2"],
    )

    mock_cache = MagicMock()
    mock_cache.delete_cache = MagicMock()

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._hash_token_if_needed",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.hash_token",
        lambda token: token,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    result, _keys_deleted = await delete_verification_tokens(
        tokens=["hashed-token-1", "hashed-token-2"],
        user_api_key_cache=mock_cache,
        user_api_key_dict=_admin_user(),
    )

    assert "failed_tokens" in result, "response must contain 'failed_tokens' key"
    assert result["failed_tokens"] == [], "no failures expected for admin full deletion"
    assert set(result["deleted_keys"]) == {"hashed-token-1", "hashed-token-2"}


# ---------------------------------------------------------------------------
# Test 2 – non-admin, all authorized, all deleted → failed_tokens is []
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_tokens_non_admin_all_succeed_returns_empty_failed_tokens(
    monkeypatch,
):
    """
    Non-admin user deletes a token they own; DB reports success.
    `failed_tokens` should be an empty list.
    """
    key1 = _make_token("hashed-token-1", user_id="user-123")
    mock_prisma = _mock_prisma(keys=[key1], deleted_tokens=["hashed-token-1"])

    mock_cache = MagicMock()
    mock_cache.delete_cache = MagicMock()

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._hash_token_if_needed",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.hash_token",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.can_modify_verification_token",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    result, _ = await delete_verification_tokens(
        tokens=["hashed-token-1"],
        user_api_key_cache=mock_cache,
        user_api_key_dict=_regular_user("user-123"),
    )

    assert "failed_tokens" in result
    assert result["failed_tokens"] == []
    assert "hashed-token-1" in result["deleted_keys"]


# ---------------------------------------------------------------------------
# Test 3 – non-admin, one token not found in DB → failed_tokens is populated
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_tokens_non_admin_token_not_in_db_returns_failed_tokens(
    monkeypatch,
):
    """
    Non-admin requests deletion of two tokens, but the DB only finds one of
    them (token-2 was already deleted or never existed).  The missing token
    must appear in `failed_tokens` and no exception should be raised.

    This is the scenario the `failed_tokens` field was introduced to handle:
    previously the function would raise Exception("Failed to delete all tokens").
    """
    key1 = _make_token("hashed-token-1", user_id="user-123")

    mock_prisma = AsyncMock()
    # DB find_many returns only key1 — token-2 is not found
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[key1])
    mock_prisma.delete_data = AsyncMock(return_value=["hashed-token-1"])
    mock_prisma.db.litellm_deletedverificationtoken.create_many = AsyncMock()

    mock_cache = MagicMock()
    mock_cache.delete_cache = MagicMock()

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._hash_token_if_needed",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.hash_token",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.can_modify_verification_token",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    result, _ = await delete_verification_tokens(
        tokens=["hashed-token-1", "hashed-token-2"],
        user_api_key_cache=mock_cache,
        user_api_key_dict=_regular_user("user-123"),
    )

    assert "failed_tokens" in result
    assert "hashed-token-2" in result["failed_tokens"], (
        "token-2 was not found in the DB and must appear in failed_tokens"
    )
    assert "hashed-token-1" in result["deleted_keys"]


# ---------------------------------------------------------------------------
# Test 4 – admin, DB bulk-delete returns fewer tokens → failed_tokens populated
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_tokens_admin_partial_db_failure_returns_failed_tokens(
    monkeypatch,
):
    """
    PROXY_ADMIN requests deletion of two tokens; the DB bulk-delete only
    removes one (e.g. the other was concurrently deleted).  The unremoved
    token must appear in `failed_tokens` — previously it would be silently
    swallowed since the admin path never compared returned vs. requested counts.
    """
    key1 = _make_token("hashed-token-1")
    key2 = _make_token("hashed-token-2")
    # DB reports only token-1 as deleted
    mock_prisma = _mock_prisma(
        keys=[key1, key2],
        deleted_tokens=["hashed-token-1"],
    )

    mock_cache = MagicMock()
    mock_cache.delete_cache = MagicMock()

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._hash_token_if_needed",
        lambda token: token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.hash_token",
        lambda token: token,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    result, _ = await delete_verification_tokens(
        tokens=["hashed-token-1", "hashed-token-2"],
        user_api_key_cache=mock_cache,
        user_api_key_dict=_admin_user(),
    )

    assert "failed_tokens" in result
    assert "hashed-token-2" in result["failed_tokens"], (
        "token-2 was not deleted by the DB and must appear in failed_tokens for admins too"
    )
    assert "hashed-token-1" in result["deleted_keys"]
