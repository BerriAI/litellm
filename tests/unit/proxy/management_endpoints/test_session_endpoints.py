"""
Tests for POST /session/logout and revoke_ui_session_keys
(litellm/proxy/management_endpoints/session_endpoints.py).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Response

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_VerificationToken, UserAPIKeyAuth
from litellm.proxy.management_endpoints.session_endpoints import (
    revoke_ui_session_keys,
    session_logout,
)

HASHED_TOKEN = "hashed-session-token"
USER_ID = "user-123"


def _session_row(token: str = HASHED_TOKEN, user_id: str = USER_ID) -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=user_id)


def _make_prisma(
    find_unique_row: LiteLLM_VerificationToken | None = None,
    find_many_rows: list[LiteLLM_VerificationToken] | None = None,
) -> MagicMock:
    prisma = MagicMock()
    table = prisma.db.litellm_verificationtoken
    table.find_unique = AsyncMock(return_value=find_unique_row)
    table.find_many = AsyncMock(return_value=find_many_rows or [])
    table.delete_many = AsyncMock(return_value=1)
    return prisma


def _ui_session_caller(token: str | None = HASHED_TOKEN) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=USER_ID)


def _patched_globals(prisma):
    return (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", prisma
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.proxy_logging_obj", None
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock()
        ),
    )


@pytest.mark.asyncio
async def test_session_logout_revokes_presented_session():
    prisma = _make_prisma(find_unique_row=_session_row())
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints._persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session revoked."
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": HASHED_TOKEN}
    # Audit record persisted before the row is gone.
    persist_mock.assert_awaited_once()
    assert persist_mock.await_args.kwargs["keys"][0].token == HASHED_TOKEN
    # Cache evicted + broadcast even on the delete path.
    evict_mock.assert_awaited_once()
    assert tuple(evict_mock.await_args.kwargs["hashed_tokens"]) == (HASHED_TOKEN,)


@pytest.mark.asyncio
async def test_session_logout_clears_token_cookie():
    prisma = _make_prisma(find_unique_row=_session_row())
    fastapi_response = Response()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints._persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        await session_logout(
            response=fastapi_response,
            user_api_key_dict=_ui_session_caller(),
        )

    set_cookie_headers = [v.decode() for k, v in fastapi_response.raw_headers if k == b"set-cookie"]
    assert any(h.startswith('token="";') or h.startswith("token=;") for h in set_cookie_headers)


@pytest.mark.asyncio
async def test_session_logout_refuses_non_ui_session_key():
    """The endpoint must not become a generic key-deletion oracle: a normal
    virtual key (no UI team id) is refused outright."""
    prisma = _make_prisma()
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=UserAPIKeyAuth(token=HASHED_TOKEN, team_id="some-real-team", user_id=USER_ID),
            )

    assert exc_info.value.status_code == 403
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_session_logout_is_idempotent_when_row_already_gone():
    prisma = _make_prisma(find_unique_row=None)
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session already revoked."
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()
    # The cache entry may outlive the row; evict regardless.
    evict_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_logout_requires_db():
    p2 = patch("litellm.proxy.proxy_server.proxy_logging_obj", None)
    p3 = patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    with (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", None
        ),
        p2,
        p3,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=_ui_session_caller(),
            )

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_revokes_all_and_broadcasts():
    rows = [_session_row(token="t1"), _session_row(token="t2"), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints._persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 3
    find_kwargs = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs
    assert find_kwargs["where"] == {"user_id": USER_ID, "team_id": UI_SESSION_TOKEN_TEAM_ID}
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t2", "t3"]}}
    persist_mock.assert_awaited_once()
    evict_mock.assert_awaited_once()
    assert evict_mock.await_args.kwargs["hashed_tokens"] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_keeps_callers_session():
    rows = [_session_row(token="t1"), _session_row(token=HASHED_TOKEN), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints._persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
            keep_hashed_token=HASHED_TOKEN,
        )

    assert revoked == 2
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t3"]}}


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_when_no_sessions():
    prisma = _make_prisma(find_many_rows=[])
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_failure_is_swallowed():
    """The password write has already committed when this runs; a revocation
    failure must not fail the caller's request."""
    prisma = _make_prisma(find_many_rows=[_session_row(token="t1")])
    prisma.db.litellm_verificationtoken.delete_many = AsyncMock(side_effect=RuntimeError("db down"))
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints._persist_deleted_verification_tokens",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_without_db():
    with patch(  # test-quality-ok: helper reads proxy_server module globals; no injection seam
        "litellm.proxy.proxy_server.prisma_client", None
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0
