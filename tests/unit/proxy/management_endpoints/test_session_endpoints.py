"""
Tests for POST /session/logout and revoke_ui_session_keys
(litellm/proxy/management_endpoints/session_endpoints.py).
"""

from collections.abc import Awaitable, Callable, Mapping
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_VerificationToken, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.login_throttle import LoginThrottle
from litellm.proxy.auth.login_utils import authenticate_user
from litellm.proxy.management_endpoints.session_endpoints import (
    end_earlier_ui_sessions,
    revoke_ui_session_keys,
    session_logout,
)
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

HASHED_TOKEN = "hashed-session-token"
USER_ID = "user-123"


def _unlimited_throttle() -> LoginThrottle:
    return LoginThrottle(
        client_ip="1.2.3.4",
        source_limit=None,
        user_limit=10_000,
        window_seconds=60,
        block_seconds=300,
        counters=InMemoryCache(),
        blocks=InMemoryCache(),
    )


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
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
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
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
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
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
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
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
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
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
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


_FIRST_SIGN_IN: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _SessionKeyTable:
    """In-memory UI session key rows; ``mint`` stands in for the key a login writes."""

    def __init__(self) -> None:
        self.rows: list[LiteLLM_VerificationToken] = []
        self.names: dict[str, str] = {}
        self.logins = 0
        self.mint_fails = False

    async def mint(self, **key_fields: object) -> dict[str, object]:
        if self.mint_fails:
            raise RuntimeError("key write failed")
        self.logins += 1
        name: Final = f"session-{self.logins}"
        raw_key: Final = f"sk-{name}"
        user_id: Final = cast(str, key_fields["user_id"])
        self.add(name, hash_token(raw_key), user_id, _FIRST_SIGN_IN + timedelta(seconds=self.logins))
        return {"token": raw_key, "user_id": user_id}

    def add(self, name: str, hashed: str, user_id: str, created_at: datetime) -> None:
        self.names[hashed] = name
        self.rows.append(
            LiteLLM_VerificationToken(
                token=hashed, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=user_id, created_at=created_at
            )
        )

    async def find_many(self, where: Mapping[str, object]) -> list[LiteLLM_VerificationToken]:
        return [row for row in self.rows if row.user_id == where["user_id"] and row.team_id == where["team_id"]]

    async def delete_many(self, where: Mapping[str, Mapping[str, list[str]]]) -> int:
        ended: Final = set(where["token"]["in"])
        self.rows = [row for row in self.rows if row.token not in ended]
        return len(ended)

    def tokens(self) -> list[str]:
        return [self.names[cast(str, row.token)] for row in self.rows]


def _session_db(table: _SessionKeyTable) -> ExitStack:
    prisma: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_verificationtoken=table,
            litellm_deletedverificationtoken=SimpleNamespace(create_many=AsyncMock()),
        )
    )
    stack: Final = ExitStack()
    stack.enter_context(
        patch(
            "litellm.proxy.proxy_server.prisma_client", prisma
        )  # test-quality-ok: revocation reads the proxy's DB global
    )
    stack.enter_context(
        patch("litellm.proxy.proxy_server.proxy_logging_obj", None)  # test-quality-ok: no shared cache to evict from
    )
    return stack


async def _admin_login(table: _SessionKeyTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "admin-password")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    with (
        patch(
            "litellm.proxy.auth.login_utils.generate_key_helper_fn", table.mint
        ),  # test-quality-ok: the key write is the DB boundary
        patch(
            "litellm.proxy.auth.login_utils.user_update", AsyncMock()
        ),  # test-quality-ok: the admin row write is the DB boundary
    ):
        await authenticate_user(
            username="admin",
            password="admin-password",
            master_key="sk-master",
            prisma_client=None,
            throttle=_unlimited_throttle(),
        )


async def _password_login(table: _SessionKeyTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    user: Final = SimpleNamespace(
        user_id=USER_ID, user_email="u@example.com", user_role="internal_user", password="stored"
    )
    users: Final = MagicMock(
        return_value=SimpleNamespace(table=SimpleNamespace(find_first=AsyncMock(return_value=user)))
    )
    with (
        patch(
            "litellm.proxy.auth.login_utils.UserRepository", users
        ),  # test-quality-ok: the user lookup is the DB boundary
        patch("litellm.proxy.auth.login_utils.verify_password", return_value=True),  # test-quality-ok: skips scrypt
        patch("litellm.proxy.auth.login_utils._rehash_password_if_needed", AsyncMock()),  # test-quality-ok: DB write
        patch(
            "litellm.proxy.auth.login_utils.generate_key_helper_fn", table.mint
        ),  # test-quality-ok: the key write is the DB boundary
    ):
        await authenticate_user(
            username="u@example.com",
            password="right",
            master_key="sk-master",
            prisma_client=MagicMock(),
            throttle=_unlimited_throttle(),
            general_settings={"password_policy_check_breached_passwords": False},
        )


async def _sso_login(
    table: _SessionKeyTable, monkeypatch: pytest.MonkeyPatch, ui_access_mode: str | None = None
) -> None:
    request: Final = MagicMock(spec=Request)
    request.scope = {}
    request.base_url = "http://localhost:4000/"
    request.cookies = {}
    callback_globals: Final[tuple[tuple[str, object], ...]] = (
        ("litellm.proxy.utils.get_prisma_client_or_throw", MagicMock(return_value=MagicMock())),
        ("litellm.proxy.proxy_server.master_key", "sk-master"),
        ("litellm.proxy.proxy_server.general_settings", {}),
        ("litellm.proxy.proxy_server.premium_user", False),
        ("litellm.proxy.proxy_server.user_custom_sso", None),
        ("litellm.proxy.proxy_server.redis_usage_cache", None),
        ("litellm.proxy.proxy_server.generate_key_helper_fn", table.mint),
        ("litellm.proxy.management_endpoints.ui_sso.get_user_info_from_db", AsyncMock(return_value=None)),
        (
            "litellm.proxy.management_endpoints.ui_sso.check_and_update_if_proxy_admin_id",
            AsyncMock(return_value="internal_user"),
        ),
    )
    with ExitStack() as stack:
        for target, value in callback_globals:
            stack.enter_context(patch(target, value))  # test-quality-ok: proxy globals and DB helpers
        await SSOAuthenticationHandler.get_redirect_response_from_openid(
            result=CustomOpenID(id=USER_ID, email="u@example.com", provider="generic", team_ids=[]),
            request=request,
            received_response=None,
            generic_client_id="cid",
            ui_access_mode=ui_access_mode,
        )


_Login = Callable[[_SessionKeyTable, pytest.MonkeyPatch], Awaitable[None]]

_LOGINS: Final = (
    pytest.param(_admin_login, id="ui-admin-password"),
    pytest.param(_password_login, id="user-password"),
    pytest.param(_sso_login, id="sso"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("login", _LOGINS)
async def test_a_login_ends_the_users_earlier_session(login: _Login, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", raising=False)
    table: Final = _SessionKeyTable()

    with _session_db(table):
        await login(table, monkeypatch)
        await login(table, monkeypatch)

    assert table.tokens() == ["session-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("login", _LOGINS)
async def test_allow_multiple_sessions_setting_keeps_every_session(
    login: _Login, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", "true")
    table: Final = _SessionKeyTable()

    with _session_db(table):
        await login(table, monkeypatch)
        await login(table, monkeypatch)

    assert table.tokens() == ["session-1", "session-2"]


@pytest.mark.asyncio
async def test_a_login_leaves_other_users_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", raising=False)
    table: Final = _SessionKeyTable()
    table.add("other-user-session", "other-hash", "user-456", _FIRST_SIGN_IN)

    with _session_db(table):
        await _password_login(table, monkeypatch)

    assert table.tokens() == ["other-user-session", "session-1"]


@pytest.mark.asyncio
async def test_a_refused_sso_sign_in_keeps_the_earlier_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", raising=False)
    table: Final = _SessionKeyTable()

    with _session_db(table):
        await _sso_login(table, monkeypatch)
        with pytest.raises(HTTPException) as refused:
            await _sso_login(table, monkeypatch, ui_access_mode="admin_only")

    assert refused.value.status_code == 401
    assert table.tokens() == ["session-1", "session-2"]


@pytest.mark.asyncio
async def test_a_failed_key_write_keeps_the_earlier_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", raising=False)
    table: Final = _SessionKeyTable()

    with _session_db(table):
        await _password_login(table, monkeypatch)
        table.mint_fails = True
        with pytest.raises(RuntimeError, match="key write failed"):
            await _password_login(table, monkeypatch)

    assert table.tokens() == ["session-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finishing_order", [("sk-session-1", "sk-session-2"), ("sk-session-2", "sk-session-1")])
async def test_overlapping_sign_ins_keep_only_the_newer_session(
    finishing_order: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_UI_ALLOW_MULTIPLE_SESSIONS", raising=False)
    table: Final = _SessionKeyTable()
    await table.mint(user_id=USER_ID)
    await table.mint(user_id=USER_ID)

    with _session_db(table):
        for new_key in finishing_order:
            await end_earlier_ui_sessions(USER_ID, new_key)

    assert table.tokens() == ["session-2"]
