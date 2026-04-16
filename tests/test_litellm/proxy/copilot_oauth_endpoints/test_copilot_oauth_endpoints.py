import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.llms.github_copilot.common_utils import (
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)
from litellm.llms.github_copilot.db_authenticator import DBAuthenticator
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.copilot_oauth_endpoints import endpoints as oauth_endpoints
from litellm.proxy.copilot_oauth_endpoints.endpoints import (
    RefreshRequest,
    StartRequest,
    _sessions,
    _sessions_lock,
    oauth_cancel,
    oauth_refresh,
    oauth_status,
    start_oauth,
)


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="u-1", user_role=LitellmUserRoles.INTERNAL_USER)


def _view_only_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="view-1", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )


@pytest.fixture(autouse=True)
def _clear_sessions():
    with _sessions_lock:
        _sessions.clear()
    DBAuthenticator._api_key_cache.clear()
    yield
    with _sessions_lock:
        _sessions.clear()
    DBAuthenticator._api_key_cache.clear()


class TestStartOAuth:
    @pytest.mark.asyncio
    async def test_admin_only_rejects_internal_user(self):
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"), user_api_key_dict=_non_admin()
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_only_rejects_view_only_admin(self):
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"),
                user_api_key_dict=_view_only_admin(),
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_when_session_cap_reached(self):
        from litellm.proxy.copilot_oauth_endpoints.endpoints import SESSIONS_MAX_SIZE

        with _sessions_lock:
            for i in range(SESSIONS_MAX_SIZE):
                _sessions[f"existing-{i}"] = {
                    "status": "pending",
                    "credential_name": "c",
                    "expires_at": time.time() + 600,
                }
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"), user_api_key_dict=_admin()
            )
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_device_code_failure_cleans_up_reserved_slot(self):
        with patch.object(
            oauth_endpoints.Authenticator,
            "_get_device_code",
            side_effect=GetDeviceCodeError(status_code=500, message="gh down"),
        ):
            with pytest.raises(HTTPException):
                await start_oauth(
                    StartRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        with _sessions_lock:
            assert len(_sessions) == 0

    @pytest.mark.asyncio
    async def test_creates_session_with_github_verification_url(self):
        fake_device_code = {
            "device_code": "dc-1",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
        }

        class _FakeThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with (
            patch.object(
                oauth_endpoints.Authenticator,
                "_get_device_code",
                return_value=fake_device_code,
            ),
            patch(
                "litellm.proxy.copilot_oauth_endpoints.endpoints.threading.Thread",
                _FakeThread,
            ),
        ):
            response = await start_oauth(
                StartRequest(credential_name="my-copilot"),
                user_api_key_dict=_admin(),
            )

        assert response.user_code == "ABCD-1234"
        assert response.verification_url == "https://github.com/login/device"
        assert response.interval == 5

    @pytest.mark.asyncio
    async def test_returns_502_on_device_code_failure(self):
        with patch.object(
            oauth_endpoints.Authenticator,
            "_get_device_code",
            side_effect=GetDeviceCodeError(status_code=500, message="gh down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await start_oauth(
                    StartRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        assert exc_info.value.status_code == 502
        assert "gh down" in exc_info.value.detail


class TestStatusAndCancel:
    @pytest.mark.asyncio
    async def test_status_returns_pending(self):
        with _sessions_lock:
            _sessions["s1"] = {
                "status": "pending",
                "credential_name": "c",
                "expires_at": time.time() + 600,
            }
        response = await oauth_status(session_id="s1", user_api_key_dict=_admin())
        assert response.status == "pending"

    @pytest.mark.asyncio
    async def test_cancel_flips_pending(self):
        with _sessions_lock:
            _sessions["s1"] = {
                "status": "pending",
                "credential_name": "c",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }
        result = await oauth_cancel(session_id="s1", user_api_key_dict=_admin())
        assert result == {"success": True}
        with _sessions_lock:
            assert _sessions["s1"]["status"] == "cancelled"


class TestRefreshEndpoint:
    @pytest.mark.asyncio
    async def test_admin_only(self):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_refresh(
                RefreshRequest(credential_name="c"), user_api_key_dict=_non_admin()
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_view_only_admin(self):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_refresh(
                RefreshRequest(credential_name="c"),
                user_api_key_dict=_view_only_admin(),
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_expires_at_on_success(self):
        with patch.object(
            DBAuthenticator,
            "force_refresh_api_key",
            return_value={"token": "fresh", "expires_at": 1700000000},
        ):
            response = await oauth_refresh(
                RefreshRequest(credential_name="c"),
                user_api_key_dict=_admin(),
            )
        assert response.credential_name == "c"
        assert response.api_key_expires_at == 1700000000

    @pytest.mark.asyncio
    async def test_502_on_refresh_failure(self):
        with patch.object(
            DBAuthenticator,
            "force_refresh_api_key",
            side_effect=RefreshAPIKeyError(status_code=401, message="bad token"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await oauth_refresh(
                    RefreshRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        assert exc_info.value.status_code == 502


class TestBackgroundWorker:
    def test_worker_persists_access_token_on_success(self, monkeypatch):
        from litellm.proxy.copilot_oauth_endpoints.endpoints import (
            _run_device_code_flow,
        )

        session_id = "s1"
        with _sessions_lock:
            _sessions[session_id] = {
                "status": "pending",
                "credential_name": "c",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }

        auth = MagicMock()
        auth._poll_for_access_token.return_value = "gho_fresh_token"

        store_mock = MagicMock()
        monkeypatch.setattr(
            DBAuthenticator, "store_access_token", lambda self, tok: store_mock(tok)
        )

        _run_device_code_flow(
            session_id=session_id,
            credential_name="c",
            device_code_info={"device_code": "dc", "user_code": "UC"},
            authenticator=auth,
        )

        store_mock.assert_called_once_with("gho_fresh_token")
        with _sessions_lock:
            assert _sessions[session_id]["status"] == "success"

    def test_worker_marks_error_on_poll_failure(self):
        from litellm.proxy.copilot_oauth_endpoints.endpoints import (
            _run_device_code_flow,
        )

        session_id = "s1"
        with _sessions_lock:
            _sessions[session_id] = {
                "status": "pending",
                "credential_name": "c",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }

        auth = MagicMock()
        auth._poll_for_access_token.side_effect = GetAccessTokenError(
            status_code=408, message="timed out"
        )

        _run_device_code_flow(
            session_id=session_id,
            credential_name="c",
            device_code_info={"device_code": "dc", "user_code": "UC"},
            authenticator=auth,
        )

        with _sessions_lock:
            assert _sessions[session_id]["status"] == "error"
            assert "timed out" in _sessions[session_id]["message"]
