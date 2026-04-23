import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.chatgpt_oauth_endpoints import endpoints as oauth_endpoints
from litellm.proxy.chatgpt_oauth_endpoints.endpoints import (
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
    return UserAPIKeyAuth(
        user_id="admin-1",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="user-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _view_only_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="view-1",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


@pytest.fixture(autouse=True)
def _clear_sessions():
    with _sessions_lock:
        _sessions.clear()
    yield
    with _sessions_lock:
        _sessions.clear()


class TestStartOAuth:
    @pytest.mark.asyncio
    async def test_admin_only_rejects_internal_user(self):
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"),
                user_api_key_dict=_non_admin(),
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_only_rejects_view_only_admin(self):
        """PROXY_ADMIN_VIEW_ONLY must not be able to start OAuth flows."""
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"),
                user_api_key_dict=_view_only_admin(),
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_when_session_cap_reached(self):
        from litellm.proxy.chatgpt_oauth_endpoints.endpoints import SESSIONS_MAX_SIZE

        with _sessions_lock:
            for i in range(SESSIONS_MAX_SIZE):
                _sessions[f"existing-{i}"] = {
                    "status": "pending",
                    "credential_name": "c",
                    "expires_at": time.time() + 600,
                }
        with pytest.raises(HTTPException) as exc_info:
            await start_oauth(
                StartRequest(credential_name="c"),
                user_api_key_dict=_admin(),
            )
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_device_code_failure_cleans_up_reserved_slot(self):
        """
        When the device-code network call fails, the reserved session slot
        must be released so the cap doesn't leak.
        """
        from litellm.llms.chatgpt.common_utils import GetDeviceCodeError

        with patch.object(
            oauth_endpoints.Authenticator,
            "_request_device_code",
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
    async def test_creates_session_and_spawns_worker(self):
        fake_device_code = {
            "device_auth_id": "d-1",
            "user_code": "ABCD-1234",
            "interval": "5",
        }

        # Stub the background async task so start_oauth doesn't actually run
        # the poll → exchange → persist flow during the test.
        async def _noop(*args, **kwargs):
            return None

        with (
            patch.object(
                oauth_endpoints.Authenticator,
                "_request_device_code",
                return_value=fake_device_code,
            ),
            patch(
                "litellm.proxy.chatgpt_oauth_endpoints.endpoints._run_device_code_flow_async",
                _noop,
            ),
        ):
            response = await start_oauth(
                StartRequest(credential_name="my-creds"),
                user_api_key_dict=_admin(),
            )

        assert response.user_code == "ABCD-1234"
        assert response.interval == 5
        assert response.verification_url.endswith("/codex/device")

        with _sessions_lock:
            assert response.session_id in _sessions
            entry = _sessions[response.session_id]
        assert entry["status"] == "pending"
        assert entry["credential_name"] == "my-creds"

    @pytest.mark.asyncio
    async def test_returns_502_on_device_code_failure(self):
        from litellm.llms.chatgpt.common_utils import GetDeviceCodeError

        with patch.object(
            oauth_endpoints.Authenticator,
            "_request_device_code",
            side_effect=GetDeviceCodeError(status_code=500, message="upstream down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await start_oauth(
                    StartRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        assert exc_info.value.status_code == 502
        assert "upstream down" in exc_info.value.detail


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_returns_pending_session(self):
        with _sessions_lock:
            _sessions["s1"] = {
                "status": "pending",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
            }
        response = await oauth_status(session_id="s1", user_api_key_dict=_admin())
        assert response.status == "pending"
        assert response.credential_name == "my-creds"

    @pytest.mark.asyncio
    async def test_404_on_unknown(self):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_status(session_id="nope", user_api_key_dict=_admin())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_admin_only(self):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_status(session_id="any", user_api_key_dict=_non_admin())
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_view_only_admin(self):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_status(session_id="any", user_api_key_dict=_view_only_admin())
        assert exc_info.value.status_code == 403


class TestCancelEndpoint:
    @pytest.mark.asyncio
    async def test_cancel_flips_pending_to_cancelled(self):
        with _sessions_lock:
            _sessions["s1"] = {
                "status": "pending",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }
        result = await oauth_cancel(session_id="s1", user_api_key_dict=_admin())
        assert result == {"success": True}
        with _sessions_lock:
            assert _sessions["s1"]["status"] == "cancelled"
            assert _sessions["s1"]["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_leaves_success_untouched(self):
        with _sessions_lock:
            _sessions["s1"] = {
                "status": "success",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }
        await oauth_cancel(session_id="s1", user_api_key_dict=_admin())
        with _sessions_lock:
            assert _sessions["s1"]["status"] == "success"


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
    async def test_404_when_no_stored_refresh_token(self):
        from litellm.llms.chatgpt.db_authenticator import DBAuthenticator

        with patch.object(DBAuthenticator, "_read_auth_file", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await oauth_refresh(
                    RefreshRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_successful_refresh_returns_new_expires_at(self):
        from litellm.llms.chatgpt.db_authenticator import DBAuthenticator

        stored = {
            "access_token": "old",
            "refresh_token": "r1",
            "expires_at": 1699000000,
        }
        refreshed_stored = {
            "access_token": "new",
            "refresh_token": "r1",
            "expires_at": 1700000000,
        }
        reads = [stored, refreshed_stored]

        with (
            patch.object(
                DBAuthenticator, "_read_auth_file", side_effect=lambda: reads.pop(0)
            ),
            patch.object(
                DBAuthenticator,
                "_refresh_tokens",
                return_value={"access_token": "new", "refresh_token": "r1"},
            ),
        ):
            response = await oauth_refresh(
                RefreshRequest(credential_name="c"),
                user_api_key_dict=_admin(),
            )
        assert response.credential_name == "c"
        assert response.expires_at == 1700000000

    @pytest.mark.asyncio
    async def test_502_on_refresh_failure(self):
        from litellm.llms.chatgpt.common_utils import RefreshAccessTokenError
        from litellm.llms.chatgpt.db_authenticator import DBAuthenticator

        with (
            patch.object(
                DBAuthenticator,
                "_read_auth_file",
                return_value={"access_token": "a", "refresh_token": "r"},
            ),
            patch.object(
                DBAuthenticator,
                "_refresh_tokens",
                side_effect=RefreshAccessTokenError(
                    status_code=400, message="refresh failed"
                ),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await oauth_refresh(
                    RefreshRequest(credential_name="c"),
                    user_api_key_dict=_admin(),
                )
        assert exc_info.value.status_code == 502


class TestBackgroundWorker:
    @pytest.mark.asyncio
    async def test_worker_marks_success_and_persists(self, monkeypatch):
        from litellm.proxy.chatgpt_oauth_endpoints.endpoints import (
            _run_device_code_flow_async,
        )

        session_id = "s1"
        with _sessions_lock:
            _sessions[session_id] = {
                "status": "pending",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }

        auth = MagicMock()
        auth._poll_for_authorization_code.return_value = {
            "authorization_code": "ac",
            "code_challenge": "cc",
            "code_verifier": "cv",
        }
        auth._exchange_code_for_tokens.return_value = {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
        }
        auth._build_auth_record.return_value = {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
            "account_id": "acct-1",
            "expires_at": 1700000000,
        }

        persist_mock = MagicMock()

        async def _fake_persist(item):
            persist_mock(item)

        monkeypatch.setattr(
            "litellm.proxy.chatgpt_oauth_endpoints.endpoints.persist_credential_to_db",
            _fake_persist,
        )
        monkeypatch.setattr(litellm, "credential_list", [])

        await _run_device_code_flow_async(
            session_id=session_id,
            credential_name="my-creds",
            device_code={"interval": "5"},
            authenticator=auth,
        )

        with _sessions_lock:
            assert _sessions[session_id]["status"] == "success"
        persist_mock.assert_called_once()
        persisted_item = persist_mock.call_args.args[0]
        assert persisted_item.credential_name == "my-creds"
        assert any(c.credential_name == "my-creds" for c in litellm.credential_list)

    @pytest.mark.asyncio
    async def test_worker_marks_error_on_db_persist_failure(self, monkeypatch):
        """
        If tokens are obtained but the DB write fails, the session should
        flip to ``error`` with an informative message (the in-memory cache
        was already updated; next retry via UI will retry the DB write).
        """
        from litellm.proxy.chatgpt_oauth_endpoints.endpoints import (
            _run_device_code_flow_async,
        )

        session_id = "s1"
        with _sessions_lock:
            _sessions[session_id] = {
                "status": "pending",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }

        auth = MagicMock()
        auth._poll_for_authorization_code.return_value = {
            "authorization_code": "ac",
            "code_challenge": "cc",
            "code_verifier": "cv",
        }
        auth._exchange_code_for_tokens.return_value = {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
        }
        auth._build_auth_record.return_value = {
            "access_token": "a",
            "refresh_token": "r",
            "id_token": "i",
            "account_id": "acct-1",
            "expires_at": 1700000000,
        }

        async def _boom(item):
            raise RuntimeError("prisma disconnected")

        monkeypatch.setattr(
            "litellm.proxy.chatgpt_oauth_endpoints.endpoints.persist_credential_to_db",
            _boom,
        )
        monkeypatch.setattr(litellm, "credential_list", [])

        await _run_device_code_flow_async(
            session_id=session_id,
            credential_name="my-creds",
            device_code={"interval": "5"},
            authenticator=auth,
        )

        with _sessions_lock:
            assert _sessions[session_id]["status"] == "error"
            assert "DB persist failed" in _sessions[session_id]["message"]

    @pytest.mark.asyncio
    async def test_worker_marks_error_on_auth_failure(self, monkeypatch):
        from litellm.llms.chatgpt.common_utils import GetAccessTokenError
        from litellm.proxy.chatgpt_oauth_endpoints.endpoints import (
            _run_device_code_flow_async,
        )

        session_id = "s1"
        with _sessions_lock:
            _sessions[session_id] = {
                "status": "pending",
                "credential_name": "my-creds",
                "expires_at": time.time() + 600,
                "cancelled": False,
            }

        auth = MagicMock()
        auth._poll_for_authorization_code.side_effect = GetAccessTokenError(
            status_code=408, message="timed out"
        )

        await _run_device_code_flow_async(
            session_id=session_id,
            credential_name="my-creds",
            device_code={"interval": "5"},
            authenticator=auth,
        )

        with _sessions_lock:
            assert _sessions[session_id]["status"] == "error"
            assert "timed out" in _sessions[session_id]["message"]
