"""
Tests for LiteLLM proxy realtime WebRTC HTTP endpoints:
- POST /v1/realtime/client_secrets
- POST /v1/realtime/calls
"""

import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.realtime_endpoints.endpoints import (
    _decode_realtime_token_payload,
    _encode_realtime_token_payload,
)

# --- Unit tests: token encode/decode helpers ---


def test_encode_realtime_token_payload():
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_abc123",
        model_id="gpt-4o-realtime-preview",
        user_id="user-1",
        team_id="team-1",
        expires_at=1234567890,
    )
    decoded = json.loads(payload)
    assert decoded["v"] == "realtime_v1"
    assert decoded["ephemeral_key"] == "epk_abc123"
    assert decoded["model_id"] == "gpt-4o-realtime-preview"
    assert decoded["user_id"] == "user-1"
    assert decoded["team_id"] == "team-1"
    assert decoded["expires_at"] == 1234567890


def test_encode_realtime_token_payload_none_optional_fields():
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_xyz",
        model_id="gpt-4o-realtime",
        user_id=None,
        team_id=None,
        expires_at=None,
    )
    decoded = json.loads(payload)
    assert decoded["user_id"] == ""
    assert decoded["team_id"] == ""
    assert decoded["expires_at"] is None


def test_decode_realtime_token_payload_valid():
    future_expires_at = int(time.time()) + 3600
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_abc",
        model_id="gpt-4o",
        user_id=None,
        team_id=None,
        expires_at=future_expires_at,
    )
    decrypted = json.loads(payload)  # simulate decrypted value
    result = _decode_realtime_token_payload(json.dumps(decrypted))
    assert result is not None
    assert result["ephemeral_key"] == "epk_abc"
    assert result["model_id"] == "gpt-4o"
    assert result["expires_at"] == future_expires_at


def test_decode_realtime_token_payload_invalid_version():
    payload = json.dumps(
        {
            "v": "realtime_v2",
            "ephemeral_key": "epk",
            "model_id": "gpt-4o",
        }
    )
    assert _decode_realtime_token_payload(payload) is None


def test_decode_realtime_token_payload_invalid_json():
    assert _decode_realtime_token_payload("not-json") is None


def test_decode_realtime_token_payload_missing_ephemeral_key():
    payload = json.dumps({"v": "realtime_v1", "model_id": "gpt-4o"})
    assert _decode_realtime_token_payload(payload) is None


def test_decode_realtime_token_payload_ephemeral_key_not_string():
    payload = json.dumps(
        {
            "v": "realtime_v1",
            "ephemeral_key": 123,
            "model_id": "gpt-4o",
        }
    )
    assert _decode_realtime_token_payload(payload) is None


# --- Integration tests: proxy endpoints (mocked upstream) ---


@pytest.fixture
def proxy_app(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master-key")
    return proxy_server.app


@pytest.fixture
def mock_route_request_client_secrets():
    """Mock route_request to return a fake upstream client_secrets response."""
    future_expires_at = int(time.time()) + 3600
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = (
        f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
    )
    mock_resp.content = f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'.encode()
    mock_resp.headers = {}
    mock_resp.json.return_value = {
        "value": "upstream_ephemeral_key",
        "expires_at": future_expires_at,
    }

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    return _mock_route


@pytest.fixture
def mock_route_request_realtime_calls():
    """Mock route_request to return a fake SDP answer."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 201
    mock_resp.content = b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n"
    mock_resp.headers = {"content-type": "application/sdp"}

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    return _mock_route


@pytest.fixture
def mock_add_litellm_data():
    async def _mock(data, **kwargs):
        return data

    return _mock


@pytest.fixture
def mock_pre_call_hook():
    async def _mock(user_api_key_dict, data, call_type):
        return data

    return _mock


def test_client_secrets_requires_auth(proxy_app):
    """POST /v1/realtime/client_secrets returns 401 without Authorization."""
    from fastapi import HTTPException

    def _raise_401():
        raise HTTPException(status_code=401, detail="Unauthorized")

    proxy_app.dependency_overrides[user_api_key_auth] = _raise_401
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/realtime/client_secrets",
            json={"model": "gpt-4o-realtime-preview"},
        )
        assert response.status_code == 401
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_success_with_mock(
    proxy_app,
    mock_route_request_client_secrets,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """POST /v1/realtime/client_secrets returns 200 with valid auth and mocked upstream."""
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", team_id="test-team"
    )
    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=mock_route_request_client_secrets,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "gpt-4o-realtime-preview"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "value" in data
        assert data["expires_at"] is not None
        assert data["expires_at"] > int(time.time())  # Should be in the future
        # Proxy encrypts the upstream value, so returned value should differ
        assert data["value"] != "upstream_ephemeral_key"
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_transcription_rejects_disallowed_nested_model(
    proxy_app,
):
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-4o-realtime-preview"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "model": "gpt-4o-realtime-preview",
                    "session": {
                        "type": "transcription",
                        "model": "gpt-4o-realtime-preview",
                        "audio": {
                            "input": {
                                "transcription": {
                                    "model": "gpt-realtime-whisper"
                                }
                            }
                        },
                    },
                },
            )

        assert response.status_code == 403
        assert "Tried to access gpt-realtime-whisper" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_transcription_routes_on_nested_model(
    proxy_app,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-4o-realtime-preview", "gpt-realtime-whisper"],
    )
    captured = {}
    future_expires_at = int(time.time()) + 3600

    async def _capturing_route(*args, **kwargs):
        captured["data"] = kwargs.get("data")

        async def _inner():
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.text = (
                f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
            )
            resp.content = (
                f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
            ).encode()
            resp.headers = {}
            resp.json.return_value = {
                "value": "upstream_ephemeral_key",
                "expires_at": future_expires_at,
            }
            return resp

        return _inner()

    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_capturing_route,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "model": "gpt-4o-realtime-preview",
                    "session": {
                        "type": "transcription",
                        "model": "gpt-4o-realtime-preview",
                        "audio": {
                            "input": {
                                "transcription": {
                                    "model": "gpt-realtime-whisper"
                                }
                            }
                        },
                    },
                },
            )

        assert response.status_code == 200
        assert captured["data"]["model"] == "gpt-realtime-whisper"
        session = captured["data"]["session"]
        assert session["type"] == "transcription"
        assert "model" not in session
        assert (
            session["audio"]["input"]["transcription"]["model"]
            == "gpt-realtime-whisper"
        )
        encrypted_value = response.json()["value"]
        decoded = _decode_realtime_token_payload(
            decrypt_value_helper(
                encrypted_value,
                key="client_secret.value",
                exception_type="debug",
            )
            or ""
        )
        assert decoded is not None
        assert decoded["model_id"] == "gpt-realtime-whisper"
        assert decoded["session_type"] == "transcription"
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


def test_realtime_calls_requires_auth(proxy_app):
    """POST /v1/realtime/calls returns 401 without Authorization.

    Note: /realtime/calls does NOT use the user_api_key_auth dependency —
    it checks the Bearer token manually (an encrypted ephemeral key from
    /realtime/client_secrets).  So no dependency override is needed here.
    """
    client = TestClient(proxy_app)
    response = client.post(
        "/v1/realtime/calls",
        content=b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n",
    )
    assert response.status_code == 401


def test_realtime_calls_invalid_token_returns_401(proxy_app):
    """POST /v1/realtime/calls returns 401 with invalid Bearer token."""
    client = TestClient(proxy_app)
    response = client.post(
        "/v1/realtime/calls",
        headers={"Authorization": "Bearer invalid-token-not-encrypted"},
        content=b"v=0\r\n",
    )
    assert response.status_code == 401
    assert "Invalid or expired token" in response.json().get("error", "")


@pytest.mark.asyncio
async def test_realtime_calls_success_with_valid_encrypted_token(
    proxy_app,
    mock_route_request_realtime_calls,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """POST /v1/realtime/calls returns 201 with valid encrypted token from client_secrets."""
    # Build a valid encrypted token (same format as client_secrets returns)
    future_expires_at = int(time.time()) + 3600
    token_payload = _encode_realtime_token_payload(
        ephemeral_key="fake_upstream_epk",
        model_id="gpt-4o-realtime-preview",
        user_id=None,
        team_id=None,
        expires_at=future_expires_at,
    )
    encrypted_token = encrypt_value_helper(token_payload)

    client = TestClient(proxy_app)
    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=mock_route_request_realtime_calls,
        ),
        patch(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()

        response = client.post(
            "/v1/realtime/calls",
            headers={"Authorization": f"Bearer {encrypted_token}"},
            content=b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
        )

    assert response.status_code == 201
    assert response.content.startswith(b"v=0")
    assert b"application/sdp" in response.headers.get("content-type", "").encode()


def test_token_payload_carries_session_type():
    """The encrypted token records the session kind so /realtime/calls can replay it."""
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk",
        model_id="gpt-realtime-whisper",
        user_id=None,
        team_id=None,
        expires_at=None,
        session_type="transcription",
    )
    decoded = _decode_realtime_token_payload(payload)
    assert decoded is not None
    assert decoded["session_type"] == "transcription"


@pytest.mark.asyncio
async def test_realtime_calls_replays_transcription_session_type(
    proxy_app,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """
    A token minted for a transcription session must drive /realtime/calls to send
    session.type == "transcription" upstream, not the default "realtime".
    """
    captured = {}

    async def _capturing_route(*args, **kwargs):
        captured["session"] = kwargs.get("data", {}).get("session")

        async def _inner():
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 201
            resp.content = b"v=0\r\n"
            resp.headers = {"content-type": "application/sdp"}
            return resp

        return _inner()

    token_payload = _encode_realtime_token_payload(
        ephemeral_key="epk",
        model_id="gpt-realtime-whisper",
        user_id=None,
        team_id=None,
        expires_at=int(time.time()) + 3600,
        session_type="transcription",
    )
    encrypted_token = encrypt_value_helper(token_payload)

    client = TestClient(proxy_app)
    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=_capturing_route,
        ),
        patch(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()

        client.post(
            "/v1/realtime/calls",
            headers={"Authorization": f"Bearer {encrypted_token}"},
            content=b"v=0\r\n",
        )

    assert captured["session"]["type"] == "transcription"
    assert (
        captured["session"]["audio"]["input"]["transcription"]["model"]
        == "gpt-realtime-whisper"
    )


# --- transcription_sessions endpoint ---


@pytest.fixture
def mock_route_request_transcription_sessions():
    """Mock route_request to return a fake transcription_sessions upstream response."""
    future_expires_at = int(time.time()) + 3600
    body = {
        "id": "sess_abc",
        "object": "realtime.transcription_session",
        "client_secret": {
            "value": "upstream_ephemeral_key",
            "expires_at": future_expires_at,
        },
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = json.dumps(body)
    mock_resp.content = json.dumps(body).encode()
    mock_resp.headers = {}
    mock_resp.json.return_value = body

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    return _mock_route


def test_transcription_sessions_requires_auth(proxy_app):
    """POST /v1/realtime/transcription_sessions returns 401 without Authorization."""
    from fastapi import HTTPException

    def _raise_401():
        raise HTTPException(status_code=401, detail="Unauthorized")

    proxy_app.dependency_overrides[user_api_key_auth] = _raise_401
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/realtime/transcription_sessions",
            json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
        )
        assert response.status_code == 401
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_rejects_disallowed_resolved_model(
    proxy_app,
):
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-4o-realtime-preview"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "input_audio_transcription": {"model": "gpt-realtime-whisper"}
                },
            )

        assert response.status_code == 403
        assert "Tried to access gpt-realtime-whisper" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_rejects_disallowed_team_model_scope(
    proxy_app,
):
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    team = LiteLLM_TeamTableCachedObj(
        team_id="team-a",
        models=["gpt-4o-realtime-preview"],
    )
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        team_id="team-a",
        models=["*"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new=AsyncMock(return_value=team),
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_membership",
                new=AsyncMock(return_value=None),
            ),
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "input_audio_transcription": {"model": "gpt-realtime-whisper"}
                },
            )

        assert response.status_code == 403
        assert "team" in response.text.lower()
        assert "Tried to access gpt-realtime-whisper" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_rejects_disallowed_project_model_scope(
    proxy_app,
):
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj

    project = LiteLLM_ProjectTableCachedObj(
        project_id="project-a",
        models=["gpt-4o-realtime-preview"],
        created_by="test-user",
        updated_by="test-user",
    )
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        project_id="project-a",
        models=["*"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
            patch(
                "litellm.proxy.auth.auth_checks.get_project_object",
                new=AsyncMock(return_value=project),
            ),
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "input_audio_transcription": {"model": "gpt-realtime-whisper"}
                },
            )

        assert response.status_code == 403
        assert "project" in response.text.lower()
        assert "Tried to access gpt-realtime-whisper" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_rejects_disallowed_team_member_model_scope(
    proxy_app,
):
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_TeamMembership,
        LiteLLM_TeamTableCachedObj,
    )

    team = LiteLLM_TeamTableCachedObj(team_id="team-a", models=["*"])
    membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="team-a",
        litellm_budget_table=LiteLLM_BudgetTable(
            allowed_models=["gpt-4o-realtime-preview"],
        ),
    )
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        team_id="team-a",
        models=["*"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new=AsyncMock(return_value=team),
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_membership",
                new=AsyncMock(return_value=membership),
            ),
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "input_audio_transcription": {"model": "gpt-realtime-whisper"}
                },
            )

        assert response.status_code == 403
        assert "Team member not allowed to access model" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_realtime_transcription_websocket_default_model_checks_key_scope():
    from litellm.proxy import proxy_server

    websocket = MagicMock()
    websocket.headers = {}
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()

    await proxy_server.realtime_websocket_endpoint(
        websocket=websocket,
        model=None,
        intent="transcription",
        user_api_key_dict=UserAPIKeyAuth(models=["gpt-4o-realtime-preview"]),
    )

    websocket.accept.assert_not_awaited()
    websocket.close.assert_awaited_once()
    _, close_kwargs = websocket.close.call_args
    assert close_kwargs["code"] == 1008
    assert "not allowed to access model" in close_kwargs["reason"]


@pytest.mark.asyncio
async def test_realtime_transcription_websocket_default_model_checks_team_scope():
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    team = LiteLLM_TeamTableCachedObj(
        team_id="team-a",
        models=["gpt-4o-realtime-preview"],
    )
    websocket = MagicMock()
    websocket.headers = {}
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()

    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(return_value=team),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new=AsyncMock(return_value=None),
        ),
    ):
        await proxy_server.realtime_websocket_endpoint(
            websocket=websocket,
            model=None,
            intent="transcription",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="test-user",
                team_id="team-a",
                models=["*"],
            ),
        )

    websocket.accept.assert_not_awaited()
    websocket.close.assert_awaited_once()
    _, close_kwargs = websocket.close.call_args
    assert close_kwargs["code"] == 1008
    assert "not allowed to access model" in close_kwargs["reason"]


@pytest.mark.asyncio
async def test_transcription_sessions_encrypts_client_secret(
    proxy_app,
    mock_route_request_transcription_sessions,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """
    POST /v1/realtime/transcription_sessions returns 200 and the ephemeral key
    under client_secret.value must be encrypted (never the raw upstream key).
    """
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", team_id="test-team"
    )
    captured_route_type = {}

    async def _capturing_route(*args, **kwargs):
        captured_route_type["route_type"] = kwargs.get("route_type")
        return await mock_route_request_transcription_sessions(*args, **kwargs)

    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_capturing_route,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "gpt-realtime-whisper"},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["client_secret"]["value"] != "upstream_ephemeral_key"
        # The encrypted value must decrypt back to a payload carrying the raw key.
        decrypted = decrypt_value_helper(
            data["client_secret"]["value"],
            key="client_secret.value",
            exception_type="debug",
        )
        assert decrypted is not None
        assert "upstream_ephemeral_key" in decrypted
        # Routed through the dedicated transcription_sessions route type.
        assert (
            captured_route_type["route_type"]
            == "acreate_realtime_transcription_session"
        )
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


def test_session_type_coerced_for_unknown_value():
    """An unrecognized session_type in the token falls back to 'realtime'."""
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk",
        model_id="gpt-4o",
        user_id=None,
        team_id=None,
        expires_at=None,
        session_type="INJECTED_TYPE",
    )
    # Force-deserialize and check the coercion that happens in proxy_realtime_calls.
    decoded = json.loads(payload)
    session_type = decoded.get("session_type") or "realtime"
    if session_type not in ("realtime", "transcription"):
        session_type = "realtime"
    assert session_type == "realtime"


@pytest.mark.asyncio
async def test_client_secrets_realtime_default_model_blocked_when_not_in_key_scope(
    proxy_app,
):
    """
    Regression: omitting both model and session.model must NOT bypass the authz
    check. The endpoint defaults to gpt-4o-realtime-preview; a key that cannot
    reach that model must receive 403.
    """
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["some-other-model"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={},
            )

        assert response.status_code == 403
        assert "gpt-4o-realtime-preview" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_realtime_explicit_model_blocked_when_not_in_key_scope(
    proxy_app,
):
    """An explicit model not in the key's allowed list must also be rejected."""
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-4o-realtime-preview"],
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch("litellm.proxy.proxy_server.route_request") as mock_route_request,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "gpt-4o-realtime-mini"},
            )

        assert response.status_code == 403
        assert "gpt-4o-realtime-mini" in response.text
        mock_route_request.assert_not_called()
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_realtime_default_model_allowed_when_in_key_scope(
    proxy_app,
    mock_route_request_client_secrets,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """Omitting model should succeed when the default (gpt-4o-realtime-preview) is in scope."""
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-4o-realtime-preview"],
    )
    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=mock_route_request_client_secrets,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={},
            )

        assert response.status_code == 200
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_returns_upstream_error_verbatim(
    proxy_app,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """Non-200 upstream response is forwarded unchanged (no encryption attempted)."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.content = b'{"error":"bad_request"}'
    mock_resp.headers = {}
    mock_resp.json.return_value = {"error": "bad_request"}
    mock_resp.text = '{"error":"bad_request"}'

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", team_id="test-team"
    )
    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_mock_route,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
            )
        assert response.status_code == 400
        assert response.content == b'{"error":"bad_request"}'
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_transcription_sessions_wraps_route_exception(
    proxy_app,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """A route exception is wrapped in a ProxyException with a human-readable message."""
    from fastapi import HTTPException

    async def _raise_http(*args, **kwargs):
        raise HTTPException(status_code=403, detail="Model not allowed")

    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user"
    )
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_raise_http,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
            )
        assert response.status_code == 403
        assert "Model not allowed" in response.text
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)
