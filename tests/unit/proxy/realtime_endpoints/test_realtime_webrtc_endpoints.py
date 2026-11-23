"""
Tests for LiteLLM proxy realtime WebRTC HTTP endpoints:
- POST /v1/realtime/client_secrets
- POST /v1/realtime/calls
"""

import functools
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Final, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.routing import Match

import litellm
from litellm.litellm_core_utils.realtime_streaming import TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE
from litellm.proxy._lazy_features import LAZY_FEATURES, attach_lazy_features, reserve_lazy_slot
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.realtime_endpoints.endpoints import (
    _ALLOWED_SESSION_TYPES,
    _coerce_realtime_session_type,
    _decode_realtime_token_payload,
    _encode_realtime_token_payload,
    _prepare_client_secret_session,
)
from litellm.types.realtime import (
    RealtimeAudioInputConfig,
    RealtimeAudioTranscriptionConfig,
    RealtimeClientSecretRequest,
    RealtimeSessionAudioConfig,
    RealtimeSessionConfig,
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
    mock_resp.text = f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
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


class AddLitellmDataToRequest(Protocol):
    def __call__(self, data: dict[str, object], **kwargs: object) -> Awaitable[dict[str, object]]: ...


class PreCallHook(Protocol):
    def __call__(
        self, user_api_key_dict: UserAPIKeyAuth, data: dict[str, object], call_type: str
    ) -> Awaitable[dict[str, object]]: ...


@pytest.fixture
def mock_add_litellm_data() -> AddLitellmDataToRequest:
    async def _mock(data: dict[str, object], **kwargs: object) -> dict[str, object]:
        return data

    return _mock


@pytest.fixture
def mock_pre_call_hook() -> PreCallHook:
    async def _mock(user_api_key_dict: UserAPIKeyAuth, data: dict[str, object], call_type: str) -> dict[str, object]:
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
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user", team_id="test-team")
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
                        "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}},
                    },
                },
            )

        assert response.status_code == 403
        assert "The requested model 'gpt-realtime-whisper' is not available for this API key" in response.text
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
            resp.text = f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
            resp.content = (f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}').encode()
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
                        "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}},
                    },
                },
            )

        assert response.status_code == 200
        assert captured["data"]["model"] == "gpt-realtime-whisper"
        session = captured["data"]["session"]
        assert session["type"] == "transcription"
        assert "model" not in session
        assert session["audio"]["input"]["transcription"]["model"] == "gpt-realtime-whisper"
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


@pytest.mark.parametrize("authorization", ["Basic {token}", "bearer{token}", "bearer"])
def test_realtime_calls_without_a_bearer_credential_returns_401(proxy_app: FastAPI, authorization: str) -> None:
    token: Final = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-4o-realtime-preview",
            user_id=None,
            team_id=None,
            expires_at=None,
        )
    )
    response: Final = TestClient(proxy_app).post(
        "/v1/realtime/calls",
        headers={"Authorization": authorization.format(token=token)},
        content=b"v=0\r\n",
    )
    assert response.status_code == 401
    assert response.json() == {"error": "Missing or invalid Authorization header"}


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
@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
async def test_realtime_calls_success_with_valid_encrypted_token(
    proxy_app,
    mock_route_request_realtime_calls,
    mock_add_litellm_data,
    mock_pre_call_hook,
    scheme: str,
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
            headers={"Authorization": f"{scheme} {encrypted_token}"},
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
    assert captured["session"]["audio"]["input"]["transcription"]["model"] == "gpt-realtime-whisper"


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
@pytest.mark.parametrize(
    "body",
    [
        {"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
        {"audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}}},
    ],
)
async def test_transcription_sessions_rejects_disallowed_resolved_model(
    proxy_app,
    body,
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
                json=body,
            )

        assert response.status_code == 403
        assert "The requested model 'gpt-realtime-whisper' is not available for this API key" in response.text
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
                json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
            )

        assert response.status_code == 403
        assert "team" in response.text.lower()
        assert "The requested model 'gpt-realtime-whisper' is not available for this API key" in response.text
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
                json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
            )

        assert response.status_code == 403
        assert "project" in response.text.lower()
        assert "The requested model 'gpt-realtime-whisper' is not available for this API key" in response.text
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
                json={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
            )

        assert response.status_code == 403
        assert "is not available for this API key" in response.text
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
    assert "is not available for this API key" in close_kwargs["reason"]


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
    assert "is not available for this API key" in close_kwargs["reason"]


@pytest.mark.asyncio
async def test_realtime_websocket_phase2_failure_sends_error_event_and_reasoned_close():
    """Regression for the realtime accept-then-silence hang: a phase-2 failure
    (routing / upstream credential resolution) used to close 1011 with the bare
    reason "Internal server error" and no error event, leaving the client with
    no clue what happened. The client must get an OpenAI-style error event and
    a close reason naming the failure."""
    from litellm.proxy import proxy_server

    events = []

    websocket = MagicMock()
    websocket.headers = {}
    websocket.scope = {"headers": []}
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock(side_effect=lambda payload: events.append(("send_text", payload)))
    websocket.close = AsyncMock(side_effect=lambda **kwargs: events.append(("close", kwargs)))

    mock_processor = MagicMock()
    mock_processor.common_processing_pre_call_logic = AsyncMock(
        return_value=({"model": "gpt-4o-realtime-preview"}, MagicMock())
    )

    with (
        patch(
            "litellm.proxy.proxy_server.can_key_call_resolved_model",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
        ),
        patch(
            "litellm.proxy.proxy_server.route_request",
            new=AsyncMock(side_effect=RuntimeError("vertex token refresh exploded")),
        ),
    ):
        await proxy_server.realtime_websocket_endpoint(
            websocket=websocket,
            model="gpt-4o-realtime-preview",
            intent=None,
            guardrails=None,
            user_api_key_dict=UserAPIKeyAuth(models=["*"]),
        )

    websocket.accept.assert_awaited_once()
    assert [name for name, _ in events] == ["send_text", "close"]

    error_event = json.loads(events[0][1])
    assert error_event["type"] == "error"
    assert error_event["error"]["type"] == "server_error"
    assert "vertex token refresh exploded" in error_event["error"]["message"]

    close_kwargs = events[1][1]
    assert close_kwargs["code"] == 1011
    assert "vertex token refresh exploded" in close_kwargs["reason"]
    assert len(close_kwargs["reason"].encode("utf-8")) <= 123


@pytest.mark.asyncio
async def test_realtime_websocket_phase2_failure_on_closed_socket_does_not_escape():
    """The lower handler layer may have already closed the client socket before
    the phase-2 handler runs (it closes on backend failures itself, then can
    re-raise). Send and close must each be guarded: the close is still
    attempted after a failed send, and neither failure escapes to the ASGI
    layer."""
    from litellm.proxy import proxy_server

    websocket = MagicMock()
    websocket.headers = {}
    websocket.scope = {"headers": []}
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock(side_effect=RuntimeError('Cannot call "send" once a close message has been sent.'))
    websocket.close = AsyncMock(side_effect=RuntimeError('Cannot call "send" once a close message has been sent.'))

    mock_processor = MagicMock()
    mock_processor.common_processing_pre_call_logic = AsyncMock(
        return_value=({"model": "gpt-4o-realtime-preview"}, MagicMock())
    )

    with (
        patch(
            "litellm.proxy.proxy_server.can_key_call_resolved_model",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
        ),
        patch(
            "litellm.proxy.proxy_server.route_request",
            new=AsyncMock(side_effect=RuntimeError("vertex token refresh exploded")),
        ),
    ):
        await proxy_server.realtime_websocket_endpoint(
            websocket=websocket,
            model="gpt-4o-realtime-preview",
            intent=None,
            guardrails=None,
            user_api_key_dict=UserAPIKeyAuth(models=["*"]),
        )

    websocket.close.assert_awaited_once()
    _, close_kwargs = websocket.close.call_args
    assert close_kwargs["code"] == 1011
    assert "vertex token refresh exploded" in close_kwargs["reason"]


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
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user", team_id="test-team")
    captured_route_type = {}

    async def _capturing_route(*args, **kwargs):
        captured_route_type["route_type"] = kwargs.get("route_type")
        captured_route_type["session"] = kwargs["data"]["transcription_session"]
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
                    "model": "gpt-realtime-whisper",
                    "audio": {"input": {"transcription": {"model": "gpt-live-transcribe", "language": "en"}}},
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
        assert captured_route_type["route_type"] == "acreate_realtime_transcription_session"
        assert captured_route_type["session"]["input_audio_transcription"]["model"] == "gpt-realtime-whisper"
        assert captured_route_type["session"]["audio"]["input"]["transcription"] == {
            "model": "gpt-realtime-whisper",
            "language": "en",
        }
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"model": "gpt-4o-realtime-preview", "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}}},
        {
            "input_audio_transcription": {"model": "gpt-4o-realtime-preview"},
            "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}},
        },
        {"model": "gpt-4o-realtime-preview", "input_audio_transcription": {"model": "gpt-realtime-whisper"}},
    ],
)
async def test_transcription_sessions_route_the_allowed_model_over_a_nested_model_the_key_cannot_call(
    proxy_app,
    mock_route_request_transcription_sessions,
    mock_add_litellm_data,
    mock_pre_call_hook,
    body,
):
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", models=["gpt-4o-realtime-preview"]
    )
    route: Final = AsyncMock(side_effect=mock_route_request_transcription_sessions)

    try:
        client = TestClient(proxy_app)
        with (
            patch("litellm.proxy.proxy_server.route_request", route),
            patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/transcription_sessions",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json=body,
            )

        assert response.status_code == 200, response.text
        route.assert_awaited_once()
        routed_session: Final = route.await_args.kwargs["data"]["transcription_session"]
        assert {
            routed_session.get("input_audio_transcription", {}).get("model"),
            routed_session.get("audio", {}).get("input", {}).get("transcription", {}).get("model"),
        } - {None} == {"gpt-4o-realtime-preview"}
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


def test_session_type_coerced_for_unknown_value():
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk",
        model_id="gpt-4o",
        user_id=None,
        team_id=None,
        expires_at=None,
        session_type="INJECTED_TYPE",
    )
    decoded = json.loads(payload)
    assert decoded["session_type"] == "INJECTED_TYPE"
    assert _coerce_realtime_session_type("INJECTED_TYPE") == "realtime"
    assert _coerce_realtime_session_type(None) == "realtime"
    for allowed_session_type in _ALLOWED_SESSION_TYPES:
        assert _coerce_realtime_session_type(allowed_session_type) == allowed_session_type


@pytest.mark.asyncio
async def test_translation_client_secret_rejects_disallowed_nested_transcription_model() -> None:
    req = RealtimeClientSecretRequest(
        model="gpt-realtime-translate",
        session=RealtimeSessionConfig(
            type="translation",
            model="gpt-realtime-translate",
            audio=RealtimeSessionAudioConfig(
                input=RealtimeAudioInputConfig(
                    transcription=RealtimeAudioTranscriptionConfig(model="gpt-live-transcribe"),
                )
            ),
        ),
    )

    with pytest.raises(Exception, match=r"gpt-live-transcribe.*not available for this API key"):
        await _prepare_client_secret_session(
            req=req,
            user_api_key_dict=UserAPIKeyAuth(models=["gpt-realtime-translate"]),
            llm_model_list=None,
            llm_router=None,
            forced_session_type="translation",
        )


@pytest.mark.asyncio
async def test_translation_client_secret_binds_authorized_nested_transcription_model() -> None:
    req = RealtimeClientSecretRequest(
        model="gpt-realtime-translate",
        session=RealtimeSessionConfig(
            type="translation",
            model="gpt-realtime-translate",
            audio=RealtimeSessionAudioConfig(
                input=RealtimeAudioInputConfig(
                    transcription=RealtimeAudioTranscriptionConfig(model="gpt-realtime-whisper"),
                )
            ),
        ),
    )

    model, session_data, session_type = await _prepare_client_secret_session(
        req=req,
        user_api_key_dict=UserAPIKeyAuth(models=["gpt-realtime-translate", "gpt-realtime-whisper"]),
        llm_model_list=None,
        llm_router=None,
        forced_session_type="translation",
    )

    assert model == "gpt-realtime-translate"
    assert session_type == "translation"
    assert session_data is not None
    assert session_data["model"] == "gpt-realtime-translate"
    assert session_data["audio"]["input"]["transcription"]["model"] == "gpt-realtime-whisper"


@pytest.mark.parametrize(
    "path",
    (
        "/openai/v1/realtime/client_secrets",
        "/openai/v1/realtime/calls",
        "/openai/v1/realtime/translations/client_secrets",
        "/openai/v1/realtime/translations/calls",
    ),
)
def test_openai_prefixed_realtime_http_paths_stay_with_the_openai_pass_through(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import openai_proxy_route

    monkeypatch.setenv("LITELLM_DISABLE_LAZY_ROUTES", "true")
    features = tuple(feature for feature in LAZY_FEATURES if feature.name in ("llm_passthrough", "realtime"))
    app = FastAPI()
    reserve_lazy_slot(app, "llm_passthrough", features=features)
    attach_lazy_features(app, features)
    scope = {"type": "http", "method": "POST", "path": path, "root_path": "", "headers": [], "query_string": b""}

    with TestClient(app):
        resolved = next(route for route in app.router.routes if route.matches(scope)[0] is Match.FULL)

    assert getattr(resolved, "endpoint", None) is openai_proxy_route


@pytest.mark.parametrize(
    "path",
    [
        "/v1/realtime/translations/client_secrets",
        "/realtime/translations/client_secrets",
    ],
)
def test_translation_client_secret_aliases_bind_token_family(
    proxy_app,
    mock_route_request_client_secrets,
    mock_add_litellm_data,
    mock_pre_call_hook,
    path,
):
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-realtime-translate"],
    )
    captured = {}

    async def capture_route(*args, **kwargs):
        captured["route_type"] = kwargs["route_type"]
        captured["data"] = kwargs["data"]
        return await mock_route_request_client_secrets(*args, **kwargs)

    try:
        client = TestClient(proxy_app)
        with (
            patch(  # test-quality-ok: endpoint test captures the proxy routing boundary
                "litellm.proxy.proxy_server.route_request", side_effect=capture_route
            ),
            patch(  # test-quality-ok: endpoint test isolates request metadata enrichment
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch(  # test-quality-ok: endpoint test isolates the process-wide proxy logger
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as logging,
        ):
            logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            logging.post_call_failure_hook = AsyncMock()
            response = client.post(
                path,
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "gpt-realtime-translate"},
            )

        assert response.status_code == 200
        assert captured["route_type"] == "acreate_realtime_translation_client_secret"
        assert captured["data"]["session"] == {
            "type": "translation",
            "model": "gpt-realtime-translate",
        }
        decrypted = decrypt_value_helper(
            response.json()["value"],
            key="client_secret.value",
            exception_type="debug",
        )
        decoded = _decode_realtime_token_payload(decrypted or "")
        assert decoded is not None
        assert decoded["session_type"] == "translation"
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/realtime/translations/calls",
        "/realtime/translations/calls",
    ],
)
def test_translation_calls_aliases_route_translation_session(
    proxy_app,
    mock_route_request_realtime_calls,
    mock_add_litellm_data,
    mock_pre_call_hook,
    path,
):
    token = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="ek_test",
            model_id="gpt-realtime-translate",
            user_id="test-user",
            team_id=None,
            expires_at=int(time.time()) + 3600,
            session_type="translation",
        )
    )
    captured = {}

    async def capture_route(*args, **kwargs):
        captured["route_type"] = kwargs["route_type"]
        captured["data"] = kwargs["data"]
        return await mock_route_request_realtime_calls(*args, **kwargs)

    client = TestClient(proxy_app)
    with (
        patch(  # test-quality-ok: endpoint test captures the proxy routing boundary
            "litellm.proxy.proxy_server.route_request", side_effect=capture_route
        ),
        patch(  # test-quality-ok: endpoint test isolates request metadata enrichment
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ),
        patch(  # test-quality-ok: endpoint test isolates the process-wide proxy logger
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as logging,
    ):
        logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        logging.post_call_failure_hook = AsyncMock()
        response = client.post(
            path,
            headers={"Authorization": f"Bearer {token}"},
            content=b"v=0\r\n",
        )

    assert response.status_code == 201
    assert captured["route_type"] == "arealtime_translation_calls"
    assert captured["data"]["session"] == {
        "type": "translation",
        "model": "gpt-realtime-translate",
    }


@pytest.mark.parametrize(
    ("disable_global_guardrails", "expected_status", "expected_body"),
    [(True, 201, "v=0"), (False, 400, TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE)],
)
def test_translation_calls_honor_the_minting_keys_guardrail_opt_out(
    proxy_app: FastAPI,
    mock_route_request_client_secrets: object,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    monkeypatch: pytest.MonkeyPatch,
    disable_global_guardrails: bool,
    expected_status: int,
    expected_body: str,
) -> None:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy import proxy_server
    from litellm.types.guardrails import GuardrailEventHooks

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-realtime-translate",
                    "litellm_params": {"model": "openai/gpt-realtime-translate", "api_key": "sk-upstream"},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [CustomGuardrail(guardrail_name="global-pre-call", event_hook=GuardrailEventHooks.pre_call, default_on=True)],
    )
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user",
        models=["gpt-realtime-translate"],
        metadata={"disable_global_guardrails": disable_global_guardrails},
    )
    client: Final = TestClient(proxy_app, raise_server_exceptions=False)
    try:
        with (
            patch(  # test-quality-ok: endpoint test stands in for the upstream client_secrets call
                "litellm.proxy.proxy_server.route_request", side_effect=mock_route_request_client_secrets
            ),
            patch(  # test-quality-ok: endpoint test isolates request metadata enrichment while minting
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch(  # test-quality-ok: endpoint test isolates the process-wide proxy logger while minting
                "litellm.proxy.proxy_server.proxy_logging_obj"
            ) as logging,
        ):
            logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            logging.post_call_failure_hook = AsyncMock()
            minted: Final = client.post(
                "/v1/realtime/translations/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "gpt-realtime-translate"},
            )
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)

    with respx.mock(assert_all_called=False) as upstream:
        upstream.post("https://api.openai.com/v1/realtime/translations/calls").mock(
            return_value=httpx.Response(201, content=b"v=0\r\n", headers={"content-type": "application/sdp"})
        )
        answered: Final = client.post(
            "/v1/realtime/translations/calls",
            headers={"Authorization": f"Bearer {minted.json()['value']}"},
            content=b"v=0\r\n",
        )

    assert minted.status_code == 200
    assert answered.status_code == expected_status
    assert expected_body in answered.text


@pytest.mark.parametrize(
    "session_type,path",
    [
        ("realtime", "/v1/realtime/translations/calls"),
        ("translation", "/v1/realtime/calls"),
    ],
)
def test_realtime_calls_reject_cross_family_token(proxy_app, session_type, path):
    model = "gpt-realtime-translate" if session_type == "translation" else "gpt-realtime-2"
    token = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="ek_test",
            model_id=model,
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            session_type=session_type,
        )
    )
    response = TestClient(proxy_app).post(
        path,
        headers={"Authorization": f"Bearer {token}"},
        content=b"v=0\r\n",
    )
    assert response.status_code == 401
    assert response.json()["error"] == "Token is not valid for this Realtime endpoint"


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

    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user", team_id="test-team")
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

    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user")
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


def test_realtime_calls_upstream_rejection_answers_an_openai_typed_error(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    monkeypatch: pytest.MonkeyPatch,
):
    """A bare HTTPException carries no type or param, so the tail used to ship the
    literal string "None" in both fields of the error the browser client reads."""
    token_payload = _encode_realtime_token_payload(
        ephemeral_key="fake_upstream_epk",
        model_id="gpt-4o-realtime-preview",
        user_id=None,
        team_id=None,
        expires_at=int(time.time()) + 3600,
    )
    encrypted_token = encrypt_value_helper(token_payload)

    async def failing_route_request(*args: object, **kwargs: object) -> None:
        raise HTTPException(
            status_code=404,
            detail={"error": "realtime: Invalid model name passed in model=gpt-4o-realtime-preview"},
        )

    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
    proxy_logging.post_call_failure_hook = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.route_request", failing_route_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", mock_add_litellm_data)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging)

    response = TestClient(proxy_app).post(
        "/v1/realtime/calls",
        headers={"Authorization": f"Bearer {encrypted_token}"},
        content=b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
    )

    assert response.status_code == 404
    assert (response.json()["error"]["type"], response.json()["error"]["param"]) == ("invalid_request_error", None)


def test_transcription_sessions_rejection_answers_an_openai_typed_error(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    monkeypatch: pytest.MonkeyPatch,
):
    """A model the router cannot serve surfaces as a bare HTTPException, which this tail
    used to relabel with the literal string "None" for both type and param."""

    async def failing_route_request(*args: object, **kwargs: object) -> None:
        raise HTTPException(
            status_code=400,
            detail={"error": "realtime: Invalid model name passed in model=no-such-transcribe"},
        )

    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
    proxy_logging.post_call_failure_hook = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.route_request", failing_route_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", mock_add_litellm_data)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging)
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user")
    try:
        response = TestClient(proxy_app, raise_server_exceptions=False).post(
            "/v1/realtime/transcription_sessions",
            headers={"Authorization": "Bearer sk-test-master-key"},
            json={"input_audio_transcription": {"model": "no-such-transcribe"}},
        )
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)

    assert response.status_code == 400
    assert (response.json()["error"]["type"], response.json()["error"]["param"]) == ("invalid_request_error", None)


_AZURE_REALTIME_DEPLOYMENTS: Final = (
    ("azure-realtime-alias", "azure/qa-realtime-deployment"),
    ("azure-transcribe-alias", "azure/qa-whisper-deployment"),
    ("azure-translate-alias", "azure/qa-translate-deployment"),
)


def _post_through_the_azure_deployments(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: object,
    deployments: tuple[tuple[str, str], ...] = _AZURE_REALTIME_DEPLOYMENTS,
) -> tuple[httpx.Response, respx.Route]:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_list: Final = [  # mutable-ok: Router takes a list of deployments
        {
            "model_name": alias,
            "litellm_params": {
                "model": deployment,
                "api_base": "https://realtime-qa.openai.azure.com",
                "api_key": "azure-test-key",
            },
        }
        for alias, deployment in deployments
    ]
    monkeypatch.setattr(proxy_server, "llm_router", litellm.Router(model_list=model_list))
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    upstream: Final = respx_mock.route(host="realtime-qa.openai.azure.com").mock(
        return_value=httpx.Response(200, json={"value": "ek_azure", "expires_at": int(time.time()) + 3600})
    )
    monkeypatch.setitem(
        proxy_app.dependency_overrides,
        user_api_key_auth,
        lambda: UserAPIKeyAuth(user_id="test-user", models=[alias for alias, _ in deployments]),
    )
    with (
        patch(  # test-quality-ok: endpoint test isolates request metadata enrichment
            "litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data
        ),
        patch(  # test-quality-ok: endpoint test isolates the process-wide proxy logger
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as logging,
    ):
        logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        logging.post_call_failure_hook = AsyncMock()
        response: Final = TestClient(proxy_app).post(
            path, headers={"Authorization": "Bearer sk-test-master-key"}, json=body
        )
    return response, upstream


def _upstream_session_value(upstream: respx.Route, session_path: tuple[str, ...]) -> object:
    return functools.reduce(
        lambda node, key: node[key], session_path, json.loads(upstream.calls.last.request.content)["session"]
    )


@pytest.mark.parametrize(
    ("path", "body", "upstream_model_path", "expected_model"),
    (
        (
            "/v1/realtime/client_secrets",
            {"session": {"type": "realtime", "model": "azure-realtime-alias"}},
            ("model",),
            "qa-realtime-deployment",
        ),
        (
            "/v1/realtime/client_secrets",
            {"model": "azure-realtime-alias"},
            ("model",),
            "qa-realtime-deployment",
        ),
        (
            "/v1/realtime/client_secrets",
            {
                "session": {
                    "type": "transcription",
                    "audio": {"input": {"transcription": {"model": "azure-transcribe-alias"}}},
                }
            },
            ("audio", "input", "transcription", "model"),
            "qa-whisper-deployment",
        ),
        (
            "/v1/realtime/translations/client_secrets",
            {"model": "azure-translate-alias"},
            ("model",),
            "qa-translate-deployment",
        ),
        (
            "/v1/realtime/translations/client_secrets",
            {
                "model": "azure-translate-alias",
                "session": {"audio": {"input": {"transcription": {"model": "azure-transcribe-alias"}}}},
            },
            ("audio", "input", "transcription", "model"),
            "qa-whisper-deployment",
        ),
    ),
)
def test_client_secrets_mint_a_secret_for_the_azure_deployment_model(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: dict[str, object],
    upstream_model_path: tuple[str, ...],
    expected_model: str,
) -> None:
    response, upstream = _post_through_the_azure_deployments(
        proxy_app, mock_add_litellm_data, mock_pre_call_hook, respx_mock, monkeypatch, path, body
    )

    assert response.status_code == 200, response.text
    assert _upstream_session_value(upstream, upstream_model_path) == expected_model


@pytest.mark.parametrize(
    ("body", "session_path", "expected"),
    (
        (
            {
                "session": {
                    "type": "realtime",
                    "model": "azure-realtime-alias",
                    "audio": {"input": {"turn_detection": None}},
                }
            },
            ("audio", "input", "turn_detection"),
            None,
        ),
        (
            {
                "session": {
                    "type": "transcription",
                    "audio": {"input": {"transcription": {"model": "azure-transcribe-alias"}, "turn_detection": None}},
                }
            },
            ("audio", "input", "turn_detection"),
            None,
        ),
        ({"session": {"type": "conversation", "model": "azure-realtime-alias"}}, ("type",), "realtime"),
        (
            {
                "session": {
                    "type": "transcription",
                    "audio": {"input": {"transcription": {"model": "azure-transcribe-alias", "delay": "auto"}}},
                }
            },
            ("audio", "input", "transcription", "delay"),
            "auto",
        ),
    ),
)
def test_client_secrets_forward_every_session_value_the_caller_set(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    body: Mapping[str, object],
    session_path: tuple[str, ...],
    expected: object,
) -> None:
    response, upstream = _post_through_the_azure_deployments(
        proxy_app,
        mock_add_litellm_data,
        mock_pre_call_hook,
        respx_mock,
        monkeypatch,
        "/v1/realtime/client_secrets",
        body,
    )

    assert response.status_code == 200, response.text
    assert _upstream_session_value(upstream, session_path) == expected


def test_client_secrets_without_a_model_let_the_upstream_pick_its_default_through_a_wildcard_deployment(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, upstream = _post_through_the_azure_deployments(
        proxy_app,
        mock_add_litellm_data,
        mock_pre_call_hook,
        respx_mock,
        monkeypatch,
        "/v1/realtime/client_secrets",
        {"session": {"type": "realtime"}},
        deployments=(("*", "openai/*"),),
    )

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls.last.request.content)["session"] == {"type": "realtime"}


def test_client_secrets_send_the_deployment_model_in_the_flat_transcription_field(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, upstream = _post_through_the_azure_deployments(
        proxy_app,
        mock_add_litellm_data,
        mock_pre_call_hook,
        respx_mock,
        monkeypatch,
        "/v1/realtime/client_secrets",
        {"session": {"type": "transcription", "input_audio_transcription": {"model": "azure-transcribe-alias"}}},
    )

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls.last.request.content)["session"] == {
        "type": "transcription",
        "input_audio_transcription": {"model": "qa-whisper-deployment"},
    }


@pytest.mark.parametrize(
    ("path", "body"),
    (
        ("/v1/realtime/client_secrets", {"session": "not-a-session"}),
        ("/v1/realtime/translations/client_secrets", {"session": {"audio": {"input": "not-an-input"}}}),
        ("/v1/realtime/transcription_sessions", {"audio": {"input": {"transcription": "not-a-transcription"}}}),
    ),
)
def test_realtime_session_routes_answer_400_for_a_malformed_body(
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: Mapping[str, object],
) -> None:
    response, upstream = _post_through_the_azure_deployments(
        proxy_app, mock_add_litellm_data, mock_pre_call_hook, respx_mock, monkeypatch, path, body
    )

    assert (response.status_code, response.json()["error"]["type"]) == (400, "invalid_request_error")
    assert upstream.called is False


def _upstream_sdp_answer_route(location: str | None) -> Callable[..., Awaitable[Awaitable[httpx.Response]]]:
    answer: Final = httpx.Response(
        201,
        content=b"v=0\r\n",
        headers={"content-type": "application/sdp", **({} if location is None else {"location": location})},
    )

    async def _route(*args: object, **kwargs: object) -> Awaitable[httpx.Response]:
        async def _inner() -> httpx.Response:
            return answer

        return _inner()

    return _route


@pytest.mark.parametrize(
    ("caller_auth", "location", "expected_sidebands"),
    [
        (
            {"token": "caller-key-hash", "team_id": "team-a"},
            "/v1/realtime/calls/rtc_abc",
            [("rtc_abc", "caller-key-hash", "deployment-1")],
        ),
        (None, "/v1/realtime/calls/rtc_abc", []),
        ({"token": "caller-key-hash"}, None, []),
    ],
    ids=["billed-to-the-minting-key", "token-without-a-caller-key", "no-call-id-from-upstream"],
)
def test_realtime_calls_meter_the_call_on_a_sideband_for_the_key_that_minted_the_token(
    proxy_app,
    monkeypatch: pytest.MonkeyPatch,
    mock_add_litellm_data,
    mock_pre_call_hook,
    caller_auth: dict[str, object] | None,
    location: str | None,
    expected_sidebands: list[tuple[str, str, str]],
) -> None:
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    started: Final[list[tuple[str, str | None, str | None]]] = []

    async def _record(
        call_id: str, model: str, caller_auth: UserAPIKeyAuth, deployment_id: str | None, is_translation: bool = False
    ) -> bool:
        started.append((call_id, caller_auth.token, deployment_id))
        return True

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _record)
    encrypted_token: Final = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-realtime",
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            caller_auth=caller_auth,
            deployment_id="deployment-1",
        )
    )

    with (
        patch("litellm.proxy.proxy_server.route_request", side_effect=_upstream_sdp_answer_route(location)),
        patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()
        response: Final = TestClient(proxy_app).post(
            "/v1/realtime/calls", headers={"Authorization": f"Bearer {encrypted_token}"}, content=b"v=0\r\n"
        )

    assert response.status_code == 201, response.text
    assert response.headers.get("location") == location
    assert started == expected_sidebands


@pytest.mark.parametrize(
    ("sideband_joins", "expected_status"), [(True, 201), (False, 503)], ids=["metered", "refused-when-unmetered"]
)
def test_translation_calls_are_metered_on_a_translation_sideband(
    proxy_app,
    monkeypatch: pytest.MonkeyPatch,
    mock_add_litellm_data,
    mock_pre_call_hook,
    sideband_joins: bool,
    expected_status: int,
) -> None:
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    started: Final[list[tuple[str, str | None, bool]]] = []

    async def _record(
        call_id: str, model: str, caller_auth: UserAPIKeyAuth, deployment_id: str | None, is_translation: bool = False
    ) -> bool:
        started.append((call_id, caller_auth.token, is_translation))
        return sideband_joins

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _record)
    encrypted_token: Final = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-realtime-translate",
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            session_type="translation",
            caller_auth={"token": "caller-key-hash"},
        )
    )

    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=_upstream_sdp_answer_route("/v1/realtime/translations/calls/rtc_abc"),
        ),
        patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()
        response: Final = TestClient(proxy_app).post(
            "/v1/realtime/translations/calls",
            headers={"Authorization": f"Bearer {encrypted_token}"},
            content=b"v=0\r\n",
        )

    assert response.status_code == expected_status, response.text
    assert started == [("rtc_abc", "caller-key-hash", True)]


def test_realtime_calls_refuse_a_call_the_proxy_could_not_join_to_meter(
    proxy_app, monkeypatch: pytest.MonkeyPatch, mock_add_litellm_data, mock_pre_call_hook
) -> None:
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    async def _sideband_does_not_join(**_call: object) -> bool:
        return False

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _sideband_does_not_join)
    encrypted_token: Final = encrypt_value_helper(
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-realtime",
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            caller_auth={"token": "caller-key-hash"},
        )
    )

    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=_upstream_sdp_answer_route("/v1/realtime/calls/rtc_abc"),
        ),
        patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()
        response: Final = TestClient(proxy_app).post(
            "/v1/realtime/calls", headers={"Authorization": f"Bearer {encrypted_token}"}, content=b"v=0\r\n"
        )

    assert response.status_code == 503, response.text
    assert b"v=0" not in response.content, "without the SDP answer the browser cannot start an unmetered call"
    assert "could not join this call to meter it" in response.text


def _post_realtime_call(proxy_app, mock_add_litellm_data, mock_pre_call_hook, token_payload: str) -> httpx.Response:
    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=_upstream_sdp_answer_route("/v1/realtime/calls/rtc_abc"),
        ),
        patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()
        return TestClient(proxy_app).post(
            "/v1/realtime/calls",
            headers={"Authorization": f"Bearer {encrypt_value_helper(token_payload)}"},
            content=b"v=0\r\n",
        )


@pytest.mark.parametrize("setting", [False, "false", " False "], ids=["a-boolean", "a-string", "a-padded-string"])
def test_realtime_calls_start_without_a_sideband_when_webrtc_metering_is_turned_off(
    proxy_app, monkeypatch: pytest.MonkeyPatch, mock_add_litellm_data, mock_pre_call_hook, setting: bool | str
) -> None:
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    started: Final[list[object]] = []

    async def _record(**call: object) -> bool:
        started.append(call)
        return False

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _record)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"meter_webrtc_calls": setting})

    response: Final = _post_realtime_call(
        proxy_app,
        mock_add_litellm_data,
        mock_pre_call_hook,
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-realtime",
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            caller_auth={"token": "caller-key-hash"},
        ),
    )

    assert (response.status_code, response.content) == (201, b"v=0\r\n"), response.text
    assert started == []


def _stored_keys(monkeypatch: pytest.MonkeyPatch, keys: dict[str, UserAPIKeyAuth]) -> None:
    from litellm.proxy._types import ProxyErrorTypes, ProxyException
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    async def _get_key_object(hashed_token: str, **_lookup: object) -> UserAPIKeyAuth:
        if hashed_token not in keys:
            raise ProxyException(message="gone", type=ProxyErrorTypes.token_not_found_in_db, param="key", code=401)
        return keys[hashed_token].model_copy()

    monkeypatch.setattr(realtime_endpoints, "get_key_object", _get_key_object)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", object())


def test_a_token_minted_by_a_stored_key_carries_the_key_and_only_what_the_request_added() -> None:
    from litellm.proxy.realtime_endpoints.endpoints import _caller_auth_for_token

    stored: Final = UserAPIKeyAuth(token="caller-key-hash", team_id="team-a", rpm_limit=10, models=["realtime-group"])
    caller: Final = stored.model_copy(update={"end_user_id": "customer-7"})

    assert _caller_auth_for_token(caller, stored) == {"end_user_id": "customer-7"}
    assert _caller_auth_for_token(caller)["rpm_limit"] == 10, "a caller with no stored key is carried whole"


class _UnreachableDatabase:
    def __getattr__(self, name: str) -> object:
        raise ConnectionError(f"the database is down, so {name} cannot be read")


@pytest.mark.asyncio
async def test_a_key_the_proxy_has_cached_is_read_for_a_realtime_call_while_the_database_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.realtime_endpoints.endpoints import _caller_auth_from_token
    from litellm.redis.stores.auth_cache import AuthObject

    key_cache: Final = UserApiKeyCache()
    await key_cache.async_set_cache(
        AuthObject.virtual_key("caller-key-hash"),
        UserAPIKeyAuth(token="caller-key-hash", team_id="team-a", rpm_limit=3),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", key_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _UnreachableDatabase())

    caller: Final = await _caller_auth_from_token(
        {"key": "caller-key-hash", "caller_auth": {"end_user_id": "customer-7"}}, None, None
    )

    assert caller is not None
    assert (caller.token, caller.team_id, caller.rpm_limit, caller.end_user_id) == (
        "caller-key-hash",
        "team-a",
        3,
        "customer-7",
    )


@pytest.mark.asyncio
async def test_the_master_key_is_never_looked_up_as_a_stored_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.realtime_endpoints.endpoints import _stored_key

    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _UnreachableDatabase())

    assert await _stored_key(UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS).token) is None


@pytest.mark.parametrize(
    ("stored", "expected_status", "expected_caller"),
    [
        (
            UserAPIKeyAuth(token="caller-key-hash", team_id="team-a", rpm_limit=3),
            201,
            [("caller-key-hash", "team-a", 3, "customer-7")],
        ),
        (None, 401, []),
        (UserAPIKeyAuth(token="caller-key-hash", blocked=True), 401, []),
        (UserAPIKeyAuth(token="caller-key-hash", expires=datetime(2020, 1, 1, tzinfo=timezone.utc)), 401, []),
    ],
    ids=["read-as-it-is-now", "deleted", "blocked", "expired"],
)
def test_realtime_calls_read_the_minting_key_as_it_is_stored_when_the_call_starts(
    proxy_app,
    monkeypatch: pytest.MonkeyPatch,
    mock_add_litellm_data,
    mock_pre_call_hook,
    stored: UserAPIKeyAuth | None,
    expected_status: int,
    expected_caller: list[tuple[str, str, int, str]],
) -> None:
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    started: Final[list[tuple[str | None, str | None, int | None, str | None]]] = []

    async def _record(
        call_id: str, model: str, caller_auth: UserAPIKeyAuth, deployment_id: str | None, is_translation: bool = False
    ) -> bool:
        started.append((caller_auth.token, caller_auth.team_id, caller_auth.rpm_limit, caller_auth.end_user_id))
        return True

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _record)
    _stored_keys(monkeypatch, {} if stored is None else {"caller-key-hash": stored})

    response: Final = _post_realtime_call(
        proxy_app,
        mock_add_litellm_data,
        mock_pre_call_hook,
        _encode_realtime_token_payload(
            ephemeral_key="fake_upstream_epk",
            model_id="gpt-realtime",
            user_id=None,
            team_id=None,
            expires_at=int(time.time()) + 3600,
            caller_auth={"end_user_id": "customer-7"},
            key="caller-key-hash",
        ),
    )

    assert response.status_code == expected_status, response.text
    assert started == expected_caller


@pytest.mark.parametrize("route", ["/v1/realtime", "/v1/realtime/translations"], ids=["realtime", "translation"])
def test_a_webrtc_call_stays_on_the_deployment_that_minted_its_client_secret(
    route: str,
    proxy_app: FastAPI,
    mock_add_litellm_data: AddLitellmDataToRequest,
    mock_pre_call_hook: PreCallHook,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.realtime_endpoints import endpoints as realtime_endpoints

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_list: Final = [  # mutable-ok: Router takes a list of deployments
        {
            "model_name": "realtime-group",
            "litellm_params": {
                "model": "azure/qa-realtime-deployment",
                "api_base": f"https://account-{name}.openai.azure.com",
                "api_key": f"key-{name}",
                "realtime_protocol": "GA",
            },
            "model_info": {"id": f"deployment-{name}"},
        }
        for name in ("a", "b", "c")
    ]
    monkeypatch.setattr(proxy_server, "llm_router", litellm.Router(model_list=model_list))
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    respx_mock.route(path__regex=r".*/client_secrets$").mock(
        return_value=httpx.Response(200, json={"value": "ek_azure", "expires_at": int(time.time()) + 3600})
    )
    respx_mock.route(path__regex=r".*/calls$").mock(
        return_value=httpx.Response(
            201,
            content=b"v=0\r\n",
            headers={"content-type": "application/sdp", "location": "/v1/realtime/calls/rtc_abc"},
        )
    )
    metered_on: Final[list[str | None]] = []  # mutable-ok: collects the deployment each sideband was pinned to

    async def _record(
        call_id: str, model: str, caller_auth: UserAPIKeyAuth, deployment_id: str | None, is_translation: bool = False
    ) -> bool:
        metered_on.append(deployment_id)
        return True

    monkeypatch.setattr(realtime_endpoints, "_start_realtime_billing_sideband", _record)
    monkeypatch.setitem(
        proxy_app.dependency_overrides,
        user_api_key_auth,
        lambda: UserAPIKeyAuth(token="caller-key-hash", models=["realtime-group"]),
    )
    client: Final = TestClient(proxy_app)
    with (
        patch(  # test-quality-ok: endpoint test isolates request metadata enrichment
            "litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data
        ),
        patch(  # test-quality-ok: endpoint test isolates the process-wide proxy logger
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as logging,
    ):
        logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        logging.post_call_failure_hook = AsyncMock()
        for _ in range(8):
            minted = client.post(
                f"{route}/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "realtime-group"},
            )
            assert minted.status_code == 200, minted.text
            answered = client.post(
                f"{route}/calls",
                headers={"Authorization": f"Bearer {minted.json()['value']}"},
                content=b"v=0\r\n",
            )
            assert answered.status_code == 201, answered.text

    hosts: Final = [call.request.url.host for call in respx_mock.calls]
    minting_hosts: Final = hosts[0::2]
    assert hosts[1::2] == minting_hosts, "the SDP exchange must reach the account that minted the client secret"
    assert metered_on == [f"deployment-{host.split('.')[0].removeprefix('account-')}" for host in minting_hosts]
