from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app
from litellm.proxy.utils import hash_token

client = TestClient(app)


def _mock_user(user_id="user-1", email="alice@example.com", password="scrypt:hash"):
    user = MagicMock()
    user.user_id = user_id
    user.user_email = email
    user.password = password
    return user


@pytest.mark.asyncio
async def test_forgot_password_existing_user_sends_email(mocker, monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=_mock_user())
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=0)
    mock_prisma.db.litellm_passwordresettoken.create = AsyncMock(
        return_value=MagicMock(
            dict=lambda: {
                "token_hash": "h",
                "user_id": "user-1",
                "requested_ip": "testclient",
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc),
                "used_at": None,
            }
        )
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert "message" in response.json()
    mock_send_email.assert_awaited_once()
    assert mock_send_email.call_args.kwargs["receiver_email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_forgot_password_unknown_email_same_response_no_email_sent(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "unknown@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "If an account exists for this email, a password reset link has been sent."
    }
    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_forgot_password_sso_only_user_same_response_no_email_sent(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(
        return_value=_mock_user(password=None)
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "If an account exists for this email, a password reset link has been sent."
    }
    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_forgot_password_rate_limited_by_email(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=_mock_user())
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )
    mock_cache = mocker.patch("litellm.proxy.proxy_server.user_api_key_cache")
    mock_cache.async_increment_cache = AsyncMock(side_effect=[4.0, 1.0])

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 429


@pytest.mark.asyncio
async def test_forgot_password_no_proxy_base_url_same_response_no_email_sent(mocker, monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=_mock_user())
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=0)
    mock_prisma.db.litellm_passwordresettoken.create = AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "If an account exists for this email, a password reset link has been sent."
    }
    mock_send_email.assert_not_awaited()
    mock_prisma.db.litellm_passwordresettoken.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_reset_token_valid(mocker):
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=_mock_user())
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.get("/user/reset_password/validate", params={"token": "raw-token"})

    assert response.status_code == 200
    assert response.json() == {"user_email": "alice@example.com"}


@pytest.mark.asyncio
async def test_validate_reset_token_invalid_returns_generic_400(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.get("/user/reset_password/validate", params={"token": "bogus"})

    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "This link is invalid or has expired."}}


def _install_tx_context(mock_prisma):
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=mock_prisma.db)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_prisma.db.tx = MagicMock(return_value=tx_cm)


@pytest.mark.asyncio
async def test_reset_password_happy_path(mocker):
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=1)
    mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=_mock_user())
    _install_tx_context(mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "raw-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 200
    mock_prisma.db.litellm_usertable.update.assert_awaited_once()
    _, update_kwargs = mock_prisma.db.litellm_usertable.update.call_args
    assert update_kwargs["data"]["password"] != "correct horse battery staple"


@pytest.mark.asyncio
async def test_reset_password_second_claim_fails(mocker):
    """A token already marked used_at fails the atomic update_many (updated_count == 0)."""
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=0)
    _install_tx_context(mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "raw-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 400
    mock_prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_reset_password_expired_token_rejected(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "expired-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 400
