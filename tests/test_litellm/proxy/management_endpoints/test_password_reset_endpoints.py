import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app
from litellm.proxy.utils import hash_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """The endpoint rate limits against the module-global `user_api_key_cache`, whose
    in-memory state would otherwise leak between tests reusing the same email/IP."""
    from litellm.proxy.proxy_server import user_api_key_cache

    user_api_key_cache.in_memory_cache.flush_cache()
    yield
    user_api_key_cache.in_memory_cache.flush_cache()


def _mock_user(user_id="user-1", email="alice@example.com", password="scrypt:hash"):
    user = MagicMock()
    user.user_id = user_id
    user.user_email = email
    user.password = password
    return user


def _mock_prisma_for_forgot_password():
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
    return mock_prisma


@pytest.mark.asyncio
async def test_forgot_password_existing_user_sends_email(mocker, monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    mock_prisma = _mock_prisma_for_forgot_password()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert "message" in response.json()
    mock_send_email.assert_called_once()
    assert mock_send_email.call_args.kwargs["receiver_email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_forgot_password_emailed_token_matches_stored_hash(mocker, monkeypatch):
    """Regression: the link in the email must carry the raw token whose hash_token()
    is what got persisted, under the configured PROXY_BASE_URL reset path. Mutating
    either side (emailing the wrong token, storing the raw token) must fail here."""
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    mock_prisma = _mock_prisma_for_forgot_password()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    html = mock_send_email.call_args.kwargs["html"]
    expected_prefix = "https://proxy.example.com/ui/reset-password?token="
    match = re.search(rf"{re.escape(expected_prefix)}([A-Za-z0-9_-]+)", html)
    assert match is not None, f"no reset link starting with {expected_prefix!r} found in {html!r}"
    emailed_token = match.group(1)

    mock_prisma.db.litellm_passwordresettoken.create.assert_awaited_once()
    stored_hash = mock_prisma.db.litellm_passwordresettoken.create.call_args.kwargs["data"]["token_hash"]
    assert stored_hash == hash_token(emailed_token)
    assert stored_hash != emailed_token


@pytest.mark.asyncio
async def test_forgot_password_rate_limits_on_forwarded_ip_behind_trusted_proxy(mocker, monkeypatch):
    """Regression: behind a reverse proxy the direct TCP peer is the LB for every
    request, so the per-IP limit must key off the X-Forwarded-For hop resolved
    against the operator's trusted_proxy_ranges, not request.client.host."""
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    mock_prisma = _mock_prisma_for_forgot_password()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.general_settings", {"trusted_proxy_ranges": ["10.0.0.0/8"]})
    mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )
    mock_cache = mocker.patch("litellm.proxy.proxy_server.user_api_key_cache")
    mock_cache.async_increment_cache = AsyncMock(return_value=1.0)
    lb_client = TestClient(app, client=("10.1.2.3", 44444))

    response = lb_client.post(
        "/user/forgot_password",
        json={"email": "alice@example.com"},
        headers={"x-forwarded-for": "203.0.113.9"},
    )

    assert response.status_code == 200
    rate_limit_keys = [call.kwargs["key"] for call in mock_cache.async_increment_cache.call_args_list]
    assert "password_reset_rl:ip:203.0.113.9" in rate_limit_keys
    assert "password_reset_rl:ip:10.1.2.3" not in rate_limit_keys
    stored = mock_prisma.db.litellm_passwordresettoken.create.call_args.kwargs["data"]
    assert stored["requested_ip"] == "203.0.113.9"


@pytest.mark.asyncio
async def test_forgot_password_ignores_forwarded_ip_without_trusted_proxies(mocker, monkeypatch):
    """With no trusted_proxy_ranges configured, X-Forwarded-For is attacker-controlled
    and must be ignored so the limit cannot be evaded by rotating the header."""
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
    mock_prisma = _mock_prisma_for_forgot_password()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.general_settings", {})
    mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )
    mock_cache = mocker.patch("litellm.proxy.proxy_server.user_api_key_cache")
    mock_cache.async_increment_cache = AsyncMock(return_value=1.0)
    lb_client = TestClient(app, client=("10.1.2.3", 44444))

    response = lb_client.post(
        "/user/forgot_password",
        json={"email": "alice@example.com"},
        headers={"x-forwarded-for": "203.0.113.9"},
    )

    assert response.status_code == 200
    rate_limit_keys = [call.kwargs["key"] for call in mock_cache.async_increment_cache.call_args_list]
    assert "password_reset_rl:ip:10.1.2.3" in rate_limit_keys
    assert "password_reset_rl:ip:203.0.113.9" not in rate_limit_keys


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
    mock_send_email.assert_not_called()


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
    mock_send_email.assert_not_called()


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
    mock_send_email.assert_not_called()
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


@pytest.mark.asyncio
async def test_reset_password_empty_new_password_rejected(mocker):
    """An empty string must never become a working password; the frontend's zod
    .min(1) is bypassable, so the request model has to reject it server-side."""
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

    response = client.post("/user/reset_password", json={"token": "raw-token", "new_password": ""})

    assert response.status_code == 422
    mock_prisma.db.litellm_usertable.update.assert_not_called()
