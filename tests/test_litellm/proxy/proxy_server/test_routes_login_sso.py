"""Pin tests for proxy_server.py login/SSO routes (PR3).

Routes covered:
- GET /fallback/login
- POST /login
- POST /v2/login
- POST /v3/login
- POST /v3/login/exchange
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .conftest import normalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_login_mocks(monkeypatch, raise_on_auth: bool = False) -> None:
    """Patch authenticate_user + create_ui_token_object at their import paths.

    Both /login, /v2/login and /v3/login do a *local* (in-function) import of
    these helpers, so we patch the module they live in.
    """
    from litellm.proxy import proxy_server as ps

    async def _fake_auth(username, password, master_key, prisma_client, auth_method=None):
        if raise_on_auth:
            raise Exception("boom-auth-failure")
        fake = MagicMock()
        fake.user_id = "u-1"
        fake.user_email = "test@example.invalid"
        fake.user_role = "proxy_admin"
        fake.key = "sk-fake-ui-key"
        return fake

    def _fake_token_object(login_result, general_settings, premium_user):
        return {
            "user_id": "u-1",
            "user_email": "test@example.invalid",
            "user_role": "proxy_admin",
            "premium_user": premium_user,
            "key": "sk-fake-ui-key",
        }

    monkeypatch.setattr("litellm.proxy.auth.login_utils.authenticate_user", _fake_auth)
    monkeypatch.setattr(
        "litellm.proxy.auth.login_utils.create_ui_token_object", _fake_token_object
    )
    monkeypatch.setattr(ps, "master_key", "sk-test-master")
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "premium_user", False)


# ---------------------------------------------------------------------------
# GET /fallback/login
# ---------------------------------------------------------------------------


def test_fallback_login_returns_html_form(client, monkeypatch):
    """Pin: GET /fallback/login returns an HTML login form with status 200."""
    monkeypatch.delenv("UI_USERNAME", raising=False)
    response = client.get("/fallback/login")
    body_lower = response.text.lower()
    shape = {
        "status": response.status_code,
        "content_type_html": response.headers.get("content-type", "").startswith(
            "text/html"
        ),
        "has_form": "<form" in body_lower or "username" in body_lower,
    }
    assert shape == {
        "status": 200,
        "content_type_html": True,
        "has_form": True,
    }


def test_fallback_login_returns_html_form_with_ui_username_set(client, monkeypatch):
    """Both branches (UI_USERNAME set or not) return the same HTML form."""
    monkeypatch.setenv("UI_USERNAME", "admin")
    response = client.get("/fallback/login")
    body_lower = response.text.lower()
    shape = {
        "status": response.status_code,
        "content_type_html": response.headers.get("content-type", "").startswith(
            "text/html"
        ),
        "has_form_or_username": "<form" in body_lower or "username" in body_lower,
    }
    assert shape == {
        "status": 200,
        "content_type_html": True,
        "has_form_or_username": True,
    }


def test_fallback_login_shows_credentials_hint_by_default(client, monkeypatch):
    """Control: without the flag, /fallback/login still renders the hint."""
    monkeypatch.delenv("UI_USERNAME", raising=False)
    monkeypatch.delenv("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", raising=False)
    response = client.get("/fallback/login")
    assert response.status_code == 200
    assert "Default Credentials" in response.text
    assert "MASTER_KEY" in response.text


def test_fallback_login_hides_credentials_hint_via_env_flag(client, monkeypatch):
    """Pin: LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT removes the hint on /fallback/login."""
    monkeypatch.delenv("UI_USERNAME", raising=False)
    monkeypatch.setenv("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", "true")
    response = client.get("/fallback/login")
    assert response.status_code == 200
    assert "Default Credentials" not in response.text
    assert "MASTER_KEY" not in response.text
    # the login form itself must still render
    assert "username" in response.text.lower()


def test_fallback_login_invalid_method_405(client):
    """POST against the GET-only /fallback/login is rejected (error path)."""
    response = client.post("/fallback/login")
    assert response.status_code == 405
    body = (
        response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else {}
    )
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


def test_login_form_success_redirects_with_token_cookie(client, monkeypatch):
    """Pin: POST /login with valid form returns a 303 redirect to /ui/ and
    sets the 'token' cookie."""
    _install_login_mocks(monkeypatch)
    response = client.post(
        "/login",
        data={"username": "admin", "password": "password"},
        follow_redirects=False,
    )
    location = response.headers.get("location", "")
    set_cookie = response.headers.get("set-cookie", "")
    shape = {
        "status": response.status_code,
        "location_has_ui": "/ui/" in location,
        "location_has_login_success": "login=success" in location,
        "has_token_cookie": "token=" in set_cookie,
    }
    assert shape == {
        "status": 303,
        "location_has_ui": True,
        "location_has_login_success": True,
        "has_token_cookie": True,
    }


def test_login_form_authenticate_raises_500(client, monkeypatch):
    """Error path: authenticate_user raising causes a 500 (handler has no try/except)."""
    _install_login_mocks(monkeypatch, raise_on_auth=True)
    response = client.post(
        "/login",
        data={"username": "admin", "password": "wrong"},
        follow_redirects=False,
    )
    # raise_server_exceptions=False -> TestClient returns 500 with body
    assert response.status_code == 500
    # Body must be non-empty so a future refactor that drops the error body
    # would trip this gate.
    assert len(response.content) > 0
    assert response.headers.get("content-type") is not None


# ---------------------------------------------------------------------------
# POST /v2/login
# ---------------------------------------------------------------------------


def test_v2_login_success_returns_token_and_redirect(client, monkeypatch):
    """Pin: POST /v2/login returns JSON {redirect_url, token} + sets token cookie."""
    _install_login_mocks(monkeypatch)
    response = client.post(
        "/v2/login",
        json={"username": "admin", "password": "password"},
    )
    assert response.status_code == 200
    assert normalize(
        response.json(), volatile=frozenset({"token", "redirect_url"})
    ) == {"redirect_url": "<VOLATILE>", "token": "<VOLATILE>"}
    body = response.json()
    set_cookie = response.headers.get("set-cookie", "")
    shape = {
        "redirect_url_has_ui": "/ui/" in body.get("redirect_url", ""),
        "redirect_url_has_login_success": "login=success"
        in body.get("redirect_url", ""),
        "token_in_body": bool(body.get("token")),
        "token_cookie_set": "token=" in set_cookie,
    }
    assert shape == {
        "redirect_url_has_ui": True,
        "redirect_url_has_login_success": True,
        "token_in_body": True,
        "token_cookie_set": True,
    }


def test_v2_login_authenticate_failure_500(client, monkeypatch):
    """Error path: authenticate_user raising -> ProxyException -> 500 with structured error."""
    _install_login_mocks(monkeypatch, raise_on_auth=True)
    response = client.post(
        "/v2/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert response.status_code == 500
    body = response.json()
    # Non-status assertion: response shape should carry an error
    assert "error" in body or "detail" in body
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# POST /v3/login
# ---------------------------------------------------------------------------


def test_v3_login_without_control_plane_url_404(client, monkeypatch):
    """Pin: /v3/login is gated on general_settings['control_plane_url'] — 404 when absent."""
    _install_login_mocks(monkeypatch)
    # _install_login_mocks sets general_settings to {} — re-affirm
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "general_settings", {})

    response = client.post(
        "/v3/login",
        json={"username": "admin", "password": "password"},
    )
    assert response.status_code == 404
    body = response.json()
    # Detail carries the structured ProxyException error
    detail = body.get("detail", {})
    if isinstance(detail, dict):
        message = detail.get("error", {})
        if isinstance(message, dict):
            message_str = message.get("message", "")
        else:
            message_str = str(message)
    else:
        message_str = str(detail)
    assert "control_plane_url" in str(body)


def test_v3_login_success_returns_code(client, monkeypatch):
    """Pin: /v3/login with control_plane_url returns {code, expires_in}."""
    from litellm.proxy import proxy_server as ps

    _install_login_mocks(monkeypatch)
    monkeypatch.setattr(
        ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"}
    )
    # Force the local (non-redis) cache path
    monkeypatch.setattr(ps, "redis_usage_cache", None)
    fake_cache = MagicMock()
    fake_cache.async_set_cache = AsyncMock()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_cache)

    response = client.post(
        "/v3/login",
        json={"username": "admin", "password": "password"},
    )
    assert response.status_code == 200
    body = response.json()
    # Strong assertion via normalize with extended volatile set ("code" is volatile)
    assert normalize(body, volatile=frozenset({"code", "expires_in"})) == {
        "code": "<VOLATILE>",
        "expires_in": "<VOLATILE>",
    }
    shape = {
        "has_code": isinstance(body.get("code"), str) and len(body["code"]) > 0,
        "expires_in_60": body.get("expires_in") == 60,
        "cache_set_called": fake_cache.async_set_cache.await_count == 1,
    }
    assert shape == {
        "has_code": True,
        "expires_in_60": True,
        "cache_set_called": True,
    }


def test_v3_login_authenticate_failure_500(client, monkeypatch):
    """Error path: with control_plane_url set, authenticate_user raises -> 500."""
    from litellm.proxy import proxy_server as ps

    _install_login_mocks(monkeypatch, raise_on_auth=True)
    monkeypatch.setattr(
        ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"}
    )

    response = client.post(
        "/v3/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert response.status_code == 500
    body = response.json()
    assert isinstance(body, dict)
    assert "error" in body or "detail" in body


# ---------------------------------------------------------------------------
# POST /v3/login/exchange
# ---------------------------------------------------------------------------


def test_v3_login_exchange_without_control_plane_url_404(client, monkeypatch):
    """Pin: /v3/login/exchange gated on control_plane_url — 404 when absent."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "general_settings", {})

    response = client.post("/v3/login/exchange", json={"code": "abc"})
    assert response.status_code == 404
    body = response.json()
    assert "control_plane_url" in str(body)
    assert isinstance(body, dict)


def test_v3_login_exchange_missing_code_400(client, monkeypatch):
    """Error path: missing 'code' in body -> 400 with 'Missing' message."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(
        ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"}
    )

    response = client.post("/v3/login/exchange", json={})
    assert response.status_code == 400
    body = response.json()
    assert isinstance(body, dict)
    assert "Missing" in str(body) or "code" in str(body)


def test_v3_login_exchange_invalid_code_401(client, monkeypatch):
    """Error path: code that isn't in cache -> 401 'Invalid or expired'."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(
        ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"}
    )
    monkeypatch.setattr(ps, "redis_usage_cache", None)
    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=None)
    fake_cache.async_delete_cache = AsyncMock()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_cache)

    response = client.post("/v3/login/exchange", json={"code": "nope"})
    assert response.status_code == 401
    body = response.json()
    assert isinstance(body, dict)
    assert "Invalid" in str(body) or "expired" in str(body)


def test_v3_login_exchange_success_returns_token_and_redirect(client, monkeypatch):
    """Pin: valid code -> JSON {token, redirect_url} + token cookie + cache deleted (single-use)."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(
        ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"}
    )
    monkeypatch.setattr(ps, "redis_usage_cache", None)

    cached_payload = {
        "token": "jwt-token-xyz",
        "redirect_url": "https://litellm.example.invalid/ui/?login=success",
    }
    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=cached_payload)
    fake_cache.async_delete_cache = AsyncMock()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_cache)

    response = client.post("/v3/login/exchange", json={"code": "valid-code"})
    assert response.status_code == 200
    assert normalize(
        response.json(), volatile=frozenset({"token", "redirect_url"})
    ) == {"token": "<VOLATILE>", "redirect_url": "<VOLATILE>"}
    body = response.json()
    set_cookie = response.headers.get("set-cookie", "")
    shape = {
        "token": body.get("token"),
        "redirect_url": body.get("redirect_url"),
        "token_cookie_set": "token=" in set_cookie,
        "cache_deleted_once": fake_cache.async_delete_cache.await_count == 1,
    }
    assert shape == {
        "token": "jwt-token-xyz",
        "redirect_url": "https://litellm.example.invalid/ui/?login=success",
        "token_cookie_set": True,
        "cache_deleted_once": True,
    }
