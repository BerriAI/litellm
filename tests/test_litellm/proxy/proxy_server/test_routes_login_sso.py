"""Pin tests for proxy_server.py login/SSO routes (PR3).

Routes covered:
- GET /fallback/login
- POST /login
- POST /v2/login
- POST /v3/login
- POST /v3/login/exchange
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

    async def _fake_auth(username, password, master_key, prisma_client, throttle=None):
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
    monkeypatch.setattr("litellm.proxy.auth.login_utils.create_ui_token_object", _fake_token_object)
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
        "content_type_html": response.headers.get("content-type", "").startswith("text/html"),
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
        "content_type_html": response.headers.get("content-type", "").startswith("text/html"),
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
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


def test_login_form_success_redirects_with_token_cookie(client, monkeypatch):
    """Pin: POST /login with valid form returns a 303 redirect to /ui and
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
        "location_has_ui": "/ui" in location,
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
    assert normalize(response.json(), volatile=frozenset({"token", "redirect_url"})) == {
        "redirect_url": "<VOLATILE>",
        "token": "<VOLATILE>",
    }
    body = response.json()
    set_cookie = response.headers.get("set-cookie", "")
    shape = {
        "redirect_url_has_ui": "/ui" in body.get("redirect_url", ""),
        "redirect_url_has_login_success": "login=success" in body.get("redirect_url", ""),
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
    monkeypatch.setattr(ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"})
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
    monkeypatch.setattr(ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"})

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

    monkeypatch.setattr(ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"})

    response = client.post("/v3/login/exchange", json={})
    assert response.status_code == 400
    body = response.json()
    assert isinstance(body, dict)
    assert "Missing" in str(body) or "code" in str(body)


def test_v3_login_exchange_invalid_code_401(client, monkeypatch):
    """Error path: code that isn't in cache -> 401 'Invalid or expired'."""
    from litellm.proxy import proxy_server as ps

    monkeypatch.setattr(ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"})
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

    monkeypatch.setattr(ps, "general_settings", {"control_plane_url": "https://cp.example.invalid"})
    monkeypatch.setattr(ps, "redis_usage_cache", None)

    cached_payload = {
        "token": "jwt-token-xyz",
        "redirect_url": "https://litellm.example.invalid/ui?login=success",
    }
    fake_cache = MagicMock()
    fake_cache.async_get_cache = AsyncMock(return_value=cached_payload)
    fake_cache.async_delete_cache = AsyncMock()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_cache)

    response = client.post("/v3/login/exchange", json={"code": "valid-code"})
    assert response.status_code == 200
    assert normalize(response.json(), volatile=frozenset({"token", "redirect_url"})) == {
        "token": "<VOLATILE>",
        "redirect_url": "<VOLATILE>",
    }
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
        "redirect_url": "https://litellm.example.invalid/ui?login=success",
        "token_cookie_set": True,
        "cache_deleted_once": True,
    }


def test_login_form_honors_same_origin_return_to_cookie(client, monkeypatch):
    """The aggregate DCR connect flow preserves a same-origin return_to in the litellm_cp_return_to
    cookie; /login must RESUME there after password sign-in instead of dead-ending at the dashboard."""
    _install_login_mocks(monkeypatch)
    return_to = "/mcp/authorize?client_id=llm_dcrc_abc&response_type=code"
    response = client.post(
        "/login",
        data={"username": "admin", "password": "password"},
        cookies={"litellm_cp_return_to": return_to},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers.get("location", "") == return_to  # resumed the connect flow, not the dashboard
    assert "token=" in response.headers.get("set-cookie", "")


def test_login_form_honors_control_plane_return_to_cookie(client, monkeypatch):
    """/login resumes through the SAME resumer the SSO callback uses, so it honors BOTH shapes
    _persist_return_to_cookie is willing to store. Honoring only the relative one silently dropped
    a control-plane return_to and landed the user on the dashboard."""
    import litellm.proxy.proxy_server as ps

    _install_login_mocks(monkeypatch)
    monkeypatch.setitem(ps.general_settings, "control_plane_url", "https://cp.example.com")
    response = client.post(
        "/login",
        data={"username": "admin", "password": "password"},
        cookies={"litellm_cp_return_to": "https://cp.example.com/console"},
        follow_redirects=False,
    )
    location = response.headers.get("location", "")
    assert response.status_code == 303
    assert location.startswith("https://cp.example.com/console")
    # Cross-origin arm hands the JWT off via a one-time code rather than a cookie.
    assert "code=" in location and "login=success" in location
    assert "token=" not in response.headers.get("set-cookie", "")


def test_login_form_survives_stale_control_plane_return_to(client, monkeypatch):
    """A stale one-shot cookie must NEVER fail a completed sign-in. The resumer rejects a return_to
    that no longer matches control_plane_url (a config change between the cookie's write and this
    read); the user has already authenticated, so land on the dashboard instead of erroring."""
    import litellm.proxy.proxy_server as ps

    _install_login_mocks(monkeypatch)
    monkeypatch.setitem(ps.general_settings, "control_plane_url", "https://new-cp.example.com")
    response = client.post(
        "/login",
        data={"username": "admin", "password": "password"},
        cookies={"litellm_cp_return_to": "https://old-cp.example.com/console"},
        follow_redirects=False,
    )
    assert response.status_code == 303, "login must not break on a stale return_to cookie"
    location = response.headers.get("location", "")
    assert "old-cp.example.com" not in location
    assert "/ui" in location


def test_login_form_ignores_open_redirect_return_to(client, monkeypatch):
    """A non-same-origin return_to (open-redirect attempt) is rejected — /login falls back to the
    dashboard rather than honoring an absolute/foreign URL."""
    _install_login_mocks(monkeypatch)
    response = client.post(
        "/login",
        data={"username": "admin", "password": "password"},
        cookies={"litellm_cp_return_to": "https://evil.example.com/steal"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers.get("location", "")
    assert "evil.example.com" not in location
    assert "/ui" in location  # dashboard fallback


# ---------------------------------------------------------------------------
# Failed-login accounting across the login routes (LIT-5285)
# ---------------------------------------------------------------------------


def _install_real_auth(monkeypatch, **settings):
    """Run the real authenticate_user so the throttle inside it is exercised.

    prisma_client stays None, so every guess falls through to the credential rejection.
    """
    from litellm.proxy import proxy_server as ps

    monkeypatch.setenv("UI_USERNAME", "admin")
    monkeypatch.setenv("UI_PASSWORD", "right-password")
    monkeypatch.setattr(ps, "master_key", "sk-test-master")
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "premium_user", False)
    monkeypatch.setattr(ps, "general_settings", dict(settings))


def _form_login(client, username="admin", password="wrong"):
    return client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=False
    ).status_code


def _json_login(client, path, username="admin", password="wrong"):
    return client.post(path, json={"username": username, "password": password}).status_code


def test_budget_is_shared_across_every_login_endpoint(client, monkeypatch, reset_login_throttle):
    """The endpoint is not part of the key, so spending the budget on one route blocks the rest.

    Partitioning the counter per endpoint would silently triple the real allowance.
    """
    _install_real_auth(
        monkeypatch,
        max_failed_login_attempts=10,
        control_plane_url="https://cp.example.com",
    )

    assert [_form_login(client) for _ in range(5)] == [401] * 5
    assert [_json_login(client, "/v2/login") for _ in range(5)] == [401] * 5

    assert _json_login(client, "/v3/login") == 429, "the eleventh attempt must be refused on a third route"


def test_budget_is_shared_across_username_casing(client, monkeypatch, reset_login_throttle):
    """The database lookup is case-insensitive, so casing must not partition the counter."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=10)

    assert [_json_login(client, "/v2/login", username="admin@corp.com") for _ in range(5)] == [401] * 5
    assert [_json_login(client, "/v2/login", username="ADMIN@corp.com") for _ in range(5)] == [401] * 5

    assert _json_login(client, "/v2/login", username="Admin@corp.com") == 429


def test_a_refused_attempt_carries_retry_after(client, monkeypatch, reset_login_throttle):
    """The 429 tells the caller how long the window has left."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=2, failed_login_window_seconds=77)

    assert [_json_login(client, "/v2/login") for _ in range(2)] == [401, 401]

    refused = client.post("/v2/login", json={"username": "admin", "password": "wrong"})
    assert refused.status_code == 429
    assert refused.headers.get("retry-after") == "77"


def test_the_form_returns_a_human_readable_lockout_page(client, monkeypatch, reset_login_throttle):
    """The no-JavaScript form must render a wait page when its POST is throttled."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=2, failed_login_window_seconds=77)

    assert [_form_login(client) for _ in range(2)] == [401, 401]

    refused = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert refused.status_code == 429
    assert refused.headers.get("content-type", "").startswith("text/html")
    assert "Try again in about 77 seconds" in refused.text
    assert refused.headers.get("retry-after") == "77"


def test_a_second_username_from_the_same_source_still_gets_through(client, monkeypatch, reset_login_throttle):
    """The username counter carries no address, so one account exhausting it cannot block another."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=2)

    for _ in range(3):
        _json_login(client, "/v2/login", username="admin")

    assert _json_login(client, "/v2/login", username="someone-else@example.com") == 401


def test_a_spray_across_usernames_is_refused_on_the_source_counter(client, monkeypatch, reset_login_throttle):
    """A fresh username per guess keeps every username counter at one, so the address is what stops it."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=100, max_failed_login_attempts_per_source=4)

    sprayed = [_json_login(client, "/v2/login", username=f"sprayed-{i}@corp.com") for i in range(4)]
    assert sprayed == [401] * 4

    assert _json_login(client, "/v2/login", username="sprayed-5@corp.com") == 429


def test_the_configured_admin_password_still_signs_in_while_refused(client, monkeypatch, reset_login_throttle):
    """The operator must never be locked out of the console by traffic aimed at it."""
    from unittest.mock import AsyncMock, patch

    _install_real_auth(monkeypatch, max_failed_login_attempts=2)
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")

    assert [_json_login(client, "/v2/login") for _ in range(2)] == [401, 401]
    assert _json_login(client, "/v2/login") == 429

    with patch("litellm.proxy.auth.login_utils.user_update", new=AsyncMock()), patch(
        "litellm.proxy.auth.login_utils.generate_key_helper_fn", new=AsyncMock(return_value={"token": "sk-ui"})
    ):
        assert _json_login(client, "/v2/login", password="right-password") == 200


def test_sign_in_succeeds_again_once_the_budget_is_restored(client, monkeypatch, reset_login_throttle):
    """A cleared bucket lets the same username straight back in."""
    _install_real_auth(monkeypatch, max_failed_login_attempts=2)

    assert [_json_login(client, "/v2/login") for _ in range(2)] == [401, 401]
    assert _json_login(client, "/v2/login") == 429

    reset_login_throttle()
    assert _json_login(client, "/v2/login") == 401
