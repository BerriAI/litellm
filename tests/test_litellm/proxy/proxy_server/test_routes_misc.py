"""Pin tests for proxy_server.py misc routes (PR3).

Routes covered:
- GET /
- GET /routes
- GET /adaptive_router/state
- GET /get_logo_url
- GET /get_image
- GET /get_favicon
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .conftest import normalize


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_home_returns_200_with_body(client, auth_as):
    """GET / serves either the home string or the Swagger UI fallback —
    both return 200 with a non-empty body. This pins the contract: root
    always answers and never errors."""
    with auth_as():
        response = client.get("/")
    shape = {
        "status": response.status_code,
        "has_body": len(response.content) > 0,
        "has_content_type": bool(response.headers.get("content-type")),
    }
    assert shape == {"status": 200, "has_body": True, "has_content_type": True}


def test_home_invalid_method_405(client):
    """GET / handler is GET-only; DELETE returns 405 (error path)."""
    response = client.delete("/")
    assert response.status_code == 405
    assert len(response.content) > 0 and response.headers.get("content-type")


# ---------------------------------------------------------------------------
# GET /routes
# ---------------------------------------------------------------------------


def test_get_routes_returns_routes_list(client, auth_as):
    with auth_as():
        response = client.get("/routes")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert "routes" in body
    assert isinstance(body["routes"], list)
    assert len(body["routes"]) > 0
    sample = body["routes"][0]
    shape = {
        "has_path": "path" in sample,
        "has_methods": "methods" in sample,
        "has_endpoint": "endpoint" in sample,
    }
    assert shape == {
        "has_path": True,
        "has_methods": True,
        "has_endpoint": True,
    }


def test_get_routes_invalid_method_405(client):
    """POST against the GET-only /routes endpoint is rejected (error path)."""
    response = client.post("/routes")
    assert response.status_code == 405
    body = response.json() if response.headers.get("content-type", "").startswith(
        "application/json"
    ) else {}
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# GET /adaptive_router/state
# ---------------------------------------------------------------------------


def test_adaptive_router_state_returns_snapshots(client, auth_as, monkeypatch):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    fake_router = MagicMock()
    snap = {"router_name": "ar-1", "queue_depth": 0, "posteriors": []}
    bandit = MagicMock()
    bandit.get_state_snapshot = AsyncMock(return_value=snap)
    from litellm.types.router import TaggedPreRoutingStrategy

    fake_router.adaptive_routers = {
        "ar-1": [TaggedPreRoutingStrategy(tags=(), strategy=bandit)]
    }
    monkeypatch.setattr(ps, "llm_router", fake_router)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/adaptive_router/state")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "routers": [
            {"router_name": "ar-1", "queue_depth": 0, "posteriors": []},
        ]
    }


def test_adaptive_router_state_not_admin_forbidden(client, auth_as):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER):
        response = client.get("/adaptive_router/state")
    assert response.status_code == 403
    assert "error" in response.json().get("detail", {})


def test_adaptive_router_state_not_configured_404(client, auth_as, monkeypatch):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    fake_router = MagicMock()
    fake_router.adaptive_routers = {}
    monkeypatch.setattr(ps, "llm_router", fake_router)

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        response = client.get("/adaptive_router/state")
    assert response.status_code == 404
    assert "adaptive_router" in response.json().get("detail", {}).get("error", "")


# ---------------------------------------------------------------------------
# GET /get_logo_url
# ---------------------------------------------------------------------------


def test_get_logo_url_returns_http_url_when_set(client, monkeypatch):
    monkeypatch.setenv("UI_LOGO_PATH", "https://example.invalid/logo.png")
    response = client.get("/get_logo_url")
    assert response.status_code == 200
    assert normalize(response.json()) == {"logo_url": "https://example.invalid/logo.png"}


def test_get_logo_url_blank_when_local_path(client, monkeypatch):
    """Local filesystem paths must NOT be disclosed via this endpoint."""
    monkeypatch.setenv("UI_LOGO_PATH", "/var/lib/litellm/internal-secret-logo.png")
    response = client.get("/get_logo_url")
    assert response.status_code == 200
    assert normalize(response.json()) == {"logo_url": ""}


def test_get_logo_url_blank_when_unset(client, monkeypatch):
    monkeypatch.delenv("UI_LOGO_PATH", raising=False)
    response = client.get("/get_logo_url")
    assert response.status_code == 200
    assert normalize(response.json()) == {"logo_url": ""}


def test_get_logo_url_invalid_scheme_blank(client, monkeypatch):
    """file:// and other non-HTTP schemes are not disclosed (error/edge path)."""
    monkeypatch.setenv("UI_LOGO_PATH", "file:///etc/passwd")
    response = client.get("/get_logo_url")
    assert response.status_code == 200
    assert normalize(response.json()) == {"logo_url": ""}


# ---------------------------------------------------------------------------
# GET /get_image
# ---------------------------------------------------------------------------


def test_get_image_returns_default_logo(client, monkeypatch):
    monkeypatch.delenv("UI_LOGO_PATH", raising=False)
    response = client.get("/get_image")
    assert response.status_code == 200
    media_type = response.headers.get("content-type", "").split(";")[0]
    shape = {
        "status": response.status_code,
        "media_type_image": media_type.startswith("image/"),
        "has_body": len(response.content) > 0,
    }
    assert shape == {"status": 200, "media_type_image": True, "has_body": True}


def test_get_image_redirects_remote_url(client, monkeypatch):
    """Remote logo URLs are served via redirect — the proxy never fetches them server-side."""
    monkeypatch.setenv("UI_LOGO_PATH", "https://example.invalid/logo.png")
    response = client.get("/get_image", follow_redirects=False)
    assert response.status_code in (302, 303, 307, 308)
    assert response.headers.get("location") == "https://example.invalid/logo.png"


def test_get_image_invalid_local_path_falls_back(client, monkeypatch):
    """Non-existent UI_LOGO_PATH (error path) falls back to default logo, still 200."""
    monkeypatch.setenv("UI_LOGO_PATH", "/nonexistent/path/to/logo.png")
    response = client.get("/get_image")
    assert response.status_code == 200
    shape = {
        "status": response.status_code,
        "media_type_image": response.headers.get("content-type", "").startswith(
            "image/"
        ),
        "has_body": len(response.content) > 0,
    }
    assert shape == {"status": 200, "media_type_image": True, "has_body": True}


# ---------------------------------------------------------------------------
# GET /get_favicon
# ---------------------------------------------------------------------------


def test_get_favicon_returns_file(client):
    response = client.get("/get_favicon")
    assert response.status_code == 200
    shape = {
        "status": response.status_code,
        "has_body": len(response.content) > 0,
        "content_type_set": bool(response.headers.get("content-type")),
    }
    assert shape == {"status": 200, "has_body": True, "content_type_set": True}


def test_get_favicon_invalid_custom_path_falls_back(client, monkeypatch):
    """Bad UI_FAVICON_PATH (error/edge path) falls back to default — still 200."""
    monkeypatch.setenv("UI_FAVICON_PATH", "/nonexistent/favicon.ico")
    response = client.get("/get_favicon")
    assert response.status_code == 200
    assert len(response.content) > 0
