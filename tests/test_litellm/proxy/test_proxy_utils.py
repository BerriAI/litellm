import datetime as real_datetime
import json
import os
import sys

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.utils import get_custom_url, join_paths


def test_get_custom_url(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    custom_url = get_custom_url(request_base_url="http://0.0.0.0:4000", route="ui/")
    assert custom_url == "http://0.0.0.0:4000/litellm/ui/"


def test_proxy_only_error_true_for_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._is_proxy_only_llm_api_error(
        original_exception=Exception(),
        error_type=ProxyErrorTypes.auth_error,
        route="/v1/chat/completions",
    )


def test_proxy_only_error_true_for_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )
        is True
    )


def test_proxy_only_error_false_for_non_llm_non_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/generate",
        )
        is False
    )


def test_proxy_only_error_false_for_other_error_type():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=None,
            route="/v1/chat/completions",
        )
        is False
    )


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(
        base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path"
    )
    assert result == "http://0.0.0.0:4000/my-custom-path"


def test_join_paths_normal_join():
    """Test normal path joining"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="/api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_with_trailing_slash():
    """Test path joining with trailing slash on base_path"""
    result = join_paths(base_path="http://0.0.0.0:4000/", route="api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_empty_base():
    """Test path joining with empty base_path"""
    result = join_paths(base_path="", route="api/v1")
    assert result == "/api/v1"


def test_join_paths_empty_route():
    """Test path joining with empty route"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="")
    assert result == "http://0.0.0.0:4000"


def test_join_paths_both_empty():
    """Test path joining with both empty"""
    result = join_paths(base_path="", route="")
    assert result == "/"


def test_join_paths_nested_path():
    """Test path joining with nested paths"""
    result = join_paths(base_path="http://0.0.0.0:4000/v1", route="chat/completions")
    assert result == "http://0.0.0.0:4000/v1/chat/completions"


def _make_redirect_app():
    """Build a minimal Starlette app with the rewrite_redirect_location
    middleware and a route that issues a redirect.  This avoids loading the
    full proxy app (which may trigger DB/Redis connections) and removes the
    dependency on the Next.js UI build artefacts."""
    from urllib.parse import urlparse, urlunparse

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import RedirectResponse, Response
    from starlette.routing import Route

    async def _redirect_handler(request: Request) -> Response:
        """Simulate Starlette's redirect_slashes: redirect using the
        incoming Host header (which, behind a proxy, is the internal pod IP)."""
        host = request.headers.get("host", "testserver")
        return RedirectResponse(url=f"http://{host}/ui/")

    async def _ok_handler(request: Request) -> Response:
        return Response("ok")

    async def rewrite_redirect_location(request: Request, call_next):
        """Mirror of the middleware in proxy_server.py."""
        response = await call_next(request)
        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("location", "")
            fwd_host = request.headers.get("x-forwarded-host", "")
            fwd_proto = request.headers.get("x-forwarded-proto", "https")
            if fwd_host and location:
                parsed = urlparse(location)
                request_host = request.headers.get("host", "")
                if parsed.netloc and parsed.netloc == request_host:
                    new = parsed._replace(scheme=fwd_proto, netloc=fwd_host)
                    response.headers["location"] = urlunparse(new)
        return response

    app = Starlette(
        routes=[
            Route("/ui", _redirect_handler),
            Route("/ui/", _ok_handler),
        ],
    )
    app.middleware("http")(rewrite_redirect_location)
    return app


def test_rewrite_redirect_location_with_forwarded_host():
    """Test that redirect Location headers are rewritten using X-Forwarded-Host
    when the Location points back to the same host as the request."""
    from starlette.testclient import TestClient

    client = TestClient(_make_redirect_app())
    response = client.get(
        "/ui",
        headers={
            "x-forwarded-host": "external.company.com",
            "x-forwarded-proto": "https",
        },
        follow_redirects=False,
    )
    assert response.status_code == 307
    location = response.headers.get("location", "")
    assert "external.company.com" in location
    assert location.startswith("https://")


def test_rewrite_redirect_location_no_forwarded_host():
    """Test that redirect Location headers are NOT rewritten without X-Forwarded-Host."""
    from starlette.testclient import TestClient

    client = TestClient(_make_redirect_app())
    response = client.get("/ui", follow_redirects=False)
    assert response.status_code == 307
    location = response.headers.get("location", "")
    assert "external.company.com" not in location
    # Should still point to the original testserver host
    assert "testserver" in location


def test_rewrite_redirect_location_ignores_foreign_netloc():
    """Middleware must NOT rewrite Location headers that point to a different
    host than the incoming request, preventing open-redirect attacks."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from urllib.parse import urlparse, urlunparse

    async def _foreign_redirect(request: Request) -> Response:
        """Return a redirect whose Location points to a third-party domain."""
        return Response(
            status_code=307,
            headers={"location": "http://other-service.internal/callback"},
        )

    async def rewrite_redirect_location(request: Request, call_next):
        response = await call_next(request)
        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("location", "")
            fwd_host = request.headers.get("x-forwarded-host", "")
            fwd_proto = request.headers.get("x-forwarded-proto", "https")
            if fwd_host and location:
                parsed = urlparse(location)
                request_host = request.headers.get("host", "")
                if parsed.netloc and parsed.netloc == request_host:
                    new = parsed._replace(scheme=fwd_proto, netloc=fwd_host)
                    response.headers["location"] = urlunparse(new)
        return response

    app = Starlette(routes=[Route("/redir", _foreign_redirect)])
    app.middleware("http")(rewrite_redirect_location)

    client = TestClient(app)
    response = client.get(
        "/redir",
        headers={
            "x-forwarded-host": "evil.com",
            "x-forwarded-proto": "https",
        },
        follow_redirects=False,
    )
    assert response.status_code == 307
    location = response.headers.get("location", "")
    # Must NOT be rewritten to evil.com — the original Location stays intact
    assert "evil.com" not in location
    assert "other-service.internal" in location


def _patch_today(monkeypatch, year, month, day):
    class PatchedDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date(year, month, day)

    monkeypatch.setattr("litellm.proxy.utils.date", PatchedDate)


def test_get_projected_spend_over_limit_day_one(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = _get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)
