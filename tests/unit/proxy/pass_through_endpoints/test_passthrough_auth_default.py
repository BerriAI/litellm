"""
Regression tests for the pass-through endpoint auth-default fix
(GHSA-7h34-mmrh-6g58).

Two failures the fix closes:

1. ``PassThroughGenericEndpoint.auth`` defaulted to ``False`` — an
   admin who added a pass-through to ``general_settings`` without
   explicitly setting ``auth: true`` shipped an unauthenticated
   forwarder.
2. Setting ``auth: true`` was rejected at startup unless the operator
   had a LiteLLM Enterprise license, leaving OSS deployments with no
   safe configuration.

The fix flips the default to ``True`` (safe-by-default) and removes
the enterprise gate so OSS operators can register an authenticated
pass-through. The runtime check in ``user_api_key_auth.py`` also now
defaults to ``True`` so a config dict (raw, not Pydantic) without an
``auth`` key still requires authentication.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from litellm.proxy._types import LiteLLMRoutes, PassThroughGenericEndpoint, ProxyException
from litellm.proxy.auth.user_api_key_auth import pass_through_auth_off, pass_through_key
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _register_pass_through_endpoint,
)
from litellm.proxy.proxy_server import openai_exception_handler


def test_passthrough_auth_defaults_to_true():
    # Regression: an admin who configures a pass-through without setting
    # auth explicitly used to ship an unauthenticated forwarder. The
    # default is now safe.
    endpoint = PassThroughGenericEndpoint(
        path="/canary-forwarder",
        target="https://postman-echo.com/get",
    )
    assert endpoint.auth is True


def test_passthrough_auth_can_still_be_explicitly_disabled():
    # Operators who genuinely need an unauthenticated forwarder (e.g.
    # public webhook receiver) can opt in explicitly.
    endpoint = PassThroughGenericEndpoint(
        path="/public-webhook",
        target="https://example.com/webhook",
        auth=False,
    )
    assert endpoint.auth is False


@pytest.mark.asyncio
async def test_register_passthrough_with_auth_true_works_for_oss(monkeypatch: pytest.MonkeyPatch) -> None:
    app: Final = FastAPI(exception_handlers={ProxyException: openai_exception_handler})
    visited: Final[set[str]] = set()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-passthrough-test")

    endpoint: Final = PassThroughGenericEndpoint(
        path="/forwarder",
        target="https://example.com",
        auth=True,
    )

    await _register_pass_through_endpoint(
        endpoint=endpoint,
        app=app,
        premium_user=False,
        visited_endpoints=visited,
    )
    assert [route.path for route in app.routes if isinstance(route, APIRoute)] == ["/forwarder"]
    with TestClient(app) as client:
        response: Final = client.get(endpoint.path)
    assert response.status_code == 401, response.text
    assert response.json()["error"]["type"] == "auth_error"


@pytest.mark.asyncio
async def test_runtime_check_treats_missing_auth_key_as_authenticated():
    # The runtime dispatch in user_api_key_auth pulls
    # pass_through_endpoints from general_settings as raw dicts (the
    # Pydantic default never applies). A dict without an ``auth`` key
    # must default to "authenticated" — without this, the previous
    # behaviour (``endpoint.get("auth") is not True`` -> True -> empty
    # auth) ships an unauthenticated forwarder.
    request = MagicMock()
    request.headers = {}
    raw_endpoint_no_auth_key = {
        "path": "/forwarder",
        "target": "https://example.com",
        # ``auth`` deliberately omitted
    }

    assert not pass_through_auth_off("/forwarder", [raw_endpoint_no_auth_key])
    assert pass_through_key(request, "/forwarder", [raw_endpoint_no_auth_key], "sk-9876") == "sk-9876"


@pytest.mark.asyncio
async def test_runtime_check_explicit_auth_false_still_skips_validation():
    # Operators who explicitly set ``auth: False`` get the legacy
    # behaviour — an empty UserAPIKeyAuth, no key required.
    request = MagicMock()
    request.headers = {}
    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    assert pass_through_auth_off("/public-webhook", [raw_endpoint_auth_false])


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/eu_jev/v1/systemone", "/eu_jev/v1/systemone/extra", "/eu_jev/v1/systemone2"])
async def test_no_eu_jev_route_reads_the_litellm_user_api_key_header(route: str) -> None:
    request: Final = Request({"type": "http", "headers": [(b"litellm_user_api_key", b"sk-from-header")]})

    assert pass_through_key(request, route, None, "sk-from-bearer") == "sk-from-bearer"


def test_mapped_pass_through_route_authenticates_the_litellm_user_api_key_header() -> None:
    route: Final = f"{LiteLLMRoutes.mapped_pass_through_routes.value[0]}/v1/x"
    request: Final = Request({"type": "http", "headers": [(b"litellm_user_api_key", b"sk-proxy")]})

    assert pass_through_key(request, route, None, "provider-key") == "sk-proxy"


@pytest.mark.asyncio
async def test_passthrough_oauth_client_id_in_query_string_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_urls: list[str] = []

    class EchoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_urls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = HTTPServer(("127.0.0.1", 0), EchoHandler)
    port: Final = server.server_port
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        from litellm.proxy import proxy_server

        monkeypatch.setattr(
            proxy_server,
            "general_settings",
            {
                "allow_client_side_credentials": False,
                "pass_through_endpoints": [
                    {
                        "path": "/MS_Office/upstream/authorize",
                        "target": f"http://127.0.0.1:{port}/",
                        "methods": ["GET"],
                        "auth": False,
                    }
                ],
            },
        )

        app: Final = FastAPI(exception_handlers={ProxyException: openai_exception_handler})
        visited: Final[set[str]] = set()
        endpoint: Final = PassThroughGenericEndpoint(
            path="/MS_Office/upstream/authorize",
            target=f"http://127.0.0.1:{port}/",
            methods=["GET"],
            auth=False,
        )

        await _register_pass_through_endpoint(
            endpoint=endpoint,
            app=app,
            premium_user=False,
            visited_endpoints=visited,
        )

        with TestClient(app) as client:
            response: Final = client.get(
                "/MS_Office/upstream/authorize?response_type=code&client_id=test-client"
            )

        assert response.status_code == 200, response.text
        assert response.json() == {"status": "ok"}
        assert received_urls == ["/?response_type=code&client_id=test-client"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_passthrough_client_id_in_request_body_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_urls: list[str] = []

    class EchoHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received_urls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = HTTPServer(("127.0.0.1", 0), EchoHandler)
    port: Final = server.server_port
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        from litellm.proxy import proxy_server

        monkeypatch.setattr(
            proxy_server,
            "general_settings",
            {
                "allow_client_side_credentials": False,
                "pass_through_endpoints": [
                    {
                        "path": "/MS_Office/upstream/authorize",
                        "target": f"http://127.0.0.1:{port}/",
                        "methods": ["POST"],
                        "auth": False,
                    }
                ],
            },
        )

        app: Final = FastAPI(exception_handlers={ProxyException: openai_exception_handler})
        visited: Final[set[str]] = set()
        endpoint: Final = PassThroughGenericEndpoint(
            path="/MS_Office/upstream/authorize",
            target=f"http://127.0.0.1:{port}/",
            methods=["POST"],
            auth=False,
        )

        await _register_pass_through_endpoint(
            endpoint=endpoint,
            app=app,
            premium_user=False,
            visited_endpoints=visited,
        )

        with TestClient(app) as client:
            response: Final = client.post(
                "/MS_Office/upstream/authorize",
                json={"client_id": "test-client"},
            )

        assert response.status_code == 401, response.text
        assert "client_id is not allowed in request body" in response.text
        assert received_urls == []
    finally:
        server.shutdown()
