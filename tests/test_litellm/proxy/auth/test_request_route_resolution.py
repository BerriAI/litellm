"""
Regression tests: a malformed ``Host`` header must not influence the
route the auth gate sees.

``request.url.path`` in Starlette is constructed by interpolating the
``Host`` header into a URL string and re-parsing with ``urlsplit``, so a
Host containing ``/?`` or ``/#`` collapses ``url.path`` to ``"/"`` (the
real path falls into the query/fragment). ``"/"`` is in
``LiteLLMRoutes.public_routes``, so route-based auth gates would treat
protected routes as public. ``get_request_route`` reads
``scope["path"]`` instead — the authoritative ASGI path FastAPI uses
for routing.
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy._types import LiteLLMRoutes  # noqa: E402
from litellm.proxy.auth.auth_utils import get_request_route  # noqa: E402


def _make_request(path: str, host_header: bytes, root_path: bytes = b"") -> Request:
    """Build a minimal Starlette request with the chosen Host header."""
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "server": ("localhost", 4000),
        "client": ("127.0.0.1", 12345),
        "root_path": root_path.decode(),
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", host_header)],
    }
    return Request(scope=scope)


# All four payload variants from the advisory + extra shapes I confirmed
# in variant analysis. ``user@`` and ``[::1]`` both collapse url.path to
# "/" via the same urlsplit reparse mechanism.
_BYPASS_HOST_HEADERS = [
    b"localhost/?x=1",
    b"localhost:4000/?x=1",
    b"localhost/#test",
    b"localhost:4000/#test",
    b"user@localhost/?x=1",
    b"[::1]/?x=1",
    b"localhost\\/?x=1",
]


@pytest.mark.parametrize("host_header", _BYPASS_HOST_HEADERS)
def test_get_request_route_ignores_host_header(host_header):
    protected = "/get/internal_user_settings"
    req = _make_request(protected, host_header)
    assert get_request_route(req) == protected, (
        f"Host header {host_header!r} corrupted the auth-time path: "
        f"got {get_request_route(req)!r}, expected {protected!r}."
    )


def test_get_request_route_strips_root_path():
    # Operators run the proxy mounted under a root_path (e.g. /genai). The
    # auth gate has always compared against unprefixed paths; preserve that.
    req = _make_request("/genai/chat/completions", b"localhost", root_path=b"/genai")
    assert get_request_route(req) == "/chat/completions"


def test_get_request_route_handles_root_path_with_bad_host():
    # Combination: mounted root_path AND malicious Host. The fix must
    # still resolve the unprefixed route correctly.
    req = _make_request(
        "/genai/get/internal_user_settings",
        b"localhost/?x=1",
        root_path=b"/genai",
    )
    assert get_request_route(req) == "/get/internal_user_settings"


def test_slash_is_still_a_public_route():
    # Sanity: ``/`` IS in public_routes — proving the bypass shape
    # (corrupted path = "/") would have skipped auth before this fix.
    assert "/" in LiteLLMRoutes.public_routes.value


@pytest.fixture(scope="module")
def proxy_client():
    """One TestClient per module — TestClient construction triggers
    FastAPI route-tree build + lifespan startup, both expensive. Env
    vars are restored after the fixture exits so they don't bleed into
    other test modules in the same worker."""
    overrides = {
        "DATABASE_URL": "",
        "DISABLE_SCHEMA_UPDATE": "True",
        "LITELLM_MASTER_KEY": "sk-1234",
    }
    saved = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        os.environ[k] = v
    try:
        import litellm.proxy.proxy_server as ps

        with patch.object(ps, "master_key", "sk-1234"):
            yield TestClient(ps.app)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.parametrize("host_header", _BYPASS_HOST_HEADERS)
def test_e2e_protected_admin_route_remains_401(proxy_client, host_header):
    """End-to-end: full FastAPI app sees the malicious Host but the
    auth gate still refuses the request because get_request_route now
    returns the real path."""
    r = proxy_client.get(
        "/get/internal_user_settings",
        headers={"Host": host_header.decode("latin-1")},
    )
    assert r.status_code == 401, (
        f"Malformed Host={host_header!r} bypassed auth: returned "
        f"{r.status_code} (body: {r.text[:200]})"
    )
