"""
Test that /mcp (without trailing slash) is explicitly registered on the parent
FastAPI app so it is handled directly instead of falling through to Starlette's
Mount redirect (307 /mcp -> /mcp/).
"""

import pytest
from starlette.routing import Route


def test_should_have_bare_mcp_route_on_parent_app():
    """The parent proxy app must have an explicit /mcp route so that requests
    to /mcp (no trailing slash) are handled directly rather than redirected."""
    from litellm.proxy.proxy_server import app

    mcp_routes = [
        r
        for r in app.routes
        if isinstance(r, Route) and getattr(r, "path", "") == "/mcp"
    ]
    assert len(mcp_routes) == 1, (
        "Expected exactly one explicit /mcp Route on the parent app; "
        f"found {len(mcp_routes)}"
    )

    route = mcp_routes[0]
    assert "GET" in route.methods
    assert "POST" in route.methods
