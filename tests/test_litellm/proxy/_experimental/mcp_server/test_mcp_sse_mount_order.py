"""
Tests for MCP sub-app mount order (Fixes #22074).

The SSE transport endpoint must be mounted before the "/" catch-all in the
Starlette/FastAPI sub-app, otherwise Starlette's sequential route matching
causes "/" to shadow "/sse", making SSE transport unreachable.

Additionally, the extraneous "/mcp" and "/{mcp_server_name}/mcp" mounts
(which were either unreachable or unsupported) should not be present.
"""

import pytest
from starlette.routing import Mount


class TestMCPSubAppMountOrder:
    """Verify mount order and composition of the MCP sub-application."""

    def _get_mcp_app(self):
        """Import and return the MCP sub-app, skipping if unavailable."""
        try:
            from litellm.proxy._experimental.mcp_server.server import app

            return app
        except ImportError:
            pytest.skip("MCP server not available")

    def _get_mount_paths(self, app) -> list:
        """Return an ordered list of mount paths from the app's routes."""
        return [
            route.path for route in app.routes if isinstance(route, Mount)
        ]

    def test_sse_mount_exists(self):
        """The /sse mount must be present in the MCP sub-app."""
        app = self._get_mcp_app()
        mount_paths = self._get_mount_paths(app)
        assert "/sse" in mount_paths, (
            f"/sse mount not found in MCP sub-app routes: {mount_paths}"
        )

    def test_catch_all_mount_exists(self):
        """The catch-all '/' mount (stored as '') must be present."""
        app = self._get_mcp_app()
        mount_paths = self._get_mount_paths(app)
        # Starlette normalizes "/" to "" in the Mount.path attribute
        assert "" in mount_paths, (
            f"Catch-all '/' mount not found in MCP sub-app routes: {mount_paths}"
        )

    def test_sse_mounted_before_catch_all(self):
        """
        /sse must appear before the catch-all '/' mount.

        Starlette evaluates mounts sequentially. If '/' is listed first it
        matches every request path, making /sse unreachable.
        """
        app = self._get_mcp_app()
        mount_paths = self._get_mount_paths(app)

        sse_index = mount_paths.index("/sse")
        # Starlette normalizes "/" to ""
        catch_all_index = mount_paths.index("")

        assert sse_index < catch_all_index, (
            f"/sse (index {sse_index}) must be mounted before the catch-all "
            f"'/' (index {catch_all_index}) to remain reachable. "
            f"Current mount order: {mount_paths}"
        )

    def test_no_extraneous_mcp_mounts(self):
        """
        The sub-app should NOT contain '/mcp' or '/{mcp_server_name}/mcp'
        mounts. '/mcp' would map to the external path '/mcp/mcp' (incorrect
        double-prefix) and path parameters are not supported in Starlette
        mounts.
        """
        app = self._get_mcp_app()
        mount_paths = self._get_mount_paths(app)

        assert "/mcp" not in mount_paths, (
            f"Extraneous '/mcp' mount found in MCP sub-app: {mount_paths}"
        )
        for path in mount_paths:
            assert "{" not in path, (
                f"Path-parameter mount '{path}' found in MCP sub-app. "
                "Starlette mounts do not support path parameters."
            )

    def test_exactly_two_mounts(self):
        """
        The sub-app should have exactly two mounts: /sse and the catch-all /.
        """
        app = self._get_mcp_app()
        mount_paths = self._get_mount_paths(app)
        assert len(mount_paths) == 2, (
            f"Expected exactly 2 mounts (/sse and /), got {len(mount_paths)}: "
            f"{mount_paths}"
        )
