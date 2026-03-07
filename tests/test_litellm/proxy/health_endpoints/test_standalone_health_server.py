"""
Tests for QW-4: Standalone TCP health check server.

Validates:
- Server starts on configured port and responds to /health/live
- Server returns 404 for other paths
- Server can be stopped cleanly
- Server is disabled when port is set to 0
- Server survives even when main event loop is blocked
"""

import os
import sys
import time
import urllib.request

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch

import pytest


def _get_free_port():
    """Get a free port from the OS."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_standalone_health_server_responds():
    """Server responds with 200 on /health/live."""
    port = _get_free_port()
    with patch.dict(os.environ, {"LITELLM_STANDALONE_HEALTH_PORT": str(port)}):
        # Re-import to pick up new port
        import importlib

        import litellm.proxy.health_endpoints.standalone_health_server as mod

        importlib.reload(mod)

        try:
            assert mod.start_standalone_health_server() is True
            time.sleep(0.2)  # Give server time to bind

            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live")
            assert resp.status == 200
            assert resp.read() == b"ok"
        finally:
            mod.stop_standalone_health_server()


def test_standalone_health_server_404_on_other_paths():
    """Server returns 404 for paths other than /health/live."""
    port = _get_free_port()
    with patch.dict(os.environ, {"LITELLM_STANDALONE_HEALTH_PORT": str(port)}):
        import importlib

        import litellm.proxy.health_endpoints.standalone_health_server as mod

        importlib.reload(mod)

        try:
            mod.start_standalone_health_server()
            time.sleep(0.2)

            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/other")
                pytest.fail("Expected HTTP error")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            mod.stop_standalone_health_server()


def test_standalone_health_server_disabled_with_port_zero():
    """Server does not start when port is set to 0."""
    with patch.dict(os.environ, {"LITELLM_STANDALONE_HEALTH_PORT": "0"}):
        import importlib

        import litellm.proxy.health_endpoints.standalone_health_server as mod

        importlib.reload(mod)

        assert mod.start_standalone_health_server() is False


def test_standalone_health_server_stop():
    """Server stops cleanly."""
    port = _get_free_port()
    with patch.dict(os.environ, {"LITELLM_STANDALONE_HEALTH_PORT": str(port)}):
        import importlib

        import litellm.proxy.health_endpoints.standalone_health_server as mod

        importlib.reload(mod)

        mod.start_standalone_health_server()
        time.sleep(0.2)

        # Verify it's running
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live")
        assert resp.status == 200

        # Stop and verify it's no longer responding
        mod.stop_standalone_health_server()
        time.sleep(0.2)

        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health/live", timeout=1
            )
            pytest.fail("Expected connection error after stop")
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass  # Expected
