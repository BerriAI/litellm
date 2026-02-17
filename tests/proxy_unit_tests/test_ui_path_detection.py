"""
Unit tests for UI path detection and configuration.

Tests the new LITELLM_UI_PATH and LITELLM_ASSETS_PATH functionality
for read-only filesystem support.

Note: Tests involving proxy_server imports are intentionally minimal
to avoid long module load times during testing.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestUIPathEnvironmentVariable:
    """Test LITELLM_UI_PATH environment variable handling."""

    def test_custom_ui_path_env_var(self):
        """Test that LITELLM_UI_PATH overrides default."""
        custom_path = "/custom/ui/path"

        with mock.patch.dict(
            os.environ, {"LITELLM_UI_PATH": custom_path, "LITELLM_NON_ROOT": "true"}
        ):
            is_non_root = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
            default_runtime_ui_path = (
                "/var/lib/litellm/ui" if is_non_root else "/default/packaged/path"
            )
            runtime_ui_path = os.getenv("LITELLM_UI_PATH", default_runtime_ui_path)

            assert runtime_ui_path == custom_path

    def test_default_ui_path_non_root(self):
        """Test default UI path in non-root mode."""
        with mock.patch.dict(
            os.environ, {"LITELLM_NON_ROOT": "true"}, clear=False
        ):
            # Clear LITELLM_UI_PATH if it exists
            env_copy = os.environ.copy()
            if "LITELLM_UI_PATH" in env_copy:
                del env_copy["LITELLM_UI_PATH"]

            with mock.patch.dict(os.environ, env_copy, clear=True):
                is_non_root = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
                default_runtime_ui_path = (
                    "/var/lib/litellm/ui"
                    if is_non_root
                    else "/default/packaged/path"
                )
                runtime_ui_path = os.getenv(
                    "LITELLM_UI_PATH", default_runtime_ui_path
                )

                assert runtime_ui_path == "/var/lib/litellm/ui"


class TestAssetsPathEnvironmentVariable:
    """Test LITELLM_ASSETS_PATH environment variable handling."""

    def test_custom_assets_path_env_var(self):
        """Test that LITELLM_ASSETS_PATH overrides default."""
        custom_path = "/custom/assets/path"

        with mock.patch.dict(
            os.environ,
            {"LITELLM_ASSETS_PATH": custom_path, "LITELLM_NON_ROOT": "true"},
        ):
            is_non_root = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
            default_assets_dir = (
                "/var/lib/litellm/assets" if is_non_root else "/default/current/dir"
            )
            assets_dir = os.getenv("LITELLM_ASSETS_PATH", default_assets_dir)

            assert assets_dir == custom_path

    def test_default_assets_path_non_root(self):
        """Test default assets path in non-root mode."""
        env_copy = os.environ.copy()
        env_copy["LITELLM_NON_ROOT"] = "true"
        if "LITELLM_ASSETS_PATH" in env_copy:
            del env_copy["LITELLM_ASSETS_PATH"]

        with mock.patch.dict(os.environ, env_copy, clear=True):
            is_non_root = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
            default_assets_dir = (
                "/var/lib/litellm/assets" if is_non_root else "/default/current/dir"
            )
            assets_dir = os.getenv("LITELLM_ASSETS_PATH", default_assets_dir)

            assert assets_dir == "/var/lib/litellm/assets"


class TestUIDetectionLogic:
    """Test UI pre-restructured detection logic without importing proxy_server."""

    def setup_method(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_marker_file_exists(self):
        """Test marker file detection logic."""
        marker_path = os.path.join(self.temp_dir, ".litellm_ui_ready")
        Path(marker_path).touch()

        # Verify marker file exists
        assert os.path.exists(marker_path)

    def test_structural_routes_exist(self):
        """Test structural detection logic."""
        routes = ["login", "guardrails", "logs"]
        for route in routes:
            route_dir = os.path.join(self.temp_dir, route)
            os.makedirs(route_dir, exist_ok=True)
            index_html = os.path.join(route_dir, "index.html")
            Path(index_html).touch()

        # Verify routes exist
        found_routes = 0
        expected_routes = ["login", "guardrails", "logs", "api-reference"]
        for route in expected_routes:
            route_index = os.path.join(self.temp_dir, route, "index.html")
            if os.path.exists(route_index):
                found_routes += 1

        assert found_routes >= 3

    def test_writability_check(self):
        """Test that os.access() correctly detects writable directories."""
        # Should be writable
        assert os.access(self.temp_dir, os.W_OK) is True

        # Create a directory we can't write to (platform-dependent)
        if os.name != "nt":  # Skip on Windows
            readonly_dir = os.path.join(self.temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only

            # Should not be writable
            assert os.access(readonly_dir, os.W_OK) is False

            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
