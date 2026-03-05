"""
Tests for PrismaManager.ensure_client_generated().

Verifies that the Prisma Python client is auto-generated when missing,
and that the method is a no-op when the client already exists.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaManager


class TestEnsureClientGenerated:
    """Tests for PrismaManager.ensure_client_generated"""

    @patch("litellm.proxy.db.prisma_client.subprocess.run")
    def test_skips_generate_when_client_exists(self, mock_run):
        """When the Prisma client is already importable, prisma generate is NOT called."""
        prisma_module = MagicMock()
        prisma_module.Prisma = object()
        with patch(
            "litellm.proxy.db.prisma_client.importlib.import_module",
            return_value=prisma_module,
        ):
            PrismaManager.ensure_client_generated()
        mock_run.assert_not_called()

    @patch("litellm.proxy.db.prisma_client.sys")
    @patch("litellm.proxy.db.prisma_client.subprocess.run")
    @patch("litellm.proxy.db.prisma_client.importlib.invalidate_caches")
    def test_runs_generate_when_client_missing(self, mock_invalidate_caches, mock_run, mock_sys):
        """When the Prisma client cannot be imported, prisma generate is called."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["prisma", "generate"], returncode=0, stdout="", stderr=""
        )
        # Prevent real sys.modules mutation during test
        mock_sys.modules = {}
        prisma_module = MagicMock()
        prisma_module.Prisma = object()
        with patch(
            "litellm.proxy.db.prisma_client.importlib.import_module",
            side_effect=[ImportError("missing prisma"), prisma_module],
        ):
            PrismaManager.ensure_client_generated()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["prisma", "generate"]
        assert mock_run.call_args.kwargs.get("cwd") == PrismaManager._get_prisma_dir()
        mock_invalidate_caches.assert_called_once()

    @patch("litellm.proxy.db.prisma_client.subprocess.run")
    def test_raises_on_generate_failure(self, mock_run):
        """When prisma generate fails, a RuntimeError is raised."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["prisma", "generate"],
            returncode=1,
            stdout="",
            stderr="error: schema not found",
        )
        with patch(
            "litellm.proxy.db.prisma_client.importlib.import_module",
            side_effect=ImportError("missing prisma"),
        ):
            with pytest.raises(RuntimeError, match="prisma generate"):
                PrismaManager.ensure_client_generated()

    @patch("litellm.proxy.db.prisma_client.subprocess.run")
    def test_raises_on_timeout(self, mock_run):
        """When prisma generate times out, a RuntimeError is raised."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["prisma", "generate"], timeout=120
        )
        with patch(
            "litellm.proxy.db.prisma_client.importlib.import_module",
            side_effect=ImportError("missing prisma"),
        ):
            with pytest.raises(RuntimeError, match="timed out after"):
                PrismaManager.ensure_client_generated()

    @patch("litellm.proxy.db.prisma_client.sys")
    @patch("litellm.proxy.db.prisma_client.subprocess.run")
    @patch("litellm.proxy.db.prisma_client.importlib.invalidate_caches")
    def test_raises_when_still_not_importable_after_generate(
        self, mock_invalidate_caches, mock_run, mock_sys
    ):
        """If import still fails after generate, raise a clear RuntimeError."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["prisma", "generate"], returncode=0, stdout="", stderr=""
        )
        mock_sys.modules = {}
        with patch(
            "litellm.proxy.db.prisma_client.importlib.import_module",
            side_effect=[ImportError("missing prisma"), ImportError("still missing")],
        ):
            with pytest.raises(RuntimeError, match="import still failing"):
                PrismaManager.ensure_client_generated()
        mock_invalidate_caches.assert_called_once()
