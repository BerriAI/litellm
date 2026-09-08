import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy import prisma_migration


class TestPrismaMigration:
    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_enforces_migration_check_by_default(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict(os.environ, {}, clear=True):
            assert prisma_migration.main() == 0

        mock_run_server.assert_called_once_with(
            ("--skip_server_startup", "--enforce_prisma_migration_check"),
            standalone_mode=False,
        )

    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_disables_migration_check_when_explicitly_false(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict(os.environ, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}, clear=True):
            assert prisma_migration.main() == 0

        mock_run_server.assert_called_once_with(("--skip_server_startup",), standalone_mode=False)

    @pytest.mark.parametrize("env", [{}, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}])
    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_exits_zero_when_only_prisma_generate_fails(
        self,
        mock_run_server: MagicMock,
        mock_subprocess_run: MagicMock,
        env: dict[str, str],
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="PermissionError: [Errno 13] Permission denied: '/app/.venv/lib/python3.13/site-packages/prisma/schema.prisma'",
        )

        with patch.dict(os.environ, env, clear=True):
            assert prisma_migration.main() == 0

    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_propagates_migration_failure(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_run_server.side_effect = SystemExit(1)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="1"):
                prisma_migration.main()

        mock_subprocess_run.assert_not_called()
