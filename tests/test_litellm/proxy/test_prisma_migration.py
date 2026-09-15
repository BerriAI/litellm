import os
import sys
from pathlib import Path
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

    @patch("litellm.proxy.prisma_migration.subprocess.run")  # test-quality-ok: the spawned argv is the behavior under test
    @patch("litellm.proxy.prisma_migration.run_server")  # test-quality-ok: run_server boots the whole proxy
    def test_prisma_generate_runs_through_the_module_when_the_cli_is_not_on_path(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        empty_bin: Path = tmp_path / "emptybin"
        empty_bin.mkdir()

        with patch.dict(os.environ, {"PATH": str(empty_bin)}, clear=True):
            assert prisma_migration.main() == 0

        assert mock_subprocess_run.call_args.args[0] == (sys.executable, "-m", "prisma", "generate")

    @patch("litellm.proxy.prisma_migration.subprocess.run")  # test-quality-ok: the spawned argv is the behavior under test
    @patch("litellm.proxy.prisma_migration.run_server")  # test-quality-ok: run_server boots the whole proxy
    def test_prisma_generate_runs_the_console_script_when_it_is_on_path(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        bin_dir: Path = tmp_path / "bin"
        bin_dir.mkdir()
        script: Path = bin_dir / "prisma"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        with patch.dict(os.environ, {"PATH": str(bin_dir)}, clear=True):
            assert prisma_migration.main() == 0

        assert mock_subprocess_run.call_args.args[0] == ("prisma", "generate")
