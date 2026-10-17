import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy import prisma_migration


class TestPrismaMigration:
    @pytest.mark.parametrize("env", [{}, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}], ids=("unset", "legacy-opt-out"))
    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_runs_the_migration_job_with_no_opt_out(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock, env: dict[str, str]
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict(os.environ, env, clear=True):
            assert prisma_migration.main(can_generate=lambda: True) == 0

        mock_run_server.assert_called_once_with(("--skip_server_startup",), standalone_mode=False)

    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_exits_zero_when_only_prisma_generate_fails(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="PermissionError: [Errno 13] Permission denied: '/app/.venv/lib/python3.13/site-packages/prisma/schema.prisma'",
        )

        with patch.dict(os.environ, {}, clear=True):
            assert prisma_migration.main(can_generate=lambda: True) == 0

    @patch("litellm.proxy.prisma_migration.subprocess.run")
    @patch("litellm.proxy.prisma_migration.run_server")
    def test_main_propagates_migration_failure(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_run_server.side_effect = SystemExit(1)

        with patch.dict(os.environ, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}, clear=True):
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
            assert prisma_migration.main(can_generate=lambda: True) == 0

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
            assert prisma_migration.main(can_generate=lambda: True) == 0

        assert mock_subprocess_run.call_args.args[0] == ("prisma", "generate")

    @patch("litellm.proxy.prisma_migration.subprocess.run")  # test-quality-ok: the spawned argv is the behavior under test
    @patch("litellm.proxy.prisma_migration.run_server")  # test-quality-ok: run_server boots the whole proxy
    def test_prisma_generate_is_skipped_when_the_prisma_package_is_not_writable(
        self, mock_run_server: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        assert prisma_migration.main(can_generate=lambda: False) == 0

        mock_run_server.assert_called_once_with(("--skip_server_startup",), standalone_mode=False)
        mock_subprocess_run.assert_not_called()

    @pytest.mark.parametrize(
        ("locations", "writable", "expected"),
        [
            (("/site-packages/prisma",), frozenset({"/site-packages/prisma"}), True),
            (("/site-packages/prisma",), frozenset[str](), False),
            (("/a/prisma", "/b/prisma"), frozenset({"/a/prisma"}), False),
            ((), frozenset[str](), False),
        ],
        ids=("writable", "read-only", "one-of-two-read-only", "no-package-dir"),
    )
    def test_prisma_package_is_writable_only_when_every_package_dir_is(
        self, locations: tuple[str, ...], writable: frozenset[str], expected: bool
    ) -> None:
        spec = ModuleSpec("prisma", None, is_package=True)
        spec.submodule_search_locations = list(locations)

        assert (
            prisma_migration.prisma_package_is_writable(
                find_spec=lambda _name: spec, is_writable=lambda path: path in writable
            )
            is expected
        )

    def test_prisma_package_is_not_writable_when_prisma_is_not_installed(self) -> None:
        assert (
            prisma_migration.prisma_package_is_writable(find_spec=lambda _name: None, is_writable=lambda _path: True)
            is False
        )
