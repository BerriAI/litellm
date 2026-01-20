import sys
import os
import subprocess
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

# Add parent directory to path to allow importing litellm
sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.db.prisma_client import PrismaManager
from litellm.proxy.proxy_cli import run_server


class TestMigrationFailureHandling:
    @patch("subprocess.run")
    def test_prisma_client_permission_error_retry(self, mock_subprocess_run):
        """
        Regression Test: Verifies that PrismaManager.setup_database
        catches PermissionError during 'prisma db push' and retries with '--skip-generate'.
        """
        # Mock behavior:
        # call 1: raises CalledProcessError with "Permission denied" and "schema.prisma"
        # call 2 (retry): succeeds

        error_output = "Error: Permission denied writing to ... schema.prisma"

        mock_process_error = subprocess.CalledProcessError(
            returncode=1, cmd=["prisma", "db", "push"], stderr=error_output
        )

        mock_subprocess_run.side_effect = [
            mock_process_error,  # 1st attempt fails with permission error
            MagicMock(returncode=0),  # 2nd attempt (retry) succeeds
        ]

        # Ensure we run the 'db push' path (use_migrate=False)
        # We also need to mock should_update_prisma_schema to return True

        with patch(
            "litellm.proxy.db.prisma_client.should_update_prisma_schema",
            return_value=True,
        ):
            # Run setup_database with use_migrate=False to trigger 'prisma db push' path
            result = PrismaManager.setup_database(use_migrate=False)

            # Assert success
            assert result is True

            # Verify calls
            assert mock_subprocess_run.call_count == 2

            # Check 1st call arguments (standard push)
            args1, _ = mock_subprocess_run.call_args_list[0]
            assert "push" in args1[0]
            assert "--skip-generate" not in args1[0]

            # Check 2nd call arguments (retry with skip-generate)
            args2, _ = mock_subprocess_run.call_args_list[1]
            assert "push" in args2[0]
            assert "--skip-generate" in args2[0]

    def test_proxy_cli_exit_on_migration_fail(self):
        """
        Regression Test: Verifies that proxy_cli.run_server exits with NON-ZERO status
        if PrismaManager.setup_database returns False.
        """
        runner = CliRunner()

        # Mock setup_database to return False (Simulating failure)
        # Mock should_update_prisma_schema to return True (Ensure we hit the DB setup logic)
        with patch(
            "litellm.proxy.db.prisma_client.PrismaManager.setup_database",
            return_value=False,
        ), patch(
            "litellm.proxy.db.prisma_client.should_update_prisma_schema",
            return_value=True,
        ):
            # Mock dependencies to prevent actual server startup and handle imports
            mock_app = MagicMock()
            mock_proxy_config = MagicMock()

            # Patch sys.modules to prevent ImportErrors for proxy_server
            with patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=mock_app, ProxyConfig=mock_proxy_config
                    )
                },
            ):
                with patch(
                    "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
                ) as mock_get_args:
                    mock_get_args.return_value = {
                        "app": "app",
                        "host": "localhost",
                        "port": 8000,
                    }

                    # Set DATABASE_URL to trigger DB logic
                    with patch.dict(
                        os.environ,
                        {"DATABASE_URL": "postgresql://user:pass@localhost:5432/db"},
                    ):
                        # Execute: Run server with --local and --skip_server_startup
                        result = runner.invoke(
                            run_server, ["--local", "--skip_server_startup"]
                        )

                        # Assert: Exit code should be non-zero (failure)
                        assert (
                            result.exit_code != 0
                        ), f"Expected non-zero exit code, got {result.exit_code}. Output: {result.output}"
                        assert "Database setup failed. Exiting..." in result.output
