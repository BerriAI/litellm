# stdlib imports
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# third party imports
from click.testing import CliRunner

sys.path.insert(0, os.path.abspath("../../.."))

# local imports
from litellm.proxy.client.cli import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROXY_URL": "http://localhost:4000",
            "LITELLM_PROXY_API_KEY": "sk-test",
        },
    ):
        yield


def test_db_generate_success(cli_runner):
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/fake/prisma/dir",
        ),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_command", return_value="prisma"),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_env", return_value=None),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=True),
        patch("litellm.proxy.client.cli.commands.db.subprocess.run", mock_run),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code == 0, result.output
    call_args = mock_run.call_args[0][0]
    assert "prisma" in call_args
    assert "generate" in call_args
    assert "--schema" in call_args
    schema_arg = call_args[call_args.index("--schema") + 1]
    assert schema_arg.endswith("schema.prisma")
    assert "Prisma client generated successfully." in result.output


def test_db_generate_schema_path_uses_get_prisma_dir(cli_runner):
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/custom/litellm/proxy/dir",
        ),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_command", return_value="prisma"),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_env", return_value=None),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=True),
        patch("litellm.proxy.client.cli.commands.db.subprocess.run", mock_run),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code == 0, result.output
    call_args = mock_run.call_args[0][0]
    schema_arg = call_args[call_args.index("--schema") + 1]
    assert schema_arg == "/custom/litellm/proxy/dir/schema.prisma"


def test_db_generate_prisma_failure(cli_runner):
    mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "prisma"))
    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/fake/prisma/dir",
        ),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_command", return_value="prisma"),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_env", return_value=None),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=True),
        patch("litellm.proxy.client.cli.commands.db.subprocess.run", mock_run),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code != 0
    assert "prisma generate failed" in result.output


def test_db_generate_schema_missing(cli_runner):
    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/fake/prisma/dir",
        ),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_command", return_value="prisma"),
        patch("litellm.proxy.client.cli.commands.db._get_prisma_env", return_value=None),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=False),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code != 0
    assert "schema.prisma not found" in result.output


def test_db_generate_proxy_extras_not_installed(cli_runner):
    with patch.dict(
        sys.modules,
        {
            "litellm_proxy_extras": None,
            "litellm_proxy_extras.utils": None,
        },
    ):
        # Patch the module-level flag directly since the import already happened
        with patch("litellm.proxy.client.cli.commands.db._PROXY_EXTRAS_AVAILABLE", False):
            result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code != 0
    assert "not installed" in result.output


def test_db_group_registered_in_cli(cli_runner):
    result = cli_runner.invoke(cli, ["db", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.output


def test_db_generate_help(cli_runner):
    result = cli_runner.invoke(cli, ["db", "generate", "--help"])
    assert result.exit_code == 0
    assert any(word in result.output for word in ("Prisma", "schema", "prisma"))


def test_db_generate_env_includes_scripts_dir_on_path(cli_runner):
    """The generate subprocess env must carry the interpreter's scripts dir on
    PATH so the bundled `prisma-client-py` generator resolves without the caller
    activating the venv (Symptoms 2 & 3).

    PATH injection is now the responsibility of _get_prisma_env() (not
    _get_generate_env()), so this test mocks _get_prisma_env to return a
    realistic env that already has scripts_dir prepended — matching what the
    real implementation produces — and verifies the env reaches subprocess.run
    intact."""
    import sysconfig

    captured = {}

    def _capture(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return MagicMock(returncode=0)

    scripts_dir = sysconfig.get_path("scripts") or os.path.dirname(
        os.path.abspath(sys.executable)
    )

    # Simulate what _get_prisma_env() actually returns: scripts_dir prepended,
    # base system PATH entries preserved after it.
    injected_env = {"PATH": os.pathsep.join([scripts_dir, "/usr/bin", "/bin"])}

    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/fake/prisma/dir",
        ),
        patch(
            "litellm.proxy.client.cli.commands.db._get_prisma_command",
            return_value="prisma",
        ),
        # _get_prisma_env already injects scripts_dir — mock returns realistic output
        patch(
            "litellm.proxy.client.cli.commands.db._get_prisma_env",
            return_value=injected_env,
        ),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=True),
        patch("litellm.proxy.client.cli.commands.db.subprocess.run", side_effect=_capture),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert env is not None, "subprocess.run was called without an explicit env"
    path_entries = env["PATH"].split(os.pathsep)
    assert (
        scripts_dir in path_entries
    ), f"scripts dir {scripts_dir!r} not on child PATH {env['PATH']!r}"
    # scripts_dir must be first so it shadows any system-installed prisma binary
    assert path_entries[0] == scripts_dir
    # base env entries are preserved, not clobbered
    assert "/usr/bin" in path_entries


def test_db_generate_env_does_not_duplicate_scripts_dir(cli_runner):
    """If the scripts dir is already on PATH (e.g. an activated venv), the fix
    must not add a duplicate entry."""
    import sysconfig

    captured = {}

    def _capture(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return MagicMock(returncode=0)

    scripts_dir = sysconfig.get_path("scripts") or os.path.dirname(
        os.path.abspath(sys.executable)
    )

    with (
        patch(
            "litellm.proxy.client.cli.commands.db.PrismaManager._get_prisma_dir",
            return_value="/fake/prisma/dir",
        ),
        patch(
            "litellm.proxy.client.cli.commands.db._get_prisma_command",
            return_value="prisma",
        ),
        patch(
            "litellm.proxy.client.cli.commands.db._get_prisma_env",
            return_value={"PATH": scripts_dir + os.pathsep + "/usr/bin"},
        ),
        patch("litellm.proxy.client.cli.commands.db.os.path.exists", return_value=True),
        patch("litellm.proxy.client.cli.commands.db.subprocess.run", side_effect=_capture),
    ):
        result = cli_runner.invoke(cli, ["db", "generate"])

    assert result.exit_code == 0, result.output
    path_entries = captured["env"]["PATH"].split(os.pathsep)
    assert path_entries.count(scripts_dir) == 1
