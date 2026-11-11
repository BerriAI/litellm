"""Regression tests for Prisma subprocess PATH setup."""

import os
import subprocess
from unittest.mock import patch

import pytest
from litellm_proxy_extras.utils import _get_prisma_env


def test_get_prisma_env_prepends_interpreter_scripts_dir(monkeypatch):
    """An unactivated virtualenv still exposes prisma-client-py to Prisma."""
    scripts_dir = "/venv/bin"
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")

    with patch("sysconfig.get_path", return_value=scripts_dir):
        env = _get_prisma_env()

    assert env["PATH"].split(os.pathsep) == [
        scripts_dir,
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]


def test_get_prisma_env_promotes_and_deduplicates_scripts_dir(monkeypatch):
    """The scripts directory appears once and takes precedence over host tools."""
    scripts_dir = "/venv/bin"
    monkeypatch.setenv("PATH", f"/usr/bin:{scripts_dir}:/bin:{scripts_dir}")

    with patch("sysconfig.get_path", return_value=scripts_dir):
        env = _get_prisma_env()

    assert env["PATH"].split(os.pathsep) == [scripts_dir, "/usr/bin", "/bin"]


def test_get_prisma_env_uses_interpreter_directory_when_scripts_is_unknown(monkeypatch):
    """Platforms without a sysconfig scripts value still resolve console scripts."""
    monkeypatch.setenv("PATH", "/usr/bin")

    with (
        patch("sysconfig.get_path", return_value=None),
        patch("sys.executable", "/venv/bin/python"),
    ):
        env = _get_prisma_env()

    assert env["PATH"].split(os.pathsep)[0] == "/venv/bin"


def test_get_prisma_env_keeps_offline_configuration(monkeypatch):
    """PATH injection does not discard the existing offline Prisma settings."""
    monkeypatch.setenv("PRISMA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PATH", "/usr/bin")

    with patch("sysconfig.get_path", return_value="/venv/bin"):
        env = _get_prisma_env()

    assert env["NPM_CONFIG_PREFER_OFFLINE"] == "true"
    assert env["PATH"].split(os.pathsep)[0] == "/venv/bin"


@pytest.mark.skipif(os.name == "nt", reason="uses the POSIX Prisma shell launcher")
def test_prisma_subprocess_resolves_console_script_without_venv_activation(monkeypatch, tmp_path):
    """The constructed environment lets the shell locate the generated client."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    command = scripts_dir / "prisma-client-py"
    command.write_text("#!/bin/sh\nprintf resolved\n")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    with patch("sysconfig.get_path", return_value=str(scripts_dir)):
        env = _get_prisma_env()

    completed = subprocess.run(
        ["/bin/sh", "-c", "prisma-client-py"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout == "resolved"
