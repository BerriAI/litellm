"""Tests for _get_prisma_env() venv PATH injection (Symptom 3 of issue #26097).

Symptom 3: after `pip install 'litellm[proxy]'`, running the proxy by absolute
path in a non-activated venv causes `prisma db push` to exit 0 without creating
tables because `prisma-client-py` is not on PATH when /bin/sh forks for the
generator. The fix: _get_prisma_env() prepends the interpreter's scripts dir.
"""

import os
from unittest.mock import MagicMock, patch

from litellm_proxy_extras.utils import ProxyExtrasDBManager, _get_prisma_env

# ---------------------------------------------------------------------------
# Unit tests for _get_prisma_env() PATH injection
# ---------------------------------------------------------------------------


def test_get_prisma_env_injects_scripts_dir_at_index_0(monkeypatch):
    """Scripts dir appears at PATH index 0 when venv is not activated."""
    fake_scripts = "/fake/venv/bin"
    # Remove scripts dir from PATH so it's not already there
    stripped_path = "/usr/local/bin:/usr/bin:/bin"
    monkeypatch.setenv("PATH", stripped_path)
    with (patch("sysconfig.get_path", return_value=fake_scripts),):
        env = _get_prisma_env()

    path_entries = env["PATH"].split(os.pathsep)
    assert (
        path_entries[0] == fake_scripts
    ), f"Expected scripts dir at index 0, got: {env['PATH']}"
    # Original PATH entries preserved after the injected dir
    assert "/usr/local/bin" in path_entries
    assert "/usr/bin" in path_entries


def test_get_prisma_env_promotes_scripts_dir_to_index_0(monkeypatch):
    """Scripts dir is promoted to index 0 if already in PATH at a later position."""
    fake_scripts = "/fake/venv/bin"
    path_with_scripts_at_end = f"/usr/bin:/bin:{fake_scripts}"
    monkeypatch.setenv("PATH", path_with_scripts_at_end)
    with patch("sysconfig.get_path", return_value=fake_scripts):
        env = _get_prisma_env()

    path_entries = env["PATH"].split(os.pathsep)
    assert path_entries[0] == fake_scripts
    # Must appear exactly once
    assert path_entries.count(fake_scripts) == 1


def test_get_prisma_env_handles_empty_path(monkeypatch):
    """Empty PATH results in PATH containing only the scripts dir."""
    fake_scripts = "/fake/venv/bin"
    monkeypatch.setenv("PATH", "")
    with patch("sysconfig.get_path", return_value=fake_scripts):
        env = _get_prisma_env()

    assert env["PATH"] == fake_scripts
    assert (
        os.pathsep not in env["PATH"] or env["PATH"].rstrip(os.pathsep) == fake_scripts
    )


def test_get_prisma_env_falls_back_to_dirname_when_sysconfig_falsy(monkeypatch):
    """Falls back to dirname(sys.executable) when sysconfig returns falsy."""
    fake_exe = "/fake/venv/bin/python3"
    monkeypatch.setenv("PATH", "/usr/bin")
    with (
        patch("sysconfig.get_path", return_value=None),
        patch("sys.executable", fake_exe),
    ):
        env = _get_prisma_env()

    expected_scripts = os.path.dirname(os.path.abspath(fake_exe))
    path_entries = env["PATH"].split(os.pathsep)
    assert path_entries[0] == expected_scripts


def test_get_prisma_env_sets_offline_vars_when_enabled(monkeypatch):
    """PRISMA_OFFLINE_MODE vars are still set alongside the PATH injection."""
    monkeypatch.setenv("PRISMA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PATH", "/usr/bin")
    with patch("sysconfig.get_path", return_value="/venv/bin"):
        env = _get_prisma_env()

    assert env.get("NPM_CONFIG_PREFER_OFFLINE") == "true"
    assert "NPM_CONFIG_CACHE" in env
    # PATH injection still happens
    assert env["PATH"].startswith("/venv/bin")


def test_get_prisma_env_noop_when_scripts_dir_already_at_index_0(monkeypatch):
    """No duplicate entry when scripts dir is already at index 0."""
    fake_scripts = "/fake/venv/bin"
    monkeypatch.setenv("PATH", f"{fake_scripts}:/usr/bin:/bin")
    with patch("sysconfig.get_path", return_value=fake_scripts):
        env = _get_prisma_env()

    path_entries = env["PATH"].split(os.pathsep)
    assert path_entries[0] == fake_scripts
    assert path_entries.count(fake_scripts) == 1


# ---------------------------------------------------------------------------
# Integration: db push subprocess receives patched env with scripts_dir
# ---------------------------------------------------------------------------


def test_db_push_subprocess_receives_scripts_dir_on_path(monkeypatch, tmp_path):
    """setup_database(use_migrate=False) passes env with scripts_dir to subprocess.

    This is the Symptom 3 integration guard: the subprocess call must receive
    the patched env dict that has the venv scripts dir at PATH index 0. Without
    it, prisma db push exits 0 but the generator fails silently and tables are
    never created.
    """
    fake_scripts = "/fake/venv/bin"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/testdb")

    (tmp_path / "schema.prisma").write_text("// stub schema\n")

    captured_calls = []

    def fake_subprocess_run(args, **kwargs):
        captured_calls.append({"args": list(args), "env": kwargs.get("env", {})})
        return MagicMock(returncode=0)

    with (
        patch("sysconfig.get_path", return_value=fake_scripts),
        patch.object(
            ProxyExtrasDBManager, "_get_prisma_dir", return_value=str(tmp_path)
        ),
        patch("litellm_proxy_extras.utils._get_prisma_command", return_value="prisma"),
        patch("subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = ProxyExtrasDBManager.setup_database(use_migrate=False)

    assert result is True, "setup_database should return True on success"
    assert (
        len(captured_calls) == 1
    ), f"Expected 1 subprocess call, got {len(captured_calls)}"

    received_env = captured_calls[0]["env"]
    assert received_env is not None, "subprocess.run must be called with env="
    path_entries = received_env.get("PATH", "").split(os.pathsep)
    assert path_entries[0] == fake_scripts, (
        f"Expected scripts dir '{fake_scripts}' at PATH index 0 in subprocess env. "
        f"Got PATH: {received_env.get('PATH')}"
    )

    call_args = captured_calls[0]["args"]
    assert "db" in call_args
    assert "push" in call_args
    assert "--accept-data-loss" in call_args
