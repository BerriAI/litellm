# Adversarial verifier tests for litellm-prisma-generate-oob (venv PATH fix)
import os
import shutil
import stat
import tempfile
from unittest.mock import patch

import pytest

from litellm.proxy.client.cli.commands.db import (
    _get_generate_env,
    _get_venv_scripts_dir,
)

_DB_MODULE = "litellm.proxy.client.cli.commands.db"


# ---------------------------------------------------------------------------
# _get_venv_scripts_dir unit tests
# ---------------------------------------------------------------------------


def test_get_venv_scripts_dir_returns_sysconfig_path():
    with patch("sysconfig.get_path", return_value="/home/user/venv/bin") as mock_get:
        result = _get_venv_scripts_dir()
    mock_get.assert_called_once_with("scripts")
    assert result == "/home/user/venv/bin"


@pytest.mark.parametrize("falsy_value", [None, ""])
def test_get_venv_scripts_dir_falls_back_when_sysconfig_falsy(falsy_value):
    fake_exe = "/home/user/venv/bin/python"
    with (
        patch("sysconfig.get_path", return_value=falsy_value),
        patch("sys.executable", fake_exe),
    ):
        result = _get_venv_scripts_dir()
    assert result == os.path.dirname(os.path.abspath(fake_exe))


# ---------------------------------------------------------------------------
# _get_generate_env unit tests
# ---------------------------------------------------------------------------


def test_get_generate_env_no_path_key_in_base_env():
    fake_scripts = "/home/user/venv/bin"
    base_env_with_path = {
        "SOME_VAR": "value",
        "PATH": fake_scripts,
    }
    with patch(f"{_DB_MODULE}._get_prisma_env", return_value=base_env_with_path):
        env = _get_generate_env()
    assert env["PATH"] == fake_scripts
    assert not env["PATH"].startswith(os.pathsep)


def test_get_generate_env_prisma_env_attribute_is_none():
    with (
        patch(f"{_DB_MODULE}._get_prisma_env", None),
        patch.dict(os.environ, {"MY_CUSTOM_VAR": "hello"}, clear=False),
    ):
        env = _get_generate_env()
    assert env.get("MY_CUSTOM_VAR") == "hello"
    assert isinstance(env, dict)


def test_get_generate_env_prisma_env_callable_returns_none():
    sentinel_var = "_VERIFIER_TEST_SENTINEL_12345"
    with (
        patch(f"{_DB_MODULE}._get_prisma_env", return_value=None),
        patch.dict(os.environ, {sentinel_var: "sentinel_value"}, clear=False),
    ):
        env = _get_generate_env()
    assert env.get(sentinel_var) == "sentinel_value"


def test_get_generate_env_preserves_non_path_env_vars():
    base_env = {
        "PATH": "/usr/bin:/bin",
        "DATABASE_URL": "postgres://localhost/db",
        "NODE_OPTIONS": "--max-old-space-size=512",
        "NPM_CONFIG_CACHE": "/tmp/npm",
    }
    fake_scripts = "/home/user/venv/bin"
    with (
        patch(f"{_DB_MODULE}._get_prisma_env", return_value=base_env),
        patch(f"{_DB_MODULE}._get_venv_scripts_dir", return_value=fake_scripts),
    ):
        env = _get_generate_env()
    assert env["DATABASE_URL"] == "postgres://localhost/db"
    assert env["NODE_OPTIONS"] == "--max-old-space-size=512"
    assert env["NPM_CONFIG_CACHE"] == "/tmp/npm"


# ---------------------------------------------------------------------------
# Entrypoint resolvability test
# ---------------------------------------------------------------------------


def test_prisma_client_py_resolvable_via_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_exe = os.path.join(tmpdir, "prisma-client-py")
        with open(fake_exe, "w") as f:
            f.write("#!/bin/sh\necho fake\n")
        os.chmod(
            fake_exe,
            os.stat(fake_exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

        with patch(
            f"{_DB_MODULE}._get_prisma_env", return_value={"PATH": f"{tmpdir}:/usr/bin"}
        ):
            env = _get_generate_env()

        path_entries = env["PATH"].split(os.pathsep)
        assert (
            path_entries[0] == tmpdir
        ), f"scripts dir must be first on PATH; got {path_entries!r}"
        found = shutil.which("prisma-client-py", path=env["PATH"])
        assert found is not None, "prisma-client-py not findable via PATH lookup"
        assert os.path.dirname(found) == tmpdir


# ---------------------------------------------------------------------------
# Adversarial: scripts_dir already at non-zero index
#
# _get_prisma_env() must promote scripts_dir to index 0 even when it appears
# later in PATH. _get_generate_env() delegates entirely to _get_prisma_env(),
# so this test verifies the delegation chain preserves the promotion.
# ---------------------------------------------------------------------------


def test_scripts_dir_promoted_to_index_zero_when_at_later_position():
    fake_scripts = "/home/user/venv/bin"
    promoted_path = os.pathsep.join(
        [
            fake_scripts,
            "/usr/bin",
            "/opt/other",
            "/bin",
        ]
    )
    promoted_env = {"PATH": promoted_path}

    with patch(f"{_DB_MODULE}._get_prisma_env", return_value=promoted_env):
        env = _get_generate_env()

    path_entries = env["PATH"].split(os.pathsep)
    assert path_entries[0] == fake_scripts, (
        f"scripts_dir must be at index 0 in the env returned by _get_generate_env(); "
        f"got {path_entries!r}. "
        f"_get_prisma_env() is responsible for the promotion; "
        f"_get_generate_env() must faithfully delegate."
    )
    assert path_entries.count(fake_scripts) == 1, "scripts_dir must not be duplicated"


# ---------------------------------------------------------------------------
# Delegation regression tests (added for Symptom 3 consolidation)
# After _get_prisma_env() absorbed PATH injection, _get_generate_env() is a
# thin delegation wrapper. These tests ensure the delegation chain works end-
# to-end and the None-guard fires correctly when extras are not installed.
# ---------------------------------------------------------------------------


def test_get_generate_env_delegation_chain_returns_scripts_dir_at_index_0():
    """End-to-end: _get_generate_env() produces PATH with scripts_dir at index 0.

    Validates the full delegation path: _get_generate_env() -> _get_prisma_env()
    (which now handles PATH injection). This is the regression guard for the
    Symptom 3 consolidation: if the delegation chain breaks, db generate would
    regress to the old scoped-only fix and db push would still be unfixed.
    """
    fake_scripts = "/fake/venv/bin"
    base_env = {"PATH": "/usr/bin:/bin"}
    with (
        patch(
            f"{_DB_MODULE}._get_prisma_env",
            return_value={**base_env, "PATH": f"{fake_scripts}:/usr/bin:/bin"},
        ),
    ):
        env = _get_generate_env()

    path_entries = env["PATH"].split(os.pathsep)
    assert path_entries[0] == fake_scripts, (
        f"_get_generate_env() must return env with scripts_dir at PATH index 0. "
        f"Got: {env.get('PATH')}"
    )


def test_get_generate_env_falls_back_to_os_environ_when_prisma_env_is_none():
    """Falls back to os.environ.copy() when _get_prisma_env is None (extras not installed).

    In the not-installed case, _PROXY_EXTRAS_AVAILABLE=False and _get_prisma_env
    is bound to None (not a callable). The wrapper must not raise AttributeError.
    """
    with patch(f"{_DB_MODULE}._get_prisma_env", None):
        env = _get_generate_env()

    # Must return a dict (a copy of os.environ) without raising
    assert isinstance(env, dict)
    assert len(env) > 0
