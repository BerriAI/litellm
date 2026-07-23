"""Integration tests for `litellm-proxy db generate` (Symptom 2).

Verifies that invoking the binary by absolute path, without activating the venv,
still finds the bundled `prisma` script because `_get_generate_env()` prepends
the interpreter's scripts directory to PATH.

These tests invoke real binaries, so they live outside `tests/test_litellm/`
(the mock-only unit tree) and are skipped by default. Set
`LITELLM_RUN_INTEGRATION_TESTS=1` to run them. See the README in this directory
for details.
"""

import os
import stat
import subprocess
import sysconfig

import pytest


@pytest.fixture
def non_activated_env():
    venv_root = sysconfig.get_path("data") or os.path.dirname(
        sysconfig.get_path("scripts") or ""
    )
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    path_entries = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join(
        e for e in path_entries if not e.startswith(venv_root)
    )
    return env


@pytest.fixture
def fake_prisma_bin(tmp_path):
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    prisma = fake_bin / "prisma"
    prisma.write_text('#!/bin/sh\necho "fake-prisma called: $@"\nexit 0\n')
    prisma.chmod(prisma.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return fake_bin


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("LITELLM_RUN_INTEGRATION_TESTS"),
    reason="set LITELLM_RUN_INTEGRATION_TESTS=1 to run integration tests",
)
def test_db_generate_absolute_path_non_activated_venv(non_activated_env):
    scripts_dir = sysconfig.get_path("scripts")
    litellm_proxy_bin = os.path.join(scripts_dir, "litellm-proxy")
    if not os.path.isfile(litellm_proxy_bin):
        pytest.skip("litellm-proxy console script not installed in venv")

    result = subprocess.run(
        [litellm_proxy_bin, "db", "generate"],
        env=non_activated_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "Generating Prisma client" in result.stdout
    assert "Prisma client generated successfully." in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("LITELLM_RUN_INTEGRATION_TESTS"),
    reason="set LITELLM_RUN_INTEGRATION_TESTS=1 to run integration tests",
)
def test_db_generate_succeeds_with_scripts_dir_off_caller_path(non_activated_env):
    """The fix injects the interpreter scripts dir into PATH internally, so
    ``db generate`` finds the bundled ``prisma`` script even when the caller's
    PATH does not contain the venv scripts dir. This is the positive proof of
    Symptom 2: a stripped caller PATH no longer breaks generation.
    """
    scripts_dir = sysconfig.get_path("scripts")
    litellm_proxy_bin = os.path.join(scripts_dir, "litellm-proxy")
    if not os.path.isfile(litellm_proxy_bin):
        pytest.skip("litellm-proxy console script not installed in venv")

    # Caller PATH deliberately excludes the venv scripts dir (and thus prisma).
    test_env = non_activated_env.copy()
    test_env["PATH"] = "/usr/bin:/bin"

    result = subprocess.run(
        [litellm_proxy_bin, "db", "generate"],
        env=test_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "Generating Prisma client" in result.stdout
    assert "Prisma client generated successfully." in result.stdout
