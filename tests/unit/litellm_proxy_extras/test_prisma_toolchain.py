import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from litellm_proxy_extras.prisma_toolchain import (
    DEFAULT_PRISMA_COMMAND_TIMEOUT,
    DEFAULT_PRISMA_MIGRATE_DEPLOY_TIMEOUT,
    PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR,
    PRISMA_COMMAND_TIMEOUT_ENV_VAR,
    PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR,
    heal_incomplete_nodeenv_cache,
    node_binary_path,
    prisma_bootstrap_timeout,
    prisma_cli_available,
    prisma_command_timeout,
    prisma_migrate_deploy_timeout,
    resolve_prisma_argv,
)


FAKE_PRISMA = """#!{python}
import json
import os
import pathlib
import subprocess
import sys
import time

args = sys.argv[1:]
cache_dir = os.environ["PRISMA_NODEENV_CACHE_DIR"]
log_path = pathlib.Path(os.environ["FAKE_PRISMA_LOG"])
earlier_same_command = sum(
    1
    for line in (log_path.read_text().splitlines() if log_path.exists() else [])
    if json.loads(line)["args"][:2] == args[:2]
)
with log_path.open("a") as log:
    log.write(
        json.dumps({{"args": args, "cache_dir_present": os.path.isdir(cache_dir)}})
        + "\\n"
    )
time.sleep(float(os.environ.get("FAKE_PRISMA_SLEEP", "0")))
if args[:2] == ["migrate", "deploy"]:
    if earlier_same_command == 0:
        if os.environ.get("FAKE_PRISMA_GRANDCHILD_PIDFILE"):
            grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            pathlib.Path(os.environ["FAKE_PRISMA_GRANDCHILD_PIDFILE"]).write_text(str(grandchild.pid))
        time.sleep(float(os.environ.get("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "0")))
    elif os.environ.get("FAKE_PRISMA_LATER_DEPLOY_STDERR"):
        print(os.environ["FAKE_PRISMA_LATER_DEPLOY_STDERR"], file=sys.stderr)
        sys.exit(1)
    print("No pending migrations to apply")
if args[:2] == ["db", "push"] and earlier_same_command == 0:
    time.sleep(float(os.environ.get("FAKE_PRISMA_FIRST_PUSH_SLEEP", "0")))
sys.exit(0)
"""


def _write_fake_prisma(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "prisma"
    script.write_text(FAKE_PRISMA.format(python=sys.executable))
    script.chmod(0o755)
    return bin_dir


@pytest.fixture
def toolchain_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the toolchain at a scratch cache dir driven by a fake Prisma CLI."""
    cache_dir = tmp_path / "nodeenv"
    log_path = tmp_path / "prisma-calls.jsonl"
    bin_dir = _write_fake_prisma(tmp_path)
    monkeypatch.setenv("PRISMA_NODEENV_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("FAKE_PRISMA_LOG", str(log_path))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, raising=False)
    monkeypatch.delenv(PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, raising=False)
    monkeypatch.delenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, raising=False)
    return cache_dir, log_path


def _make_incomplete_cache(cache_dir: Path) -> None:
    (cache_dir / "lib").mkdir(parents=True)
    (cache_dir / "bin").mkdir()


FAKE_PRISMA_MODULE_MAIN = """import json
import sys

print(json.dumps({"module_argv": sys.argv[1:]}))
"""


def _write_fake_prisma_module(tmp_path: Path) -> Path:
    package_dir = tmp_path / "fakemodule" / "prisma"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "__main__.py").write_text(FAKE_PRISMA_MODULE_MAIN)
    return package_dir.parent


def _empty_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "emptybin"
    bin_dir.mkdir()
    return bin_dir


def _make_complete_cache(cache_dir: Path) -> None:
    node = node_binary_path(cache_dir)
    node.parent.mkdir(parents=True)
    node.write_text("")


def test_interrupted_toolchain_install_is_removed(
    toolchain_env: tuple[Path, Path],
) -> None:
    cache_dir, _ = toolchain_env
    _make_incomplete_cache(cache_dir)

    assert heal_incomplete_nodeenv_cache() is True
    assert not cache_dir.exists()


def test_installed_toolchain_is_left_alone(toolchain_env: tuple[Path, Path]) -> None:
    cache_dir, _ = toolchain_env
    _make_complete_cache(cache_dir)

    assert heal_incomplete_nodeenv_cache() is False
    assert node_binary_path(cache_dir).exists()


def test_absent_toolchain_is_not_an_error(toolchain_env: tuple[Path, Path]) -> None:
    cache_dir, _ = toolchain_env

    assert heal_incomplete_nodeenv_cache() is False
    assert not cache_dir.exists()


_CAN_DENY_ACCESS = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0


@pytest.mark.skipif(not _CAN_DENY_ACCESS, reason="root and Windows do not honour a 0o000 directory")
def test_unreadable_cache_dir_is_not_an_error(
    toolchain_env: tuple[Path, Path],
) -> None:
    """A cache dir this process cannot stat means nothing to heal, not a crash.

    Images bake the cache under the build user's home, and a container started
    under any other uid cannot search that directory. `Path.is_dir()` only
    swallows ENOENT-shaped errnos, so it raises `PermissionError` there and
    kills the migration before Prisma is ever invoked.
    """
    cache_dir, _ = toolchain_env
    _make_incomplete_cache(cache_dir)
    cache_dir.parent.chmod(0o000)

    try:
        assert heal_incomplete_nodeenv_cache() is False
    finally:
        cache_dir.parent.chmod(0o700)


@pytest.mark.parametrize(
    ("command_timeout", "deploy_timeout", "expected"),
    [
        ("900", None, 900.0),
        ("12", None, DEFAULT_PRISMA_MIGRATE_DEPLOY_TIMEOUT),
        ("900", "1200", 1200.0),
        ("900", "300", 300.0),
    ],
    ids=[
        "raised_command_budget_carries_over",
        "lowered_command_budget_does_not",
        "override_wins_upward",
        "override_wins_downward",
    ],
)
def test_migrate_deploy_budget_keeps_a_raised_command_budget(
    command_timeout: str, deploy_timeout: str | None, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deployments that raised the per-command budget to survive a long deploy keep that budget for deploy."""
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, command_timeout)
    if deploy_timeout is None:
        monkeypatch.delenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, deploy_timeout)

    assert prisma_migrate_deploy_timeout() == expected


@pytest.mark.parametrize(
    "raw",
    ["", "0", "-5", "not-a-number", "nan", "inf", "-inf", "1e400"],
)
@pytest.mark.parametrize(
    ("env_var", "read_timeout", "default"),
    [
        (PRISMA_COMMAND_TIMEOUT_ENV_VAR, prisma_command_timeout, DEFAULT_PRISMA_COMMAND_TIMEOUT),
        (PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, prisma_migrate_deploy_timeout, DEFAULT_PRISMA_MIGRATE_DEPLOY_TIMEOUT),
    ],
    ids=["command", "migrate_deploy"],
)
def test_unusable_timeout_override_falls_back_to_the_default(
    raw: str, env_var: str, read_timeout: Callable[[], float], default: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite override would silently disable the timeout it configures."""
    monkeypatch.setenv(env_var, raw)

    assert read_timeout() == default


def test_timeout_overrides_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "12")
    monkeypatch.setenv(PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, "900")
    monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, "1200")

    assert prisma_command_timeout() == 12
    assert prisma_bootstrap_timeout() == 900
    assert prisma_migrate_deploy_timeout() == 1200


def test_resolve_prisma_argv_leaves_an_explicit_cli_path_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(_empty_bin(tmp_path)))
    explicit = ("/app/.cache/prisma-python/prisma", "migrate", "deploy")

    assert resolve_prisma_argv(explicit) == explicit


def test_prisma_cli_is_unavailable_with_neither_script_nor_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(_empty_bin(tmp_path)))
    monkeypatch.delitem(sys.modules, "prisma", raising=False)
    monkeypatch.setattr(sys, "path", [])

    assert prisma_cli_available() is False


def test_prisma_cli_is_available_through_the_package_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(_empty_bin(tmp_path)))
    monkeypatch.delitem(sys.modules, "prisma", raising=False)
    monkeypatch.setattr(sys, "path", [str(_write_fake_prisma_module(tmp_path))])

    assert prisma_cli_available() is True


def test_prisma_cli_is_available_through_the_console_script_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(_write_fake_prisma(tmp_path)))
    monkeypatch.delitem(sys.modules, "prisma", raising=False)
    monkeypatch.setattr(sys, "path", [])

    assert prisma_cli_available() is True
