"""Migrations must survive a Node toolchain install that was killed mid-flight.

The Prisma CLI installs a private Node runtime on its first invocation. If that
install is interrupted, the cache directory is left behind without a Node
binary and Prisma skips reinstalling it forever, so every later migration
attempt fails identically. These tests pin the two behaviours that keep a
container recoverable: an incomplete cache is deleted before Prisma is
invoked, and the install gets a budget of its own rather than sharing the one
that bounds each migration command.

``prisma migrate deploy`` gets a budget of its own for the same reason: its
runtime grows with the number of pending migrations, so a fresh database that
replays every migration overran the per-command budget on slow machines and
the proxy gave up after four identical timeouts.
"""

import ast
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from litellm_proxy_extras.prisma_toolchain import (
    DEFAULT_PRISMA_COMMAND_TIMEOUT,
    DEFAULT_PRISMA_MIGRATE_DEPLOY_TIMEOUT,
    PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR,
    PRISMA_COMMAND_TIMEOUT_ENV_VAR,
    PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR,
    ensure_prisma_toolchain,
    heal_incomplete_nodeenv_cache,
    node_binary_path,
    prisma_bootstrap_timeout,
    prisma_command_timeout,
    prisma_migrate_deploy_timeout,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_EXTRAS = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"

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


def _fake_prisma_calls(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines()]


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


def _deploy_calls(log_path: Path) -> list[list[str]]:
    return [call["args"] for call in _fake_prisma_calls(log_path) if call["args"][:2] == ["migrate", "deploy"]]


def _make_incomplete_cache(cache_dir: Path) -> None:
    (cache_dir / "lib").mkdir(parents=True)
    (cache_dir / "bin").mkdir()


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


@pytest.mark.skipif(
    not _CAN_DENY_ACCESS, reason="root and Windows do not honour a 0o000 directory"
)
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


def test_bootstrap_clears_the_cache_before_invoking_prisma(
    toolchain_env: tuple[Path, Path],
) -> None:
    cache_dir, log_path = toolchain_env
    _make_incomplete_cache(cache_dir)

    result = ensure_prisma_toolchain(
        prisma_command="prisma", prisma_env=dict(os.environ)
    )

    assert result.healed_incomplete_cache is True
    assert result.ready is True
    calls = _fake_prisma_calls(log_path)
    assert len(calls) == 1
    assert calls[0]["cache_dir_present"] is False


def test_bootstrap_is_not_bounded_by_the_per_command_timeout(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, log_path = toolchain_env
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_SLEEP", "3")

    result = ensure_prisma_toolchain(
        prisma_command="prisma", prisma_env=dict(os.environ)
    )

    assert result.ready is True
    assert len(_fake_prisma_calls(log_path)) == 1


def test_bootstrap_stops_at_its_own_timeout(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_SLEEP", "30")

    started = time.monotonic()
    result = ensure_prisma_toolchain(
        prisma_command="prisma", prisma_env=dict(os.environ)
    )
    elapsed = time.monotonic() - started

    assert result.ready is False
    assert elapsed < 15


def test_setup_database_prepares_the_toolchain_before_migrating(
    toolchain_env: tuple[Path, Path],
) -> None:
    cache_dir, log_path = toolchain_env
    _make_incomplete_cache(cache_dir)

    assert ProxyExtrasDBManager.setup_database(use_migrate=True) is True

    calls = _fake_prisma_calls(log_path)
    assert [call["args"] for call in calls][:2] == [
        ["--version"],
        ["migrate", "deploy"],
    ]
    assert calls[0]["cache_dir_present"] is False


@pytest.mark.parametrize("use_v2_resolver", [False, True], ids=["v1", "v2"])
def test_migrate_deploy_is_not_bounded_by_the_per_command_timeout(
    toolchain_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    use_v2_resolver: bool,
) -> None:
    """A fresh database replays every migration, which takes longer than any bookkeeping command."""
    _, log_path = toolchain_env
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "3")

    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True
    assert _deploy_calls(log_path) == [["migrate", "deploy"]]


def test_migrate_deploy_stops_at_its_own_timeout(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy budget still bounds a deploy that hangs, so boot cannot wait forever."""
    _, log_path = toolchain_env
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "60")
    monkeypatch.setenv("FAKE_PRISMA_LATER_DEPLOY_STDERR", "Error: P3018 permission denied for schema public")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="insufficient permissions"):
        ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)
    elapsed = time.monotonic() - started

    assert len(_deploy_calls(log_path)) == 2
    assert elapsed < 30


def _process_is_gone(pid: int, within_seconds: float) -> bool:
    deadline = time.monotonic() + within_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_timed_out_migrate_deploy_takes_its_process_tree_with_it(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real CLI forks Node and a schema engine; a timeout must not leave them running."""
    _, log_path = toolchain_env
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "60")
    monkeypatch.setenv("FAKE_PRISMA_LATER_DEPLOY_STDERR", "Error: P3018 permission denied for schema public")
    monkeypatch.setenv("FAKE_PRISMA_GRANDCHILD_PIDFILE", str(pidfile))

    with pytest.raises(RuntimeError, match="insufficient permissions"):
        ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    grandchild_pid = int(pidfile.read_text())
    try:
        assert len(_deploy_calls(log_path)) == 2
        assert _process_is_gone(grandchild_pid, within_seconds=5)
    finally:
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_db_push_timeout_hint_names_the_per_command_budget(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``db push`` keeps the per-command budget, so its timeout hint has to name that variable."""
    _, log_path = toolchain_env
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_PUSH_SLEEP", "3")

    with caplog.at_level(logging.WARNING, logger="litellm_proxy_extras"):
        assert ProxyExtrasDBManager.setup_database(use_migrate=False, use_v2_resolver=False) is True

    assert [call["args"][:2] for call in _fake_prisma_calls(log_path)].count(["db", "push"]) == 2
    assert [record.getMessage() for record in caplog.records if "timed out" in record.getMessage()] == [
        f"Attempt 1 timed out. Raise {PRISMA_COMMAND_TIMEOUT_ENV_VAR} if this database needs longer to apply its schema."
    ]


@pytest.mark.parametrize(
    ("command_timeout", "deploy_timeout", "expected"),
    [
        ("900", None, 900.0),
        ("12", None, DEFAULT_PRISMA_MIGRATE_DEPLOY_TIMEOUT),
        ("900", "1200", 1200.0),
        ("900", "300", 300.0),
    ],
    ids=["raised_command_budget_carries_over", "lowered_command_budget_does_not", "override_wins_upward", "override_wins_downward"],
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


@pytest.mark.parametrize("module", ["utils.py", "replica_identity.py"])
def test_every_prisma_command_timeout_is_overridable(module: str) -> None:
    tree = ast.parse((PROXY_EXTRAS / module).read_text())
    literals = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "timeout"
        and isinstance(node.value, ast.Constant)
    ]

    assert literals == [], (
        f"{module} still hardcodes a Prisma timeout at lines {literals}; "
        "route it through prisma_command_timeout() so it can be raised without a release"
    )
