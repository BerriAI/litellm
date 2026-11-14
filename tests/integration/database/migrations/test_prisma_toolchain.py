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

import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest
from litellm_proxy_extras.prisma_toolchain import (
    PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR,
    PRISMA_COMMAND_TIMEOUT_ENV_VAR,
    PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR,
    ensure_prisma_toolchain,
    run_prisma,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager

from tests._process_helpers import process_is_gone
from tests.integration._support.process import kill_stray_process
from tests.integration._support.timing import assert_cut_short, assert_enforced_within


HUNG_BOOTSTRAP_SECONDS = 30
HUNG_DEPLOY_SECONDS = 60
COMMAND_TIMEOUT_SECONDS = 1
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
    monkeypatch.setenv(PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, str(COMMAND_TIMEOUT_SECONDS))
    monkeypatch.setenv("FAKE_PRISMA_SLEEP", str(HUNG_BOOTSTRAP_SECONDS))

    started = time.monotonic()
    result = ensure_prisma_toolchain(
        prisma_command="prisma", prisma_env=dict(os.environ)
    )
    elapsed = time.monotonic() - started

    assert result.ready is False
    assert_cut_short(elapsed, injected_delay=HUNG_BOOTSTRAP_SECONDS)
    assert_enforced_within(elapsed, configured=COMMAND_TIMEOUT_SECONDS, slack=14)


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
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "3")

    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True
    assert _deploy_calls(log_path) == [["migrate", "deploy"]]


def test_migrate_deploy_stops_at_its_own_timeout(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy budget still bounds a deploy that hangs, so boot cannot wait forever."""
    _, log_path = toolchain_env
    monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, str(COMMAND_TIMEOUT_SECONDS))
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", str(HUNG_DEPLOY_SECONDS))
    monkeypatch.setenv("FAKE_PRISMA_LATER_DEPLOY_STDERR", "Error: P3018 permission denied for schema public")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="insufficient permissions"):
        ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)
    elapsed = time.monotonic() - started

    assert len(_deploy_calls(log_path)) == 2
    assert_cut_short(elapsed, injected_delay=HUNG_DEPLOY_SECONDS)
    assert_enforced_within(elapsed, configured=2 * COMMAND_TIMEOUT_SECONDS, slack=28)


def test_a_timed_out_migrate_deploy_takes_its_process_tree_with_it(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real CLI forks Node and a schema engine; a timeout must not leave them running."""
    _, log_path = toolchain_env
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv(PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR, str(COMMAND_TIMEOUT_SECONDS))
    monkeypatch.setenv("FAKE_PRISMA_FIRST_DEPLOY_SLEEP", "60")
    monkeypatch.setenv("FAKE_PRISMA_LATER_DEPLOY_STDERR", "Error: P3018 permission denied for schema public")
    monkeypatch.setenv("FAKE_PRISMA_GRANDCHILD_PIDFILE", str(pidfile))

    with pytest.raises(RuntimeError, match="insufficient permissions"):
        ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    grandchild_pid = int(pidfile.read_text())
    try:
        assert len(_deploy_calls(log_path)) == 2
        assert process_is_gone(grandchild_pid, within_seconds=5)
    finally:
        kill_stray_process(grandchild_pid)


def test_db_push_timeout_hint_names_the_per_command_budget(
    toolchain_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``db push`` keeps the per-command budget, so its timeout hint has to name that variable."""
    _, log_path = toolchain_env
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "1")
    monkeypatch.setenv("FAKE_PRISMA_FIRST_PUSH_SLEEP", "3")

    with caplog.at_level(logging.WARNING, logger="litellm_proxy_extras"):
        assert ProxyExtrasDBManager.setup_database(use_migrate=False, use_v2_resolver=False) is True

    assert [call["args"][:2] for call in _fake_prisma_calls(log_path)].count(["db", "push"]) == 2
    assert [record.getMessage() for record in caplog.records if "timed out" in record.getMessage()] == [
        f"Attempt 1 timed out. Raise {PRISMA_COMMAND_TIMEOUT_ENV_VAR} if this database needs longer to apply its schema."
    ]










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


def test_run_prisma_uses_the_module_when_the_console_script_is_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_bin = _empty_bin(tmp_path)
    module_root = _write_fake_prisma_module(tmp_path)
    monkeypatch.setenv("PATH", str(empty_bin))

    result = run_prisma(
        ["prisma", "migrate", "deploy"],
        timeout=60,
        env={"PATH": str(empty_bin), "PYTHONPATH": str(module_root)},
    )

    assert json.loads(result.stdout) == {"module_argv": ["migrate", "deploy"]}


def test_run_prisma_prefers_the_console_script_on_path(
    toolchain_env: tuple[Path, Path], tmp_path: Path
) -> None:
    _, log_path = toolchain_env
    module_root = _write_fake_prisma_module(tmp_path)

    result = run_prisma(
        ["prisma", "--version"],
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(module_root)},
    )

    assert [call["args"] for call in _fake_prisma_calls(log_path)] == [["--version"]]
    assert "module_argv" not in result.stdout








