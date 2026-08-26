"""Migrations must survive a Node toolchain install that was killed mid-flight.

The Prisma CLI installs a private Node runtime on its first invocation. If that
install is interrupted, the cache directory is left behind without a Node
binary and Prisma skips reinstalling it forever, so every later migration
attempt fails identically. These tests pin the two behaviours that keep a
container recoverable: an incomplete cache is deleted before Prisma is
invoked, and the install gets a budget of its own rather than sharing the one
that bounds each migration command.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path

import pytest

from litellm_proxy_extras.prisma_toolchain import (
    DEFAULT_PRISMA_COMMAND_TIMEOUT,
    PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR,
    PRISMA_COMMAND_TIMEOUT_ENV_VAR,
    ensure_prisma_toolchain,
    heal_incomplete_nodeenv_cache,
    node_binary_path,
    prisma_bootstrap_timeout,
    prisma_command_timeout,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_EXTRAS = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"

FAKE_PRISMA = """#!{python}
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
cache_dir = os.environ["PRISMA_NODEENV_CACHE_DIR"]
with pathlib.Path(os.environ["FAKE_PRISMA_LOG"]).open("a") as log:
    log.write(
        json.dumps({{"args": args, "cache_dir_present": os.path.isdir(cache_dir)}})
        + "\\n"
    )
time.sleep(float(os.environ.get("FAKE_PRISMA_SLEEP", "0")))
if args[:2] == ["migrate", "deploy"]:
    print("No pending migrations to apply")
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
    return cache_dir, log_path


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


@pytest.mark.parametrize(
    "raw",
    ["", "0", "-5", "not-a-number", "nan", "inf", "-inf", "1e400"],
)
def test_unusable_timeout_override_falls_back_to_the_default(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite override would silently disable the timeout it configures."""
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, raw)

    assert prisma_command_timeout() == DEFAULT_PRISMA_COMMAND_TIMEOUT


def test_timeout_overrides_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PRISMA_COMMAND_TIMEOUT_ENV_VAR, "12")
    monkeypatch.setenv(PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, "900")

    assert prisma_command_timeout() == 12
    assert prisma_bootstrap_timeout() == 900


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
