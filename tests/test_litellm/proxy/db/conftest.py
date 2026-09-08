import json
import os
import signal
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

DB_ENV_KEYS = (
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
    "DATABASE_URL",
    "DIRECT_URL",
    "DATABASE_URL_READ_REPLICA",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_USER",
    "DATABASE_USERNAME",
    "DATABASE_NAME",
    "DATABASE_SCHEMA",
    "DATABASE_PASSWORD",
    "DATABASE_HOST_READ_REPLICA",
    "DATABASE_PORT_READ_REPLICA",
    "DATABASE_USER_READ_REPLICA",
    "DATABASE_USERNAME_READ_REPLICA",
    "DATABASE_NAME_READ_REPLICA",
    "DATABASE_SCHEMA_READ_REPLICA",
    "DATABASE_PASSWORD_READ_REPLICA",
)

_db_env_snapshot_key = pytest.StashKey[dict[str, Optional[str]]]()


def _db_env_snapshot() -> dict[str, Optional[str]]:
    return {key: os.environ.get(key) for key in DB_ENV_KEYS}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    item.stash[_db_env_snapshot_key] = _db_env_snapshot()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> Generator[None, None, None]:
    result = yield
    before = item.stash[_db_env_snapshot_key]
    leaked = {key: value for key, value in _db_env_snapshot().items() if value != before[key]}
    for key, original in before.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    assert not leaked, (
        f"{item.nodeid} leaked DB env vars past monkeypatch teardown: {leaked}. "
        "Product code under test writes DATABASE_URL(_READ_REPLICA) into os.environ as a side effect; "
        "monkeypatch only restores keys it has a record for, so a value written to a previously unset "
        "key survives the test and poisons every later test in this pytest-xdist worker process "
        "(DB-backed e2e tests arm themselves on DATABASE_URL and then fail to connect). "
        "Use the unset_database_url fixture (or monkeypatch.setenv) so restoration is registered."
    )
    return result


@pytest.fixture(autouse=True)
def reset_entra_token_provider_cache() -> Generator[None, None, None]:
    """The Entra provider factory is cached process-wide so one Azure credential serves
    the whole proxy; that cache would otherwise carry one test's stub into the next."""
    from litellm.proxy.db.token_auth import build_azure_entra_token_provider

    build_azure_entra_token_provider.cache_clear()
    yield
    build_azure_entra_token_provider.cache_clear()


@pytest.fixture
def unset_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "about-to-be-unset")
    monkeypatch.delenv("DATABASE_URL")


FAKE_PRISMA_CLI = """#!{python}
import json
import os
import pathlib
import subprocess
import sys
import time

calls_file = pathlib.Path(os.environ["FAKE_PRISMA_CALLS"])
earlier_calls = calls_file.read_text().splitlines() if calls_file.exists() else []
with calls_file.open("a") as log:
    print(json.dumps(sys.argv[1:]), file=log)
if not earlier_calls and os.environ.get("FAKE_PRISMA_HANG_FIRST"):
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    pathlib.Path(os.environ["FAKE_PRISMA_GRANDCHILD_PIDFILE"]).write_text(str(grandchild.pid))
    time.sleep(600)
sys.exit(0)
"""


@dataclass(frozen=True, slots=True)
class FakePrismaCli:
    """A stand-in `prisma` on PATH, recording every invocation.

    With FAKE_PRISMA_HANG_FIRST set it hangs on its first call from a process tree
    of its own, the way the real CLI wraps Node around a Rust schema engine, so a
    timeout that kills only the direct child leaves the rest of that tree running.
    """

    calls_file: Path
    grandchild_pidfile: Path

    @property
    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def grandchild_is_gone(self, within_seconds: float) -> bool:
        deadline = time.monotonic() + within_seconds
        while time.monotonic() < deadline:
            try:
                os.kill(int(self.grandchild_pidfile.read_text()), 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False


@pytest.fixture
def fake_prisma_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[FakePrismaCli, None, None]:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "prisma"
    script.write_text(FAKE_PRISMA_CLI.format(python=sys.executable))
    script.chmod(0o755)
    cli = FakePrismaCli(
        calls_file=tmp_path / "calls.jsonl",
        grandchild_pidfile=tmp_path / "grandchild.pid",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_PRISMA_CALLS", str(cli.calls_file))
    monkeypatch.setenv("FAKE_PRISMA_GRANDCHILD_PIDFILE", str(cli.grandchild_pidfile))
    monkeypatch.setenv("LITELLM_PRISMA_COMMAND_TIMEOUT", "1")
    monkeypatch.delenv("FAKE_PRISMA_HANG_FIRST", raising=False)
    yield cli
    if cli.grandchild_pidfile.exists():
        try:
            os.kill(int(cli.grandchild_pidfile.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
