from __future__ import annotations

import asyncio
import errno
import os
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Final

import httpx
import psutil
import psycopg
import pytest

from tests.integration._support import process as process_support
from tests.integration._support.async_server import LoopbackServer
from tests.integration._support.client import Gateway
from tests.integration._support.database_relay import (
    DatabaseRelay,
    DroppedConnectionRelay,
    HeldStatementRelay,
    database_relay,
    dropped_connection_relay,
    held_statement_relay,
)

Relay = DatabaseRelay | DroppedConnectionRelay | HeldStatementRelay
RelayFactory = Callable[[str, bytes], AbstractContextManager[tuple[Relay, str]]]


@pytest.mark.parametrize("factory", (database_relay, dropped_connection_relay, held_statement_relay))
def test_relay_closes_active_connections_and_releases_its_listener(factory: RelayFactory) -> None:
    with ExitStack() as cleanup:
        with factory(os.environ["DATABASE_URL"], b"unissued-lifecycle-trigger") as (relay, url):
            connection: Final = cleanup.enter_context(psycopg.connect(url, connect_timeout=2, autocommit=True))
            assert connection.execute("SELECT 1").fetchone() == (1,)
        with pytest.raises(psycopg.OperationalError):
            connection.execute("SELECT 1")
        with socket.socket() as replacement:
            replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            replacement.bind(("127.0.0.1", relay.port))
            replacement.listen()
        assert relay._loop.is_closed()


async def _unused_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.read()


def test_relay_bind_failure_closes_its_loop_and_allows_repeated_cleanup() -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        relay: Final = LoopbackServer(occupied.getsockname()[1], _unused_peer)
        with pytest.raises(OSError, match=os.strerror(errno.EADDRINUSE).lower()):
            relay.start()
        relay.stop()
        assert relay.loop.is_closed()


def test_upstream_readiness_failure_reaps_the_started_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(process_support, "_UPSTREAM_READY_SECONDS", 0)
    children_before: Final = frozenset(child.pid for child in psutil.Process().children(recursive=True))
    slot: Final = process_support.UpstreamSlot(tmp_path, process_support._free_port(), process_support._harness_root())
    try:
        with pytest.raises(AssertionError, match="readiness deadline exceeded"):
            slot.start()
        assert slot.process is None
        assert frozenset(child.pid for child in psutil.Process().children(recursive=True)) == children_before
    finally:
        if slot.process is not None:
            slot.stop()


@pytest.mark.parametrize("explicit_dotenv", (False, True))
def test_owned_process_ignores_ambient_dotenv_unless_its_contract_opts_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit_dotenv: bool
) -> None:
    monkeypatch.delenv("OWNED_DOTENV_SENTINEL", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    (tmp_path / ".env").write_text("OWNED_DOTENV_SENTINEL=ambient-config\n")
    with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
        environment: Final = process_support._proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"),
            {"PYTHON_DOTENV_DISABLED": "0"} if explicit_dotenv else {},
            (),
        )
    result: Final = subprocess.run(
        (
            sys.executable,
            "-P",
            "-c",
            "import os; from prisma import Prisma; Prisma(); print(os.getenv('OWNED_DOTENV_SENTINEL', 'absent'))",
        ),
        cwd=tmp_path,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == ("ambient-config" if explicit_dotenv else "absent")


def test_seeded_collection_reuses_scoped_fixtures_and_executes_every_case(tmp_path: Path) -> None:
    events: Final = tmp_path / "events.txt"
    probe: Final = tmp_path / "test_fixture_probe.py"
    probe.write_text(
        "import pytest\nfrom pathlib import Path\n"
        f"EVENTS = Path({str(events)!r})\n"
        "CONFIGS = ('D1',)*5 + ('D2',)*5 + ('D3',)*4 + ('D4',)*4\n"
        "def record(value):\n"
        "    with EVENTS.open('a') as output:\n"
        "        output.write(value + '\\n')\n"
        "@pytest.fixture(scope='module')\n"
        "def rig(request):\n"
        "    record('start ' + request.param)\n"
        "    yield request.param\n"
        "    record('stop ' + request.param)\n"
        "@pytest.mark.parametrize(('rig', 'number'), "
        "tuple((value, index) for index, value in enumerate(CONFIGS)), indirect=['rig'], scope='module')\n"
        "def test_route(rig, number):\n"
        "    assert rig == CONFIGS[number]\n"
        "    record('case ' + str(number))\n"
    )
    result: Final = subprocess.run(
        (
            sys.executable,
            "-P",
            "-m",
            "pytest",
            str(probe),
            "-p",
            "tests.integration.conftest",
            "--integration-order-seed=1848224111",
            "--timeout=15",
            "-q",
        ),
        env={
            name: value
            for name, value in os.environ.items()
            if name not in ("INTEGRATION_RESULTS_DIR", "INTEGRATION_ROUTING")
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed: Final = tuple(events.read_text().splitlines())
    configurations: Final = Counter({"D1": 1, "D2": 1, "D3": 1, "D4": 1})
    assert Counter(event.removeprefix("start ") for event in observed if event.startswith("start ")) == configurations
    assert Counter(event.removeprefix("stop ") for event in observed if event.startswith("stop ")) == configurations
    assert sorted(int(event.removeprefix("case ")) for event in observed if event.startswith("case ")) == list(
        range(18)
    )
