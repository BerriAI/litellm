from __future__ import annotations

import asyncio
import errno
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event
from types import FrameType
from typing import Final, Literal
from xml.etree import ElementTree

import httpx
import psutil
import psycopg
import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from tests.integration._support import process as process_support
from tests.integration._support.async_server import LoopbackServer
from tests.integration._support.budget import case_timeout_seconds
from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database_relay import (
    DatabaseRelay,
    DroppedConnectionRelay,
    HeldStatementRelay,
    database_relay,
    dropped_connection_relay,
    held_statement_relay,
)
from tests.integration._support.runtime import configure_executor
from tests.integration._support.sharding import ShardExecution, shard_errors
from tests.integration._support.wire import Reply, wire_server
from tests.integration.run import _without_provider_api_keys


class ExitRecord(TypedDict):
    root_pid: ReadOnly[int]
    exit_after_stop: ReadOnly[int | None]
    exception_at_record: ReadOnly[str | None]


Relay = DatabaseRelay | DroppedConnectionRelay | HeldStatementRelay
RelayFactory = Callable[[str, bytes], AbstractContextManager[tuple[Relay, str]]]
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
NESTED_RUN_SECONDS: Final = 60.0
WORKER_STACK_BOOT_SECONDS: Final = 30.0
OUTER_TEST_SLACK_SECONDS: Final = 30.0


def _nested_run_seconds(workers: int) -> float:
    return NESTED_RUN_SECONDS + (workers * WORKER_STACK_BOOT_SECONDS if workers > 1 else 0.0)


def _nested_run(*values: object, workers: int) -> object:
    return pytest.param(*values, marks=pytest.mark.timeout(_nested_run_seconds(workers) + OUTER_TEST_SLACK_SECONDS))


def _nested_pytest_environment(environment: Mapping[str, str]) -> dict[str, str]:
    inherited_run_settings: Final = frozenset(
        (
            "INTEGRATION_RESULTS_DIR",
            "INTEGRATION_ROUTING",
            "INTEGRATION_WORKER_ISOLATION",
            "INTEGRATION_COVERAGE_FILE",
            "INTEGRATION_CASE_SHARDS",
            "INTEGRATION_CASE_SHARD_INDEX",
            "CIRCLE_NODE_TOTAL",
            "CIRCLE_NODE_INDEX",
            "PYTEST_XDIST_WORKER",
            "PYTEST_XDIST_WORKER_COUNT",
            "PYTEST_XDIST_TESTRUNUID",
            "GITHUB_ACTIONS",
        )
    )
    return {name: value for name, value in environment.items() if name not in inherited_run_settings}


def test_runner_environment_drops_inherited_provider_api_keys() -> None:
    inherited: Final = {
        "OPENAI_API_KEY": "ambient-openai",
        "GEMINI_API_KEY": "ambient-gemini",
        "OTHER_API_KEY": "ambient-other",
        "LITELLM_MASTER_KEY": "sk-integration-master",
        "OPENAI_BASE_URL": "http://127.0.0.1:8190",
    }
    filtered: Final = _without_provider_api_keys(inherited)
    assert filtered == {
        "LITELLM_MASTER_KEY": "sk-integration-master",
        "OPENAI_BASE_URL": "http://127.0.0.1:8190",
    }
    assert inherited["OPENAI_API_KEY"] == "ambient-openai"


def test_owned_readiness_rejects_a_healthy_listener_from_another_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_PROXY_READY_SECONDS", "0")
    with wire_server(lambda request: Reply(body=b"healthy")) as foreign:
        with subprocess.Popen(
            (sys.executable, "-I", "-c", "import time; time.sleep(30)"), start_new_session=True
        ) as process:
            try:
                port: Final = int(foreign.url.rsplit(":", 1)[1])
                launch: Final = process_support._Launch(process, port, tmp_path / "proxy.log", time.monotonic())
                with pytest.raises(AssertionError, match="readiness deadline"):
                    process_support._wait_until_ready(launch)
                assert process.poll() is None
            finally:
                process_support.stop_child(process, grace_seconds=3)


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


class _InterruptedCleanup(BaseException):
    pass


async def test_owned_runtime_starts_the_entire_held_provider_burst_on_a_small_runner() -> None:
    arrived: Final[SimpleQueue[bool]] = SimpleQueue()
    complete: Final = Event()
    release: Final = Event()

    def held() -> None:
        arrived.put(True)
        if arrived.qsize() == 20:
            complete.set()
        assert release.wait(timeout=5), "Held executor job was never released"

    with ThreadPoolExecutor(max_workers=1) as initial:
        asyncio.get_running_loop().set_default_executor(initial)
        configure_executor()
        async with asyncio.TaskGroup() as tasks:
            burst: Final = tuple(tasks.create_task(asyncio.to_thread(held)) for _ in range(20))
            try:
                assert await asyncio.wait_for(asyncio.to_thread(complete.wait, 2), timeout=3)
                assert arrived.qsize() == len(burst)
            finally:
                release.set()


@pytest.mark.timeout(case_timeout_seconds())
def test_slow_worker_startup_survives_healthchecks_and_a_stalled_worker_is_replaced(
    gateway: Gateway, tmp_path: Path
) -> None:
    hook: Final = tmp_path / "slow_worker_startup.py"
    receipts: Final = tmp_path / "startup-receipts"
    receipts.mkdir()
    hook.write_text(
        "import os\nfrom pathlib import Path\nfrom threading import Event\n\ndef configure():\n"
        f"    root = Path({str(receipts)!r})\n"
        "    try:\n"
        "        with (root / 'first').open('x'):\n"
        "            pass\n"
        "    except FileExistsError:\n"
        "        Event().wait(6)\n"
        "    (root / str(os.getpid())).touch()\n"
    )
    overrides: Final = {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", ""))),
        "LITELLM_WORKER_STARTUP_HOOKS": "slow_worker_startup:configure",
    }
    with process_support.owned_proxy_process(gateway, tmp_path, overrides, workers=2) as owned:
        workers: Final = tuple(int(pid) for pid in STARTED_WORKER.findall(owned.log.read_text()))
        assert len(workers) == 2, owned.log.read_text()
        assert all((receipts / str(pid)).exists() for pid in workers), (
            "Proxy yielded before both workers finished startup"
        )
        assert owned.log.read_text().count("Application startup complete.") == 2
        replacement: Final = owned.stall_worker(workers[0])
        assert replacement not in workers
        assert tuple(int(pid) for pid in STARTED_WORKER.findall(owned.log.read_text()))[-1] == replacement
        assert owned.log.read_text().count("Application startup complete.") == 3
        assert not psutil.pid_exists(workers[0]), "Supervisor did not reap the stalled worker"
        response: Final = owned.gateway.request("GET", "/health/readiness")
        assert response.status_code == 200, response.text


def test_interrupted_cleanup_reaps_the_owned_root_and_its_child() -> None:
    child_code: Final = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    )
    parent_code: Final = (
        "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"child=subprocess.Popen((sys.executable,'-P','-c',{child_code!r}),stdout=subprocess.PIPE); "
        "assert child.stdout.readline()==b'ready\\n'; print('ready',flush=True); time.sleep(60)"
    )

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise _InterruptedCleanup

    with subprocess.Popen(
        (sys.executable, "-P", "-c", parent_code), start_new_session=True, stdout=subprocess.PIPE
    ) as process:
        try:
            assert process.stdout is not None
            assert process.stdout.readline() == b"ready\n"
            assert len(process_support.group_members(process.pid)) == 2
            previous_timer: Final = signal.getitimer(signal.ITIMER_REAL)
            started: Final = time.monotonic()
            previous_handler: Final = signal.signal(signal.SIGALRM, interrupt)
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.2)
                with pytest.raises(_InterruptedCleanup):
                    process_support._stop(process)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                remaining: Final = (
                    max(0.001, previous_timer[0] - (time.monotonic() - started)) if previous_timer[0] else 0
                )
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
            assert process.poll() is not None
            eventually(lambda: process_support.group_members(process.pid), lambda members: not members, seconds=5)
        finally:
            process_support.stop_process_group(process, 3)


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
        env=_nested_pytest_environment(os.environ),
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


@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("subset", (False, True))
def test_native_file_selection_reuses_fixtures_and_qualifies_an_empty_shard(
    tmp_path: Path, workers: int, subset: bool
) -> None:
    events: Final = tmp_path / "events.txt"
    root: Final = Path(__file__).resolve().parents[3]
    with TemporaryDirectory(prefix="native-selection-", dir=Path(__file__).parent) as directory:
        probe: Final = Path(directory) / "test_fixture_probe.py"
        plain: Final = Path(directory) / "test_plain_probe.py"
        probe.write_text(
            "import pytest\nfrom pathlib import Path\n"
            f"EVENTS = Path({str(events)!r})\n"
            "def record(value):\n"
            "    with EVENTS.open('a') as output:\n"
            "        output.write(value + '\\n')\n"
            "@pytest.fixture(scope='module')\n"
            "def shared():\n"
            "    record('start shared')\n"
            "    yield 'shared'\n"
            "    record('stop shared')\n"
            "@pytest.mark.parametrize('number', range(4))\n"
            "def test_shared(shared, number):\n"
            "    assert shared == 'shared'\n"
            "    record('shared ' + str(number))\n"
        )
        plain.write_text(
            "import pytest\nfrom pathlib import Path\n"
            f"EVENTS = Path({str(events)!r})\n"
            "@pytest.mark.parametrize('number', range(8))\n"
            "def test_plain(number):\n"
            "    with EVENTS.open('a') as output:\n"
            "        output.write('plain ' + str(number) + '\\n')\n"
            "def test_existing_skip():\n"
            "    pytest.skip('retained skip identity')\n"
        )
        selections: Final = (str(probe.relative_to(root)), str(plain.relative_to(root)), "")
        candidates: Final = (
            (f"{selections[0]}::test_shared[1]", f"{selections[1]}::test_existing_skip") if subset else selections[:2]
        )
        listed: Final = subprocess.run(
            (sys.executable, "-P", "tests/integration/run.py", "database", "--list", *candidates),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert listed.returncode == 0, listed.stdout + listed.stderr
        assert tuple(listed.stdout.splitlines()) == tuple(sorted(selections[:2]))
        outputs: Final = tuple(tmp_path / str(index) for index in range(3))
        files: Final = tuple(tmp_path / f"selection-{index}.txt" for index in range(3))
        for selection, path in zip(selections, files, strict=True):
            path.write_text(selection + "\n" if selection else "")
        results: Final = tuple(
            subprocess.run(
                (
                    sys.executable,
                    "-P",
                    "-m",
                    "pytest",
                    *candidates,
                    "--noconftest",
                    "-p",
                    "tests.integration.conftest",
                    "-o",
                    "addopts=",
                    "--integration-shard-count=3",
                    f"--integration-shard-index={index}",
                    f"--integration-shard-files={files[index]}",
                    "--integration-order-seed=1848224111",
                    "--hypothesis-seed=4106601",
                    "--timeout=15",
                    f"--junitxml={outputs[index] / 'junit.xml'}",
                    *(("-n", str(workers), "--dist=loadgroup") if workers > 1 else ()),
                    "-q",
                ),
                cwd=root,
                env={
                    **_nested_pytest_environment(os.environ),
                    "INTEGRATION_RESULTS_DIR": str(outputs[index]),
                    "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), str(root / "tests/e2e"))),
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            for index in range(3)
        )
        assert all(result.returncode == 0 for result in results), tuple(
            result.stdout + result.stderr for result in results
        )
        executions: Final = tuple(
            ShardExecution.model_validate_json((output / "execution.json").read_text()) for output in outputs
        )
        assert shard_errors(executions, 3) == ()
        for output, execution in zip(outputs, executions, strict=True):
            cases: Final = ElementTree.parse(output / "junit.xml").findall(".//testcase")
            assert sorted(case.attrib["name"] for case in cases) == sorted(
                nodeid.rsplit("::", 1)[-1] for nodeid in execution.collected
            )
            assert all("@integration_fixture_" not in nodeid for nodeid in execution.inventory)
        assert len(executions[0].inventory) == (2 if subset else 13)
        assert executions[2].collected == ()
        assert executions[2].complete
        assert len(executions[1].skipped) == 1
        assert executions[1].skipped[0].endswith("::test_existing_skip")
    observed: Final = tuple(events.read_text().splitlines())
    assert observed.count("start shared") == observed.count("stop shared") == 1
    assert sorted(event for event in observed if event.startswith("shared ")) == (
        ["shared 1"] if subset else [f"shared {index}" for index in range(4)]
    )
    assert sorted(event for event in observed if event.startswith("plain ")) == (
        [] if subset else [f"plain {index}" for index in range(8)]
    )


@pytest.mark.parametrize(
    ("workers", "seed"),
    tuple(_nested_run(workers, seed, workers=workers) for workers, seed in ((1, 1848224111), (2, 0), (2, 984872314))),
)
@pytest.mark.parametrize("class_parameters", (False, True))
def test_mixed_fixture_scopes_reuse_outer_and_inner_configurations(
    tmp_path: Path, workers: int, seed: int, class_parameters: bool
) -> None:
    events: Final = tmp_path / "events.txt"
    root: Final = Path(__file__).resolve().parents[3]
    with TemporaryDirectory(prefix="mixed-scopes-", dir=Path(__file__).parent) as directory:
        probe: Final = Path(directory) / "test_fixture_probe.py"
        probe.write_text(
            "import pytest\nfrom pathlib import Path\n"
            f"EVENTS = Path({str(events)!r})\n"
            "def record(value):\n"
            "    with EVENTS.open('a') as output:\n"
            "        output.write(value + '\\n')\n"
            "@pytest.fixture(scope='module', params=('M0', 'M1'))\n"
            "def module_rig(request):\n"
            "    record('start ' + request.param)\n"
            "    yield request.param\n"
            "    record('stop ' + request.param)\n"
            f"@pytest.fixture(scope='class', params={('C0', 'C1') if class_parameters else None!r})\n"
            "def class_rig(request, module_rig):\n"
            "    identity = (\n"
            "        module_rig + '/' + request.cls.__name__\n"
            "        + ('/' + request.param if hasattr(request, 'param') else '')\n"
            "    )\n"
            "    record('start ' + identity)\n"
            "    yield identity\n"
            "    record('stop ' + identity)\n"
            "class TestFirst:\n"
            "    @pytest.mark.parametrize('number', range(3))\n"
            "    def test_case(self, module_rig, class_rig, number):\n"
            "        assert class_rig.startswith(module_rig + '/')\n"
            "        record('case ' + class_rig + '/' + str(number))\n"
            "    def test_skip(self, class_rig):\n"
            "        pytest.skip('retained mixed-scope skip')\n"
            "class TestSecond:\n"
            "    @pytest.mark.parametrize('number', range(3))\n"
            "    def test_case(self, module_rig, class_rig, number):\n"
            "        assert class_rig.startswith(module_rig + '/')\n"
            "        record('case ' + class_rig + '/' + str(number))\n"
        )
        result: Final = subprocess.run(
            (
                sys.executable,
                "-P",
                "tests/integration/run.py",
                "database",
                "--results",
                str(tmp_path),
                "--workers",
                str(workers),
                "--order-seed",
                str(seed),
                str(probe.relative_to(root)),
            ),
            cwd=root,
            env=_nested_pytest_environment(os.environ),
            capture_output=True,
            text=True,
            timeout=_nested_run_seconds(workers),
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        execution: Final = ShardExecution.model_validate_json((tmp_path / "execution.json").read_text())
        assert execution.complete
        expected_passes: Final = 24 if class_parameters else 12
        expected_skips: Final = 4 if class_parameters else 2
        assert len(execution.collected) == expected_passes + expected_skips
        assert len(execution.passed) == expected_passes
        assert len(execution.skipped) == expected_skips
        cases: Final = ElementTree.parse(tmp_path / "junit.xml").findall(".//testcase")
        assert sorted(case.attrib["name"] for case in cases) == sorted(
            nodeid.rsplit("::", 1)[-1] for nodeid in execution.collected
        )
    observed: Final = tuple(events.read_text().splitlines())
    starts: Final = Counter(event.removeprefix("start ") for event in observed if event.startswith("start "))
    stops: Final = Counter(event.removeprefix("stop ") for event in observed if event.startswith("stop "))
    assert starts == stops
    assert len(starts) == (10 if class_parameters else 6)
    assert all(count == 1 for count in starts.values()), observed
    executed: Final = tuple(event for event in observed if event.startswith("case "))
    assert len(executed) == len(frozenset(executed)) == expected_passes


@pytest.mark.parametrize("workers", tuple(_nested_run(workers, workers=workers) for workers in (1, 2)))
def test_direct_and_dependent_consumers_reuse_the_same_module_parameter(tmp_path: Path, workers: int) -> None:
    source: Final = Path(__file__).resolve().parents[3]
    root: Final = tmp_path / "probe-root"
    for relative in (
        "pyproject.toml",
        "tests/__init__.py",
        "tests/_fake_openai_endpoint_server.py",
        "tests/integration/__init__.py",
        "tests/integration/conftest.py",
        "tests/integration/run.py",
        "tests/integration/proxy_config.yaml",
        "tests/integration/mcp_coverage.toml",
        "tests/integration/cost_calculation/cost_map.py",
        "tests/integration/cost_calculation/cost_tracking_case.py",
    ):
        destination: Final = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    shutil.copytree(
        source / "tests/integration/_support",
        root / "tests/integration/_support",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    events: Final = tmp_path / "events.txt"
    results: Final = tmp_path / "results"
    probe: Final = root / "tests/integration/database/test_fixture_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "import pytest\nfrom pathlib import Path\n"
        f"EVENTS = Path({str(events)!r})\n"
        "def record(value):\n"
        "    with EVENTS.open('a') as output:\n"
        "        output.write(value + '\\n')\n"
        "@pytest.fixture(scope='module', params=('A', 'B'))\n"
        "def outer(request):\n"
        "    record('start ' + request.param)\n"
        "    yield request.param\n"
        "    record('stop ' + request.param)\n"
        "@pytest.fixture(scope='module')\n"
        "def dependent(outer):\n"
        "    return outer\n"
        "@pytest.mark.parametrize('number', range(5))\n"
        "def test_parent(outer, number):\n"
        "    record(f'case {outer}-parent-{number}')\n"
        "@pytest.mark.parametrize('number', range(5))\n"
        "def test_child(outer, dependent, number):\n"
        "    assert dependent == outer\n"
        "    record(f'case {outer}-child-{number}')\n"
        "def test_retained_skip(outer):\n"
        "    pytest.skip('retained shared consumer skip')\n"
    )
    result: Final = subprocess.run(
        (
            sys.executable,
            "-P",
            str(root / "tests/integration/run.py"),
            "database",
            "--workers",
            str(workers),
            "--order-seed",
            "1",
            "--results",
            str(results),
            str(probe.relative_to(root)),
        ),
        cwd=root,
        env=_nested_pytest_environment(
            {
                **os.environ,
                "INTEGRATION_CASE_SHARDS": "2",
                "INTEGRATION_CASE_SHARD_INDEX": "1",
                "CIRCLE_NODE_TOTAL": "2",
                "CIRCLE_NODE_INDEX": "1",
            }
        ),
        capture_output=True,
        text=True,
        timeout=_nested_run_seconds(workers),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    execution: Final = ShardExecution.model_validate_json((results / "execution.json").read_bytes())
    assert execution.complete and len(execution.collected) == 22
    assert len(execution.passed) == 20 and len(execution.skipped) == 2
    assert all("::test_retained_skip[" in node for node in execution.skipped)
    junit: Final = ElementTree.parse(results / "junit.xml")
    assert len(junit.findall(".//testcase")) == 22 and len(junit.findall(".//skipped")) == 2
    assert all("@integration_fixture_" not in node.attrib["name"] for node in junit.findall(".//testcase"))
    observed: Final = tuple(events.read_text().splitlines())
    assert Counter(event for event in observed if event.startswith("start ")) == Counter({"start A": 1, "start B": 1})
    assert Counter(event for event in observed if event.startswith("stop ")) == Counter({"stop A": 1, "stop B": 1})
    executed: Final = tuple(event for event in observed if event.startswith("case "))
    assert len(executed) == len(frozenset(executed)) == 20


def test_worker_crash_keeps_the_original_junit_identity_and_an_incomplete_execution(tmp_path: Path) -> None:
    root: Final = Path(__file__).resolve().parents[3]
    with TemporaryDirectory(prefix="worker-crash-", dir=Path(__file__).parent) as directory:
        probe: Final = Path(directory) / "test_fixture_probe.py"
        probe.write_text(
            "import os, pytest\n"
            "@pytest.fixture(scope='module')\n"
            "def shared():\n"
            "    yield 'shared'\n"
            "def test_worker_crash(shared):\n"
            "    assert shared == 'shared'\n"
            "    os._exit(7)\n"
        )
        result: Final = subprocess.run(
            (
                sys.executable,
                "-P",
                "-m",
                "pytest",
                str(probe.relative_to(root)),
                "--noconftest",
                "-p",
                "tests.integration.conftest",
                "-o",
                "addopts=",
                "-n",
                "2",
                "--dist=loadgroup",
                "--max-worker-restart=0",
                f"--junitxml={tmp_path / 'junit.xml'}",
                "-q",
            ),
            cwd=root,
            env={
                **_nested_pytest_environment(os.environ),
                "INTEGRATION_RESULTS_DIR": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        execution: Final = ShardExecution.model_validate_json((tmp_path / "execution.json").read_text())
        assert not execution.complete
        assert execution.passed == ()
        assert execution.skipped == ()
        assert execution.collected == (str(probe.relative_to(root)) + "::test_worker_crash",)
        cases: Final = ElementTree.parse(tmp_path / "junit.xml").findall(".//testcase")
        assert tuple(case.attrib.get("name") for case in cases) == ("test_worker_crash",)
        assert cases[0].find("error") is not None


@pytest.mark.parametrize("assignment", ("absent", "missing", "foreign"))
def test_invalid_native_selections_fail_before_any_case_runs(
    tmp_path: Path, assignment: Literal["absent", "missing", "foreign"]
) -> None:
    root: Final = Path(__file__).resolve().parents[3]
    events: Final = tmp_path / "executed.txt"
    selection: Final = tmp_path / "node-files.txt"
    if assignment == "foreign":
        selection.write_text("tests/integration/database/test_unknown.py\n")
    with TemporaryDirectory(prefix="invalid-selection-", dir=Path(__file__).parent) as directory:
        probe: Final = Path(directory) / "test_probe.py"
        probe.write_text(f"from pathlib import Path\ndef test_case():\n    Path({str(events)!r}).write_text('ran')\n")
        result: Final = subprocess.run(
            (
                sys.executable,
                "-P",
                "-m",
                "pytest",
                str(probe.relative_to(root)),
                "--noconftest",
                "-p",
                "tests.integration.conftest",
                "-o",
                "addopts=",
                "--integration-shard-count=2",
                "--integration-shard-index=0",
                *((f"--integration-shard-files={selection}",) if assignment != "absent" else ()),
                "-q",
            ),
            cwd=root,
            env={
                **_nested_pytest_environment(os.environ),
                "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), str(root / "tests/e2e"))),
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
        assert not events.exists()


@pytest.mark.parametrize("entrypoint", ("gateway", "refused", "zero-workers"))
def test_owned_entrypoints_record_exit_and_reap_the_process_group(
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Literal["gateway", "refused", "zero-workers"],
) -> None:
    monkeypatch.setenv("INTEGRATION_RESULTS_DIR", str(tmp_path))
    config: Final = tmp_path / "config.yaml"
    if entrypoint == "refused":
        config.write_text("model_list: [\n")
        refused: Final = process_support.refused_boot_log(gateway, tmp_path, {}, config=config)
        assert refused
    else:
        rejected_config: Final = tmp_path / "rejected.yaml"
        rejected_config.write_text("model_list: [\n")
        assert process_support.refused_boot_log(gateway, tmp_path, {}, config=rejected_config)
        config.write_text(
            "model_list: []\ngeneral_settings:\n  master_key: os.environ/LITELLM_MASTER_KEY\n"
            "  database_url: os.environ/DATABASE_URL\n"
        )
        with process_support.owned_gateway_image(
            gateway, tmp_path, {}, config=config, workers=0 if entrypoint == "zero-workers" else 1
        ) as owned:
            response: Final = owned.gateway.request("GET", "/health/readiness")
            assert response.status_code == 200, response.text
        completed: Final = TypeAdapter(ExitRecord).validate_json(owned.log.with_suffix(".exit.json").read_text())
        assert completed["root_pid"] == owned.process.pid
        assert completed["exit_after_stop"] == owned.process.returncode
    records: Final = tuple(
        TypeAdapter(ExitRecord).validate_json(path.read_text()) for path in tmp_path.glob("*.exit.json")
    )
    assert records
    if entrypoint == "refused":
        assert len(records) == 1 and records[0]["exit_after_stop"] != 0
    for record in records:
        assert record["exit_after_stop"] is not None
        assert record["exception_at_record"] is None
        assert not process_support.group_members(record["root_pid"])
