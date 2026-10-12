from __future__ import annotations

import asyncio
import errno
import importlib
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Final
from unittest.mock import MagicMock

import httpx
import psutil
import pytest
import yaml
from pydantic import TypeAdapter

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import (
    OwnedProxy,
    WorkerHealthcheck,
    WorkerKill,
    healthcheck_seconds,
    kill_stray_process,
    owned_proxy_process,
    replacement_pid,
    stop_process_group,
    worker_configuration,
)

TESTS_DIR: Final = Path(__file__).resolve().parents[2]
UNRELATED_CRASH: Final = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'litellm'\n"


def test_worker_configuration_routes_stock_edges_to_its_own_servers_and_preserves_other_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    monkeypatch.setenv("INTEGRATION_PROVIDER_PORT", "8203")
    monkeypatch.setenv("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8202")
    source: Final = tmp_path / "source.yaml"
    original: Final = (
        'api_base: http://127.0.0.1:8191/v1\ntoken_url: "http://127.0.0.1:8190/_oauth/token"\n'
        "peer: http://127.0.0.1:81910/v1\ncustom: http://127.0.0.1:9000/v1\n"
    )
    source.write_text(original)
    configured: Final = worker_configuration(source, tmp_path / "worker")
    assert yaml.safe_load(configured.read_text()) == {
        "api_base": "http://127.0.0.1:8203/v1",
        "token_url": "http://127.0.0.1:8202/_oauth/token",
        "peer": "http://127.0.0.1:81910/v1",
        "custom": "http://127.0.0.1:9000/v1",
        "general_settings": {"database_connection_pool_limit": 2},
    }
    assert source.read_text() == original


@pytest.mark.parametrize("limit", (1, 5, 0, None, "", "bad"))
def test_worker_configuration_preserves_explicit_pool_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int | str | None
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    source: Final = tmp_path / "source.yaml"
    original: Final = {
        "general_settings": {"database_connection_pool_limit": limit, "database_connection_timeout": 3},
        "model_list": [{"model_name": "synthetic", "litellm_params": {"model": "openai/synthetic"}}],
    }
    source.write_text(yaml.safe_dump(original))
    configured: Final = worker_configuration(source, tmp_path / "worker")
    assert yaml.safe_load(configured.read_text()) == original
    assert yaml.safe_load(source.read_text()) == original


def test_worker_configuration_budgets_implicit_pools_without_changing_connection_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    source: Final = tmp_path / "source.yaml"
    original: Final = {"general_settings": {"database_extra_connection_params": {"connection_limit": "1"}}}
    source.write_text(yaml.safe_dump(original))
    configured: Final = worker_configuration(source, tmp_path / "worker")
    decoded: Final = TypeAdapter(dict[str, object]).validate_python(yaml.safe_load(configured.read_text()))
    assert decoded == {
        "general_settings": {
            "database_connection_pool_limit": 2,
            "database_extra_connection_params": {"connection_limit": "1"},
        }
    }
    assert yaml.safe_load(source.read_text()) == original


def test_serial_proxy_keeps_the_requested_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("INTEGRATION_WORKER_ISOLATION", raising=False)
    source: Final = tmp_path / "source.yaml"
    source.write_text("api_base: http://127.0.0.1:8191/v1\n")
    assert worker_configuration(source, tmp_path / "unused") == source
    assert source.read_text() == "api_base: http://127.0.0.1:8191/v1\n"


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    return importlib.import_module("integration._support.process")


async def _refused_bind(port: int) -> OSError | None:
    try:
        await asyncio.get_running_loop().create_server(asyncio.Protocol, "127.0.0.1", port)
    except OSError as refused:
        return refused
    return None


@pytest.fixture
def bind_error_line() -> Iterator[str]:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        refused: Final = asyncio.run(_refused_bind(held.getsockname()[1]))
        assert refused is not None and refused.errno == errno.EADDRINUSE
        yield f"ERROR:    {refused}\n"


def _written_log(directory: Path, text: str) -> Path:
    log: Final = directory / "owned-proxy.log"
    log.write_text(text)
    return log


def test_lost_port_race_matches_the_bind_error_the_server_logs(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert process_module._lost_port_race(1, _written_log(tmp_path, bind_error_line))


def test_lost_port_race_ignores_an_exit_for_another_reason(process_module: ModuleType, tmp_path: Path) -> None:
    assert not process_module._lost_port_race(1, _written_log(tmp_path, UNRELATED_CRASH))


def test_lost_port_race_needs_the_process_to_have_exited(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert not process_module._lost_port_race(None, _written_log(tmp_path, bind_error_line))


@pytest.mark.parametrize("failure", (psutil.AccessDenied, psutil.NoSuchProcess))
def test_an_unreadable_process_cannot_establish_listener_ownership(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: type[psutil.Error]
) -> None:
    member: Final = MagicMock(spec=psutil.Process)
    member.pid = os.getpid()
    member.net_connections.side_effect = failure(member.pid)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter((member,)))
    parent: Final = MagicMock()
    parent.pid = os.getpgid(member.pid)
    launch: Final = process_module._Launch(parent, 8123, tmp_path / "unused.log", 0.0)
    assert not process_module._owns_listener(launch)


def test_an_unreadable_member_does_not_hide_another_owned_listener(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unreadable: Final = MagicMock(spec=psutil.Process)
    unreadable.pid = 1
    unreadable.net_connections.side_effect = psutil.AccessDenied(1)
    readable: Final = MagicMock(spec=psutil.Process)
    readable.pid = 2
    listener: Final = MagicMock()
    listener.status = psutil.CONN_LISTEN
    listener.laddr.port = 8123
    readable.net_connections.return_value = (listener,)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter((unreadable, readable)))
    monkeypatch.setattr(os, "getpgid", lambda pid: 99)
    parent: Final = MagicMock()
    parent.pid = 99
    launch: Final = process_module._Launch(parent, 8123, tmp_path / "unused.log", 0.0)
    assert process_module._owns_listener(launch)


@pytest.mark.parametrize(
    ("status", "ready"),
    (
        (psutil.STATUS_RUNNING, True),
        (psutil.STATUS_SLEEPING, True),
        (psutil.STATUS_ZOMBIE, False),
        (psutil.STATUS_DEAD, False),
        (psutil.STATUS_STOPPED, False),
        (psutil.STATUS_TRACING_STOP, False),
        (psutil.AccessDenied, False),
        (psutil.NoSuchProcess, False),
    ),
)
def test_only_live_readable_workers_can_satisfy_readiness(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str | type[psutil.Error], ready: bool
) -> None:
    member: Final = MagicMock(spec=psutil.Process)
    member.pid = os.getpid()
    if isinstance(status, str):
        member.status.return_value = status
    else:
        member.status.side_effect = status(member.pid)
    listener: Final = MagicMock()
    listener.status = psutil.CONN_LISTEN
    listener.laddr.port = 8123
    member.net_connections.return_value = (listener,)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter((member,)))
    parent: Final = MagicMock()
    parent.pid = os.getpgid(member.pid)
    parent.poll.return_value = None
    launch: Final = process_module._Launch(parent, 8123, tmp_path / "unused.log", 0.0)
    original_client: Final = httpx.Client

    def client(*, base_url: str, timeout: float, trust_env: bool, limits: httpx.Limits) -> httpx.Client:
        return original_client(
            base_url=base_url,
            timeout=timeout,
            trust_env=trust_env,
            limits=limits,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, headers={"x-integration-worker-pid": str(member.pid)})
            ),
        )

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setenv("INTEGRATION_PROXY_READY_SECONDS", "-1")
    if ready:
        process_module._wait_until_ready(launch)
    else:
        with pytest.raises(AssertionError, match="readiness deadline"):
            process_module._wait_until_ready(launch)


@pytest.mark.parametrize(
    ("writer", "explicit_reader", "expected_reader"),
    (
        ("postgresql://writer@127.0.0.1/shared", None, "postgresql://reader@127.0.0.1/shared"),
        ("postgresql://writer@127.0.0.1/scratch", None, None),
        (
            "postgresql://writer@127.0.0.1/scratch",
            "postgresql://reader@127.0.0.1/scratch",
            "postgresql://reader@127.0.0.1/scratch",
        ),
    ),
)
def test_owned_writer_keeps_only_a_reader_paired_with_its_database(
    process_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    explicit_reader: str | None,
    expected_reader: str | None,
) -> None:
    monkeypatch.setenv("INTEGRATION_PROXY_READ_REPLICA_URL", "postgresql://reader@127.0.0.1/shared")
    overrides: Final = {
        "DATABASE_URL": writer,
        **({"DATABASE_URL_READ_REPLICA": explicit_reader} if explicit_reader else {}),
    }
    with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
        environment: Final = process_module._proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"), overrides, ()
        )
    assert environment["DATABASE_URL"] == writer
    assert environment.get("DATABASE_URL_READ_REPLICA") == expected_reader


@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("remove", (False, True))
def test_owned_runtime_configuration_preserves_custom_worker_startup_hooks(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, explicit: bool, remove: bool
) -> None:
    monkeypatch.setenv("LITELLM_WORKER_STARTUP_HOOKS", "ambient:configure")
    with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
        environment: Final = process_module._proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"),
            {"LITELLM_WORKER_STARTUP_HOOKS": "custom:configure"} if explicit else {},
            ("LITELLM_WORKER_STARTUP_HOOKS",) if remove else (),
        )
    expected: Final = "custom:configure" if explicit else ("" if remove else "ambient:configure")
    assert environment["LITELLM_WORKER_STARTUP_HOOKS"] == ",".join(
        hook
        for hook in (
            "integration._support.runtime:configure_executor",
            "integration._support.runtime:configure_readiness",
            expected,
        )
        if hook
    )


_SUPERVISOR: Final = """import multiprocessing
import time

if __name__ == "__main__":
    worker = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    worker.start()
    print(worker.pid, flush=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        multiprocessing.active_children()
        time.sleep(0.05)
"""


@contextmanager
def _supervised(tmp_path: Path, healthcheck: WorkerHealthcheck) -> Iterator[tuple[OwnedProxy, int]]:
    script: Final = tmp_path / "supervisor.py"
    script.write_text(_SUPERVISOR)
    with subprocess.Popen(
        (sys.executable, "-P", str(script)), stdout=subprocess.PIPE, start_new_session=True
    ) as supervisor:
        try:
            assert supervisor.stdout is not None
            worker: Final = int(supervisor.stdout.readline())
            with httpx.Client(base_url="http://127.0.0.1:9") as client:
                owned: Final = OwnedProxy(
                    Gateway(client, "sk-unit", "http://127.0.0.1:9"), supervisor, tmp_path / "proxy.log", healthcheck
                )
                eventually(owned.worker_pids, lambda pids: pids == frozenset((worker,)), seconds=30)
                yield owned, worker
        finally:
            stop_process_group(supervisor, 5)


def test_worker_pids_are_the_supervisors_spawned_workers_and_not_the_supervisor(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        assert owned.worker_pids() == frozenset((worker,))
        assert owned.process.pid not in owned.worker_pids()


def test_only_a_serving_worker_of_this_proxy_can_be_killed(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        with pytest.raises(AssertionError, match="not a serving worker"):
            owned.kill_worker(owned.process.pid)
        assert owned.process.poll() is None
        assert psutil.Process(worker).is_running()


def _leave_without_killing(owned: OwnedProxy, worker: int) -> None:
    with owned.replacing_worker(worker):
        pass


def test_a_replacing_block_that_never_kills_its_worker_fails_and_leaves_it_serving(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        with pytest.raises(AssertionError, match="without killing the worker"):
            _leave_without_killing(owned, worker)
        assert owned.worker_pids() == frozenset((worker,))


def _kill_of(owned: OwnedProxy, worker: int) -> WorkerKill:
    return WorkerKill(psutil.Process(worker), psutil.Process(owned.process.pid), 5, threading.Event())


def test_a_worker_kill_returns_once_the_supervisor_reaped_the_victim_and_cannot_fire_twice(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        kill: Final = _kill_of(owned, worker)
        kill()
        assert not psutil.pid_exists(worker)
        assert kill.fired.is_set()
        with pytest.raises(AssertionError, match="already killed"):
            kill()


def test_a_frozen_victim_stays_stopped_until_the_kill(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        kill: Final = _kill_of(owned, worker)
        assert kill.freeze().status() == psutil.STATUS_STOPPED
        kill()
        assert not psutil.pid_exists(worker)


def test_a_kill_under_a_suspended_supervisor_returns_with_the_victim_left_unreaped(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.LENIENT) as (owned, worker):
        kill: Final = _kill_of(owned, worker)
        with owned.suspended(owned.process.pid):
            kill()
            assert psutil.Process(worker).status() == psutil.STATUS_ZOMBIE
        eventually(lambda: psutil.pid_exists(worker), lambda exists: not exists, seconds=5)


def test_a_kill_fails_when_the_supervisor_never_reaps_the_victim(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        unreaping: Final = psutil.Process(owned.process.pid)
        kill: Final = WorkerKill(psutil.Process(worker), psutil.Process(os.getpid()), 0.5, threading.Event())
        unreaping.suspend()
        try:
            with pytest.raises(AssertionError, match="did not reap"):
                kill()
        finally:
            unreaping.resume()
        assert not kill.fired.is_set()


def test_a_strict_healthcheck_proxy_refuses_suspension_before_stopping_anything(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        with pytest.raises(AssertionError, match="LENIENT"):
            with owned.suspended(worker):
                pass
        assert psutil.Process(worker).status() != psutil.STATUS_STOPPED


def test_a_lenient_healthcheck_proxy_suspends_for_the_block_and_resumes_after(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.LENIENT) as (owned, worker):
        with owned.suspended(worker) as suspended:
            assert suspended.status() == psutil.STATUS_STOPPED
        assert psutil.Process(worker).status() != psutil.STATUS_STOPPED
        assert owned.worker_pids() == frozenset((worker,))


def test_a_worker_killed_while_suspended_fails_the_block(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.LENIENT) as (owned, worker):
        with pytest.raises(AssertionError, match="died while suspended"):
            with owned.suspended(worker):
                kill_stray_process(worker)


def test_the_worker_healthcheck_cannot_be_overridden_through_extra_arguments(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="healthcheck"):
        with owned_proxy_process(
            MagicMock(spec=Gateway), tmp_path, {}, extra_arguments=("--timeout_worker_healthcheck", "60")
        ):
            pass
    assert not tuple(tmp_path.glob("owned-proxy-*.log"))


def test_a_lenient_healthcheck_outlasts_a_strict_one() -> None:
    assert healthcheck_seconds(WorkerHealthcheck.LENIENT) > healthcheck_seconds(WorkerHealthcheck.STRICT)


@pytest.mark.parametrize(
    ("answered", "serving", "replacement"),
    (
        (30, frozenset((20, 30)), 30),
        (20, frozenset((20, 30)), None),
        (10, frozenset((20, 30)), None),
        (40, frozenset((20, 30)), None),
        (None, frozenset((20, 30)), None),
    ),
)
def test_only_a_new_serving_worker_answering_readiness_counts_as_the_replacement(
    answered: int | None, serving: frozenset[int], replacement: int | None
) -> None:
    assert replacement_pid(frozenset((10, 20)), answered, serving) == replacement
