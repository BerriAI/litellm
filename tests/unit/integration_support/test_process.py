from __future__ import annotations

import importlib
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final
from unittest.mock import MagicMock

import httpx
import psutil
import pytest
import yaml
from pydantic import TypeAdapter

from litellm.proxy.types_utils.utils import get_instance_fn
from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import (
    OWNED_STATE_VARIABLE,
    OwnedProxy,
    OwnedState,
    PrivateState,
    WorkerHealthcheck,
    WorkerKill,
    _is_gone,
    _proxy_environment,
    boots_private_state,
    healthcheck_seconds,
    kill_stray_process,
    owned_environment,
    owned_proxy_process,
    private_configuration,
    replacement_pid,
    stop_process_group,
    worker_configuration,
)

TESTS_DIR: Final = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("config_contents", ("[]\n", "general_settings: []\n", "general_settings: {}\n"))
def test_worker_configuration_preserves_sibling_auth_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_contents: str,
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    monkeypatch.setenv("INTEGRATION_PROVIDER_PORT", "8203")
    monkeypatch.setenv("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8202")
    source: Final = tmp_path / "source.yaml"
    source.write_text(config_contents)
    auth_state: Final = tmp_path / "sibling_auth.json"
    auth_state.write_text("authenticated")
    (tmp_path / "sibling_auth.py").write_text(
        "from pathlib import Path\n"
        "def authenticate():\n"
        "    return Path(__file__).with_suffix('.json').read_text()\n"
    )

    configured: Final = worker_configuration(source, tmp_path / "worker")

    assert get_instance_fn("sibling_auth.authenticate", config_file_path=str(configured))() == "authenticated"
    auth_state.write_text("updated")
    assert get_instance_fn("sibling_auth.authenticate", config_file_path=str(configured))() == "updated"


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


def test_worker_configuration_keeps_modules_beside_a_test_config_importable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    source: Final = tmp_path / "proxy.yaml"
    source.write_text(yaml.safe_dump({"general_settings": {"custom_auth": "sibling_auth.authenticate"}}))
    (tmp_path / "sibling_auth.py").write_text("def authenticate() -> str:\n    return 'sibling'\n")
    configured: Final = worker_configuration(source, tmp_path / "gateway")
    assert configured != source
    assert get_instance_fn("sibling_auth.authenticate", config_file_path=str(configured))() == "sibling"


def test_worker_configuration_writes_the_repository_config_copy_into_the_worker_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    monkeypatch.setenv("INTEGRATION_PROVIDER_PORT", "8203")
    monkeypatch.setenv("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8202")
    monkeypatch.setenv("INTEGRATION_PROXY_ROOT", str(tmp_path / "product"))
    configured: Final = worker_configuration(TESTS_DIR / "integration/proxy_config.yaml", tmp_path / "worker")
    assert configured.parent == tmp_path / "worker"


def test_worker_configuration_writes_the_proxy_checkout_config_copy_into_the_worker_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    checkout: Final = tmp_path / "checkout"
    monkeypatch.setenv("INTEGRATION_PROXY_ROOT", str(checkout))
    source: Final = checkout / "tests/integration/proxy_config.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("model_list: []\n")
    configured: Final = worker_configuration(source, tmp_path / "worker")
    assert configured.parent == tmp_path / "worker"


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


def test_readiness_never_sends_a_request_to_a_port_the_launch_does_not_own(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(psutil, "process_iter", lambda: iter(()))
    parent: Final = MagicMock()
    parent.pid = 99
    parent.poll.return_value = None
    launch: Final = process_module._Launch(parent, 8123, tmp_path / "unused.log", 0.0)
    original_client: Final = httpx.Client

    def foreign_listener(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"probed a listener another process owns: {request.url}")

    def client(*, base_url: str, timeout: float, trust_env: bool, limits: httpx.Limits) -> httpx.Client:
        return original_client(
            base_url=base_url,
            timeout=timeout,
            trust_env=trust_env,
            limits=limits,
            transport=httpx.MockTransport(foreign_listener),
        )

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setenv("INTEGRATION_PROXY_READY_SECONDS", "-1")
    with pytest.raises(AssertionError, match="readiness deadline"):
        process_module._wait_until_ready(launch)


def test_boot_phases_mark_when_prisma_ran_the_port_bound_and_each_worker_answered(
    process_module: ModuleType,
) -> None:
    probe: Final = process_module._Probe
    timeline: Final = (
        probe(0.5, listening=False, prisma=False, ready_pid=None),
        probe(4.0, listening=False, prisma=True, ready_pid=None),
        probe(5.0, listening=False, prisma=False, ready_pid=None),
        probe(5.5, listening=False, prisma=True, ready_pid=None),
        probe(6.5, listening=False, prisma=False, ready_pid=None),
        probe(7.0, listening=True, prisma=False, ready_pid=None),
        probe(10.0, listening=True, prisma=False, ready_pid=41),
        probe(11.0, listening=True, prisma=False, ready_pid=42),
    )
    phases: Final = reduce(process_module._observe, timeline, process_module.BootPhases())
    assert phases == process_module.BootPhases(
        prisma_started=4.0,
        prisma_finished=6.5,
        listening=7.0,
        first_ready=10.0,
        ready_pids=frozenset((41, 42)),
    )


def test_boot_phases_leave_prisma_unset_when_no_prisma_child_was_seen(process_module: ModuleType) -> None:
    probe: Final = process_module._Probe
    timeline: Final = (
        probe(1.0, listening=False, prisma=False, ready_pid=None),
        probe(2.0, listening=True, prisma=False, ready_pid=7),
    )
    phases: Final = reduce(process_module._observe, timeline, process_module.BootPhases())
    assert (phases.prisma_started, phases.prisma_finished, phases.listening, phases.first_ready) == (
        None,
        None,
        2.0,
        2.0,
    )


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


WORKER_STATE: Final = MappingProxyType(
    {"DATABASE_URL": "postgresql://admin@127.0.0.1/worker", "REDIS_HOST": "127.0.0.1", "REDIS_PORT": "6379"}
)
PRIVATE_DATABASE: Final = "postgresql://admin@127.0.0.1/owned_private"


@contextmanager
def _private_state(directory: Path, inherited: Mapping[str, str]) -> Iterator[PrivateState]:
    yield PrivateState(
        PRIVATE_DATABASE,
        MappingProxyType({"DATABASE_URL": PRIVATE_DATABASE, "REDIS_HOST": "127.0.0.2", "REDIS_PORT": "6390"}),
    )


@pytest.mark.parametrize(
    ("state", "overrides", "remove_environment", "marked", "private"),
    (
        (None, {}, (), False, True),
        (None, {"DATABASE_URL": "postgresql://admin@127.0.0.1/scratch"}, (), False, False),
        (None, {}, ("DATABASE_URL",), False, False),
        (OwnedState.SHARED_WITH_WORKER, {}, (), False, False),
        (None, {}, (), True, False),
        (OwnedState.PRIVATE, {}, (), True, True),
    ),
    ids=("default", "explicit-database", "stripped-database", "shared", "marked-module", "private-in-marked-module"),
)
def test_an_owned_proxy_boots_private_state_unless_it_shares_or_brings_its_own_database(
    state: OwnedState | None,
    overrides: Mapping[str, str],
    remove_environment: tuple[str, ...],
    marked: bool,
    private: bool,
) -> None:
    marker: Final = {OWNED_STATE_VARIABLE: OwnedState.SHARED_WITH_WORKER.value} if marked else {}
    inherited: Final = {**WORKER_STATE, **marker}
    assert boots_private_state(state, overrides, remove_environment, inherited) is private


def test_a_default_owned_proxy_runs_on_its_own_database_and_redis(tmp_path: Path) -> None:
    with (
        httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client,
        owned_environment(tmp_path, {}, (), None, WORKER_STATE, _private_state) as owned,
    ):
        environment: Final = _proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"), owned.overrides, (), WORKER_STATE
        )
    assert environment["DATABASE_URL"] == PRIVATE_DATABASE != WORKER_STATE["DATABASE_URL"]
    assert (environment["REDIS_HOST"], environment["REDIS_PORT"]) == ("127.0.0.2", "6390")
    assert owned.database_url == PRIVATE_DATABASE


@pytest.mark.parametrize(
    ("overrides", "remove_environment", "redis"),
    (
        ({}, ("REDIS_HOST", "REDIS_PORT"), (None, None)),
        ({"REDIS_HOST": "redis.caller", "REDIS_PORT": "7000"}, (), ("redis.caller", "7000")),
    ),
    ids=("stripped", "caller-owned"),
)
def test_a_private_owned_proxy_leaves_redis_to_a_caller_that_chose_it(
    tmp_path: Path,
    overrides: Mapping[str, str],
    remove_environment: tuple[str, ...],
    redis: tuple[str | None, str | None],
) -> None:
    with (
        httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client,
        owned_environment(tmp_path, overrides, remove_environment, None, WORKER_STATE, _private_state) as owned,
    ):
        environment: Final = _proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"), owned.overrides, remove_environment, WORKER_STATE
        )
    assert environment["DATABASE_URL"] == PRIVATE_DATABASE
    assert (environment.get("REDIS_HOST"), environment.get("REDIS_PORT")) == redis


@pytest.mark.parametrize(
    ("router_settings", "retries"),
    (({"allowed_fails": 3}, 0), ({"num_retries": 2}, 2), (None, 0)),
    ids=("inherits-worker-default", "keeps-explicit", "no-router-settings"),
)
def test_a_private_owned_proxy_keeps_the_worker_retry_default_unless_its_config_sets_one(
    tmp_path: Path, router_settings: dict[str, int] | None, retries: int
) -> None:
    source: Final = tmp_path / "source.yaml"
    router: Final = {"router_settings": router_settings} if router_settings else {}
    source.write_text(yaml.safe_dump({"model_list": [], **router}))
    configured: Final = yaml.safe_load(private_configuration(source, tmp_path / "proxy").read_text())
    assert configured["router_settings"]["num_retries"] == retries
    assert configured["model_list"] == []


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


@contextmanager
def _unreaped_child() -> Iterator[psutil.Process]:
    with subprocess.Popen((sys.executable, "-I", "-c", "import time; time.sleep(60)")) as child:
        try:
            yield psutil.Process(child.pid)
        finally:
            child.kill()


def test_a_worker_kill_returns_once_the_victim_is_dead_before_anything_reaps_it_and_cannot_fire_twice() -> None:
    with _unreaped_child() as victim:
        kill: Final = WorkerKill(victim, threading.Event())
        kill()
        assert victim.status() == psutil.STATUS_ZOMBIE
        assert kill.fired.is_set()
        with pytest.raises(AssertionError, match="already killed"):
            kill()


def test_a_frozen_victim_stays_stopped_until_the_kill() -> None:
    with _unreaped_child() as victim:
        kill: Final = WorkerKill(victim, threading.Event())
        assert kill.freeze().status() == psutil.STATUS_STOPPED
        kill()
        assert victim.status() == psutil.STATUS_ZOMBIE


_LEADER_EXITS_FIRST: Final = """import ctypes
import socket
import threading

listener = socket.create_server(("127.0.0.1", 0))


def hold() -> None:
    connection, _ = listener.accept()
    connection.recv(1)


threading.Thread(target=hold).start()
print(listener.getsockname()[1], flush=True)
ctypes.CDLL(None).pthread_exit(None)
"""


def _closed_by_peer(connection: socket.socket) -> bool:
    return bool(select.select((connection,), (), (), 0)[0])


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux keeps a zombie thread-group leader's other threads")
def test_a_process_is_not_gone_while_a_thread_outliving_its_leader_still_holds_its_sockets() -> None:
    with subprocess.Popen((sys.executable, "-I", "-c", _LEADER_EXITS_FIRST), stdout=subprocess.PIPE) as child:
        try:
            assert child.stdout is not None
            port: Final = int(child.stdout.readline())
            with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
                leader: Final = psutil.Process(child.pid)
                eventually(leader.status, lambda status: status == psutil.STATUS_ZOMBIE, seconds=10)
                assert not _closed_by_peer(connection)
                assert not _is_gone(leader)
                connection.sendall(b"x")
                eventually(lambda: _is_gone(leader), bool, seconds=10)
                assert _closed_by_peer(connection)
        finally:
            child.kill()


def _kill_after_freezing_the_supervisor(owned: OwnedProxy, worker: int, supervisor: psutil.Process) -> None:
    with owned.replacing_worker(worker) as kill:
        supervisor.suspend()
        kill()


def test_a_replacing_block_fails_when_the_supervisor_never_reaps_the_killed_worker(tmp_path: Path) -> None:
    with _supervised(tmp_path, WorkerHealthcheck.STRICT) as (owned, worker):
        unreaping: Final = psutil.Process(owned.process.pid)
        try:
            with pytest.raises(AssertionError, match="did not reap"):
                _kill_after_freezing_the_supervisor(owned, worker, unreaping)
        finally:
            unreaping.resume()


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
