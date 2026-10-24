import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from queue import SimpleQueue
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(300)

_REMOTE_ONLY_MODEL: Final = "zz-remote-only-sdk-model"
_CHILD_SECONDS: Final = 120

_SCRIPT: Final = textwrap.dedent(
    """
    import os, sys, threading, time
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info

    variant, model = sys.argv[1:3]

    def join_fetch():
        for thread in threading.enumerate():
            if thread.name == "litellm-model-cost-map-fetch":
                thread.join(timeout=60)

    info = get_model_cost_map_source_info()
    print(info["source"], info["fallback_reason"], model in litellm.model_cost, flush=True)
    if variant == "adopt":
        sys.stdin.readline()
        join_fetch()
        print(get_model_cost_map_source_info()["source"], model in litellm.model_cost, flush=True)
    elif variant == "get_llm_provider":
        print(litellm.get_llm_provider(model)[1], flush=True)
    elif variant == "get_model_info":
        print(litellm.get_model_info(model, custom_llm_provider="openai")["litellm_provider"], flush=True)
    elif variant == "fork":
        pid = os.fork()
        if pid == 0:
            join_fetch()
            os._exit(0 if model in litellm.model_cost else 1)
        print(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), flush=True)
    elif variant == "raw_fork":
        import ctypes, signal
        pid = ctypes.PyDLL(None).fork()
        if pid == 0:
            signal.alarm(60)
            try:
                print(litellm.cost_per_token(model="gpt-4o", prompt_tokens=1000, completion_tokens=0)[0], flush=True)
                os._exit(0)
            finally:
                os._exit(1)
        print(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), flush=True)
    elif variant == "forks_during_retry_wait":
        while get_model_cost_map_source_info()["fallback_reason"] == "Remote fetch in progress":
            time.sleep(0.01)
        children = []
        for _ in range(6):
            pid = os.fork()
            if pid == 0:
                join_fetch()
                os._exit(0 if model in litellm.model_cost else 1)
            children.append(pid)
        join_fetch()
        print(*(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) for pid in children), flush=True)
    else:
        raise SystemExit(f"unknown variant {variant}")
    """
)


def _bundled_map() -> dict:
    return json.loads(files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text())


def _remote_gpt_4o_input_cost() -> float:
    return 3 * _bundled_map()["gpt-4o"]["input_cost_per_token"]


def _remote_map() -> bytes:
    bundled: Final = _bundled_map()
    return json.dumps(
        {
            **bundled,
            "gpt-4o": {**bundled["gpt-4o"], "input_cost_per_token": _remote_gpt_4o_input_cost()},
            _REMOTE_ONLY_MODEL: {"litellm_provider": "openai", "mode": "chat"},
        }
    ).encode()


@dataclass(frozen=True, slots=True)
class _CostMapHost:
    wire: Wire
    first_request_answered: threading.Event


@pytest.fixture
def cost_map_host() -> Iterator[_CostMapHost]:
    body: Final = _remote_map()
    gate: Final = threading.Event()
    first_seen: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if not first_seen.is_set():
            first_seen.set()
            gate.wait(timeout=_CHILD_SECONDS)
        return Reply(body=body)

    with wire_server(respond) as wire:
        try:
            yield _CostMapHost(wire=wire, first_request_answered=gate)
        finally:
            gate.set()


@dataclass(frozen=True, slots=True)
class _Child:
    after_import: str
    rest: tuple[str, ...]


def _run_child(host: _CostMapHost, variant: str, tmp_path: Path, answer_after_import: bool) -> _Child:
    stderr_path: Final = tmp_path / "stderr.log"
    with (
        stderr_path.open("w") as stderr,
        subprocess.Popen(
            [sys.executable, "-P", "-c", _SCRIPT, variant, _REMOTE_ONLY_MODEL],
            env={
                **os.environ,
                "LITELLM_LOCAL_MODEL_COST_MAP": "False",
                "NO_PROXY": "*",
                "LITELLM_MODEL_COST_MAP_URL": f"{host.wire.url}/model_prices_and_context_window.json",
            },
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        ) as child,
    ):
        assert child.stdout is not None
        after_import: Final = child.stdout.readline().strip()
        if answer_after_import:
            host.first_request_answered.set()
        rest: Final = child.communicate(input="\n", timeout=_CHILD_SECONDS)[0]
    assert child.returncode == 0, stderr_path.read_text()
    return _Child(after_import=after_import, rest=tuple(rest.split()))


def test_import_returns_before_the_remote_map_arrives_and_adopts_it_after(
    cost_map_host: _CostMapHost, tmp_path: Path
) -> None:
    child: Final = _run_child(cost_map_host, "adopt", tmp_path, answer_after_import=True)

    assert child.after_import == "local Remote fetch in progress False"
    assert child.rest == ("remote", "True")


@pytest.mark.parametrize("lookup", ["get_llm_provider", "get_model_info"])
def test_a_lookup_right_after_import_finds_a_model_only_the_remote_map_has(
    cost_map_host: _CostMapHost, tmp_path: Path, lookup: str
) -> None:
    child: Final = _run_child(cost_map_host, lookup, tmp_path, answer_after_import=True)

    assert child.after_import == "local Remote fetch in progress False"
    assert child.rest == ("openai",)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_worker_forked_while_the_remote_map_is_in_flight_fetches_it_itself(
    cost_map_host: _CostMapHost, tmp_path: Path
) -> None:
    child: Final = _run_child(cost_map_host, "fork", tmp_path, answer_after_import=False)

    assert child.after_import == "local Remote fetch in progress False"
    assert child.rest == ("0",)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_worker_forked_without_the_fork_hooks_while_the_remote_map_is_in_flight_prices_from_it(
    cost_map_host: _CostMapHost, tmp_path: Path
) -> None:
    child: Final = _run_child(cost_map_host, "raw_fork", tmp_path, answer_after_import=False)

    assert child.after_import == "local Remote fetch in progress False"
    assert float(child.rest[0]) == pytest.approx(1000 * _remote_gpt_4o_input_cost())
    assert child.rest[1:] == ("0",)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_workers_forked_during_a_rate_limited_retry_wait_spread_their_retries(tmp_path: Path) -> None:
    body: Final = _remote_map()
    arrivals: Final[SimpleQueue[float]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        arrivals.put(time.monotonic())
        if arrivals.qsize() == 1:
            return Reply(status=429, headers={"Retry-After": "2"})
        return Reply(body=body)

    with wire_server(respond) as wire:
        child: Final = _run_child(
            _CostMapHost(wire=wire, first_request_answered=threading.Event()),
            "forks_during_retry_wait",
            tmp_path,
            answer_after_import=False,
        )
    retries: Final = sorted(tuple(arrivals.get_nowait() for _ in range(arrivals.qsize()))[1:])

    assert child.rest == ("0",) * 6
    assert len(retries) == 7
    assert retries[5] - retries[0] > 0.1
