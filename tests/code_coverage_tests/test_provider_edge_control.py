from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from capture_policy import SCENARIO_BYTES, ScenarioIdentity, canonical_scenario_id
from capture_session import CaptureResult
from capture_store import StoreFailure
from fixture_bundle import (
    BundleRecorder,
    LoadedBundle,
    RecordedHttpResponse,
    RecordedRequest,
    load_bundle,
    prepare_bundle,
)
from provider_edge import REPLAY_MISS_STATUS, _persist, start_provider_edge
from provider_edge_control import ControlRequest, ControlServer, EdgeController
from provider_edge_remote import RemoteEdge
from test_provider_capture import ScriptedReservations, completion_payload, fake_provider, successful_response
from test_provider_edge import CHAT_PATH, call_edge, provider_url


@contextmanager
def remote_controller(controller: EdgeController, mounts: dict[str, str]) -> Generator[RemoteEdge, None, None]:
    running: Final = start_provider_edge(
        controller.backend({}), mounts=mounts, guard=controller, observation=controller
    )
    control: Final = ControlServer(controller)
    thread: Final = threading.Thread(target=control.serve_forever, daemon=True)
    thread.start()
    try:
        yield RemoteEdge(f"http://127.0.0.1:{control.server_port}", f"http://127.0.0.1:{running.edge.port}")
    finally:
        control.shutdown()
        control.server_close()
        running.shutdown()
        thread.join(timeout=5)


class TestTrustedRemoteEdge:
    def test_capture_and_replay_keep_lifecycle_and_observed_counts(self, tmp_path: Path) -> None:
        identity: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")
        node: Final = canonical_scenario_id(identity.node)
        recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        controller: Final = EdgeController(
            {node: identity}, "owner", 200, store=ScriptedReservations(), recorder=recorder
        )
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider)}
            with remote_controller(controller, mounts) as remote:
                denied: Final = call_edge(remote.edge, "POST", CHAT_PATH, body=b"{}")
                assert denied.status_code == REPLAY_MISS_STATUS
                remote.begin(node)
                remote.phase(node, "setup", True)
                observer: Final = remote.command(ControlRequest(action="observe", node=node, marker="blue"))
                assert observer.observation_id is not None
                captured: Final = call_edge(
                    remote.edge,
                    "POST",
                    CHAT_PATH,
                    body=b'{"prompt":"blue"}',
                    headers={"content-type": "application/json"},
                )
                assert captured.status_code == 200
                observed: Final = remote.command(
                    ControlRequest(action="count", node=node, observation_id=observer.observation_id)
                )
                assert observed.count == 1
                assert call_edge(remote.edge, "POST", "/control", body=b"{}").status_code == 404
                remote.phase(node, "call", True)
                remote.phase(node, "teardown", True)
                assert len(controller.results) == 1 and controller.results[0].publishable
                with pytest.raises(RuntimeError, match="already started"):
                    remote.begin(node)
            bundle: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(bundle, LoadedBundle)
            replay: Final = EdgeController({node: identity}, "replay", 200, replay_bundle=bundle)
            with remote_controller(replay, mounts) as remote:
                remote.begin("/opt/runner/" + node)
                remote.phase(node, "setup", True)
                response: Final = call_edge(
                    remote.edge,
                    "POST",
                    CHAT_PATH,
                    body=b'{"prompt":"blue"}',
                    headers={"content-type": "application/json"},
                )
                assert response.status_code == 200 and response.body == captured.body
                remote.phase(node, "call", True)
                remote.phase(node, "teardown", True)
            assert len(provider.hits) == 1

    def test_finalizer_failure_is_a_failed_remote_outcome(self, tmp_path: Path) -> None:
        identity: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")
        node: Final = canonical_scenario_id(identity.node)
        recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        controller: Final = EdgeController(
            {node: identity}, "owner", 200, store=ScriptedReservations(), recorder=recorder
        )
        with fake_provider() as provider, remote_controller(controller, {"openai": provider_url(provider)}) as remote:
            remote.begin(node)
            remote.phase(node, "setup", True)
            assert (
                call_edge(
                    remote.edge,
                    "POST",
                    CHAT_PATH,
                    body=b'{"prompt":"blue"}',
                    headers={"content-type": "application/json"},
                ).status_code
                == 200
            )
            remote.phase(node, "call", True)
            with pytest.raises(RuntimeError, match="teardown must all pass"):
                remote.phase(node, "teardown", False)
            assert not controller.results[0].publishable
            assert call_edge(remote.edge, "POST", CHAT_PATH, body=b"{}").status_code == REPLAY_MISS_STATUS
            assert len(provider.hits) == 1

    @pytest.mark.parametrize(
        "control",
        ["http://0.0.0.0:8081", "http://10.1.0.2:8081", "https://127.0.0.1:8081", "http://user@127.0.0.1:8081"],
    )
    def test_driver_cannot_address_non_loopback_management(self, control: str) -> None:
        with pytest.raises(ValueError, match="loopback"):
            RemoteEdge(control, "http://10.1.0.2:8080")


@pytest.mark.parametrize("fault", ["duplicate", "release_denied", "release_uncertain"])
def test_rejected_lifecycle_cannot_export_success(tmp_path: Path, fault: str) -> None:
    class Store(ScriptedReservations):
        def release(self, *, scenario_key: str, owner: str) -> StoreFailure | None:
            return (
                None
                if fault == "duplicate"
                else StoreFailure("lease release failed", uncertain=fault == "release_uncertain")
            )

    identity: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")
    node: Final = canonical_scenario_id(identity.node)
    recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
    assert isinstance(recorder, BundleRecorder)
    saved: Final = []
    controller: Final = EdgeController(
        {node: identity}, "owner", 200, store=Store(), recorder=recorder, outcome_sink=saved.append
    )
    assert controller.command(ControlRequest(action="begin", node=node)).ok
    assert controller.command(ControlRequest(action="phase", node=node, phase="setup", passed=True)).ok
    assert controller.before_attempt() is None
    controller.response_finished(successful_response())
    if fault == "duplicate":
        assert not controller.command(ControlRequest(action="phase", node=node, phase="setup", passed=True)).ok
    controller.command(ControlRequest(action="phase", node=node, phase="call", passed=True))
    assert not controller.command(ControlRequest(action="phase", node=node, phase="teardown", passed=True)).ok
    assert not controller.command(ControlRequest(action="status")).ok
    assert saved and not saved[0][0].publishable


@pytest.mark.parametrize("fail_finalizer, fail_call", [(False, False), (True, False), (False, True)])
def test_real_pytest_hooks_persist_finalizer_outcome(tmp_path: Path, fail_finalizer: bool, fail_call: bool) -> None:
    harness: Final = Path(__file__).resolve().parents[1] / "e2e"
    suite: Final = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_bytes((harness / "conftest.py").read_bytes())
    (suite / "test_synthetic.py").write_text("""import os
import pytest
import requests
from e2e_config import unique_marker

@pytest.fixture
def resource():
    yield
    assert os.environ["FAIL_FINALIZER"] == "0", "synthetic finalizer failure"

@pytest.mark.e2e
def test_one(resource):
    assert unique_marker() == os.environ["EXPECTED_MARKER"]
    response = requests.post(os.environ["EDGE_DATA"] + "/openai/v1/chat/completions", json={"prompt":"blue"}, timeout=5)
    assert response.status_code == 200
    assert os.environ["FAIL_CALL"] == "0", "synthetic body failure"
""")
    identity: Final = ScenarioIdentity("test_synthetic.py::test_one", "a" * 64, "synthetic")
    node: Final = canonical_scenario_id(identity.node)
    recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
    assert isinstance(recorder, BundleRecorder)
    saved: Final = []
    controller: Final = EdgeController(
        {node: identity}, "owner", 200, store=ScriptedReservations(), recorder=recorder, outcome_sink=saved.append
    )
    with fake_provider() as provider, remote_controller(controller, {"openai": provider_url(provider)}) as remote:
        environment: Final = {
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(harness),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "E2E_FIXTURE_MODE": " Record ",
            "E2E_PROVIDER_EDGE_CONTROL_URL": remote.control_url,
            "E2E_PROVIDER_EDGE_DATA_URL": f"http://127.0.0.1:{remote.edge.port}",
            "EDGE_DATA": f"http://127.0.0.1:{remote.edge.port}",
            "LITELLM_PROXY_URL": provider_url(provider),
            "FAIL_FINALIZER": "1" if fail_finalizer else "0",
            "FAIL_CALL": "1" if fail_call else "0",
            "EXPECTED_MARKER": hashlib.sha1(f"{node}#0".encode()).hexdigest()[:12],
        }
        result: Final = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "test_synthetic.py"],
            cwd=suite,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == (1 if fail_finalizer or fail_call else 0), result.stdout + result.stderr
    assert len(saved) == 1 and len(saved[0]) == 1
    outcome: Final = saved[0][0]
    assert outcome.outcome.setup
    assert outcome.outcome.call is not fail_call
    assert outcome.outcome.teardown is not fail_finalizer
    assert outcome.publishable == (not fail_finalizer and not fail_call)


def test_aggregate_recording_limit_stops_before_second_disk_write(tmp_path: Path) -> None:
    identity: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")
    node: Final = canonical_scenario_id(identity.node)
    recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
    assert isinstance(recorder, BundleRecorder)
    controller: Final = EdgeController({node: identity}, "owner", 200, store=ScriptedReservations(), recorder=recorder)
    backend: Final = controller.backend({})
    from provider_edge import RecordEdge

    assert isinstance(backend, RecordEdge)
    request: Final = RecordedRequest(method="post", path="/openai/v1/chat/completions", headers={})
    response: Final = RecordedHttpResponse(
        status_code=200,
        headers={},
        body_b64=base64.b64encode(completion_payload("x" * (3 * 1024 * 1024))).decode(),
    )
    assert controller.command(ControlRequest(action="begin", node=node)).ok
    assert controller.command(ControlRequest(action="phase", node=node, phase="setup", passed=True)).ok
    assert controller.before_attempt() is None
    _persist(backend, node, request, response)
    assert 0 < controller.remaining_bytes() < SCENARIO_BYTES // 2
    assert controller.before_attempt() is None
    with pytest.raises(OSError, match="capture byte limit"):
        _persist(backend, node, request, response)
    controller.command(ControlRequest(action="phase", node=node, phase="call", passed=True))
    assert not controller.command(ControlRequest(action="phase", node=node, phase="teardown", passed=True)).ok
    assert not controller.results[0].publishable
    assert sum(path.stat().st_size for path in recorder.root.rglob("*.json")) < SCENARIO_BYTES


def test_inflight_teardown_drains_before_releasing_lease(tmp_path: Path) -> None:
    events: list[str] = []

    class Store(ScriptedReservations):
        def complete(self, *, scenario_key: str, owner: str, attempt_id: str, successful: bool) -> str | None:
            events.append("settled")
            return None

        def release(self, *, scenario_key: str, owner: str) -> StoreFailure | None:
            events.append("released")
            return None

    identity = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")
    node = canonical_scenario_id(identity.node)
    recorder = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
    assert isinstance(recorder, BundleRecorder)
    saved: list[tuple[CaptureResult, ...]] = []
    controller = EdgeController(
        {node: identity}, "owner", 200, store=Store(), recorder=recorder, outcome_sink=saved.append
    )
    assert controller.command(ControlRequest(action="begin", node=node)).ok
    assert controller.command(ControlRequest(action="phase", node=node, phase="setup", passed=True)).ok
    assert controller.begin_request() is None
    assert controller.before_attempt() is None
    assert controller.command(ControlRequest(action="phase", node=node, phase="call", passed=True)).ok
    ended = controller.command(ControlRequest(action="phase", node=node, phase="teardown", passed=True))
    assert not ended.ok and ended.error == "scenario ended with in-flight provider requests"
    assert events == [] and saved == []
    assert controller.begin_request() == "no active trusted scenario"
    assert controller.test_key() == node
    controller.response_finished(successful_response())
    controller.end_request()
    assert events == ["settled", "released"]
    assert len(saved) == 1 and not saved[0][0].publishable
    assert saved[0][0].error == "scenario lifecycle was rejected"
    assert controller.command(ControlRequest(action="status")).count == 1
    assert not controller.command(ControlRequest(action="begin", node=node)).ok
