from __future__ import annotations

import base64
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from capture_policy import ScenarioIdentity, ScenarioOutcome, canonical_scenario_id, publication_error
from capture_session import AttemptDenied, AttemptReserved, AttemptUncertain, CaptureSession, Reservation
from capture_snapshot import (
    CaptureProvenance,
    ScenarioSnapshot,
    SnapshotFailure,
    build_snapshot,
    materialize_snapshot,
    refresh_due,
    verify_snapshot,
)
from capture_store import StoreFailure
from fixture_bundle import (
    BundleRecorder,
    LoadedBundle,
    RecordedHttpResponse,
    RecordedRequest,
    RecordedStreamedResponse,
    load_bundle,
    prepare_bundle,
)
from fixture_mode import current_test_key
from provider_edge import REPLAY_MISS_STATUS, RecordEdge, ReplayEdge, ReplaySource, _persist
from test_provider_edge import CHAT_PATH, call_edge, fake_provider, provider_url, running_edge


@dataclass
class ScriptedReservations:
    cap: int = 12
    uncertain: bool = False
    calls: tuple[str, ...] = field(default=(), init=False)

    def reserve(self, *, scenario_key: str, owner: str, attempt_id: str) -> Reservation:
        self.calls = (*self.calls, attempt_id)
        if self.uncertain:
            return AttemptUncertain("reservation outcome uncertain")
        if len(self.calls) > self.cap:
            return AttemptDenied("durable budget exhausted")
        return AttemptReserved(attempt_id)

    def complete(self, *, scenario_key: str, owner: str, attempt_id: str, successful: bool) -> str | None:
        return None

    def acquire(self, *, scenario_key: str, owner: str, expires_at: int) -> StoreFailure | None:
        return None

    def release(self, *, scenario_key: str, owner: str) -> StoreFailure | None:
        return None


def successful_response() -> RecordedHttpResponse:
    return RecordedHttpResponse(status_code=200, headers={}, body_b64=base64.b64encode(b'{"answer":"blue"}').decode())


class TestBoundedCapture:
    def test_uncertain_reservation_stops_all_following_attempts(self) -> None:
        store: Final = ScriptedReservations(uncertain=True)
        session: Final = CaptureSession(ScenarioIdentity("test_example.py::test_one", "a" * 64, "test"), "owner", store)
        assert session.before_attempt() == "reservation outcome uncertain"
        assert session.before_attempt() == "reservation outcome uncertain"
        assert len(store.calls) == 1
        assert not session.finish(ScenarioOutcome(True, True, True)).publishable

    def test_cap_plus_one_is_denied_before_http(self, tmp_path: Path) -> None:
        store: Final = ScriptedReservations(cap=1)
        scenario: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "test")
        session: Final = CaptureSession(scenario, "owner", store, max_attempts=1)
        recorder: Final = prepare_bundle(tmp_path / "bounded", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        backend: Final = RecordEdge(
            recorder,
            threading.Lock(),
            test_key=lambda: scenario.node,
            before_attempt=session.before_attempt,
            response_finished=session.response_finished,
            response_byte_limit=1024,
        )
        with fake_provider() as provider:
            with running_edge(backend, {"openai": provider_url(provider)}) as edge:
                first: Final = call_edge(
                    edge, "POST", CHAT_PATH, body=b'{"prompt":"blue"}', headers={"content-type": "application/json"}
                )
                assert first.status_code == 200
                second: Final = call_edge(
                    edge, "POST", CHAT_PATH, body=b'{"prompt":"blue"}', headers={"content-type": "application/json"}
                )
                assert second.status_code == REPLAY_MISS_STATUS
                assert b"attempt cap" in second.body
            assert len(provider.hits) == 1
        assert len(store.calls) == 1
        assert not session.finish(ScenarioOutcome(True, True, True)).publishable

    def test_failure_is_not_recovered_by_a_second_attempt(self) -> None:
        store: Final = ScriptedReservations()
        session: Final = CaptureSession(ScenarioIdentity("test_example.py::test_one", "a" * 64, "test"), "owner", store)
        assert session.before_attempt() is None
        session.response_finished(RecordedHttpResponse(status_code=503, headers={}, body_b64=""))
        assert session.before_attempt() == "provider response was not successful"
        assert len(store.calls) == 1
        assert not session.finish(ScenarioOutcome(True, True, True)).publishable

    def test_finalizer_failure_and_uncertain_inflight_outcome_cannot_publish(self) -> None:
        session: Final = CaptureSession(
            ScenarioIdentity("test_example.py::test_one", "a" * 64, "test"), "owner", ScriptedReservations()
        )
        assert session.before_attempt() is None
        assert session.finish(ScenarioOutcome(True, True, True)).error == "outbound outcome is uncertain"
        assert session.before_attempt() == "scenario is closed"
        session.response_finished(successful_response())
        assert not session.finish(ScenarioOutcome(True, True, True)).publishable

    def test_success_requires_complete_trusted_outcome(self) -> None:
        for teardown, expected in ((False, False), (True, True)):
            session: Final = CaptureSession(
                ScenarioIdentity("test_example.py::test_one", "a" * 64, "test"), "owner", ScriptedReservations()
            )
            assert session.before_attempt() is None
            session.response_finished(successful_response())
            result: Final = session.finish(ScenarioOutcome(True, True, teardown))
            assert result.publishable is expected
            assert result.response_count == 1
            assert len(result.attempts) == 1


class TestScenarioIdentity:
    def test_test_roots_normalize_without_changing_parameters(self) -> None:
        expected: Final = "tests/e2e/router/test_cache.py::TestCache::test_hit[prompt/2031-04-05]"
        assert canonical_scenario_id(expected) == expected
        assert canonical_scenario_id("/opt/runner/" + expected) == expected
        assert canonical_scenario_id(expected.removeprefix("tests/e2e/")) == expected
        assert canonical_scenario_id(expected.replace("2031-04-05", "2031-04-06")) != expected

    @pytest.mark.parametrize("node", ["", "test_cache.py", "../test_cache.py::test_hit", "x/../test.py::test_hit"])
    def test_invalid_scenario_ids_are_rejected(self, node: str) -> None:
        with pytest.raises(ValueError):
            canonical_scenario_id(node)

    def test_contract_and_profile_change_identity_but_candidate_revision_does_not_key_it(self) -> None:
        node: Final = "router/test_cache.py::test_hit"
        identity: Final = ScenarioIdentity(node, "a" * 64, "openai-test-v1")
        assert identity.key == ScenarioIdentity("/opt/runner/tests/e2e/" + node, "a" * 64, "openai-test-v1").key
        assert identity.key != ScenarioIdentity(node, "b" * 64, "openai-test-v1").key
        assert identity.key != ScenarioIdentity(node, "a" * 64, "openai-test-v2").key

    def test_explicit_record_and_replay_identity_does_not_use_pytest_process(self, tmp_path: Path) -> None:
        scenario: Final = "tests/e2e/router/test_cache.py::test_hit"
        recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock(), test_key=lambda: scenario), mounts) as edge:
                captured: Final = call_edge(
                    edge,
                    "POST",
                    CHAT_PATH,
                    body=b'{"prompt":"synthetic"}',
                    headers={"content-type": "application/json"},
                )
                assert captured.status_code == 200
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            wrong: Final = ReplaySource(loaded, test_key=current_test_key)
            with running_edge(ReplayEdge(wrong), mounts) as edge:
                assert (
                    call_edge(
                        edge,
                        "POST",
                        CHAT_PATH,
                        body=b'{"prompt":"synthetic"}',
                        headers={"content-type": "application/json"},
                    ).status_code
                    == REPLAY_MISS_STATUS
                )
            source: Final = ReplaySource(loaded, test_key=lambda: scenario)
            with running_edge(ReplayEdge(source), mounts) as edge:
                replayed: Final = call_edge(
                    edge,
                    "POST",
                    CHAT_PATH,
                    body=b'{"prompt":"synthetic"}',
                    headers={"content-type": "application/json"},
                )
                assert replayed.status_code == 200
                assert replayed.body == captured.body
            assert source.leftover_error(scenario) is None
            assert len(provider.hits) == 1


class TestPublicationPolicy:
    @pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
    def test_every_trusted_phase_must_pass(self, phase: str) -> None:
        outcome: Final = ScenarioOutcome(setup=phase != "setup", call=phase != "call", teardown=phase != "teardown")
        assert publication_error(outcome, ()) is not None

    def test_empty_or_failed_capture_cannot_be_published(self) -> None:
        outcome: Final = ScenarioOutcome(setup=True, call=True, teardown=True)
        assert publication_error(outcome, ()) == "capture contains no interactions"
        failed: Final = RecordedHttpResponse(status_code=503, headers={}, body_b64=base64.b64encode(b"failed").decode())
        assert publication_error(outcome, (failed,)) == "provider response was not successful"

    @pytest.mark.parametrize(
        "bad",
        [b'{"error":"failed"}', b'{"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":9}}', b"not-json"],
    )
    def test_wrong_success_bodies_do_not_publish(self, bad: bytes) -> None:
        response: Final = RecordedHttpResponse(status_code=200, headers={}, body_b64=base64.b64encode(bad).decode())
        assert publication_error(ScenarioOutcome(True, True, True), (response,)) is not None

    @pytest.mark.parametrize("tail", [b"", b'data: {"error":"failed"}\n\n', b"data: [DONE]\n"])
    def test_incomplete_or_error_stream_does_not_publish(self, tail: bytes) -> None:
        data: Final = b'data: {"choices":[{"delta":{"content":"blue"}}]}\n\n' + tail
        response: Final = RecordedStreamedResponse(
            status_code=200, headers={}, chunks_b64=[base64.b64encode(data).decode()]
        )
        assert publication_error(ScenarioOutcome(True, True, True), (response,)) is not None

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_complete_stream_across_arbitrary_chunk_boundaries_publishes(self, provider: str) -> None:
        payload: Final = (
            b': keepalive\ndata: {"choices":[{"index":0,"delta":{"content":"blue"},"finish_reason":null}],"usage":null}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\ndata: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\ndata: [DONE]\n\n'
            if provider == "openai"
            else b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":2,"output_tokens":0}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"blue"}}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        for split in range(1, len(payload)):
            response: Final = RecordedStreamedResponse(
                status_code=200,
                headers={},
                chunks_b64=[base64.b64encode(part).decode() for part in (payload[:split], payload[split:])],
            )
            assert publication_error(ScenarioOutcome(True, True, True), (response,)) is None

    @pytest.mark.parametrize(
        "payload",
        [
            b': heartbeat\ndata: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
            b"data: [DONE]\n\n",
            b': heartbeat\ndata: {"error":{"message":"failed"}}\n\ndata: {"choices":[{"delta":{"content":"blue"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\ndata: {"choices":[{"index":0,"delta":{"content":"late"},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
            b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
            b'data: [DONE]\n\ndata: {"type":"message_stop"}\n\n',
            b'data: {"choices":null}\n\ndata: [DONE]\n\n',
            b'data: {"choices":[{"finish_reason":"length"}]}\n\ndata: [DONE]\n\n',
        ],
    )
    def test_malformed_completion_settles_attempt_as_failure(self, payload: bytes) -> None:
        response: Final = RecordedStreamedResponse(
            status_code=200, headers={}, chunks_b64=[base64.b64encode(payload).decode()]
        )
        session: Final = CaptureSession(
            ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic"), "owner", ScriptedReservations()
        )
        assert session.before_attempt() is None
        session.response_finished(response)
        result: Final = session.finish(ScenarioOutcome(True, True, True))
        assert not result.publishable
        assert result.error != "outbound outcome is uncertain"


class TestSnapshotBoundary:
    def test_capture_download_fresh_directory_replay_and_faults(self, tmp_path: Path) -> None:
        identity: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic-v1")
        session: Final = CaptureSession(identity, "owner", ScriptedReservations())
        recorder: Final = prepare_bundle(tmp_path / "capture", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider)}
            backend: Final = RecordEdge(
                recorder,
                threading.Lock(),
                test_key=lambda: canonical_scenario_id(identity.node),
                before_attempt=session.before_attempt,
                response_finished=session.response_finished,
            )
            with running_edge(backend, mounts) as edge:
                response: Final = call_edge(
                    edge, "POST", CHAT_PATH, body=b'{"prompt":"blue"}', headers={"content-type": "application/json"}
                )
                assert response.status_code == 200
            bundle: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(bundle, LoadedBundle)
            result: Final = session.finish(ScenarioOutcome(True, True, True))
            snapshot: Final = build_snapshot(
                result,
                bundle,
                CaptureProvenance(
                    test_revision="b" * 40, candidate_revision="c" * 40, runner_digest="sha256:" + "d" * 64
                ),
            )
            assert isinstance(snapshot, bytes)
            digest: Final = hashlib.sha256(snapshot).hexdigest()
            now: Final = datetime.now(timezone.utc)
            verified: Final = verify_snapshot(snapshot, expected_sha256=digest, identity=identity, now=now)
            assert isinstance(verified, ScenarioSnapshot)
            assert not refresh_due(verified, now=now)
            assert refresh_due(verified, now=now + timedelta(days=1))
            assert isinstance(
                verify_snapshot(snapshot + b" ", expected_sha256=digest, identity=identity, now=now), SnapshotFailure
            )
            assert isinstance(
                verify_snapshot(snapshot, expected_sha256=digest, identity=identity, now=now + timedelta(days=7)),
                SnapshotFailure,
            )
            assert isinstance(
                verify_snapshot(
                    snapshot,
                    expected_sha256=digest,
                    identity=ScenarioIdentity(identity.node, "f" * 64, "synthetic-v1"),
                    now=now,
                ),
                SnapshotFailure,
            )
            materialize_snapshot(verified, tmp_path / "fresh")
            with pytest.raises(FileExistsError):
                materialize_snapshot(verified, tmp_path / "fresh")
            loaded: Final = load_bundle(tmp_path / "fresh", profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            source: Final = ReplaySource(loaded, test_key=lambda: canonical_scenario_id(identity.node))
            with running_edge(ReplayEdge(source), mounts) as edge:
                replayed: Final = call_edge(
                    edge, "POST", CHAT_PATH, body=b'{"prompt":"blue"}', headers={"content-type": "application/json"}
                )
                assert replayed.status_code == 200
                assert replayed.body == response.body
            assert source.leftover_error(canonical_scenario_id(identity.node)) is None
            assert len(provider.hits) == 1


@pytest.mark.parametrize("disk_failure", [True, False])
def test_persist_must_reach_disk_before_success(tmp_path: Path, disk_failure: bool) -> None:
    root: Final = tmp_path / "bundle"
    if disk_failure:
        root.write_text("occupied by a file")
    session: Final = CaptureSession(
        ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic"), "owner", ScriptedReservations()
    )
    backend: Final = RecordEdge(
        BundleRecorder(root, profile="stateless_v1"), threading.Lock(), response_finished=session.response_finished
    )
    request: Final = RecordedRequest(method="post", path="/openai/v1/chat/completions", headers={})
    assert session.before_attempt() is None
    if disk_failure:
        with pytest.raises(OSError):
            _persist(backend, session.identity.node, request, successful_response())
    else:
        _persist(backend, session.identity.node, request, successful_response())
        assert len(tuple(root.rglob("*.json"))) == 1
    result: Final = session.finish(ScenarioOutcome(True, True, True))
    assert result.publishable is not disk_failure
    assert result.response_count == (0 if disk_failure else 1)
