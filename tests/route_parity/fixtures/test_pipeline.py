from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from tests.route_parity.fixtures.pipeline import (
    RecordingInvocation,
    RecordingTarget,
    build_recording_jobs,
    record_fixtures,
)
from tests.route_parity.fixtures.recording import UpstreamEndpoint
from tests.route_parity.fixtures.store import fixture_path
from tests.route_parity.recorded_http import RecordedResponse


class _FixtureInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str

    def canonical_input(self) -> dict[str, object]:
        return {"identifier": self.identifier}


class _ParityCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    litellm_input: _FixtureInput
    provider_responses: tuple[RecordedResponse, ...]


class _Upstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, status: int = 200) -> None:
        super().__init__(("127.0.0.1", 0), _UpstreamHandler)
        self.response_status: Final = status

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length: Final = int(self.headers.get("content-length") or "0")
        self.rfile.read(length)
        body: Final = b"{}"
        server: Final = self.server
        assert isinstance(server, _Upstream)
        self.send_response(server.response_status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _upstream(status: int = 200) -> Generator[_Upstream]:
    server: Final = _Upstream(status)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@dataclass(frozen=True, slots=True)
class _OrderedInvocation:
    order: Literal["slow", "fast"]
    slow_started: threading.Event
    fast_finished: threading.Event

    def execute(self, provider_url: str, case_input: _FixtureInput) -> None:
        if self.order == "slow":
            self.slow_started.set()
            if not self.fast_finished.wait(timeout=2):
                raise TimeoutError("fast recording did not finish")
        else:
            if not self.slow_started.wait(timeout=2):
                raise TimeoutError("slow recording did not start")
        response: Final = httpx.post(f"{provider_url}/record", json={"id": case_input.identifier}, timeout=5)
        response.raise_for_status()
        if self.order == "fast":
            self.fast_finished.set()


@dataclass(frozen=True, slots=True)
class _Invocation:
    def execute(self, provider_url: str, case_input: _FixtureInput) -> None:
        response: Final = httpx.post(f"{provider_url}/record", json={"id": case_input.identifier}, timeout=5)
        response.raise_for_status()


def _target(
    name: str,
    upstream_url: str,
    case_input: _FixtureInput,
    invocation: RecordingInvocation[_FixtureInput],
) -> RecordingTarget[_FixtureInput]:
    return RecordingTarget(
        name=name,
        upstream=UpstreamEndpoint(base_url=upstream_url),
        strategy=st.just(case_input),
        invocation=invocation,
        required_inputs=(case_input,),
    )


def test_build_jobs_keeps_required_inputs_before_generated_inputs_and_deduplicates(tmp_path: Path) -> None:
    required: Final = _FixtureInput(identifier="required")
    generated: Final = _FixtureInput(identifier="generated")
    target: Final = RecordingTarget(
        name="ordered",
        upstream=UpstreamEndpoint(base_url="https://provider.invalid"),
        strategy=st.just(generated),
        invocation=_Invocation(),
        required_inputs=(required, required),
    )

    jobs: Final = build_recording_jobs((target,), tmp_path, examples=1)

    assert tuple(job.case_input.identifier for job in jobs) == ("required", "generated")


def test_progress_follows_completion_order(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    slow_started: Final = threading.Event()
    fast_finished: Final = threading.Event()
    with _upstream() as upstream:
        targets: Final = (
            _target(
                "slow",
                upstream.url,
                _FixtureInput(identifier="slow"),
                _OrderedInvocation("slow", slow_started, fast_finished),
            ),
            _target(
                "fast",
                upstream.url,
                _FixtureInput(identifier="fast"),
                _OrderedInvocation("fast", slow_started, fast_finished),
            ),
        )
        with caplog.at_level(logging.INFO, logger="tests.route_parity.fixtures.pipeline"):
            summary: Final = record_fixtures(targets, tmp_path, 1, 2, _ParityCase)

    progress: Final = tuple(record.message for record in caplog.records if record.message.startswith("["))
    assert len(summary.recorded) == 2
    assert summary.exit_code == 0
    assert "recorded fast" in progress[0]
    assert "recorded slow" in progress[1]
    assert caplog.records[0].message == "Recording 2 fixtures across 2 targets with concurrency 2"
    assert caplog.records[-1].message == "Finished 2 fixtures: 2 recorded, 0 cached, 0 failed"


def test_failure_does_not_stop_independent_recordings(tmp_path: Path) -> None:
    stale_input: Final = _FixtureInput(identifier="stale")
    stale_directory: Final = tmp_path / "stale"
    stale_directory.mkdir()
    fixture_path(stale_directory, stale_input).with_suffix(".json").write_text(
        '{"schema_version": 0}\n', encoding="utf-8"
    )
    with _upstream() as upstream:
        targets: Final = (
            _target("stale", upstream.url, stale_input, _Invocation()),
            _target("valid", upstream.url, _FixtureInput(identifier="valid"), _Invocation()),
        )
        summary: Final = record_fixtures(targets, tmp_path, 1, 2, _ParityCase)

    assert len(summary.recorded) == 1
    assert summary.recorded[0].target_name == "valid"
    assert len(summary.failed) == 1
    assert summary.failed[0].target_name == "stale"
    assert summary.exit_code == 1


@pytest.mark.parametrize("status", (408, 429, 500, 503))
def test_transient_response_is_not_cached_and_can_be_retried(tmp_path: Path, status: int) -> None:
    case_input: Final = _FixtureInput(identifier="retry")
    with _upstream(status) as upstream:
        target: Final = _target("retry", upstream.url, case_input, _Invocation())
        failed: Final = record_fixtures((target,), tmp_path, 1, 1, _ParityCase)
    assert failed.exit_code == 1
    assert not fixture_path(tmp_path / "retry", case_input).exists()
    with _upstream() as healthy_upstream:
        healthy_target: Final = _target("retry", healthy_upstream.url, case_input, _Invocation())
        retried: Final = record_fixtures((healthy_target,), tmp_path, 1, 1, _ParityCase)
    assert retried.exit_code == 0
    assert len(retried.recorded) == 1


def test_provider_rejected_response_can_be_recorded(tmp_path: Path) -> None:
    with _upstream(400) as upstream:
        target: Final = _target("rejected", upstream.url, _FixtureInput(identifier="invalid"), _Invocation())
        summary: Final = record_fixtures((target,), tmp_path, 1, 1, _ParityCase)
    assert summary.exit_code == 0
    assert len(summary.recorded) == 1
