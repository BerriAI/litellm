from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Literal

import httpx
import psutil
import pytest
from integration._support.client import eventually
from integration._support.process import group_members, owned_proxy_process
from test_airia_guardrail import (
    JSON,
    Rig,
    _assert_no_sink,
    _assert_provider,
    _assert_sink,
    _body,
    _create_rig,
    _json_object,
    _marker,
    _raw,
)

Endpoint = Literal["chat", "messages", "responses"]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _create_rig(tmp_path_factory)


def _request(rig: Rig, index: int, *, guardrail: str = "airia-pre") -> tuple[str, httpx.Response]:
    marker: Final = _marker()
    endpoint: Final[Endpoint] = ("chat", "messages", "responses")[index % 3]
    stream: Final = index % 2 == 0
    match endpoint:
        case "chat":
            return marker, _raw(
                rig,
                "/v1/chat/completions",
                _body(marker, rig.chat_model, guardrail, stream=stream),
                stream=stream,
            )
        case "messages":
            message_body: Final = {
                "model": rig.messages_model,
                "messages": [{"role": "user", "content": marker}],
                "max_tokens": 32,
                "guardrails": [guardrail],
                "stream": stream,
            }
            return marker, _raw(rig, "/v1/messages", message_body, stream=stream)
        case "responses":
            response_body: Final = {
                "model": rig.responses_model,
                "input": marker,
                "guardrails": [guardrail],
                "stream": stream,
            }
            return marker, _raw(rig, "/v1/responses", response_body, stream=stream)


def _slow_call(rig: Rig, marker: str) -> tuple[str, httpx.Response, float]:
    started: Final = time.monotonic()
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-slow"))
    return marker, response, time.monotonic() - started


def test_c1_sink_outage_mid_mixed_endpoint_burst_fails_closed_and_recovers(rig: Rig) -> None:
    original_port: Final = int(rig.sink.url.rsplit(":", 1)[1])
    with ThreadPoolExecutor(max_workers=8) as pool:
        before_outage: Final = tuple(pool.map(lambda index: _request(rig, index), range(10)))
        rig.sink.stop()
        during_outage: Final = tuple(pool.map(lambda index: _request(rig, index), range(10, 20)))
        rig.sink.start(original_port)
        after_recovery: Final = tuple(pool.map(lambda index: _request(rig, index), range(20, 30)))
        for marker, response in during_outage:
            assert response.status_code == 400, response.text
            error: Final = _json_object(JSON.validate_json(response.content))
            assert error.get("error") or error.get("detail"), error
            _assert_no_sink(rig, response)
            assert _assert_provider(rig, marker, expected=0) == ()

        for marker, response in (*before_outage, *after_recovery):
            assert response.status_code == 200, response.text
            _assert_sink(rig, marker, response)
            _assert_provider(rig, marker)


def test_c2_slow_sink_is_bounded_and_proxy_recovers(rig: Rig) -> None:
    started: Final = time.monotonic()
    markers: Final = tuple(_marker() + " AIRIA-SLOW" for _ in range(8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        slow_results: Final = tuple(pool.map(lambda marker: _slow_call(rig, marker), markers))
    elapsed: Final = time.monotonic() - started
    assert elapsed < 5, f"slow-sink burst took {elapsed:.2f}s"
    for marker, response, request_elapsed in slow_results:
        assert request_elapsed < 5, f"slow-sink request took {request_elapsed:.2f}s: {response.text}"
        assert response.status_code == 400, response.text
        _assert_sink(rig, marker, response)
        assert _assert_provider(rig, marker, expected=0) == ()
    control_marker: Final = _marker()
    control: Final = _raw(rig, "/v1/chat/completions", _body(control_marker, rig.chat_model, "airia-pre"))
    assert control.status_code == 200, control.text
    _assert_sink(rig, control_marker, control)
    _assert_provider(rig, control_marker)


def test_c3_surviving_worker_keeps_enforcing_airia(rig: Rig) -> None:
    workers: Final = tuple(
        member
        for member in group_members(rig.proxy.process.pid)
        if member.pid != rig.proxy.process.pid
        and member.is_running()
        and member.status() != psutil.STATUS_ZOMBIE
        and any("spawn_main" in part for part in member.cmdline())
    )
    assert len(workers) >= 2, workers
    killed_pid: Final = workers[0].pid
    survivor_pid: Final = workers[1].pid
    slow_markers: Final = (_marker() + " AIRIA-SLOW AIRIA-BLOCK", _marker() + " AIRIA-SLOW AIRIA-BLOCK")
    with ThreadPoolExecutor(max_workers=8) as pool:
        slow_requests: Final = tuple(
            pool.submit(
                _raw,
                rig,
                "/v1/chat/completions",
                _body(marker, rig.chat_model, "airia-slow"),
            )
            for marker in slow_markers
        )
        slow_calls: Final = eventually(
            lambda: tuple(rig.sink.matching(marker) for marker in slow_markers),
            lambda calls: all(len(marker_calls) >= 1 for marker_calls in calls),
        )
        assert tuple(len(marker_calls) for marker_calls in slow_calls) == (1, 1), slow_calls
        slow_call_ids: Final = tuple(
            _json_object(JSON.validate_python(marker_calls[0].body))["litellm_call_id"] for marker_calls in slow_calls
        )
        os.kill(killed_pid, signal.SIGKILL)
        active_pids: Final = eventually(
            lambda: tuple(
                member.pid
                for member in group_members(rig.proxy.process.pid)
                if member.pid != rig.proxy.process.pid
                and member.is_running()
                and member.status() != psutil.STATUS_ZOMBIE
            ),
            lambda pids: survivor_pid in pids and killed_pid not in pids,
        )
        assert survivor_pid in active_pids and killed_pid not in active_pids, active_pids
        for marker, request, marker_calls, call_id in zip(
            slow_markers,
            slow_requests,
            slow_calls,
            slow_call_ids,
            strict=True,
        ):
            assert isinstance(call_id, str)
            request_error: Final = request.exception(timeout=10)
            assert request_error is None or isinstance(request_error, httpx.TransportError), request_error
            if request_error is None:
                response: Final = request.result(timeout=0)
                assert response.status_code == 400, response.text
                _assert_sink(rig, marker, response)
            else:
                assert rig.sink.matching_call_id(call_id) == marker_calls
        for marker in slow_markers:
            assert _assert_provider(rig, marker, expected=0) == ()

        markers: Final = tuple(_marker() + " AIRIA-BLOCK" for _ in range(8))
        burst: Final = tuple(
            pool.map(
                lambda marker: (
                    marker,
                    _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-pre")),
                ),
                markers,
            )
        )
        for marker, response in burst:
            assert response.status_code == 400, response.text
            _assert_sink(rig, marker, response)
            assert _assert_provider(rig, marker, expected=0) == ()


def test_c4_owned_proxy_restart_resumes_airia_enforcement(rig: Rig, tmp_path: Path) -> None:
    config: Final = tmp_path / "restart.yaml"
    config.write_text(rig.config.read_text())
    with owned_proxy_process(
        rig.gateway,
        tmp_path,
        {},
        config=config,
        workers=2,
        remove_environment=("AIRIA_API_KEY", "AIRIA_GATEWAY_URL", "AIRIA_TIMEOUT"),
    ) as first:
        first_marker: Final = _marker()
        first_response: Final = first.gateway.request(
            "POST",
            "/v1/chat/completions",
            _body(first_marker, rig.chat_model, "airia-pre"),
        )
        assert first_response.status_code == 200, first_response.text
        rig.record_response("/v1/chat/completions", first_response)
        _assert_sink(rig, first_marker, first_response)
        _assert_provider(rig, first_marker)

    assert first.process.poll() is not None
    with owned_proxy_process(
        rig.gateway,
        tmp_path,
        {},
        config=config,
        workers=2,
        remove_environment=("AIRIA_API_KEY", "AIRIA_GATEWAY_URL", "AIRIA_TIMEOUT"),
    ) as restarted:
        restarted_marker: Final = _marker() + " AIRIA-BLOCK"
        restarted_response: Final = restarted.gateway.request(
            "POST",
            "/v1/chat/completions",
            _body(restarted_marker, rig.chat_model, "airia-pre"),
        )
        assert restarted_response.status_code == 400, restarted_response.text
        rig.record_response("/v1/chat/completions", restarted_response)
        _assert_sink(rig, restarted_marker, restarted_response)
        assert _assert_provider(rig, restarted_marker, expected=0) == ()
