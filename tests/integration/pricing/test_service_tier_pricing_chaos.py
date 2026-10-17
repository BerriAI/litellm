from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import httpx
import psutil
import pytest
from pydantic import JsonValue

import tests.integration._support.service_tier_pricing as pricing
from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.upstream import JsonResponse, SseResponse


@dataclass(frozen=True, slots=True)
class MatrixRequest:
    path: str
    payload: dict[str, JsonValue]
    stream: bool
    surface: Literal["chat", "responses"]


def _unused_port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


def _start_scripted_upstream(port: int) -> subprocess.Popen[bytes]:
    repository_root: Final = Path(__file__).resolve().parents[3]
    process: Final = subprocess.Popen(  # test-quality-ok: the scripted upstream imports only local integration helpers
        [
            sys.executable,
            "-m",
            "integration._support.upstream",
            "--port",
            str(port),
        ],
        cwd=repository_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(repository_root),
                    str(repository_root / "tests"),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    upstream_url: Final = f"http://127.0.0.1:{port}"

    def ready_status() -> int:
        assert process.poll() is None, "Owned scripted upstream exited before readiness"
        try:
            with httpx.Client(base_url=upstream_url, trust_env=False) as client:
                return client.get("/health", timeout=2).status_code
        except httpx.TransportError:
            return 0

    try:
        assert eventually(ready_status, lambda status: status == 200, seconds=30) == 200
    except BaseException:
        _stop_scripted_upstream(process, force=True)
        raise
    return process


def _stop_scripted_upstream(process: subprocess.Popen[bytes], *, force: bool = False) -> None:
    if process.poll() is not None:
        return
    if force:
        process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _accumulate_observations(
    upstream_url: str,
    observations: list[dict[str, JsonValue]],
) -> tuple[dict[str, JsonValue], ...]:
    observations.extend(pricing.observations_at(upstream_url))  # mutable-ok: retain destructive queue drains
    return tuple(observations)


def _register_owned_scenario(
    upstream_url: str,
    identifier: str,
    response: JsonResponse | SseResponse,
) -> None:
    with httpx.Client(base_url=upstream_url, trust_env=False) as client:
        registered: Final = client.post(
            "/__scenarios",
            json={
                "scenario_id": identifier,
                "response": response.model_dump(mode="json"),
            },
        )
        assert registered.status_code == 200, registered.text


def _capture_http(
    gateway: Gateway,
    key: str,
    request: MatrixRequest,
) -> pricing.MatrixOutcome | None:
    try:
        return pricing.invoke_http(
            gateway,
            path=request.path,
            payload=request.payload,
            key=key,
            stream=request.stream,
            surface=request.surface,
        )
    except httpx.TransportError:
        return None


def _burst_request(index: int, model_name: str) -> MatrixRequest:
    request_kind: Final = index % 3
    if request_kind == 0:
        chat_payload: Final[dict[str, JsonValue]] = {
            **pricing.chat_payload("priority"),
            "model": model_name,
            "messages": [{"role": "user", "content": f"tier chaos request {index}"}],
        }
        return MatrixRequest(
            path=pricing.MATRIX_CHAT_PATH,
            payload=chat_payload,
            stream=False,
            surface="chat",
        )
    if request_kind == 1:
        stream_payload: Final[dict[str, JsonValue]] = {
            **pricing.chat_payload("priority", stream=True),
            "model": model_name,
            "messages": [{"role": "user", "content": f"tier chaos request {index}"}],
        }
        return MatrixRequest(
            path=pricing.MATRIX_CHAT_PATH,
            payload=stream_payload,
            stream=True,
            surface="chat",
        )
    response_payload: Final[dict[str, JsonValue]] = {
        "model": model_name,
        "input": f"tier chaos request {index}",
        "max_output_tokens": pricing.MATRIX_COMPLETION_TOKENS,
        "service_tier": "priority",
        "cache": {"no-cache": True},
    }
    return MatrixRequest(
        path=pricing.MATRIX_RESPONSES_PATH,
        payload=response_payload,
        stream=False,
        surface="responses",
    )


def _deployment(
    model_name: str,
    scenario_id: str,
    upstream_url: str,
) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/{pricing.MATRIX_BACKEND_MODEL}",
            "api_key": scenario_id,
            "api_base": f"{upstream_url}/{scenario_id}",
            "input_cost_per_token": pricing.CUSTOM_STANDARD_INPUT_RATE,
            "output_cost_per_token": pricing.CUSTOM_STANDARD_OUTPUT_RATE,
        },
        "model_info": {},
    }


def _owned_model_names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{uuid.uuid4().hex}" for _ in range(count))


def _owned_scenario_ids(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{uuid.uuid4().hex}" for _ in range(count))


def _owned_model_config(
    model_names: tuple[str, ...],
    scenario_ids: tuple[str, ...],
    upstream_url: str,
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _deployment(model_names[index], scenario_ids[index], upstream_url) for index in range(len(model_names))
    )


def _assert_all_forwarded(requests: tuple[dict[str, JsonValue], ...]) -> None:
    assert requests
    for request in requests:
        pricing.assert_forwarded_body(
            object_value(request["body"]),
            expected_tier="priority",
            tier_present=True,
        )


def _completed_success(outcome: pricing.MatrixOutcome | None) -> bool:
    return outcome is not None and outcome.status == 200 and ("usage" in outcome.body or "[DONE]" in outcome.text)


def _assert_cost_rows(
    key: str,
    outcomes: tuple[pricing.MatrixOutcome | None, ...],
    requests: tuple[MatrixRequest, ...],
) -> None:
    paired_outcomes: Final = tuple(zip(outcomes, requests, strict=True))
    successful: Final = tuple(outcome for outcome, _ in paired_outcomes if _completed_success(outcome))
    successful_requests: Final = tuple(request for outcome, request in paired_outcomes if _completed_success(outcome))
    failed: Final = tuple(outcome for outcome, _ in paired_outcomes if not _completed_success(outcome))
    expected_cost: Final = pricing.expected_cost(
        "priority",
        pricing.MATRIX_PROMPT_TOKENS,
        pricing.MATRIX_COMPLETION_TOKENS,
        custom_standard=False,
    )
    for outcome, request in zip(successful, successful_requests, strict=True):
        pricing.assert_billing(
            outcome,
            expected_cost=expected_cost,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
            allow_missing_header=request.stream and request.surface == "chat",
        )
    assert all(outcome.request_id is not None for outcome in successful), successful
    successful_ids: Final = tuple(cast(str, outcome.request_id) for outcome in successful)
    rows: Final = eventually(
        lambda: pricing.spend_rows_for_key(key),
        lambda current: len(current) == len(outcomes),
        seconds=70,
    )
    assert len(rows) == len(outcomes), rows
    success_rows: Final = tuple(row for row in rows if row["status"] == "success")
    assert len(success_rows) == len(successful), rows
    assert {str(row["request_id"]) for row in success_rows} == set(successful_ids), rows
    failed_ids: Final = frozenset(
        outcome.request_id for outcome in failed if outcome is not None and outcome.request_id is not None
    )
    assert all(row["status"] != "success" or str(row["request_id"]) not in failed_ids for row in rows), rows
    failed_rows: Final = tuple(row for row in rows if row["status"] != "success")
    expected_failure_costs: Final = tuple(
        pricing.expected_cost(
            "priority",
            int(str(row["prompt_tokens"] or 0)),
            int(str(row["completion_tokens"] or 0)),
            custom_standard=False,
        )
        for row in failed_rows
    )
    assert all(
        float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6)
        for row, expected_cost in zip(failed_rows, expected_failure_costs, strict=True)
    ), tuple(zip(failed_rows, expected_failure_costs, strict=True))
    total_spend: Final = sum(float(str(row["spend"])) for row in rows)
    expected_total_spend: Final = len(successful) * expected_cost + sum(expected_failure_costs)
    assert total_spend == pytest.approx(expected_total_spend, rel=1e-6), rows


def _uvicorn_workers(process: subprocess.Popen[bytes]) -> tuple[psutil.Process, ...]:
    return tuple(
        child
        for child in psutil.Process(process.pid).children(recursive=True)
        if child.is_running() and "spawn_main" in " ".join(child.cmdline())
    )


@pytest.mark.timeout(240)
def test_upstream_restart_mid_mixed_burst_preserves_exact_tier_billing(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    port: Final = _unused_port()
    upstream_url: Final = f"http://127.0.0.1:{port}"
    model_names: Final = _owned_model_names("tier-chaos-upstream", 3)
    scenario_ids: Final = _owned_scenario_ids("tier-chaos-upstream", 3)
    response_scenarios: Final = (
        pricing.chat_json_response("priority"),
        pricing.chat_sse_response(frame_delay_ms=2_000),
        pricing.responses_json_response("priority"),
    )
    first_upstream: Final = _start_scripted_upstream(port)
    try:
        for index in range(len(scenario_ids)):
            _register_owned_scenario(upstream_url, scenario_ids[index], response_scenarios[index])
        config: Final = pricing.write_proxy_config(
            tmp_path,
            models=_owned_model_config(model_names, scenario_ids, upstream_url),
            filename="tier-upstream-restart.yaml",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                key: Final = scenario.key()
                requests: Final = tuple(
                    _burst_request(index, model_names[index % len(model_names)]) for index in range(30)
                )
                with ThreadPoolExecutor(max_workers=30) as executor:
                    futures: Final = tuple(
                        executor.submit(_capture_http, candidate, key, request) for request in requests
                    )
                    first_response: Final = futures[0].result(timeout=30)
                    assert _completed_success(first_response), first_response
                    observations: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: keep earlier queue drains
                    eventually(
                        lambda: _accumulate_observations(upstream_url, observations),
                        lambda calls: any(object_value(call["body"]).get("stream") is True for call in calls),
                        seconds=30,
                    )
                    _accumulate_observations(upstream_url, observations)
                    burst_observations: Final = tuple(observations)
                    assert burst_observations
                    assert any(object_value(call["body"]).get("stream") is True for call in burst_observations), (
                        burst_observations
                    )
                    _assert_all_forwarded(burst_observations)
                    burst_observation_count: Final = len(burst_observations)
                    _stop_scripted_upstream(first_upstream, force=True)
                    burst_outcomes: Final = tuple(future.result(timeout=90) for future in futures)
                assert any(not _completed_success(outcome) for outcome in burst_outcomes), burst_outcomes
                recovery_upstream: Final = _start_scripted_upstream(port)
                try:
                    for index in range(len(scenario_ids)):
                        _register_owned_scenario(upstream_url, scenario_ids[index], response_scenarios[index])
                    recovery_requests: Final = tuple(
                        _burst_request(30 + index, model_names[index % len(model_names)]) for index in range(5)
                    )
                    recovery_outcomes: Final = tuple(
                        _capture_http(candidate, key, request) for request in recovery_requests
                    )
                    assert all(_completed_success(outcome) for outcome in recovery_outcomes), recovery_outcomes
                    _accumulate_observations(upstream_url, observations)
                    recovery_observations: Final = tuple(observations[burst_observation_count:])
                    assert len(recovery_observations) == len(recovery_outcomes), recovery_observations
                    _assert_all_forwarded(recovery_observations)
                    _assert_cost_rows(
                        key,
                        (*burst_outcomes, *recovery_outcomes),
                        (*requests, *recovery_requests),
                    )
                finally:
                    _stop_scripted_upstream(recovery_upstream)
    finally:
        _stop_scripted_upstream(first_upstream)


@pytest.mark.timeout(240)
def test_two_worker_proxy_survives_worker_kill_and_keeps_tier_billing(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    port: Final = _unused_port()
    upstream_url: Final = f"http://127.0.0.1:{port}"
    model_name: Final = _owned_model_names("tier-chaos-worker", 1)[0]
    scenario_id: Final = _owned_scenario_ids("tier-chaos-worker", 1)[0]
    response: Final = pricing.chat_sse_response(frame_delay_ms=500)
    upstream_process: Final = _start_scripted_upstream(port)
    try:
        _register_owned_scenario(upstream_url, scenario_id, response)
        config: Final = pricing.write_proxy_config(
            tmp_path,
            models=(_deployment(model_name, scenario_id, upstream_url),),
            filename="tier-worker-kill.yaml",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: _uvicorn_workers(owned.process),
                lambda values: len(values) == 2,
                seconds=30,
            )
            survivor: Final = workers[1]
            with candidate.scenario() as scenario:
                key: Final = scenario.key()
                requests: Final = tuple(_burst_request(index * 3 + 1, model_name) for index in range(30))
                with ThreadPoolExecutor(max_workers=30) as executor:
                    futures: Final = tuple(
                        executor.submit(_capture_http, candidate, key, request) for request in requests
                    )
                    observations: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: keep earlier queue drains
                    eventually(
                        lambda: _accumulate_observations(upstream_url, observations),
                        lambda calls: len(calls) >= 4,
                        seconds=30,
                    )
                    victim: Final = workers[0]
                    victim.send_signal(signal.SIGKILL)
                    assert (
                        eventually(
                            victim.is_running,
                            lambda running: not running,
                            seconds=10,
                        )
                        is False
                    )
                    burst_outcomes: Final = tuple(future.result(timeout=90) for future in futures)
                assert survivor.is_running(), survivor
                refreshed_workers: Final = eventually(
                    lambda: _uvicorn_workers(owned.process),
                    lambda values: len(values) == 2 and survivor.pid in {worker.pid for worker in values},
                    seconds=30,
                )
                assert survivor.pid in {worker.pid for worker in refreshed_workers}, refreshed_workers
                recovery_requests: Final = tuple(_burst_request(100 + index * 3, model_name) for index in range(5))
                recovery_outcomes: Final = tuple(
                    _capture_http(candidate, key, request) for request in recovery_requests
                )
                assert all(_completed_success(outcome) for outcome in recovery_outcomes), recovery_outcomes
                _accumulate_observations(upstream_url, observations)
                observed_requests: Final = tuple(observations)
                successful_count: Final = sum(_completed_success(outcome) for outcome in burst_outcomes)
                assert len(observed_requests) >= successful_count + len(recovery_outcomes), (
                    len(observed_requests),
                    successful_count,
                )
                _assert_all_forwarded(observed_requests)
                burst_spend_pairs: Final = tuple(
                    (outcome, request)
                    for outcome, request in zip(burst_outcomes, requests, strict=True)
                    if outcome is not None
                )
                _assert_cost_rows(
                    key,
                    tuple(outcome for outcome, _ in burst_spend_pairs) + recovery_outcomes,
                    tuple(request for _, request in burst_spend_pairs) + recovery_requests,
                )
    finally:
        _stop_scripted_upstream(upstream_process)
