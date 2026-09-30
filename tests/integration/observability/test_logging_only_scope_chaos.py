from __future__ import annotations

import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate, repeat
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import psutil
import pytest
from _logging_only_scope_support import (
    JSON_OBJECT,
    CallerResult,
    ChaosCall,
    _assert_response_id,
    _call_client,
    _chaos_calls,
    _chaos_control_configuration,
    _chaos_model_list,
    _chaos_models,
    _chaos_spend_minimums,
    _configuration,
    _direction,
    _directions_for_audit_leg,
    _drain_upstream,
    _guardrail_entries,
    _is_base_audit_leg,
    _json_contains_exact_string,
    _policy_call_id_matches,
    _spend_rows,
    _spend_rows_for_calls,
    _spend_rows_matching_call,
    wire_server,
)
from _logging_only_scope_support import (
    _record_audit_properties as _record_audit_properties,
)
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request
from openai import APIConnectionError as OpenAIAPIConnectionError
from pydantic import JsonValue


def test_K1_policy_edge_restart_mid_burst_keeps_output_observation_fail_open(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k1-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        policy_port: Final = reservation.getsockname()[1]
    with gateway.scenario() as scenario:
        deployments: Final = _chaos_models(scenario, marker)
        models: Final = tuple(deployment.model_name for deployment in deployments)
        model_list: Final = _chaos_model_list(deployments)
        chaos_calls: Final = _chaos_calls(deployments, marker)
        expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
        control_config: Final = _chaos_control_configuration(tmp_path, identity, model_list)
        with owned_proxy(gateway, tmp_path, {}, config=control_config, workers=2) as control_proxy:
            controls: Final = tuple(
                _call_client(
                    call.client_kind,
                    call.endpoint,
                    control_proxy,
                    call.model,
                    call.prompt,
                    call.stream,
                    f"{call.call_id}-control",
                )
                for call in chaos_calls
            )
        control_upstream: Final = _drain_upstream(gateway.upstream_url)
        assert len(control_upstream) == 30, control_upstream
        assert (
            tuple(
                sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in control_upstream)
                for call in chaos_calls
            )
            == (1,) * 30
        ), control_upstream
        config: Final = _configuration(
            tmp_path,
            identity,
            f"http://127.0.0.1:{policy_port}",
            "output",
            model_list=model_list,
        )
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate:
            starts: Final = tuple(threading.Event() for _ in range(3))

            def run(index: int) -> tuple[int, CallerResult]:
                call: Final = chaos_calls[index]
                phase: Final = index // 10
                assert starts[phase].wait(timeout=90), (index, phase)
                return index, _call_client(
                    call.client_kind,
                    call.endpoint,
                    candidate,
                    call.model,
                    call.prompt,
                    call.stream,
                    call.call_id,
                )

            with ThreadPoolExecutor(max_workers=30) as pool:
                futures: Final = tuple(pool.submit(run, index) for index in range(30))
                try:
                    with wire_server(policy, port=policy_port) as initial_edge:
                        starts[0].set()
                        first: Final = tuple(futures[index].result(timeout=90) for index in range(10))
                        eventually(
                            lambda: initial_edge.received.qsize(),
                            lambda count: count == 10 * len(expected_directions),
                            seconds=30,
                        )
                        tuple(
                            _spend_rows(model, minimum)
                            for model, minimum in zip(models, _chaos_spend_minimums(models, chaos_calls[:10], 5))
                        )
                    starts[1].set()
                    middle: Final = tuple(futures[index].result(timeout=90) for index in range(10, 20))
                    tuple(
                        _spend_rows(model, minimum)
                        for model, minimum in zip(models, _chaos_spend_minimums(models, chaos_calls[:20], 5))
                    )
                    with wire_server(policy, port=policy_port) as recovered_edge:
                        starts[2].set()
                        recovered: Final = tuple(futures[index].result(timeout=90) for index in range(20, 30))
                        eventually(
                            lambda: recovered_edge.received.qsize(),
                            lambda count: count == 10 * len(expected_directions),
                            seconds=30,
                        )
                        tuple(_spend_rows(model, 10) for model in models)
                finally:
                    for start in starts:
                        start.set()
            results: Final = first + middle + recovered
            assert tuple(index for index, _ in results) == tuple(range(30)), results
            assert all(result.status == controls[index].status for index, result in results), results
            assert all(result.text == controls[index].text for index, result in results), results
            candidate_ids: Final = tuple(result.response_id for _, result in results)
            assert len(set(candidate_ids)) == 30, candidate_ids
            observed_upstream: Final = _drain_upstream(gateway.upstream_url)
            assert len(observed_upstream) == 30, observed_upstream
            assert (
                tuple(
                    sum(
                        _json_contains_exact_string(observation["body"], call.prompt)
                        for observation in observed_upstream
                    )
                    for call in chaos_calls
                )
                == (1,) * 30
            ), observed_upstream
            expected_success_ids: Final = frozenset(
                call.call_id for call in chaos_calls if call.index < 10 or call.index >= 20
            )
            edge_payloads: Final = tuple(
                JSON_OBJECT.validate_json(call.body) for call in initial_edge.drain() + recovered_edge.drain()
            )
            assert len(edge_payloads) == 20 * len(expected_directions), edge_payloads
            successful_calls: Final = tuple(call for call in chaos_calls if call.call_id in expected_success_ids)
            for call in successful_calls:
                payloads_for_call: Final = tuple(
                    payload for payload in edge_payloads if _policy_call_id_matches(payload, call.call_id)
                )
                assert tuple(sorted(_direction(payload) for payload in payloads_for_call)) == tuple(
                    sorted(expected_directions)
                ), (
                    call.call_id,
                    payloads_for_call,
                )
                assert all(
                    payload["texts"]
                    == ([call.prompt] if _direction(payload) == "request" else [results[call.index][1].text])
                    for payload in payloads_for_call
                ), payloads_for_call
            rows: Final = _spend_rows_for_calls(
                models,
                tuple(
                    (
                        chaos_calls[index].model,
                        response_id,
                        chaos_calls[index].call_id,
                    )
                    for index, response_id in enumerate(candidate_ids)
                ),
            )
            for index, call in enumerate(chaos_calls):
                matching_rows: Final = _spend_rows_matching_call(rows, call.model, call.call_id)
                assert len(matching_rows) == 1, (call.call_id, matching_rows)
                row: Final = matching_rows[0]
                expected_status: Final = "guardrail_failed_to_respond" if 10 <= index < 20 else "success"
                entries: Final = _guardrail_entries(row)
                assert tuple(
                    (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"]) for entry in entries
                ) == tuple((identity, "logging_only", expected_status) for _ in expected_directions), (index, entries)


def test_K2_policy_edge_delay_does_not_delay_concurrent_callers(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"logging-scope-k2-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex

    def policy(_request: Request) -> Reply:
        time.sleep(2)
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        deployments: Final = _chaos_models(scenario, marker)
        models: Final = tuple(deployment.model_name for deployment in deployments)
        model_list: Final = _chaos_model_list(deployments)
        chaos_calls: Final = _chaos_calls(deployments, marker)
        expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
        control_config: Final = _chaos_control_configuration(tmp_path, identity, model_list)
        with owned_proxy(gateway, tmp_path, {}, config=control_config, workers=2) as control_proxy:
            controls: Final = tuple(
                _call_client(
                    call.client_kind,
                    call.endpoint,
                    control_proxy,
                    call.model,
                    call.prompt,
                    call.stream,
                    f"{call.call_id}-control",
                )
                for call in chaos_calls
            )
        control_upstream: Final = _drain_upstream(gateway.upstream_url)
        assert len(control_upstream) == 30, control_upstream
        assert (
            tuple(
                sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in control_upstream)
                for call in chaos_calls
            )
            == (1,) * 30
        ), control_upstream
        with wire_server(policy) as edge:
            config: Final = _configuration(tmp_path, identity, edge.url, "output", model_list=model_list)
            with owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate:

                def run(call: ChaosCall) -> tuple[int, CallerResult, float]:
                    started: Final = time.monotonic()
                    result: Final = _call_client(
                        call.client_kind,
                        call.endpoint,
                        candidate,
                        call.model,
                        call.prompt,
                        call.stream,
                        call.call_id,
                    )
                    return call.index, result, time.monotonic() - started

                with ThreadPoolExecutor(max_workers=30) as pool:
                    results: Final = tuple(pool.map(run, chaos_calls))
                assert tuple(index for index, _, _ in results) == tuple(range(30)), results
                assert all(result.status == controls[index].status for index, result, _ in results), results
                assert all(result.text == controls[index].text for index, result, _ in results), results
                assert all(duration < 2 for _, _, duration in results), results
                eventually(
                    lambda: edge.received.qsize(),
                    lambda count: count == 30 * len(expected_directions),
                    seconds=30,
                )
                edge_calls: Final = edge.drain()
                payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in edge_calls)
                for call in chaos_calls:
                    payloads_for_call: Final = tuple(
                        payload for payload in payloads if _policy_call_id_matches(payload, call.call_id)
                    )
                    assert tuple(sorted(_direction(payload) for payload in payloads_for_call)) == tuple(
                        sorted(expected_directions)
                    ), (
                        call,
                        payloads_for_call,
                    )
                    assert all(
                        payload["texts"]
                        == ([call.prompt] if _direction(payload) == "request" else [controls[call.index].text])
                        for payload in payloads_for_call
                    ), payloads_for_call
                response_ids: Final = tuple(result.response_id for _, result, _ in results)
                assert len(set(response_ids)) == 30, response_ids
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert len(upstream) == 30, upstream
                assert (
                    tuple(
                        sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in upstream)
                        for call in chaos_calls
                    )
                    == (1,) * 30
                ), upstream
                rows: Final = _spend_rows_for_calls(
                    models,
                    tuple(
                        (call.model, result.response_id, call.call_id)
                        for call, (_, result, _) in zip(chaos_calls, results)
                    ),
                )
                for call, (_, result, _) in zip(chaos_calls, results):
                    matching_rows: Final = _spend_rows_matching_call(rows, call.model, call.call_id)
                    assert len(matching_rows) == 1, (call, result.response_id, matching_rows)
                    row: Final = matching_rows[0]
                    entries: Final = _guardrail_entries(row)
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "success") for _ in expected_directions), entries


@pytest.mark.timeout(180)
def test_K3_two_worker_sigkill_checks_post_kill_spend_rows(
    gateway: Gateway,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"logging-scope-k3-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex
    scan_started: Final = threading.Event()
    release_scans: Final = threading.Event()

    def policy(_request: Request) -> Reply:
        scan_started.set()
        assert release_scans.wait(timeout=60), identity
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        deployments: Final = _chaos_models(scenario, marker)
        models: Final = tuple(deployment.model_name for deployment in deployments)
        model_list: Final = _chaos_model_list(deployments)
        calls: Final = _chaos_calls(deployments, marker)
        expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
        expected_entries: Final = tuple((identity, "logging_only", "success") for _ in expected_directions)
        control_config: Final = _chaos_control_configuration(tmp_path, identity, model_list)
        with owned_proxy(gateway, tmp_path, {}, config=control_config, workers=2) as control_proxy:
            controls: Final = tuple(
                _call_client(
                    call.client_kind,
                    call.endpoint,
                    control_proxy,
                    call.model,
                    call.prompt,
                    call.stream,
                    f"{call.call_id}-control",
                )
                for call in calls
            )
        assert len(_drain_upstream(gateway.upstream_url)) == 30
        with wire_server(policy) as edge:
            config: Final = _configuration(tmp_path, identity, edge.url, "output", model_list=model_list)
            with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
                candidate: Final = owned.gateway
                root: Final = psutil.Process(owned.process.pid)
                workers: Final = eventually(
                    lambda: tuple(
                        child
                        for child in root.children(recursive=True)
                        if any("spawn_main" in part for part in child.cmdline())
                    ),
                    lambda children: len(children) == 2,
                    seconds=30,
                )

                def run(call: ChaosCall) -> tuple[int, CallerResult | None, str | None]:
                    try:
                        result: Final = _call_client(
                            call.client_kind,
                            call.endpoint,
                            candidate,
                            call.model,
                            call.prompt,
                            call.stream,
                            call.call_id,
                        )
                        return call.index, result, None
                    except (OpenAIAPIConnectionError, AnthropicAPIConnectionError, httpx.RemoteProtocolError) as error:
                        return call.index, None, str(error)

                with ThreadPoolExecutor(max_workers=30) as pool:
                    futures: Final = tuple(pool.submit(run, call) for call in calls)
                    try:
                        assert eventually(lambda: scan_started.is_set(), bool, seconds=30)
                        eventually(lambda: edge.received.qsize(), lambda count: count >= 5, seconds=30)
                        workers[0].send_signal(signal.SIGKILL)
                        killed_workers, surviving_workers = psutil.wait_procs((workers[0],), timeout=10)
                        assert len(killed_workers) == 1 and not surviving_workers, (
                            killed_workers,
                            surviving_workers,
                        )
                    finally:
                        release_scans.set()
                    outcomes: Final = tuple(future.result(timeout=90) for future in futures)
                assert owned.process.poll() is None, "Proxy supervisor exited after a worker was killed"
                successful: Final = tuple(
                    (calls[index], result) for index, result, error in outcomes if result is not None and error is None
                )
                assert successful, outcomes
                assert all(
                    result.status == controls[call.index].status and result.text == controls[call.index].text
                    for call, result in successful
                ), successful
                pre_kill_expected: Final = tuple(
                    (call.model, result.response_id, call.call_id) for call, result in successful
                )
                pre_kill_rows: Final = _spend_rows_for_calls(
                    models,
                    pre_kill_expected,
                    tolerate_missing=True,
                )
                pre_kill_rows_by_call: Final = tuple(
                    (call, _spend_rows_matching_call(pre_kill_rows, call.model, call.call_id)) for call in calls
                )
                assert all(len(rows) <= 1 for _, rows in pre_kill_rows_by_call), pre_kill_rows_by_call
                pre_kill_missing_rows: Final = sum(not rows for _, rows in pre_kill_rows_by_call)
                record_property("k3_pre_kill_missing_spend_rows", pre_kill_missing_rows)
                for call, matching_rows in pre_kill_rows_by_call:
                    if not matching_rows:
                        continue
                    entries: Final = _guardrail_entries(matching_rows[0])
                    assert tuple(
                        sorted(
                            (
                                entry["guardrail_name"],
                                entry["guardrail_mode"],
                                entry["guardrail_status"],
                            )
                            for entry in entries
                        )
                    ) == tuple(sorted(expected_entries)), (call, entries)

                post_kill_templates: Final = calls[:6]
                post_kill_calls: Final = tuple(
                    ChaosCall(
                        index=call.index,
                        endpoint=call.endpoint,
                        client_kind=call.client_kind,
                        model=call.model,
                        stream=call.stream,
                        prompt=f"synthetic K post-kill burst {marker}-{call.index}",
                        call_id=f"{marker}-k-post-kill-{call.index}",
                    )
                    for call in post_kill_templates
                )

                def run_post_kill(call: ChaosCall) -> CallerResult:
                    return _call_client(
                        call.client_kind,
                        call.endpoint,
                        candidate,
                        call.model,
                        call.prompt,
                        call.stream,
                        call.call_id,
                    )

                with ThreadPoolExecutor(max_workers=len(post_kill_calls)) as pool:
                    post_kill_futures: Final = tuple(pool.submit(run_post_kill, call) for call in post_kill_calls)
                    post_kill_results: Final = tuple(future.result(timeout=90) for future in post_kill_futures)
                assert all(
                    result.status == controls[call.index].status and result.text == controls[call.index].text
                    for call, result in zip(post_kill_calls, post_kill_results)
                ), post_kill_results
                served: Final = successful + tuple(zip(post_kill_calls, post_kill_results))
                response_ids: Final = tuple(result.response_id for _, result in served)
                assert len(set(response_ids)) == len(response_ids), response_ids
                served_calls: Final = tuple(call for call, _ in served)
                requested_call_ids: Final = frozenset(call.call_id for call in calls + post_kill_calls)
                upstream: Final = _drain_upstream(gateway.upstream_url)
                assert all(
                    sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in upstream) == 1
                    for call in served_calls
                ), upstream

                def accumulate_policy_payloads(
                    collected: tuple[dict[str, JsonValue], ...], _: None
                ) -> tuple[dict[str, JsonValue], ...]:
                    return collected + tuple(JSON_OBJECT.validate_json(call.body) for call in edge.drain())

                payload_batches: Final = accumulate(repeat(None), accumulate_policy_payloads, initial=())

                def has_expected_post_kill_scans(collected: tuple[dict[str, JsonValue], ...], call: ChaosCall) -> bool:
                    payloads_for_call: Final = tuple(
                        payload for payload in collected if _policy_call_id_matches(payload, call.call_id)
                    )
                    return all(
                        sum(_direction(payload) == direction for payload in payloads_for_call)
                        >= expected_directions.count(direction)
                        for direction in expected_directions
                    )

                payloads: Final = eventually(
                    lambda: next(payload_batches),
                    lambda collected: all(has_expected_post_kill_scans(collected, call) for call in post_kill_calls),
                    seconds=30,
                )
                assert all(
                    any(_policy_call_id_matches(payload, call_id) for call_id in requested_call_ids)
                    and _direction(payload) in expected_directions
                    for payload in payloads
                ), payloads
                for call, result in zip(post_kill_calls, post_kill_results):
                    payloads_for_call: Final = tuple(
                        payload for payload in payloads if _policy_call_id_matches(payload, call.call_id)
                    )
                    assert tuple(sorted(_direction(payload) for payload in payloads_for_call)) == tuple(
                        sorted(expected_directions)
                    ), (call, payloads_for_call)
                    assert all(
                        payload["texts"]
                        == ([call.prompt] if _direction(payload) == "request" else [controls[call.index].text])
                        for payload in payloads_for_call
                    ), payloads_for_call

                post_kill_expected: Final = tuple(
                    (call.model, result.response_id, call.call_id)
                    for call, result in zip(post_kill_calls, post_kill_results)
                )
                rows: Final = _spend_rows_for_calls(
                    models,
                    post_kill_expected,
                    tolerate_missing=True,
                )
                all_candidate_calls: Final = calls + post_kill_calls
                rows_by_call: Final = tuple(
                    (call, _spend_rows_matching_call(rows, call.model, call.call_id)) for call in all_candidate_calls
                )
                assert all(len(matching_rows) <= 1 for _, matching_rows in rows_by_call), rows_by_call
                for call, matching_rows in rows_by_call:
                    if not matching_rows:
                        continue
                    entries: Final = _guardrail_entries(matching_rows[0])
                    assert tuple(
                        sorted(
                            (
                                entry["guardrail_name"],
                                entry["guardrail_mode"],
                                entry["guardrail_status"],
                            )
                            for entry in entries
                        )
                    ) == tuple(sorted(expected_entries)), (call, entries)
                for call, result in successful:
                    matching_rows: Final = _spend_rows_matching_call(rows, call.model, call.call_id)
                    if matching_rows:
                        _assert_response_id(
                            call.endpoint,
                            str(matching_rows[0]["request_id"]),
                            result.response_id,
                            marker if call.endpoint == "responses" else call.call_id,
                        )
                for call, result in zip(post_kill_calls, post_kill_results):
                    matching_rows: Final = _spend_rows_matching_call(rows, call.model, call.call_id)
                    assert len(matching_rows) == 1, (result.response_id, matching_rows)
                    _assert_response_id(
                        call.endpoint,
                        str(matching_rows[0]["request_id"]),
                        result.response_id,
                        marker if call.endpoint == "responses" else call.call_id,
                    )


def test_K4_proxy_restart_after_fifteen_responses_records_lost_ids(
    gateway: Gateway,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"logging-scope-k4-{uuid.uuid4().hex}"
    marker: Final = uuid.uuid4().hex

    def policy(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with gateway.scenario() as scenario:
        deployments: Final = _chaos_models(scenario, marker)
        models: Final = tuple(deployment.model_name for deployment in deployments)
        model_list: Final = _chaos_model_list(deployments)
        calls: Final = _chaos_calls(deployments, marker)
        expected_directions: Final = _directions_for_audit_leg(("request", "response"), "output")
        control_config: Final = _chaos_control_configuration(tmp_path, identity, model_list)
        with owned_proxy(gateway, tmp_path, {}, config=control_config, workers=2) as control_proxy:
            controls: Final = tuple(
                _call_client(
                    call.client_kind,
                    call.endpoint,
                    control_proxy,
                    call.model,
                    call.prompt,
                    call.stream,
                    f"{call.call_id}-control",
                )
                for call in calls
            )
        control_upstream: Final = _drain_upstream(gateway.upstream_url)
        assert len(control_upstream) == 30, control_upstream
        assert (
            tuple(
                sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in control_upstream)
                for call in calls
            )
            == (1,) * 30
        ), control_upstream
        with wire_server(policy) as edge:
            config: Final = _configuration(tmp_path, identity, edge.url, "output", model_list=model_list)
            restart_gate: Final = threading.Event()
            second_wave_ready: Final = threading.Event()
            second_wave_barrier: Final = threading.Barrier(15, action=second_wave_ready.set)
            restarted_gateways: Final[SimpleQueue[Gateway]] = SimpleQueue()

            def gateway_for_call(call: ChaosCall, first_gateway: Gateway) -> Gateway:
                if call.index < 15:
                    return first_gateway
                second_wave_barrier.wait(timeout=90)
                assert restart_gate.wait(timeout=90)
                return restarted_gateways.get()

            def run(call: ChaosCall, first_gateway: Gateway) -> tuple[int, CallerResult]:
                candidate: Final = gateway_for_call(call, first_gateway)
                result: Final = _call_client(
                    call.client_kind,
                    call.endpoint,
                    candidate,
                    call.model,
                    call.prompt,
                    call.stream,
                    call.call_id,
                )
                return call.index, result

            with ThreadPoolExecutor(max_workers=30) as pool:
                with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as first_proxy:
                    first_proxy_port: Final = first_proxy.gateway.client.base_url.port
                    assert first_proxy_port is not None
                    futures: Final = tuple(pool.submit(run, call, first_proxy.gateway) for call in calls)
                    first_results: Final = tuple(futures[index].result(timeout=90)[1] for index in range(15))
                    assert all(
                        result.status == controls[call.index].status and result.text == controls[call.index].text
                        for call, result in zip(calls[:15], first_results)
                    ), first_results
                    assert eventually(lambda: second_wave_ready.is_set(), bool, seconds=30)
                    eventually(
                        lambda: edge.received.qsize(),
                        lambda count: count == 15 * len(expected_directions),
                        seconds=30,
                    )
                    first_wave_expected: Final = tuple(
                        (call.model, result.response_id, call.call_id)
                        for call, result in zip(calls[:15], first_results)
                    )
                    first_wave_rows: Final = _spend_rows_for_calls(
                        models,
                        first_wave_expected,
                        tolerate_missing=True,
                    )
                    assert all(
                        len(_spend_rows_matching_call(first_wave_rows, model, call_id)) <= 1
                        for model, _, call_id in first_wave_expected
                    ), first_wave_rows
                    first_wave_present_response_ids: Final = frozenset(
                        response_id
                        for model, response_id, call_id in first_wave_expected
                        if len(_spend_rows_matching_call(first_wave_rows, model, call_id)) == 1
                    )
                    first_wave_lost_response_ids: Final = (
                        frozenset(response_id for _, response_id, _ in first_wave_expected)
                        - first_wave_present_response_ids
                    )
                    record_property(
                        "K4_PRE_RESTART_LOST_RESPONSE_IDS",
                        tuple(sorted(first_wave_lost_response_ids)),
                    )
                    record_property(
                        f"K4_PRE_RESTART_LOST_ROW_COUNT_{'base' if _is_base_audit_leg() else 'head'}",
                        len(first_wave_lost_response_ids),
                    )
                assert first_proxy.process.poll() is not None, first_proxy.process.pid
                eventually(
                    lambda: tuple(
                        connection
                        for connection in psutil.net_connections(kind="tcp")
                        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == first_proxy_port
                    ),
                    lambda listeners: not listeners,
                    seconds=30,
                )
                with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as restarted_proxy:
                    for _ in range(15):
                        restarted_gateways.put(restarted_proxy.gateway)
                    restart_gate.set()
                    second_results: Final = tuple(futures[index].result(timeout=90)[1] for index in range(15, 30))
                    assert all(
                        result.status == controls[call.index].status and result.text == controls[call.index].text
                        for call, result in zip(calls[15:], second_results)
                    ), second_results
                    eventually(
                        lambda: edge.received.qsize(),
                        lambda count: count == 30 * len(expected_directions),
                        seconds=30,
                    )
            results: Final = first_results + second_results
            expected_response_ids: Final = frozenset(result.response_id for result in results)
            assert len(expected_response_ids) == 30, results
            upstream: Final = _drain_upstream(gateway.upstream_url)
            assert len(upstream) == 30, upstream
            assert (
                tuple(
                    sum(_json_contains_exact_string(observation["body"], call.prompt) for observation in upstream)
                    for call in calls
                )
                == (1,) * 30
            ), upstream
            edge_payloads: Final = tuple(JSON_OBJECT.validate_json(call.body) for call in edge.drain())
            assert len(edge_payloads) == 30 * len(expected_directions), edge_payloads
            for call, result in zip(calls, results):
                payloads_for_call: Final = tuple(
                    payload for payload in edge_payloads if _policy_call_id_matches(payload, call.call_id)
                )
                assert tuple(sorted(_direction(payload) for payload in payloads_for_call)) == tuple(
                    sorted(expected_directions)
                ), (
                    call,
                    payloads_for_call,
                )
                assert all(
                    payload["texts"] == ([call.prompt] if _direction(payload) == "request" else [result.text])
                    for payload in payloads_for_call
                ), payloads_for_call
            post_restart_expected: Final = tuple(
                (call.model, result.response_id, call.call_id) for call, result in zip(calls[15:], second_results)
            )
            rows: Final = _spend_rows_for_calls(
                models,
                post_restart_expected,
                tolerate_missing=True,
            )
            record_property("K4_RESPONSE_IDS", tuple(sorted(expected_response_ids)))
            for call, result in zip(calls, results):
                matching_rows: Final = _spend_rows_matching_call(rows, call.model, call.call_id)
                assert len(matching_rows) <= 1, (result.response_id, matching_rows)
                if call.index >= 15:
                    assert len(matching_rows) == 1, (call.call_id, result.response_id, matching_rows)
                if matching_rows:
                    entries: Final = _guardrail_entries(matching_rows[0])
                    assert tuple(
                        (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                        for entry in entries
                    ) == tuple((identity, "logging_only", "success") for _ in expected_directions), (
                        call.call_id,
                        entries,
                    )
