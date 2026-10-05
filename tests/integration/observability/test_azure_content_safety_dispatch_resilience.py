import concurrent.futures
import socket
import threading
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy
from integration._support.redis_process import owned_redis
from integration._support.wire import Wire, wire_server
from integration.observability.azure_dispatch_support import (
    ATTACK_MARKER,
    AzureBehavior,
    azure_handler,
    azure_texts,
    dispatch_proxy,
    dispatch_rig,
    provider_handler,
    provider_texts,
    write_dispatch_config,
)
from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class CallSpec:
    phase: str
    prompt: str
    path: str
    body: dict[str, JsonValue]
    stream: bool


@dataclass(frozen=True, slots=True)
class CallResult:
    spec: CallSpec
    status: int
    text: str
    headers: dict[str, str]


def _model(scenario: Scenario, provider: Wire, name: str = "openai/gpt-4o-mini") -> str:
    return scenario.model(model=name, api_base=provider.url + "/v1", api_key="synthetic-provider-key")


def _anthropic_model(scenario: Scenario, provider: Wire) -> str:
    return scenario.model(
        model="anthropic/claude-sonnet-4-5-20250929",
        api_base=provider.url,
        api_key="synthetic-provider-key",
    )


def _assert_spend(request_id: str, response_text: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT litellm_call_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert len(rows) == 1, response_text


def _request(
    candidate: Gateway,
    path: str,
    body: dict[str, JsonValue],
    *,
    stream: bool = False,
) -> tuple[int, str, dict[str, str]]:
    if not stream:
        response: Final = candidate.request("POST", path, body)
        return response.status_code, response.text, dict(response.headers)
    with candidate.client.stream(
        "POST",
        path,
        json=body,
        headers={"Authorization": f"Bearer {candidate.key}"},
    ) as response:
        response.read()
        return response.status_code, response.text, dict(response.headers)


def _safe_request(candidate: Gateway, model: str, prompt: str) -> httpx.Response | None:
    try:
        return candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": ["shield"],
                "cache": {"no-cache": True},
            },
        )
    except httpx.HTTPError:
        return None


def _worker_processes(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    descendants: Final = tuple(psutil.Process(owned.process.pid).children(recursive=True))
    return tuple(
        process
        for process in descendants
        if process.is_running()
        and any("spawn_main" in argument for argument in process.cmdline())
    )


def _operation_specs(
    phase: str,
    operation: str,
    openai_model: str,
    anthropic_model: str,
    attack: bool,
) -> tuple[CallSpec, ...]:
    marker: Final = ATTACK_MARKER if attack else "synthetic-benign"
    if operation == "chat":
        return tuple(
            CallSpec(
                phase,
                f"{phase} chat {marker} {index}",
                "/v1/chat/completions",
                {
                    "model": openai_model,
                    "messages": [{"role": "user", "content": f"{phase} chat {marker} {index}"}],
                    "guardrails": ["tuple-writer", "shield"],
                    "cache": {"no-cache": True},
                },
                False,
            )
            for index in range(2)
        )
    if operation == "chat-stream":
        return tuple(
            CallSpec(
                phase,
                f"{phase} chat stream {marker} {index}",
                "/v1/chat/completions",
                {
                    "model": openai_model,
                    "messages": [{"role": "user", "content": f"{phase} chat stream {marker} {index}"}],
                    "guardrails": ["tuple-writer", "shield"],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
                True,
            )
            for index in range(2)
        )
    if operation == "messages":
        return tuple(
            CallSpec(
                phase,
                f"{phase} messages {marker} {index}",
                "/v1/messages",
                {
                    "model": anthropic_model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"{phase} messages {marker} {index}"}],
                    "guardrails": ["tuple-writer", "shield"],
                    "cache": {"no-cache": True},
                },
                False,
            )
            for index in range(2)
        )
    if operation == "responses":
        return tuple(
            CallSpec(
                phase,
                f"{phase} responses {marker} {index}",
                "/v1/responses",
                {
                    "model": openai_model,
                    "input": f"{phase} responses {marker} {index}",
                    "guardrails": ["shield"],
                    "cache": {"no-cache": True},
                },
                False,
            )
            for index in range(2)
        )
    return tuple(
        CallSpec(
            phase,
            f"{phase} list {marker} {index}",
            "/v1/chat/completions",
            {
                "model": openai_model,
                "messages": [{"role": "user", "content": f"{phase} list {marker} {index}"}],
                "guardrails": ["shield"],
                "cache": {"no-cache": True},
            },
            False,
        )
        for index in range(2)
    )


def _phase_specs(
    phase: str, openai_model: str, anthropic_model: str, attack: bool
) -> tuple[CallSpec, ...]:
    return (
        *_operation_specs(phase, "chat", openai_model, anthropic_model, attack),
        *_operation_specs(phase, "chat-stream", openai_model, anthropic_model, attack),
        *_operation_specs(phase, "messages", openai_model, anthropic_model, attack),
        *_operation_specs(phase, "responses", openai_model, anthropic_model, attack),
        *_operation_specs(phase, "list", openai_model, anthropic_model, attack),
    )


def _wait_and_request(candidate: Gateway, gate: threading.Event, spec: CallSpec) -> CallResult:
    assert gate.wait(timeout=45), f"{spec.phase} request gate was not released"
    status, text, headers = _request(candidate, spec.path, spec.body, stream=spec.stream)
    return CallResult(spec, status, text, headers)


def test_h20_yaml_default_on_scans_tuple_attack_and_benign(
    gateway: Gateway, tmp_path: Path
) -> None:
    with dispatch_rig(gateway, tmp_path, default_on=True) as (owned, azure, provider, _):
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = _model(scenario, provider)
            attack: Final = f"default-on {ATTACK_MARKER} {uuid.uuid4().hex}"
            blocked: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": attack}],
                    "cache": {"no-cache": True},
                },
            )
            assert blocked.status_code == 400, blocked.text
            assert azure_texts(azure) == (attack,), blocked.text
            assert provider.drain() == (), blocked.text
            benign: Final = f"default-on benign {uuid.uuid4().hex}"
            allowed: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": benign}],
                    "cache": {"no-cache": True},
                },
            )
            assert allowed.status_code == 200, allowed.text
            assert azure_texts(azure) == (benign,), allowed.text
            assert provider_texts(provider) == (benign,), allowed.text
            _assert_spend(allowed.headers["x-litellm-call-id"], allowed.text)


def test_x1_azure_server_stop_restart_during_mixed_burst(
    gateway: Gateway, tmp_path: Path
) -> None:
    with ExitStack() as resources:
        redis: Final = resources.enter_context(owned_redis(tmp_path))
        provider: Final = resources.enter_context(wire_server(provider_handler))
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            azure_port: Final = int(reservation.getsockname()[1])
        first_azure_lifetime: Final = ExitStack()
        try:
            azure: Final = first_azure_lifetime.enter_context(wire_server(azure_handler(), port=azure_port))
            owned: Final = resources.enter_context(dispatch_proxy(gateway, tmp_path, redis, azure.url, workers=2))
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                openai_model: Final = _model(scenario, provider)
                anthropic_model: Final = _anthropic_model(scenario, provider)
                up_specs: Final = _phase_specs("up", openai_model, anthropic_model, True)
                down_specs: Final = _phase_specs("down", openai_model, anthropic_model, True)
                recovery_specs: Final = _phase_specs("recovery", openai_model, anthropic_model, False)
                up_gate: Final = threading.Event()
                down_gate: Final = threading.Event()
                recovery_gate: Final = threading.Event()
                specs: Final = (*up_specs, *down_specs, *recovery_specs)
                gates: Final = (
                    *((up_gate,) * len(up_specs)),
                    *((down_gate,) * len(down_specs)),
                    *((recovery_gate,) * len(recovery_specs)),
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                    futures: Final = tuple(
                        executor.submit(_wait_and_request, candidate, gate, spec)
                        for gate, spec in zip(gates, specs, strict=True)
                    )
                    up_gate.set()
                    up_results: Final = tuple(future.result(timeout=70) for future in futures[:10])
                    assert all(result.status == 400 for result in up_results), tuple(
                        result.text for result in up_results
                    )
                    first_azure_texts: Final = azure_texts(azure)
                    first_azure_lifetime.close()
                    down_gate.set()
                    down_results: Final = tuple(future.result(timeout=70) for future in futures[10:20])
                    assert all(result.status != 200 for result in down_results), tuple(
                        result.text for result in down_results
                    )
                    restarted_azure_lifetime: Final = ExitStack()
                    try:
                        restarted_azure: Final = restarted_azure_lifetime.enter_context(
                            wire_server(azure_handler(), port=azure_port)
                        )
                        recovery_gate.set()
                        recovery_results: Final = tuple(future.result(timeout=70) for future in futures[20:])
                        assert all(result.status == 200 for result in recovery_results), tuple(
                            result.text for result in recovery_results
                        )
                        restarted_texts: Final = azure_texts(restarted_azure)
                        assert len(restarted_texts) == len(recovery_results), recovery_results[-1].text
                        assert set(restarted_texts) == {result.spec.prompt for result in recovery_results}, (
                            recovery_results[-1].text
                        )
                    finally:
                        restarted_azure_lifetime.close()
                    assert set(first_azure_texts) == {result.spec.prompt for result in up_results}, up_results[0].text
                    provider_requests: Final = provider_texts(provider)
                    assert all(result.spec.prompt not in provider_requests for result in down_results), (
                        down_results[0].text
                    )
                    assert set(provider_requests) == {result.spec.prompt for result in recovery_results}, (
                        recovery_results[-1].text
                    )
                    for result in (*up_results, *down_results, *recovery_results):
                        if result.status == 200:
                            _assert_spend(result.headers["x-litellm-call-id"], result.text)
        finally:
            first_azure_lifetime.close()


def test_x2_slow_azure_edge_completes_twenty_concurrent_requests(
    gateway: Gateway, tmp_path: Path
) -> None:
    with dispatch_rig(gateway, tmp_path, behavior=AzureBehavior(delay_seconds=0.2)) as (
        owned,
        azure,
        provider,
        _,
    ):
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = _model(scenario, provider)
            prompts: Final = tuple(f"slow edge {uuid.uuid4().hex}" for _ in range(20))
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures: Final = tuple(
                    executor.submit(
                        candidate.request,
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "guardrails": ["shield"],
                            "cache": {"no-cache": True},
                        },
                    )
                    for prompt in prompts
                )
                responses: Final = tuple(future.result(timeout=70) for future in futures)
            assert all(response.status_code == 200 for response in responses), tuple(
                response.text for response in responses
            )
            scanned_prompts: Final = azure_texts(azure)
            assert len(scanned_prompts) == len(prompts), responses[-1].text
            assert set(scanned_prompts) == set(prompts), responses[-1].text
            assert set(provider_texts(provider)) == set(prompts), responses[-1].text
            for response in responses:
                _assert_spend(response.headers["x-litellm-call-id"], response.text)


def test_x3_killing_one_worker_leaves_the_other_serving(
    gateway: Gateway, tmp_path: Path
) -> None:
    entered: Final = threading.Event()
    with dispatch_rig(gateway, tmp_path, behavior=AzureBehavior(delay_seconds=0.15, entered=entered)) as (
        owned,
        azure,
        provider,
        _,
    ):
        candidate: Final = owned.gateway
        workers: Final = _worker_processes(owned)
        assert len(workers) == 2, tuple(process.cmdline() for process in workers)
        with candidate.scenario() as scenario:
            model: Final = _model(scenario, provider)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                inflight: Final = tuple(
                    executor.submit(_safe_request, candidate, model, f"worker in-flight {uuid.uuid4().hex}")
                    for _ in range(4)
                )
                assert entered.wait(timeout=30), "No request reached the slow Azure edge"
                killed_worker: Final = workers[0]
                survivor: Final = workers[1]
                killed_worker.kill()
                killed_worker.wait(timeout=10)
                assert survivor.is_running(), "The second proxy worker exited with the killed worker"
                prompts: Final = tuple(f"worker survivor {uuid.uuid4().hex}" for _ in range(8))
                futures: Final = tuple(
                    executor.submit(_safe_request, candidate, model, prompt)
                    for prompt in prompts
                )
                responses: Final = tuple(future.result(timeout=70) for future in (*inflight, *futures))
            surviving_responses: Final = tuple(response for response in responses[4:] if response is not None)
            assert all(response.status_code == 200 for response in surviving_responses), tuple(
                response.text for response in surviving_responses
            )
            assert len(surviving_responses) == len(prompts), tuple(
                response.text for response in surviving_responses if response is not None
            )
            assert set(azure_texts(azure)).issuperset(prompts), surviving_responses[-1].text
            assert set(provider_texts(provider)).issuperset(prompts), surviving_responses[-1].text
            for response in surviving_responses:
                _assert_spend(response.headers["x-litellm-call-id"], response.text)


def test_x4_proxy_restart_preserves_completed_spend_rows(
    gateway: Gateway, tmp_path: Path
) -> None:
    entered: Final = threading.Event()
    arrived: Final = threading.Semaphore(0)
    release: Final = threading.Event()
    behavior: Final = AzureBehavior(entered=entered, arrived=arrived, release=release, barrier_marker="X4_WAIT")
    with ExitStack() as resources:
        redis: Final = resources.enter_context(owned_redis(tmp_path))
        azure: Final = resources.enter_context(wire_server(azure_handler(behavior)))
        provider: Final = resources.enter_context(wire_server(provider_handler))
        first_proxy_lifetime: Final = ExitStack()
        try:
            first: Final = first_proxy_lifetime.enter_context(
                dispatch_proxy(gateway, tmp_path, redis, azure.url, workers=2)
            )
            with gateway.scenario() as scenario:
                model: Final = _model(scenario, provider)
                completed_prompts: Final = tuple(f"X4 completed {uuid.uuid4().hex}" for _ in range(5))
                completed: Final = tuple(
                    first.gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "guardrails": ["shield"],
                            "cache": {"no-cache": True},
                        },
                    )
                    for prompt in completed_prompts
                )
                assert all(response.status_code == 200 for response in completed), tuple(
                    response.text for response in completed
                )
                for response in completed:
                    _assert_spend(response.headers["x-litellm-call-id"], response.text)
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    interrupted_prompts: Final = tuple(f"X4_WAIT {uuid.uuid4().hex}" for _ in range(10))
                    interrupted: Final = tuple(
                        executor.submit(_safe_request, first.gateway, model, prompt)
                        for prompt in interrupted_prompts
                    )
                    arrived_count: Final = sum(arrived.acquire(timeout=30) for _ in interrupted_prompts)
                    assert arrived_count == len(interrupted_prompts), (
                        f"Only {arrived_count} of {len(interrupted_prompts)} interrupted requests reached Azure"
                    )
                    first.process.terminate()
                    release.set()
                    first_proxy_lifetime.close()
                    interrupted_results: Final = tuple(future.result(timeout=70) for future in interrupted)
                    second_proxy_lifetime: Final = ExitStack()
                    try:
                        second: Final = second_proxy_lifetime.enter_context(
                            dispatch_proxy(gateway, tmp_path, redis, azure.url, workers=2)
                        )
                        recovery_prompt: Final = f"X4 restarted benign {uuid.uuid4().hex}"
                        recovered: Final = second.gateway.request(
                            "POST",
                            "/v1/chat/completions",
                            {
                                "model": model,
                                "messages": [{"role": "user", "content": recovery_prompt}],
                                "guardrails": ["shield"],
                                "cache": {"no-cache": True},
                            },
                        )
                        assert recovered.status_code == 200, recovered.text
                        for response in completed:
                            _assert_spend(response.headers["x-litellm-call-id"], response.text)
                        _assert_spend(recovered.headers["x-litellm-call-id"], recovered.text)
                        azure_received: Final = azure_texts(azure)
                        azure_prompts: Final = frozenset(azure_received)
                        provider_received: Final = provider_texts(provider)
                        provider_prompts: Final = frozenset(provider_received)
                        expected_provider_prompts: Final = frozenset((*completed_prompts, recovery_prompt))
                        expected_edge_prompts: Final = tuple(
                            sorted((*completed_prompts, *interrupted_prompts, recovery_prompt))
                        )
                        assert tuple(sorted(azure_received)) == expected_edge_prompts, recovered.text
                        assert provider_prompts <= azure_prompts, recovered.text
                        assert expected_provider_prompts <= provider_prompts, recovered.text
                        for prompt, response in zip(interrupted_prompts, interrupted_results, strict=True):
                            if response is not None and response.status_code == 200:
                                assert prompt in provider_prompts, response.text
                    finally:
                        second_proxy_lifetime.close()
        finally:
            release.set()
            first_proxy_lifetime.close()
