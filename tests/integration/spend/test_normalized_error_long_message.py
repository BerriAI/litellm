import asyncio
import json
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

CRAFTED_MODEL: Final = ("exceeded " * 32_000)[:288_000]
HOSTILE_5KB_MODEL: Final = ("exceeded budget " * 400)[:5_000]
FAST_SECONDS: Final = 10.0
LIVELINESS_MAX_SECONDS: Final = 5.0
ROW_SECONDS: Final = 70
CHAT: Final = "/v1/chat/completions"
MESSAGES: Final = "/v1/messages"
RESPONSES: Final = "/v1/responses"


def _body(path: str, model: str, marker: str, stream: bool = False) -> dict[str, JsonValue]:
    content: Final = f"normalized error audit {marker}"
    match path:
        case "/v1/messages":
            return {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
            }
        case "/v1/responses":
            return {"model": model, "input": content, "stream": stream}
        case _:
            return {"model": model, "messages": [{"role": "user", "content": content}], "stream": stream}


@dataclass(frozen=True, slots=True)
class _Timed:
    response: httpx.Response
    seconds: float


def _timed_post(client: httpx.Client, path: str, body: Mapping[str, JsonValue], key: str) -> _Timed:
    started: Final = time.perf_counter()
    response: Final = client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
    return _Timed(response, time.perf_counter() - started)


@contextmanager
def _patient_client(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=120, trust_env=False) as client:
        yield client


def _error_information(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT status, metadata->'error_information' AS info FROM \"LiteLLM_SpendLogs\" WHERE request_id=%s",
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=ROW_SECONDS,
    )
    assert rows[0]["status"] == "failure", rows
    return object_value(rows[0]["info"])


def _assert_crafted_failure(timed: _Timed, expected_status: int = 400) -> None:
    response: Final = timed.response
    assert response.status_code == expected_status, response.text[:300]
    assert "Invalid model name passed in" in response.text, response.text[:300]
    assert timed.seconds < FAST_SECONDS, f"crafted 288 KB model took {timed.seconds:.2f}s"
    info: Final = _error_information(response.headers["x-litellm-call-id"])
    assert info["normalized_error"] == "400_INVALID_REQUEST" and info["error_code"] == "400", info


@pytest.mark.parametrize("path", [CHAT, MESSAGES, RESPONSES])
def test_crafted_288kb_model_fails_fast_and_logs_invalid_request(gateway: Gateway, path: str) -> None:
    with gateway.scenario() as scenario, _patient_client(gateway) as client:
        key: Final = scenario.key()
        _assert_crafted_failure(_timed_post(client, path, _body(path, CRAFTED_MODEL, uuid.uuid4().hex), key))


@pytest.mark.parametrize("path", [CHAT, MESSAGES, RESPONSES])
def test_crafted_288kb_model_with_stream_true_fails_fast(gateway: Gateway, path: str) -> None:
    with gateway.scenario() as scenario, _patient_client(gateway) as client:
        key: Final = scenario.key()
        body: Final = _body(path, CRAFTED_MODEL, uuid.uuid4().hex, stream=True)
        _assert_crafted_failure(_timed_post(client, path, body, key))


def test_crafted_288kb_model_through_async_openai_sdk_fails_fast(gateway: Gateway) -> None:
    async def call(key: str) -> tuple[openai.BadRequestError, float]:
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, timeout=120)
        started: Final = time.perf_counter()
        try:
            with pytest.raises(openai.BadRequestError) as raised:
                await client.chat.completions.create(
                    model=CRAFTED_MODEL, messages=[{"role": "user", "content": f"audit {uuid.uuid4().hex}"}]
                )
            return raised.value, time.perf_counter() - started
        finally:
            await client.close()

    with gateway.scenario() as scenario:
        error, seconds = asyncio.run(call(scenario.key()))
        assert seconds < FAST_SECONDS, f"crafted 288 KB model took {seconds:.2f}s"
        assert "Invalid model name passed in" in str(error), str(error)[:300]
        info: Final = _error_information(error.response.headers["x-litellm-call-id"])
        assert info["normalized_error"] == "400_INVALID_REQUEST", info


def test_crafted_288kb_model_through_anthropic_sdk_fails_fast(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=scenario.key(), timeout=120)
        started: Final = time.perf_counter()
        with pytest.raises(anthropic.BadRequestError) as raised:
            client.messages.create(
                model=CRAFTED_MODEL, max_tokens=8, messages=[{"role": "user", "content": f"audit {uuid.uuid4().hex}"}]
            )
        seconds: Final = time.perf_counter() - started
        assert seconds < FAST_SECONDS, f"crafted 288 KB model took {seconds:.2f}s"
        assert "Invalid model name passed in" in str(raised.value), str(raised.value)[:300]
        info: Final = _error_information(raised.value.response.headers["x-litellm-call-id"])
        assert info["normalized_error"] == "400_INVALID_REQUEST", info


def _poll_liveliness(client: httpx.Client, stop: threading.Event) -> list[float]:
    latencies: Final[list[float]] = []  # mutable-ok: thread-local sample buffer drained once by the caller
    while not stop.is_set():
        started = time.perf_counter()
        assert client.get("/health/liveliness").status_code == 200
        latencies.append(time.perf_counter() - started)
        stop.wait(0.1)
    return latencies


def _cmdline(process: psutil.Process) -> str:
    try:
        return " ".join(process.cmdline())
    except psutil.Error:
        return ""


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(child.pid for child in psutil.Process(owned.process.pid).children() if "spawn_main" in _cmdline(child))


def test_two_concurrent_crafted_requests_do_not_stall_liveliness_on_a_two_worker_proxy(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned, _patient_client(owned.gateway) as client:
        assert len(_worker_pids(owned)) == 2, _worker_pids(owned)
        stop: Final = threading.Event()
        with ThreadPoolExecutor(max_workers=3) as pool:
            liveliness: Final = pool.submit(_poll_liveliness, client, stop)
            crafted: Final = tuple(
                pool.submit(_timed_post, client, CHAT, _body(CHAT, CRAFTED_MODEL, uuid.uuid4().hex), gateway.key)
                for _ in range(2)
            )
            results: Final = tuple(future.result() for future in crafted)
            stop.set()
            latencies: Final = liveliness.result()
        for timed in results:
            _assert_crafted_failure(timed)
        assert latencies and max(latencies) < LIVELINESS_MAX_SECONDS, f"liveliness max {max(latencies):.2f}s"


def _completion(request: Request) -> Reply:
    body: Final = object_value(json.loads(request.body or b"{}"))
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model", "unknown"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
            }
        ).encode()
    )


def _rate_limited(message: str) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        return Reply(
            status=429,
            body=json.dumps({"error": {"message": message, "type": "rate_limit_error", "code": "429"}}).encode(),
        )

    return respond


def _budget_denied_row(key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, metadata->'error_information' AS info FROM \"LiteLLM_SpendLogs\" "
            "WHERE api_key=%s AND status='failure'",
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == 1,
        seconds=ROW_SECONDS,
    )
    return object_value(rows[0]["info"])


def _exhaust(
    gateway: Gateway, client: httpx.Client, model: str, key: str, table: str, column: str, identity: str
) -> None:
    first: Final = gateway.chat(model, key=key, text=f"spend {uuid.uuid4().hex}")
    assert object_value(first["usage"])["total_tokens"] == 40, first
    eventually(
        lambda: read_rows(f'SELECT spend FROM "{table}" WHERE {column}=%s', (identity,)),
        lambda values: len(values) == 1 and float(string_value(str(values[0]["spend"]))) >= 0.06,
        seconds=ROW_SECONDS,
    )
    denied: Final = eventually(
        lambda: client.post(
            CHAT, json=_body(CHAT, model, uuid.uuid4().hex), headers={"Authorization": f"Bearer {key}"}
        ),
        lambda response: response.status_code in {400, 422},
        seconds=ROW_SECONDS,
    )
    assert denied.json()["error"]["type"] == "budget_exceeded", denied.text
    info: Final = _budget_denied_row(key)
    assert info["normalized_error"] == "429_BUDGET_EXCEEDED", info
    assert "budget" in string_value(info["error_message"]).lower(), info


def test_exhausted_key_budget_denial_clusters_as_budget_exceeded(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _patient_client(gateway) as client:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        _exhaust(gateway, client, model, key, "LiteLLM_VerificationToken", "token", sha256(key.encode()).hexdigest())


def test_exhausted_team_budget_denial_clusters_as_budget_exceeded(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _patient_client(gateway) as client:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model], max_budget=0.06)
        key: Final = scenario.key(team_id=team, models=[model])
        _exhaust(gateway, client, model, key, "LiteLLM_TeamTable", "team_id", team)


def _upstream_failure_row(gateway: Gateway, message: str) -> dict[str, JsonValue]:
    with wire_server(_rate_limited(message)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        failed: Final = gateway.request("POST", CHAT, _body(CHAT, model, uuid.uuid4().hex), key=key)
        assert failed.status_code == 429 and message in failed.json()["error"]["message"], failed.text[:300]
        assert len(wire.drain()) == 1
        return _error_information(failed.headers["x-litellm-call-id"])


@pytest.mark.parametrize(
    "message",
    [
        "Budget has been exceeded! Current cost: 11.0, Max budget: 10.0",
        "ExceededBudget: User=audit over budget. Spend=12.5, Budget=10.0",
        "Exceeded budget for provider openai: 105.2 >= 100.0",
        "exceeded" + "x" * 64 + "budget",
    ],
)
def test_upstream_budget_wording_clusters_as_budget_exceeded(gateway: Gateway, message: str) -> None:
    info: Final = _upstream_failure_row(gateway, message)
    assert info["normalized_error"] == "429_BUDGET_EXCEEDED", info


def test_upstream_exceeded_and_budget_65_chars_apart_still_clusters_as_budget_exceeded(gateway: Gateway) -> None:
    info: Final = _upstream_failure_row(gateway, "exceeded" + "x" * 65 + "budget")
    assert info["normalized_error"] == "429_BUDGET_EXCEEDED", info


def test_upstream_exceeded_and_budget_on_different_lines_cluster_by_exception_class(gateway: Gateway) -> None:
    info: Final = _upstream_failure_row(gateway, "exceeded the limit\nbudget unaffected")
    assert info["normalized_error"] == "429_RATE_LIMIT_EXCEEDED", info


def test_hostile_model_values_are_rejected_without_taking_the_proxy_down(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _patient_client(gateway) as client:
        key: Final = scenario.key()
        hostile_values: Final[tuple[JsonValue, ...]] = (5, ["gpt-4o-mini"])
        for hostile in hostile_values:
            rejected = _timed_post(client, CHAT, {"model": hostile, "messages": []}, key)
            assert rejected.response.status_code == 400 and "must be a string" in rejected.response.text
            assert _error_information(rejected.response.headers["x-litellm-call-id"])["normalized_error"] == (
                "400_INVALID_REQUEST"
            )
        empty: Final = _timed_post(client, CHAT, _body(CHAT, "", uuid.uuid4().hex), key)
        assert empty.response.status_code == 400, empty.response.text
        assert _error_information(empty.response.headers["x-litellm-call-id"])["normalized_error"] == (
            "400_INVALID_REQUEST"
        )
        repeated: Final = tuple(
            _timed_post(client, CHAT, _body(CHAT, HOSTILE_5KB_MODEL, uuid.uuid4().hex), key) for _ in range(2)
        )
        call_ids: Final = tuple(timed.response.headers["x-litellm-call-id"] for timed in repeated)
        assert len(set(call_ids)) == 2 and all(timed.response.status_code == 400 for timed in repeated)
        assert all(timed.seconds < FAST_SECONDS for timed in repeated), [timed.seconds for timed in repeated]
        codes: Final = tuple(_error_information(call_id)["normalized_error"] for call_id in call_ids)
        assert len(set(codes)) == 1 and codes[0] in {"400_INVALID_REQUEST", "429_BUDGET_EXCEEDED"}, codes
        unauthenticated: Final = client.post(CHAT, json=_body(CHAT, "gpt-4o-mini", "x"))
        assert unauthenticated.status_code == 401, unauthenticated.text
        assert client.get("/health/liveliness").status_code == 200


@dataclass(frozen=True, slots=True)
class _BurstResult:
    label: str
    status: int | None
    call_id: str | None
    response_id: str | None
    seconds: float


def _burst_call(client: httpx.Client, label: str, path: str, body: Mapping[str, JsonValue], key: str) -> _BurstResult:
    started: Final = time.perf_counter()
    try:
        response: Final = client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
    except httpx.TransportError:
        return _BurstResult(label, None, None, None, time.perf_counter() - started)
    identity: Final = object_value(response.json()).get("id") if response.status_code == 200 else None
    return _BurstResult(
        label,
        response.status_code,
        response.headers.get("x-litellm-call-id"),
        identity if isinstance(identity, str) else None,
        time.perf_counter() - started,
    )


def _burst(
    client: httpx.Client, happy_model: str, happy_key: str, open_key: str, during: Callable[[], None]
) -> tuple[_BurstResult, ...]:
    crafted: Final = tuple(
        (f"crafted-{path}-{index}", path, _body(path, CRAFTED_MODEL, uuid.uuid4().hex, stream=index % 2 == 1), open_key)
        for path in (CHAT, MESSAGES, RESPONSES)
        for index in range(4)
    )
    happy: Final = tuple(
        (f"happy-{index}", CHAT, _body(CHAT, happy_model, f"happy-{index}"), happy_key) for index in range(8)
    )
    late: Final = tuple(
        (f"late-{index}", CHAT, _body(CHAT, happy_model, f"late-{index}"), happy_key) for index in range(8)
    )
    with (
        ThreadPoolExecutor(max_workers=28) as pool,
        httpx.Client(base_url=client.base_url, timeout=client.timeout, trust_env=False) as fresh,
    ):
        first: Final = tuple(
            pool.submit(_burst_call, client, label, path, body, key) for label, path, body, key in crafted + happy
        )
        wait(first, return_when=FIRST_COMPLETED)
        during()
        second: Final = tuple(
            pool.submit(_burst_call, fresh, label, path, body, key) for label, path, body, key in late
        )
        return tuple(future.result() for future in first + second)


def _assert_rows_land_exactly_once(results: tuple[_BurstResult, ...], prefix: str) -> None:
    landed: Final = tuple(result for result in results if result.label.startswith(prefix) and result.status == 200)
    assert landed, results
    response_ids: Final = tuple(string_value(result.response_id) for result in landed)
    assert len(set(response_ids)) == len(response_ids), response_ids
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s::text[])',
            ("{" + ",".join(response_ids) + "}",),
        ),
        lambda values: len(values) == len(response_ids),
        seconds=ROW_SECONDS,
    )
    assert sorted(string_value(row["request_id"]) for row in rows) == sorted(response_ids), rows
    assert all(row["status"] == "success" for row in rows), rows


def test_killing_one_worker_mid_burst_leaves_the_other_serving_crafted_and_happy_traffic(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        wire_server(_completion) as wire,
        owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned,
        owned.gateway.scenario() as scenario,
        _patient_client(owned.gateway) as client,
    ):
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        workers: Final = _worker_pids(owned)
        assert len(workers) == 2, workers

        def kill_one_worker() -> None:
            os.kill(workers[0], signal.SIGKILL)

        results: Final = _burst(client, model, key, scenario.key(), kill_one_worker)
        dropped: Final = tuple(result for result in results if result.status is None)
        assert len(dropped) < len(results), results
        crafted: Final = tuple(result for result in results if result.label.startswith("crafted") and result.status)
        assert crafted and all(result.status == 400 and result.seconds < FAST_SECONDS for result in crafted), crafted
        late: Final = tuple(result for result in results if result.label.startswith("late"))
        assert all(result.status == 200 for result in late), late
        _assert_rows_land_exactly_once(results, "late")
        survivor: Final = tuple(pid for pid in _worker_pids(owned) if pid != workers[0])
        assert survivor, "no worker left serving"
        after: Final = owned.gateway.chat(model, key=key, text=f"after kill {uuid.uuid4().hex}")
        assert isinstance(after["id"], str) and after["id"].startswith("chatcmpl-"), after
        assert client.get("/health/liveliness").status_code == 200


def test_upstream_returning_503_mid_burst_logs_every_failure_with_its_own_cluster_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    def overloaded(_request: Request) -> Reply:
        return Reply(
            status=503,
            body=b'{"error":{"message":"Controlled provider outage","type":"server_error","code":"503"}}',
        )

    with (
        wire_server(overloaded) as wire,
        owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned,
        owned.gateway.scenario() as scenario,
        _patient_client(owned.gateway) as client,
    ):
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        results: Final = _burst(client, model, key, scenario.key(), lambda: None)
        assert all(result.status is not None for result in results), results
        happy: Final = tuple(result for result in results if result.label.startswith("happy"))
        assert all(result.status == 503 for result in happy), happy
        late: Final = tuple(result for result in results if result.label.startswith("late"))
        assert all(result.status == 503 for result in late), late
        seen: Final = tuple(request.body.decode() for request in wire.drain())
        assert all(any(f"normalized error audit {result.label}" in body for body in seen) for result in happy + late), (
            seen
        )
        crafted: Final = tuple(result for result in results if result.label.startswith("crafted"))
        assert all(result.status == 400 and result.seconds < FAST_SECONDS for result in crafted), crafted
        codes: Final = {
            result.label: _error_information(string_value(result.call_id))["normalized_error"] for result in results
        }
        assert all(code == "503_PROVIDER_OVERLOADED" for label, code in codes.items() if label.startswith("happy")), (
            codes
        )
        assert all(code == "400_INVALID_REQUEST" for label, code in codes.items() if label.startswith("crafted")), codes
        assert client.get("/health/liveliness").status_code == 200
