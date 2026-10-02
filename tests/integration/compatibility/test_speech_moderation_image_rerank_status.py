from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import JsonResponse


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    path: str
    route: str
    required: tuple[str, ...]
    body: dict[str, JsonValue]
    model: str


_SPEECH: Final = _Case(
    "speech", "/v1/audio/speech", "/audio/speech", ("input",), {"voice": "alloy"}, "openai/gpt-4o-mini-tts"
)
_MODERATION: Final = _Case(
    "moderations", "/v1/moderations", "/moderations", ("input",), {"input": "moderate this"}, "openai/gpt-4o-mini"
)
_IMAGE: Final = _Case(
    "images", "/v1/images/generations", "/image/generations", ("prompt",), {"prompt": "audit"}, "openai/gpt-image-1"
)
_RERANK: Final = _Case(
    "rerank",
    "/v1/rerank",
    "/rerank",
    ("query", "documents"),
    {"query": "rank", "documents": ["first", "second"]},
    "cohere/rerank-v4.0",
)
_BODIES: Final[dict[str, dict[str, JsonValue]]] = {
    "moderations": {
        "id": "modr-$UNIQUE_ID",
        "model": "omni-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    },
    "images": {"created": 1, "data": [{"url": "https://images.invalid/audit.png"}]},
    "rerank": {"id": "rerank-$UNIQUE_ID", "results": [{"index": 0, "relevance_score": 0.5}], "meta": {}},
}
_MISSING: Final = (
    pytest.param(
        _SPEECH,
        "input",
        id="speech",
    ),
    pytest.param(
        _MODERATION,
        "input",
        id="moderations",
    ),
    pytest.param(
        _IMAGE,
        "prompt",
        id="images",
    ),
    pytest.param(
        _RERANK,
        "query",
        id="rerank-query",
    ),
    pytest.param(
        _RERANK,
        "documents",
        id="rerank-documents",
    ),
)
_VALID: Final = (
    pytest.param(_MODERATION, id="moderations"),
    pytest.param(_IMAGE, id="images"),
    pytest.param(_RERANK, id="rerank"),
)


class _Observations:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(client.get(f"{self.url}/__observations").json())
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self.items = (*self.items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self.items

    def for_scenario(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(item for item in self.items if f"/{identity}/" in str(item.get("path")))


def _response(body: dict[str, JsonValue] | None = None) -> JsonResponse:
    return JsonResponse(content_type="application/json", body=body or {"id": "audit-$UNIQUE_ID", "object": "response"})


def _assert_scripted_response(case: _Case, caller: dict[str, JsonValue]) -> None:
    scripted: Final = _BODIES[case.name]
    if case.name == "moderations":
        expected_results: Final = scripted["results"]
        actual_results: Final = caller["results"]
        assert isinstance(expected_results, list) and isinstance(actual_results, list)
        expected_result: Final = object_value(expected_results[0])
        actual_result: Final = object_value(actual_results[0])
        assert actual_result["flagged"] is expected_result["flagged"]
    elif case.name == "images":
        expected_data: Final = scripted["data"]
        actual_data: Final = caller["data"]
        assert isinstance(expected_data, list) and isinstance(actual_data, list)
        expected_item: Final = object_value(expected_data[0])
        actual_item: Final = object_value(actual_data[0])
        assert actual_item["url"] == expected_item["url"]
    else:
        expected_results = scripted["results"]
        actual_results = caller["results"]
        assert isinstance(expected_results, list) and isinstance(actual_results, list)
        expected_result = object_value(expected_results[0])
        actual_result = object_value(actual_results[0])
        assert actual_result["index"] == expected_result["index"]
        assert actual_result["relevance_score"] == expected_result["relevance_score"]


def _model(scenario: Scenario, case: _Case, spec: JsonResponse) -> tuple[str, str]:
    identity: Final = f"audit-{uuid.uuid4().hex}"
    handle: Final = register_scenario(identity, spec)
    scenario.cleanups.callback(delete_scenario, handle)
    alias: Final = scenario.model(model=case.model, api_base=handle.api_base(), api_key=identity)
    return alias, identity


def _owned_model(
    scenario: Scenario,
    url: str,
    case: _Case,
    script: JsonResponse,
) -> tuple[str, str, ScenarioHandle]:
    identity: Final = f"chaos-{case.name}-{uuid.uuid4().hex}"
    handle: Final = _register_owned(url, identity, script)
    scenario.cleanups.callback(_delete_owned, handle)
    alias: Final = scenario.model(model=case.model, api_base=handle.api_base(), api_key=identity)
    return alias, identity, handle


def _expected_error(route: str, parameter: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"{route}: Missing required parameter: '{parameter}'.",
            "type": "invalid_request_error",
            "param": parameter,
            "code": "400",
        }
    }


def _post(gateway: Gateway, path: str, body: dict[str, JsonValue], call_id: str | None = None) -> httpx.Response:
    headers: Final = {} if call_id is None else {"x-litellm-call-id": call_id}
    return gateway.request("POST", path, body, headers=headers)


def _ready_missing(
    gateway: Gateway,
    path: str,
    body: dict[str, JsonValue],
    expected: dict[str, JsonValue],
    call_id: str | None = None,
) -> httpx.Response:
    return eventually(
        lambda: _post(gateway, path, body, call_id),
        lambda response: response.status_code == 400 and response.json() == expected,
        seconds=30,
    )


@pytest.mark.parametrize(("case", "parameter"), _MISSING)
def test_missing_required_body_field_returns_exact_400(gateway: Gateway, case: _Case, parameter: str) -> None:
    with gateway.scenario() as scenario:
        alias, identity = _model(scenario, case, _response())
        body: Final = {key: value for key, value in {**case.body, "model": alias}.items() if key != parameter}
        expected: Final = _expected_error(case.route, parameter)
        response: Final = _ready_missing(gateway, case.path, body, expected)
        assert response.status_code == 400, response.text
        assert response.json() == expected, response.text
        observations: Final = _Observations(gateway.upstream_url)
        observations.read()
        assert observations.for_scenario(identity) == ()


@pytest.mark.parametrize(
    ("case", "parameter"),
    (
        pytest.param(_IMAGE, "prompt", id="images"),
        pytest.param(_RERANK, "query", id="rerank-query"),
        pytest.param(_RERANK, "documents", id="rerank-documents"),
    ),
)
def test_image_and_rerank_missing_fields_echo_call_id(gateway: Gateway, case: _Case, parameter: str) -> None:
    call_id: Final = f"audit-call-{case.name}-{parameter}"
    with gateway.scenario() as scenario:
        alias, identity = _model(scenario, case, _response())
        body: Final = {key: value for key, value in {**case.body, "model": alias}.items() if key != parameter}
        expected: Final = _expected_error(case.route, parameter)
        response: Final = _ready_missing(gateway, case.path, body, expected, call_id=call_id)
        assert response.status_code == 400, response.text
        assert response.json() == expected, response.text
        assert response.headers.get("x-litellm-call-id") == call_id, response.headers
        observations: Final = _Observations(gateway.upstream_url)
        observations.read()
        assert observations.for_scenario(identity) == ()


@pytest.mark.parametrize("case", _VALID)
def test_valid_body_reaches_scripted_upstream(gateway: Gateway, case: _Case) -> None:
    scripted: Final = _response(_BODIES[case.name])
    with gateway.scenario() as scenario:
        alias, identity = _model(scenario, case, scripted)
        body: Final = {**case.body, "model": alias}
        response: Final = eventually(
            lambda: _post(gateway, case.path, body),
            lambda result: not (result.status_code == 400 and "Invalid model name" in result.text),
            seconds=30,
        )
        assert response.status_code == 200, response.text
        caller: Final = JSON_OBJECT.validate_python(response.json())
        _assert_scripted_response(case, caller)
        observations: Final = _Observations(gateway.upstream_url)
        captured: Final = eventually(
            observations.read, lambda _items: len(observations.for_scenario(identity)) == 1, seconds=20
        )
        matches: Final = observations.for_scenario(identity)
        assert len(matches) == 1, captured
        outbound: Final = object_value(matches[0]["body"])
        assert all(outbound.get(name) == body[name] for name in case.required), captured


def test_image_generation_null_prompt_reaches_upstream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, identity = _model(scenario, _IMAGE, _response(_BODIES["images"]))
        response: Final = _post(gateway, _IMAGE.path, {"model": alias, "prompt": None})
        assert response.status_code == 200, response.text
        _assert_scripted_response(_IMAGE, JSON_OBJECT.validate_python(response.json()))
        observations: Final = _Observations(gateway.upstream_url)
        captured: Final = eventually(
            observations.read, lambda _items: len(observations.for_scenario(identity)) == 1, seconds=20
        )
        matches: Final = observations.for_scenario(identity)
        assert len(matches) == 1, captured
        outbound: Final = object_value(matches[0]["body"])
        assert "prompt" in outbound and outbound["prompt"] is None, captured


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_openai_sdk_missing_moderations_input_returns_bad_request(gateway: Gateway, client_kind: str) -> None:
    with gateway.scenario() as scenario:
        alias, identity = _model(scenario, _MODERATION, _response())
        _ready_missing(gateway, "/v1/moderations", {"model": alias}, _expected_error("/moderations", "input"))
        if client_kind == "sync":
            with OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
                with pytest.raises(BadRequestError) as raised:
                    client.post("/v1/moderations", body={"model": alias}, cast_to=httpx.Response)
        else:

            async def request() -> None:
                async with AsyncOpenAI(
                    base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
                ) as client:
                    await client.post("/v1/moderations", body={"model": alias}, cast_to=httpx.Response)

            with pytest.raises(BadRequestError) as raised:
                asyncio.run(request())
        assert raised.value.status_code == 400
        observations: Final = _Observations(gateway.upstream_url)
        observations.read()
        assert observations.for_scenario(identity) == ()


def _healthy(url: str) -> int:
    try:
        return httpx.get(f"{url}/health", timeout=2, trust_env=False).status_code
    except httpx.TransportError:
        return 0


@contextmanager
def _upstream(directory: Path) -> Iterator[tuple[subprocess.Popen[bytes], str]]:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = int(reserve.getsockname()[1])
    root: Final = Path(__file__).resolve().parents[2]
    url: Final = f"http://127.0.0.1:{port}"
    with (directory / "upstream.log").open("w") as log:
        process: Final = subprocess.Popen(
            [sys.executable, "-P", "-m", "integration._support.upstream", "--port", str(port)],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        eventually(lambda: _healthy(url), lambda code: code == 200, seconds=30)
        yield process, url
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGCONT)
            process.terminate()
            process.wait(timeout=10)


def _register_owned(url: str, identity: str, response: JsonResponse) -> ScenarioHandle:
    result: Final = httpx.post(
        f"{url}/__scenarios",
        json={"scenario_id": identity, "response": response.model_dump(mode="json")},
        timeout=10,
        trust_env=False,
    )
    result.raise_for_status()
    return ScenarioHandle(identity, url)


def _delete_owned(handle: ScenarioHandle) -> None:
    httpx.delete(
        f"{handle.control_url}/__scenarios/{handle.scenario_id}", timeout=10, trust_env=False
    ).raise_for_status()


def _workers(process: subprocess.Popen[bytes]) -> tuple[psutil.Process, ...]:
    return tuple(psutil.Process(process.pid).children(recursive=True))


def _process_tree_line(process: psutil.Process) -> str:
    try:
        return f"{process.pid} {' '.join(process.cmdline())}"
    except psutil.Error:
        return f"{process.pid} <exited>"


def _worker_alive(worker: psutil.Process) -> bool:
    try:
        return worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _chat_body(model: str, marker: str, valid: bool) -> dict[str, JsonValue]:
    return {"model": model, "user": marker, **({"messages": [{"role": "user", "content": marker}]} if valid else {})}


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,))


def test_upstream_pause_and_worker_kill_preserve_required_body_status(
    gateway: Gateway,
    tmp_path: Path,
    record_property: pytest.RecordProperty,
) -> None:
    with (
        _upstream(tmp_path) as (upstream, url),
        owned_proxy_process(
            gateway,
            tmp_path,
            {"INTEGRATION_UPSTREAM_URL": url},
            workers=2,
        ) as owned,
    ):
        candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, url)
        with candidate.scenario() as scenario:
            script: Final = JsonResponse(
                content_type="application/json",
                body={
                    "id": "chatcmpl-$UNIQUE_ID",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
            chat_case: Final = _Case(
                "chat", "/v1/chat/completions", "/chat/completions", ("messages",), {}, "openai/gpt-4o-mini"
            )
            chat_model, chat_identity, _chat_handle = _owned_model(scenario, url, chat_case, script)
            probe: Final = eventually(
                lambda: _post(candidate, "/v1/chat/completions", _chat_body(chat_model, "probe", True)),
                lambda response: response.status_code == 200,
                seconds=30,
            )
            assert probe.status_code == 200, probe.text
            process_root: Final = psutil.Process(owned.process.pid)
            processes: Final = (process_root, *process_root.children(recursive=True))
            process_tree: Final = "\n".join(_process_tree_line(process) for process in processes)
            record_property("owned_proxy_process_tree", process_tree)
            workers: Final = eventually(
                lambda: _workers(owned.process),
                lambda children: len(children) >= 2,
                seconds=30,
            )
            observations: Final = _Observations(url)
            missing_cases: Final = (
                (_SPEECH, "input"),
                (_SPEECH, "input"),
                (_MODERATION, "input"),
                (_MODERATION, "input"),
                (_IMAGE, "prompt"),
                (_IMAGE, "prompt"),
                (_RERANK, "query"),
                (_RERANK, "query"),
                (_RERANK, "documents"),
                (_RERANK, "documents"),
            )
            missing_models: Final = tuple(
                _owned_model(scenario, url, case, script) for case, _parameter in missing_cases
            )
            missing_calls: Final = tuple(
                (
                    case,
                    parameter,
                    model,
                    identity,
                    case.path,
                    {key: value for key, value in {**case.body, "model": model}.items() if key != parameter},
                    _expected_error(case.route, parameter),
                )
                for (case, parameter), (model, identity, _handle) in zip(missing_cases, missing_models)
            )
            markers: Final = tuple(f"burst-{uuid.uuid4().hex}" for _ in range(20))
            valid_calls: Final = tuple(
                ("/v1/chat/completions", _chat_body(chat_model, marker, True), marker) for marker in markers
            )
            upstream.send_signal(signal.SIGSTOP)
            try:
                with ThreadPoolExecutor(max_workers=10) as pool:
                    missing_futures: Final = tuple(
                        pool.submit(_post, candidate, path, body)
                        for _case, _parameter, _model, _identity, path, body, _expected in missing_calls
                    )
                    missing: Final = tuple(
                        (call, future.result(timeout=15)) for call, future in zip(missing_calls, missing_futures)
                    )
                    assert all(
                        response.status_code == 400 and response.json() == expected
                        for (_case, _parameter, _model, _identity, _path, _body, expected), response in missing
                    ), [response.text for _call, response in missing]
                    paused_statuses: Final = tuple(response.status_code for _call, response in missing)
                    record_property(
                        "chaos_paused_missing_status_counts",
                        str({status: paused_statuses.count(status) for status in sorted(set(paused_statuses))}),
                    )
            finally:
                upstream.send_signal(signal.SIGCONT)
            with ThreadPoolExecutor(max_workers=20) as pool:
                valid_futures: Final = tuple(
                    pool.submit(_post, candidate, path, body) for path, body, _marker in valid_calls
                )
                valid: Final = tuple(future.result(timeout=30) for future in valid_futures)
            assert all(response.status_code == 200 for response in valid), [response.text for response in valid]
            record_property("chaos_burst_size", len(missing_calls) + len(valid_calls))
            resumed_statuses: Final = tuple(response.status_code for response in valid)
            record_property(
                "chaos_resumed_valid_status_counts",
                str({status: resumed_statuses.count(status) for status in sorted(set(resumed_statuses))}),
            )
            eventually(
                observations.read,
                lambda _items: all(
                    sum(
                        object_value(item["body"]).get("user") == marker
                        for item in observations.for_scenario(chat_identity)
                    )
                    == 1
                    for _path, _body, marker in valid_calls
                ),
                seconds=30,
            )
            missing_observations: Final = {
                identity: len(observations.for_scenario(identity))
                for _case, _parameter, _model, identity, _path, _body, _expected in missing_calls
            }
            assert all(count == 0 for count in missing_observations.values()), missing_observations
            record_property(
                "chaos_missing_split",
                str(tuple(f"{case.name}:{parameter}" for case, parameter in missing_cases)),
            )
            record_property("chaos_missing_upstream_observation_counts", str(missing_observations))
            request_ids: Final = tuple(str(JSON_OBJECT.validate_python(response.json())["id"]) for response in valid)
            spend_rows: Final = tuple(
                eventually(
                    lambda request_id=request_id: _spend_rows(request_id),
                    lambda values: len(values) == 1,
                    seconds=60,
                )
                for request_id in request_ids
            )
            assert all(rows[0]["request_id"] == request_id for rows, request_id in zip(spend_rows, request_ids)), (
                spend_rows
            )
            record_property("chaos_spend_query", 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s')
            record_property("chaos_spend_response_id_count", len(request_ids))
            record_property("chaos_spend_row_counts", str(tuple(len(rows) for rows in spend_rows)))
            workers[0].kill()
            eventually(lambda: _worker_alive(workers[0]), lambda alive: not alive, seconds=10)
            post_kill: Final = tuple(
                _post(candidate, path, body)
                for _case, _parameter, _model, _identity, path, body, _expected in missing_calls[:5]
            )
            assert all(
                response.status_code == 400 and response.json() == expected
                for response, (
                    _case,
                    _parameter,
                    _model,
                    _identity,
                    _path,
                    _body,
                    expected,
                ) in zip(post_kill, missing_calls[:5])
            ), [response.text for response in post_kill]
            recovered: Final = _post(candidate, "/v1/chat/completions", _chat_body(chat_model, "recovered", True))
            assert recovered.status_code == 200, recovered.text
