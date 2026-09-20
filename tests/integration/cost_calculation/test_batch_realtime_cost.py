from __future__ import annotations

import asyncio
import json
import os
import time
from hashlib import sha256
from typing import Final

import pytest
import websockets
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.assertions import assert_exact
from integration.cost_calculation.conftest import poll_rows, poll_rows_where, read_rows_now
from integration.cost_calculation.cost_tracking_case import (
    BATCH_CASES,
    REALTIME_CASES,
    BatchCostCase,
    JsonResponse,
    RealtimeCostCase,
    RealtimeResponse,
    RoutedResponse,
    TextResponse,
)
from pydantic import JsonValue


def _register_deployment(
    scenario: Scenario,
    litellm_model: str,
    response: JsonResponse | TextResponse | RealtimeResponse,
    marker: str,
    *,
    realtime: bool,
) -> tuple[str, str]:
    scenario_id: Final = f"cost-{marker}-{sha256(os.urandom(16)).hexdigest()[:12]}"
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    control_url: Final = os.environ["INTEGRATION_UPSTREAM_URL"].rstrip("/")
    created: Final = scenario.gateway.post(
        "/model/new",
        JSON_OBJECT.validate_python(
            {
                "model_name": f"cost-{marker}-{sha256(scenario_id.encode()).hexdigest()[:12]}",
                "litellm_params": {
                    "model": litellm_model,
                    "api_key": scenario_id if realtime else "sk-scripted-provider",
                    "api_base": control_url if realtime else handle.api_base(),
                },
            }
        ),
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return string_value(created["model_name"]), identity


def _batch_response(case: BatchCostCase) -> JsonResponse | RoutedResponse:
    request_id: Final = "$REQUEST_ID"
    lines: Final = tuple(
        json.dumps(line.render(index, case.model, request_id), separators=(",", ":"))
        for index, line in enumerate(case.output_lines, start=1)
    )
    counts: Final = {
        "total": case.request_count,
        "completed": case.completed_count,
        "failed": case.failed_count,
    }
    has_output: Final = any(line.status_code == 200 for line in case.output_lines)
    has_failed: Final = any(line.status_code != 200 for line in case.output_lines)
    batch: Final = {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": "completed",
        "output_file_id": "file-out-$REQUEST_ID" if has_output else None,
        "error_file_id": "file-err-$REQUEST_ID" if has_failed else None,
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1,
        "expires_at": 1,
        "request_counts": counts,
        "metadata": None,
    }
    routes: Final = {
        "POST /files": JsonResponse(
            content_type="application/json",
            body={
                "id": "file-in-$REQUEST_ID",
                "object": "file",
                "purpose": "batch",
                "bytes": 100,
                "created_at": 1,
                "filename": "in.jsonl",
                "status": "processed",
            },
        ),
        "POST /batches": JsonResponse(
            content_type="application/json",
            body={
                **batch,
                "status": "validating",
                "output_file_id": None,
                "error_file_id": None,
            },
        ),
        "GET /batches/batch-$REQUEST_ID": JsonResponse(
            content_type="application/json",
            body=batch,
        ),
        **(
            {
                "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                    content_type="application/jsonl",
                    body="\n".join(lines) + ("\n" if lines else ""),
                )
            }
            if has_output
            else {}
        ),
    }
    return RoutedResponse(
        content_type="application/x-routed",
        routes=routes,
    )


def _batch_input_lines(case: BatchCostCase, model_name: str) -> bytes:
    count: Final = case.request_count
    return (
        "\n".join(
            json.dumps(
                {
                    "custom_id": f"r{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "batch integration"}],
                    },
                },
                separators=(",", ":"),
            )
            for index in range(1, count + 1)
        )
        + "\n"
    ).encode()


@pytest.mark.parametrize(
    "case",
    tuple(pytest.param(case, marks=pytest.mark.covers(case.covers), id=case.name) for case in BATCH_CASES),
)
def test_batch_costs(gateway: Gateway, case: BatchCostCase) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model_name, identity = _register_deployment(
            scenario,
            case.litellm_model,
            _batch_response(case),
            case.name,
            realtime=False,
        )
        file_response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model_name},
            {"file": ("in.jsonl", _batch_input_lines(case, model_name), "application/jsonl")},
            key=key,
        )
        assert file_response.is_success, file_response.text
        file_body: Final = JSON_OBJECT.validate_json(file_response.content)
        batch_response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(file_body["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model_name,
            },
            key=key,
        )
        assert batch_response.is_success, batch_response.text
        batch_body: Final = JSON_OBJECT.validate_json(batch_response.content)
        batch_id: Final = string_value(batch_body["id"])
        first_retrieval: Final = gateway.request("GET", f"/v1/batches/{batch_id}", key=key)
        second_retrieval: Final = gateway.request("GET", f"/v1/batches/{batch_id}", key=key)
        assert first_retrieval.is_success, first_retrieval.text
        assert second_retrieval.is_success, second_retrieval.text
        retrieval_rows: Final = poll_rows_where(key, 1, lambda row: row.call_type == "aretrieve_batch")
        assert len(retrieval_rows) == 1
        rows: Final = read_rows_now(key)
        assert all(row.spend == 0.0 for row in rows if row.call_type != "aretrieve_batch")
        row: Final = retrieval_rows[0]
        assert row.status == "success"
        assert row.call_type == "aretrieve_batch"
        assert row.model_id == identity
        assert_exact(case.name, "application/json", case.expected, row, second_retrieval)
        time.sleep(3)
        assert len(tuple(row for row in read_rows_now(key) if row.call_type == "aretrieve_batch")) == 1


def _realtime_response(case: RealtimeCostCase) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        session_model=case.session_model,
        events=tuple(turn.render(index, "$REQUEST_ID") for index, turn in enumerate(case.turns, start=1)),
    )


async def _run_realtime(url: str, key: str, model_name: str, turn_count: int) -> dict[str, JsonValue]:
    async with websockets.connect(
        f"{url.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/realtime?model={model_name}",
        additional_headers={"Authorization": f"Bearer {key}"},
    ) as websocket:
        session: Final = JSON_OBJECT.validate_json(await websocket.recv())
        for _ in range(turn_count):
            await websocket.send(json.dumps({"type": "response.create"}))
            while True:
                event: Final = JSON_OBJECT.validate_json(await websocket.recv())
                if event.get("type") == "response.done":
                    break
        return session


@pytest.mark.parametrize(
    "case",
    tuple(pytest.param(case, marks=pytest.mark.covers(case.covers), id=case.name) for case in REALTIME_CASES),
)
def test_realtime_costs(gateway: Gateway, case: RealtimeCostCase) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model_name, identity = _register_deployment(
            scenario,
            case.litellm_model,
            _realtime_response(case),
            case.name,
            realtime=True,
        )
        session: Final = asyncio.run(
            _run_realtime(
                os.environ["INTEGRATION_PROXY_URL"].rstrip("/"),
                key,
                model_name,
                len(case.turns),
            )
        )
        session_model: Final = object_value(session["session"])["model"]
        assert session_model == (case.session_model or case.model)
        row: Final = poll_rows(key, 1)[0]
        assert row.status == "success"
        assert row.call_type == "_arealtime"
        assert row.model_id == identity
        assert_exact(case.name, "application/json", case.expected, row, None)
