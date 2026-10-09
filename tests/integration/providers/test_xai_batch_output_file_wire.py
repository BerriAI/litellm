from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

XAI_MODEL: Final = "xai/grok-4.20-0309-non-reasoning"
XAI_BATCH_ID: Final = "batch_$REQUEST_ID"
_INPUT: Final = "".join(
    json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "grok-4.20-0309-non-reasoning", "messages": [{"role": "user", "content": "ping"}]},
        }
    )
    + "\n"
    for custom_id in ("req-1", "req-2")
)


def _completion(custom_id: str, content: str) -> dict[str, JsonValue]:
    return {
        "batch_request_id": custom_id,
        "batch_result": {
            "response": {
                "chat_get_completion": {
                    "id": f"chatcmpl-{custom_id}",
                    "object": "chat.completion",
                    "model": "grok-4.20-0309-non-reasoning",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            }
        },
    }


def _xai_batch(*, completed: bool) -> dict[str, JsonValue]:
    return {
        "batch_id": XAI_BATCH_ID,
        "name": "litellm-batch",
        "create_time": "2026-10-09T00:00:00Z",
        "expire_time": "2026-10-10T00:00:00Z",
        "state": {
            "num_requests": 2 if completed else 0,
            "num_pending": 0,
            "num_success": 2 if completed else 0,
            "num_error": 0,
            "num_cancelled": 0,
        },
        "input_file_id": "file_in_$REQUEST_ID",
    }


def _xai_routes(*, results_status: int = 200) -> RoutedResponse:
    results: Final = (
        JsonResponse(
            content_type="application/json",
            body={
                "results": [_completion("req-1", "first"), _completion("req-2", "second")],
                "pagination_token": None,
            },
        )
        if results_status == 200
        else JsonResponse(
            content_type="application/json",
            body={"code": "internal", "error": "results unavailable"},
            status=results_status,
        )
    )
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /v1/files": JsonResponse(
                content_type="application/json",
                body={
                    "id": "file_in_$REQUEST_ID",
                    "bytes": 240,
                    "created_at": 1791500000,
                    "filename": "input.jsonl",
                    "purpose": "",
                },
            ),
            "GET /v1/files/file_in_$REQUEST_ID/content": TextResponse(content_type="application/jsonl", body=_INPUT),
            "POST /v1/batches": JsonResponse(content_type="application/json", body=_xai_batch(completed=False)),
            f"GET /v1/batches/{XAI_BATCH_ID}": JsonResponse(
                content_type="application/json", body=_xai_batch(completed=True)
            ),
            f"GET /v1/batches/{XAI_BATCH_ID}/results": results,
        },
    )


@dataclass(frozen=True, slots=True)
class _XAIBatch:
    owner_key: str
    batch_id: str
    upstream: ScenarioHandle


def _create_xai_batch(scenario: Scenario, *, results_status: int = 200) -> _XAIBatch:
    handle: Final = register_scenario(
        f"xai-batch-output-{uuid.uuid4().hex}", _xai_routes(results_status=results_status)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(model=XAI_MODEL, api_key="integration-xai-key", api_base=handle.api_base())
    owner_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), models=[model])
    uploaded: Final = scenario.gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("input.jsonl", _INPUT.encode(), "application/jsonl")},
        key=owner_key,
    )
    assert uploaded.status_code == 200, uploaded.text
    created: Final = scenario.gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=owner_key,
    )
    assert created.status_code == 200, created.text
    return _XAIBatch(owner_key, string_value(JSON_OBJECT.validate_json(created.content)["id"]), handle)


def _upstream_paths(gateway: Gateway, scenario_id: str) -> tuple[str, ...]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    assert response.status_code == 200, response.text
    requests: Final = JSON_OBJECT.validate_json(response.content)["requests"]
    assert isinstance(requests, list), response.text
    return tuple(
        f"{request.get('method')} {request.get('path')}"
        for request in map(object_value, requests)
        if scenario_id in str(request.get("path"))
    )


def _results_reads(gateway: Gateway, batch: _XAIBatch, raw_batch_id: str) -> int:
    return sum(path.endswith(f"/{raw_batch_id}/results") for path in _upstream_paths(gateway, batch.upstream.scenario_id))


def _stable_results_reads(gateway: Gateway, batch: _XAIBatch, raw_batch_id: str) -> int:
    def _read_twice() -> tuple[int, int]:
        first: Final = _results_reads(gateway, batch, raw_batch_id)
        time.sleep(3)
        return first, _results_reads(gateway, batch, raw_batch_id)

    return eventually(_read_twice, lambda reads: reads[0] == reads[1], seconds=30)[1]


def test_xai_batch_output_file_is_sized_by_the_batch_job_and_never_downloaded_or_looked_up_on_read(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_xai_batch(scenario)
        raw_batch_id: Final = XAI_BATCH_ID.replace("$REQUEST_ID", batch.upstream.scenario_id)
        retrieved_response: Final = gateway.request("GET", f"/v1/batches/{batch.batch_id}", key=batch.owner_key)
        assert retrieved_response.status_code == 200, retrieved_response.text
        retrieved: Final = JSON_OBJECT.validate_json(retrieved_response.content)
        assert retrieved["status"] == "completed", retrieved
        output_id: Final = string_value(retrieved["output_file_id"])

        first_read: Final = gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key)
        assert first_read.status_code == 200, first_read.text
        first_detail: Final = JSON_OBJECT.validate_json(first_read.content)
        assert (first_detail["id"], first_detail["purpose"], first_detail["filename"]) == (
            output_id,
            "batch_output",
            f"{raw_batch_id}_results.jsonl",
        ), first_detail

        content: Final = gateway.request("GET", f"/v1/files/{output_id}/content", key=batch.owner_key)
        assert content.status_code == 200, content.text
        lines: Final = tuple(json.loads(line) for line in content.content.decode().splitlines())
        assert [
            (line["custom_id"], line["response"]["body"]["choices"][0]["message"]["content"]) for line in lines
        ] == [
            ("req-1", "first"),
            ("req-2", "second"),
        ], lines

        sized: Final = eventually(
            lambda: JSON_OBJECT.validate_json(
                gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key).content
            ),
            lambda detail: detail.get("bytes") == len(content.content),
            seconds=60,
        )
        assert (sized["filename"], "litellm_details_fallback" in sized) == (f"{raw_batch_id}_results.jsonl", False)

        settled_reads: Final = _stable_results_reads(gateway, batch, raw_batch_id)
        reads: Final = (
            *(gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key) for _ in range(3)),
            gateway.request("GET", f"/v1/batches/{batch.batch_id}", key=batch.owner_key),
            gateway.request("GET", "/v1/files", key=batch.owner_key, params={"purpose": "batch_output"}),
        )
        assert all(response.status_code == 200 for response in reads), [response.text for response in reads]
        listed_output: Final = next(value for value in reads[-1].json()["data"] if value.get("id") == output_id)
        assert (listed_output["filename"], listed_output["bytes"]) == (
            f"{raw_batch_id}_results.jsonl",
            len(content.content),
        ), listed_output
        assert _results_reads(gateway, batch, raw_batch_id) == settled_reads

        file_lookups: Final = tuple(
            path for path in _upstream_paths(gateway, batch.upstream.scenario_id) if "/v1/files/batch_" in path
        )
        assert file_lookups == (), f"xAI batch id was looked up as a file: {file_lookups}"


def test_xai_batch_output_file_answers_at_once_while_results_are_unavailable_and_is_sized_once_they_return(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_xai_batch(scenario, results_status=503)
        raw_batch_id: Final = XAI_BATCH_ID.replace("$REQUEST_ID", batch.upstream.scenario_id)
        retrieved_response: Final = gateway.request("GET", f"/v1/batches/{batch.batch_id}", key=batch.owner_key)
        assert retrieved_response.status_code == 200, retrieved_response.text
        output_id: Final = string_value(JSON_OBJECT.validate_json(retrieved_response.content)["output_file_id"])

        started: Final = time.monotonic()
        unavailable: Final = gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key)
        elapsed: Final = time.monotonic() - started
        assert unavailable.status_code == 200, unavailable.text
        unavailable_detail: Final = JSON_OBJECT.validate_json(unavailable.content)
        assert (unavailable_detail["filename"], unavailable_detail["bytes"]) == (
            f"{raw_batch_id}_results.jsonl",
            0,
        ), unavailable_detail
        assert elapsed < 5, elapsed

        register_scenario(batch.upstream.scenario_id, _xai_routes())
        content: Final = gateway.request("GET", f"/v1/files/{output_id}/content", key=batch.owner_key)
        assert content.status_code == 200, content.text
        recovered: Final = eventually(
            lambda: JSON_OBJECT.validate_json(
                gateway.request("GET", f"/v1/files/{output_id}", key=batch.owner_key).content
            ),
            lambda detail: detail.get("bytes") == len(content.content),
            seconds=60,
        )
        assert (recovered["purpose"], recovered["filename"]) == ("batch_output", f"{raw_batch_id}_results.jsonl")

        file_lookups: Final = tuple(
            path for path in _upstream_paths(gateway, batch.upstream.scenario_id) if "/v1/files/batch_" in path
        )
        assert file_lookups == (), f"xAI batch id was looked up as a file: {file_lookups}"


def test_xai_batch_output_file_registered_by_the_batch_job_first_carries_the_downloaded_size(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _create_xai_batch(scenario)
        raw_batch_id: Final = XAI_BATCH_ID.replace("$REQUEST_ID", batch.upstream.scenario_id)

        listed: Final = eventually(
            lambda: gateway.request("GET", "/v1/files", key=batch.owner_key, params={"purpose": "batch_output"}).json()[
                "data"
            ],
            lambda values: any(value.get("filename") == f"{raw_batch_id}_results.jsonl" for value in values),
            seconds=60,
        )
        listed_output: Final = next(
            value for value in listed if value.get("filename") == f"{raw_batch_id}_results.jsonl"
        )
        content: Final = gateway.request("GET", f"/v1/files/{listed_output['id']}/content", key=batch.owner_key)
        assert content.status_code == 200, content.text
        assert (listed_output["purpose"], listed_output["bytes"]) == ("batch_output", len(content.content)), (
            listed_output
        )

        upstream_paths: Final = _upstream_paths(gateway, batch.upstream.scenario_id)
        file_lookups: Final = tuple(path for path in upstream_paths if "/v1/files/batch_" in path)
        assert file_lookups == (), f"xAI batch id was looked up as a file: {file_lookups}"
        results_reads: Final = sum(path.endswith(f"/{raw_batch_id}/results") for path in upstream_paths)
        state_reads: Final = sum(path.endswith(f"/v1/batches/{raw_batch_id}") for path in upstream_paths)
        assert (results_reads, state_reads) == (2, 2), upstream_paths
