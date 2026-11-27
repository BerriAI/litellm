from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

REASONING_TOKENS: Final = (30, 50)
PROMPT_TOKENS: Final = 10
COMPLETION_TOKENS: Final = 100
ERROR_FILE_FAILURES: Final = 2


def _successful_line(index: int, reasoning_tokens: int) -> str:
    return json.dumps(
        {
            "id": f"batch_req_{index}",
            "custom_id": f"r{index}",
            "response": {
                "status_code": 200,
                "request_id": f"$REQUEST_ID-{index}",
                "body": {
                    "id": f"chatcmpl-$REQUEST_ID-{index}",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": PROMPT_TOKENS,
                        "completion_tokens": COMPLETION_TOKENS,
                        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                    },
                },
            },
            "error": None,
        },
        separators=(",", ":"),
    )


def _failed_line(index: int) -> str:
    return json.dumps(
        {
            "id": f"batch_req_{index}",
            "custom_id": f"r{index}",
            "response": {"status_code": 400, "request_id": f"$REQUEST_ID-{index}", "body": {"error": "bad"}},
            "error": {"code": "bad_request", "message": "failed"},
        },
        separators=(",", ":"),
    )


def _batch(status: str, *, files_ready: bool, endpoint: str = "/v1/chat/completions") -> dict[str, JsonValue]:
    return {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": endpoint,
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": status,
        "output_file_id": "file-out-$REQUEST_ID" if files_ready else None,
        "error_file_id": "file-err-$REQUEST_ID" if files_ready else None,
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1 if files_ready else None,
        "expires_at": 1,
        "request_counts": {"total": 5, "completed": 2, "failed": 3},
        "metadata": None,
    }


def _provider_routes(created_endpoint: str = "/v1/chat/completions") -> RoutedResponse:
    output_lines: Final = (
        _successful_line(1, REASONING_TOKENS[0]),
        _failed_line(2),
        _successful_line(3, REASONING_TOKENS[1]),
    )
    error_lines: Final = tuple(_failed_line(index) for index in range(4, 4 + ERROR_FILE_FAILURES))
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
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
                body=_batch("validating", files_ready=False, endpoint=created_endpoint),
            ),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json", body=_batch("completed", files_ready=True)
            ),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(output_lines) + "\n"
            ),
            "GET /files/file-err-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(error_lines) + "\n"
            ),
        },
    )


def _input_file(model_name: str) -> bytes:
    return (
        "\n".join(
            json.dumps(
                {
                    "custom_id": f"r{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": model_name, "messages": [{"role": "user", "content": "batch observability"}]},
                },
                separators=(",", ":"),
            )
            for index in range(1, 6)
        )
        + "\n"
    ).encode()


def _retrieval_rows(key: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        read_rows(
            'SELECT prompt_tokens, completion_tokens, metadata FROM "LiteLLM_SpendLogs" '
            "WHERE api_key=%s AND call_type='aretrieve_batch'",
            (sha256(key.encode()).hexdigest(),),
        )
    )


def _metadata(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value: Final = row["metadata"]
    return object_value(JSON_OBJECT.validate_json(value) if isinstance(value, str) else value)


@dataclass(frozen=True, slots=True)
class CompletedBatch:
    key: str
    batch_id: str
    retrieval: httpx.Response


def _complete_a_batch(gateway: Gateway, scenario: Scenario, created_endpoint: str, *, managed: bool) -> CompletedBatch:
    scenario_id: Final = f"batch-observability-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, _provider_routes(created_endpoint))
    scenario.cleanups.callback(delete_scenario, handle)
    model_name: Final = scenario.model(api_base=handle.api_base())
    key: Final = scenario.key(models=[model_name])
    file_response: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names" if managed else "model": model_name},
        {"file": ("in.jsonl", _input_file(model_name), "application/jsonl")},
        key=key,
    )
    assert file_response.status_code == 200, file_response.text
    input_file_id: Final = string_value(JSON_OBJECT.validate_json(file_response.content)["id"])
    batch_response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": input_file_id, "endpoint": created_endpoint, "completion_window": "24h"}
        | ({} if managed else {"model": model_name}),
        key=key,
    )
    assert batch_response.status_code == 200, batch_response.text
    batch_id: Final = string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])
    retrieval: Final = eventually(
        lambda: gateway.request("GET", f"/v1/batches/{batch_id}", key=key),
        lambda response: response.status_code == 200 and response.json()["status"] == "completed",
        seconds=30,
    )
    assert retrieval.status_code == 200, retrieval.text
    return CompletedBatch(key=key, batch_id=batch_id, retrieval=retrieval)


@pytest.mark.covers("spend.batches.retrieval_row_aggregates_reasoning_tokens_and_per_request_counts")
def test_batch_retrieval_row_sums_reasoning_tokens_and_counts_output_and_error_file_failures(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        batch: Final = _complete_a_batch(gateway, scenario, "/v1/chat/completions", managed=False)
        retrieval: Final = batch.retrieval
        rows: Final = eventually(lambda: _retrieval_rows(batch.key), lambda values: len(values) == 1, seconds=70)
        row: Final = rows[0]
        metadata: Final = _metadata(row)
        usage: Final = object_value(metadata["usage_object"])
        assert row["prompt_tokens"] == 2 * PROMPT_TOKENS, retrieval.text
        assert row["completion_tokens"] == 2 * COMPLETION_TOKENS, retrieval.text
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == sum(REASONING_TOKENS), (
            retrieval.text,
            usage,
        )
        assert metadata["batch_successful_requests"] == 2, (retrieval.text, metadata)
        assert metadata["batch_failed_requests"] == 1 + ERROR_FILE_FAILURES, (retrieval.text, metadata)


@pytest.mark.timeout(240)
def test_managed_batch_keeps_its_created_endpoint_when_the_provider_reports_another(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        owned_proxy(gateway, tmp_path, {"PROXY_BATCH_POLLING_ENABLED": "false"}) as candidate,
        candidate.scenario() as scenario,
    ):
        batch: Final = _complete_a_batch(candidate, scenario, "/v1/embeddings", managed=True)
        stored: Final = candidate.request("GET", f"/v1/batches/{batch.batch_id}", key=batch.key)
        assert (batch.retrieval.json()["endpoint"], stored.json()["endpoint"]) == (
            "/v1/embeddings",
            "/v1/embeddings",
        ), stored.text
