from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

FIRST_LINE: Final = {"prompt_tokens": 10, "completion_tokens": 7, "reasoning_tokens": 4}
SECOND_LINE: Final = {"prompt_tokens": 5, "completion_tokens": 3, "reasoning_tokens": 2}
ERROR_FILE_LINES: Final = 2


def _succeeded_line(index: int, model: str, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int) -> str:
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
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
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
            "response": {
                "status_code": 400,
                "request_id": f"$REQUEST_ID-{index}",
                "body": {"error": {"message": "rejected line", "type": "invalid_request_error", "code": "400"}},
            },
            "error": {"code": "bad_request", "message": "rejected line"},
        },
        separators=(",", ":"),
    )


def _batch_routes(model: str) -> RoutedResponse:
    output_lines: Final = (
        _succeeded_line(1, model, **FIRST_LINE),
        _succeeded_line(2, model, **SECOND_LINE),
        _failed_line(3),
    )
    error_lines: Final = tuple(_failed_line(index) for index in range(4, 4 + ERROR_FILE_LINES))
    completed: Final = {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": "completed",
        "output_file_id": "file-out-$REQUEST_ID",
        "error_file_id": "file-err-$REQUEST_ID",
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1,
        "expires_at": 1,
        "request_counts": {"total": 5, "completed": 2, "failed": 3},
        "metadata": None,
    }
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
                body={**completed, "status": "validating", "output_file_id": None, "error_file_id": None},
            ),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(content_type="application/json", body=completed),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(output_lines) + "\n"
            ),
            "GET /files/file-err-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(error_lines) + "\n"
            ),
        },
    )


def _team_day_endpoints(gateway: Gateway, team: str, start_date: str, end_date: str) -> dict[str, object] | None:
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity",
        params={"team_ids": team, "start_date": start_date, "end_date": end_date},
    )
    if response.status_code != 200:
        return None
    days: Final = response.json()["results"]
    if not days:
        return None
    return object_value(object_value(object_value(days[0])["breakdown"])["endpoints"])


def _input_file(model: str) -> bytes:
    return (
        "\n".join(
            json.dumps(
                {
                    "custom_id": f"r{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": model, "messages": [{"role": "user", "content": "batch accounting"}]},
                },
                separators=(",", ":"),
            )
            for index in range(1, 6)
        )
        + "\n"
    ).encode()


def _metadata(value: object) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else JSON_OBJECT.validate_python(value)


@pytest.mark.covers("quota_management.spend_tracking.batch_costs.reasoning_tokens_and_error_file_failures_recorded")
def test_completed_batch_spend_row_records_reasoning_tokens_and_error_file_failures(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        scenario_id: Final = f"batch-accounting-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, _batch_routes("gpt-4o-mini"))
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(api_base=handle.api_base())
        file_response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", _input_file(model), "application/jsonl")},
            key=key,
        )
        assert file_response.status_code == 200, file_response.text
        batch_response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
            key=key,
        )
        assert batch_response.status_code == 200, batch_response.text
        batch_id: Final = string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])
        retrieval: Final = gateway.request("GET", f"/v1/batches/{batch_id}", key=key)
        assert retrieval.status_code == 200, retrieval.text
        assert retrieval.json()["status"] == "completed", retrieval.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, prompt_tokens, completion_tokens, metadata FROM "LiteLLM_SpendLogs" '
                "WHERE api_key=%s AND call_type='aretrieve_batch'",
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        metadata: Final = _metadata(row["metadata"])
        prompt_tokens: Final = FIRST_LINE["prompt_tokens"] + SECOND_LINE["prompt_tokens"]
        completion_tokens: Final = FIRST_LINE["completion_tokens"] + SECOND_LINE["completion_tokens"]
        reasoning_tokens: Final = FIRST_LINE["reasoning_tokens"] + SECOND_LINE["reasoning_tokens"]
        assert row["status"] == "success", retrieval.text
        assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt_tokens, completion_tokens), retrieval.text
        assert (metadata["batch_successful_requests"], metadata["batch_failed_requests"]) == (
            2,
            1 + ERROR_FILE_LINES,
        ), json.dumps(metadata)
        usage: Final = JSON_OBJECT.validate_python(metadata["usage_object"])
        details: Final = JSON_OBJECT.validate_python(usage["completion_tokens_details"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
        ), json.dumps(metadata)
        assert {name: value for name, value in details.items() if value is not None} == {
            "reasoning_tokens": reasoning_tokens,
            "text_tokens": completion_tokens - reasoning_tokens,
        }, json.dumps(metadata)


INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
BATCH_PROMPT_TOKENS: Final = FIRST_LINE["prompt_tokens"] + SECOND_LINE["prompt_tokens"]
BATCH_COMPLETION_TOKENS: Final = FIRST_LINE["completion_tokens"] + SECOND_LINE["completion_tokens"]
BATCH_SPEND: Final = (BATCH_PROMPT_TOKENS * INPUT_COST_PER_TOKEN + BATCH_COMPLETION_TOKENS * OUTPUT_COST_PER_TOKEN) / 2


def test_completed_batch_spend_lands_under_batches_in_team_endpoint_activity(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"batch-endpoint-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, _batch_routes("gpt-4o-mini"))
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            api_base=handle.api_base(),
            input_cost_per_token=INPUT_COST_PER_TOKEN,
            output_cost_per_token=OUTPUT_COST_PER_TOKEN,
        )
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        file_response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", _input_file(model), "application/jsonl")},
            key=key,
        )
        assert file_response.status_code == 200, file_response.text
        batch_response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
            key=key,
        )
        assert batch_response.status_code == 200, batch_response.text
        batch_id: Final = string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])
        retrieval: Final = gateway.request("GET", f"/v1/batches/{batch_id}", key=key)
        assert retrieval.status_code == 200, retrieval.text
        assert retrieval.json()["status"] == "completed", retrieval.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" '
                "WHERE api_key=%s AND call_type='aretrieve_batch'",
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert float(row["spend"]) == pytest.approx(BATCH_SPEND), dict(row)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (
            BATCH_PROMPT_TOKENS,
            BATCH_COMPLETION_TOKENS,
        ), dict(row)
        today: Final = datetime.now(timezone.utc)
        endpoints: Final = eventually(
            lambda: _team_day_endpoints(
                gateway,
                team,
                (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            ),
            lambda value: value is not None and "/batches" in value,
            seconds=70,
            return_last_on_timeout=True,
        )
        assert endpoints is not None, "team daily activity returned no endpoint breakdown for the day"
        assert set(endpoints) == {"/batches"}, endpoints
        endpoint_metrics: Final = object_value(object_value(endpoints["/batches"])["metrics"])
        assert float(endpoint_metrics["spend"]) == pytest.approx(BATCH_SPEND), endpoints
        assert endpoint_metrics["total_tokens"] == BATCH_PROMPT_TOKENS + BATCH_COMPLETION_TOKENS, endpoints
