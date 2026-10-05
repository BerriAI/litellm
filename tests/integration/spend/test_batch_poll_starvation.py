import json
import os
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
PROMPT_TOKENS: Final = 100
COMPLETION_TOKENS: Final = 50
BATCH_COST_SHARE: Final = 0.5

_INPUT_FILE: Final = JsonResponse(
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
)


def _batch(status: str, output_file_id: str | None) -> dict[str, JsonValue]:
    return {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": None,
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1 if status == "completed" else None,
        "expires_at": 1,
        "request_counts": {"total": 1, "completed": 1 if status == "completed" else 0, "failed": 0},
        "metadata": None,
    }


def _accepting_routes() -> dict[str, JsonResponse | TextResponse]:
    return {
        "POST /files": _INPUT_FILE,
        "POST /batches": JsonResponse(content_type="application/json", body=_batch("validating", None)),
    }


def _gone_at_provider_routes() -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            **_accepting_routes(),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json",
                status=404,
                body={
                    "error": {
                        "message": "No batch found with id 'batch-$REQUEST_ID'.",
                        "type": "invalid_request_error",
                        "param": "id",
                        "code": "batch_not_found",
                    }
                },
            ),
        },
    )


def _completed_routes() -> RoutedResponse:
    output_line: Final = {
        "id": "batch_req_1",
        "custom_id": "r1",
        "response": {
            "status_code": 200,
            "request_id": "$REQUEST_ID-1",
            "body": {
                "id": "chatcmpl-$REQUEST_ID-1",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        },
        "error": None,
    }
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            **_accepting_routes(),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json", body=_batch("completed", "file-out-$REQUEST_ID")
            ),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body=json.dumps(output_line, separators=(",", ":")) + "\n"
            ),
        },
    )


def _scripted_deployment(scenario: Scenario, marker: str, routes: RoutedResponse) -> str:
    scenario_id: Final = f"poll-{marker}-{sha256(os.urandom(16)).hexdigest()[:12]}"
    handle: Final = register_scenario(scenario_id, routes)
    scenario.cleanups.callback(delete_scenario, handle)
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": f"poll-{marker}-{sha256(scenario_id.encode()).hexdigest()[:12]}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-scripted-provider",
                "api_base": handle.api_base(),
                "input_cost_per_token": INPUT_COST_PER_TOKEN,
                "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
            },
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return string_value(created["model_name"])


def _submitted_batch_id(gateway: Gateway, key: str, model_name: str) -> str:
    request_line: Final = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model_name, "messages": [{"role": "user", "content": "poll starvation"}]},
    }
    file_response: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model_name},
        {"file": ("in.jsonl", (json.dumps(request_line) + "\n").encode(), "application/jsonl")},
        key=key,
    )
    assert file_response.is_success, file_response.text
    batch_response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model_name,
        },
        key=key,
    )
    assert batch_response.is_success, batch_response.text
    return string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])


def _managed_rows(batch_ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    placeholders: Final = ", ".join("%s" for _ in batch_ids)
    return read_rows(
        f'SELECT batch_processed FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id IN ({placeholders})',
        batch_ids,
    )


@pytest.mark.timeout(180)
@pytest.mark.covers("quota_management.spend_tracking.batch_costs.uncostable_rows_retire_so_newer_batches_are_costed")
def test_batches_gone_at_provider_do_not_starve_a_newer_batch_out_of_cost_polling(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        gone_batch_ids: Final = tuple(
            _submitted_batch_id(
                gateway, key, _scripted_deployment(scenario, f"gone{index}", _gone_at_provider_routes())
            )
            for index in range(MAX_OBJECTS_PER_POLL_CYCLE)
        )
        costable_batch_id: Final = _submitted_batch_id(
            gateway, key, _scripted_deployment(scenario, "costable", _completed_routes())
        )
        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT call_type, status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
                "WHERE api_key = %s AND call_type = %s",
                (sha256(key.encode()).hexdigest(), "aretrieve_batch"),
            ),
            lambda rows: len(rows) == 1,
            seconds=120,
        )
        assert spend_rows == [
            {
                "call_type": "aretrieve_batch",
                "status": "success",
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "spend": pytest.approx(
                    BATCH_COST_SHARE
                    * (PROMPT_TOKENS * INPUT_COST_PER_TOKEN + COMPLETION_TOKENS * OUTPUT_COST_PER_TOKEN)
                ),
            }
        ]
        assert _managed_rows((costable_batch_id,)) == [{"batch_processed": True}]
        assert _managed_rows(gone_batch_ids) == [{"batch_processed": True}] * MAX_OBJECTS_PER_POLL_CYCLE
