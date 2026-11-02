from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
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


def batch_routes(model: str) -> RoutedResponse:
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


def _batches_total_tokens(endpoints: dict[str, object] | None) -> int | None:
    if endpoints is None or "/batches" not in endpoints:
        return None
    metrics: Final = object_value(object_value(endpoints["/batches"])["metrics"])
    total_tokens: Final = metrics["total_tokens"]
    return int(total_tokens) if isinstance(total_tokens, (int, float, str)) else None


def batch_input_file(model: str) -> bytes:
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
        handle: Final = register_scenario(scenario_id, batch_routes("gpt-4o-mini"))
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(api_base=handle.api_base())
        file_response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", batch_input_file(model), "application/jsonl")},
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
        handle: Final = register_scenario(scenario_id, batch_routes("gpt-4o-mini"))
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
            {"file": ("in.jsonl", batch_input_file(model), "application/jsonl")},
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
            lambda value: _batches_total_tokens(value) == BATCH_PROMPT_TOKENS + BATCH_COMPLETION_TOKENS,
            seconds=70,
            return_last_on_timeout=True,
        )
        assert endpoints is not None, "team daily activity returned no endpoint breakdown for the day"
        assert set(endpoints) == {"/batches"}, endpoints
        endpoint_metrics: Final = object_value(object_value(endpoints["/batches"])["metrics"])
        assert float(endpoint_metrics["spend"]) == pytest.approx(BATCH_SPEND), endpoints
        assert endpoint_metrics["total_tokens"] == BATCH_PROMPT_TOKENS + BATCH_COMPLETION_TOKENS, endpoints


def test_batch_reconciliation_delivers_each_configured_logger_once(gateway: Gateway, tmp_path: Path) -> None:
    events: Final[SimpleQueue[dict[str, JsonValue]]] = SimpleQueue()

    def receive(request: Request) -> Reply:
        payload: Final = JSON_OBJECT.validate_json(request.body)
        standard: Final = object_value(payload.get("standard"))
        if standard.get("call_type") == "aretrieve_batch" and standard.get("response_cost"):
            events.put(payload)
        return Reply()

    config: Final = tmp_path / "batch-callbacks.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": "batch-bootstrap",
                        "litellm_params": {
                            "model": "openai/gpt-6.1-sol",
                            "api_base": gateway.upstream_url + "/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                ],
                "litellm_settings": {
                    "callbacks": [
                        "integration._support.redaction_probe.sync_probe",
                        "integration._support.redaction_probe.probe",
                        "integration._support.redaction_probe.dual_probe",
                    ],
                },
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                    "proxy_batch_polling_interval": 1,
                },
            }
        )
    )
    with scratch_database() as database, wire_server(receive) as sink:
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DATABASE_URL": database, "REDACTION_PROBE_URL": sink.url, "PROXY_BATCH_POLLING_ENABLED": "true"},
                config=config,
            ) as proxy,
            proxy.scenario() as scenario,
        ):
            scenario_id: Final = f"batch-callbacks-{uuid.uuid4().hex}"
            routes: Final = batch_routes("gpt-6.1-sol")
            handle: Final = register_scenario(scenario_id, routes)
            scenario.cleanups.callback(delete_scenario, handle)
            model: Final = scenario.model(
                model="openai/gpt-6.1-sol",
                api_base=handle.api_base(),
                input_cost_per_token=INPUT_COST_PER_TOKEN,
                output_cost_per_token=OUTPUT_COST_PER_TOKEN,
            )
            register_scenario(
                scenario_id,
                routes.model_copy(
                    update={
                        "routes": {
                            **routes.routes,
                            "GET /files/file-in-$REQUEST_ID/content": TextResponse(
                                content_type="application/jsonl", body=batch_input_file(model).decode()
                            ),
                        }
                    }
                ),
            )
            key: Final = scenario.key(models=[model])
            uploaded: Final = proxy.request_multipart(
                "/v1/files",
                {"purpose": "batch", "target_model_names": model},
                {"file": ("in.jsonl", batch_input_file(model), "application/jsonl")},
                key=key,
            )
            assert uploaded.status_code == 200, uploaded.text
            created: Final = proxy.request(
                "POST",
                "/v1/batches",
                {
                    "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "model": model,
                },
                key=key,
            )
            assert created.status_code == 200, created.text
            assert eventually(events.qsize, lambda count: count >= 3, seconds=90) == 3
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
                    "WHERE api_key=%s AND call_type='aretrieve_batch'",
                    (sha256(key.encode()).hexdigest(),),
                    database_url=database,
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert float(rows[0]["spend"]) == pytest.approx(BATCH_SPEND)
            assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (
                BATCH_PROMPT_TOKENS,
                BATCH_COMPLETION_TOKENS,
            )
        assert events.qsize() == 3
        received: Final = tuple(events.get_nowait() for _ in range(3))
        assert {(event["logger"], event["hook"]) for event in received} == {
            ("SyncRedactionProbe", "sync"),
            ("RedactionProbe", "async"),
            ("DualRedactionProbe", "async"),
        }
        for event in received:
            standard: Final = object_value(event["standard"])
            assert standard["response_cost"] == pytest.approx(BATCH_SPEND)
            usage: Final = object_value(object_value(event["response"])["usage"])
            assert (usage["prompt_tokens"], usage["completion_tokens"]) == (
                BATCH_PROMPT_TOKENS,
                BATCH_COMPLETION_TOKENS,
            )
