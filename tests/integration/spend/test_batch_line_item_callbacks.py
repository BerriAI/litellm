from __future__ import annotations

import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

PROMPT_TOKENS: Final = 10
COMPLETION_TOKENS: Final = 100
OUTPUT_SUCCESS_IDS: Final = ("r1", "r3")
OUTPUT_FAILED_IDS: Final = ("r2",)
ERROR_FILE_IDS: Final = ("r4", "r5")
ALL_CUSTOM_IDS: Final = ("r1", "r2", "r3", "r4", "r5")


def _successful_line(custom_id: str) -> str:
    return json.dumps(
        {
            "id": f"batch_req_{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": f"$REQUEST_ID-{custom_id}",
                "body": {
                    "id": f"chatcmpl-$REQUEST_ID-{custom_id}",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"answer {custom_id}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": PROMPT_TOKENS,
                        "completion_tokens": COMPLETION_TOKENS,
                        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                    },
                },
            },
            "error": None,
        },
        separators=(",", ":"),
    )


def _failed_line(custom_id: str) -> str:
    return json.dumps(
        {
            "id": f"batch_req_{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 400,
                "request_id": f"$REQUEST_ID-{custom_id}",
                "body": {"error": {"message": f"synthetic line failure {custom_id}", "code": "bad_request"}},
            },
            "error": {"code": "bad_request", "message": f"synthetic line failure {custom_id}"},
        },
        separators=(",", ":"),
    )


def _batch(status: str, *, files_ready: bool) -> dict[str, JsonValue]:
    return {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
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


def _input_lines(model_name: str, marker: str) -> tuple[str, ...]:
    return tuple(
        json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model_name, "messages": [{"role": "user", "content": f"{marker} {custom_id}"}]},
            },
            separators=(",", ":"),
        )
        for custom_id in ALL_CUSTOM_IDS
    )


def _provider_routes(input_lines: tuple[str, ...]) -> RoutedResponse:
    output_lines: Final = (_successful_line("r1"), _failed_line("r2"), _successful_line("r3"))
    error_lines: Final = tuple(_failed_line(custom_id) for custom_id in ERROR_FILE_IDS)
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
                content_type="application/json", body=_batch("validating", files_ready=False)
            ),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json", body=_batch("completed", files_ready=True)
            ),
            "GET /files/file-in-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(input_lines) + "\n"
            ),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(output_lines) + "\n"
            ),
            "GET /files/file-err-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body="\n".join(error_lines) + "\n"
            ),
        },
    )


def _line_item_config(base: Path, destination: Path) -> Path:
    config: Final = yaml.safe_load(base.read_text())
    config["litellm_settings"].update({"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1})
    config["general_settings"].update({"store_batch_line_items_in_callbacks": True})
    destination.write_text(yaml.safe_dump(config))
    return destination


def _spend_rows(key: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        read_rows(
            'SELECT call_type, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        )
    )


def _hidden(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(event["hidden_params"])


@pytest.mark.covers(
    "spend.batches.completed_batch_emits_one_callback_event_per_jsonl_line_beside_the_aggregate",
    "spend.batches.line_item_callback_events_do_not_bill_spend_twice",
)
def test_completed_batch_emits_paired_request_response_callback_events_per_jsonl_line(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "batch-line-items-" + uuid.uuid4().hex[:12]
    sink_secret: Final = "synthetic-sink-secret-" + marker
    provider_secret: Final = "synthetic-provider-secret-" + marker

    def sink(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {sink_secret}"
        return Reply()

    with (
        wire_server(sink) as endpoint,
        owned_proxy(
            gateway,
            tmp_path,
            {"GENERIC_LOGGER_ENDPOINT": endpoint.url, "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {sink_secret}"},
            config=_line_item_config(Path("tests/integration/proxy_config.yaml"), tmp_path / "line_items.yaml"),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        routed_model: Final = scenario.model(api_base=f"{gateway.upstream_url}/{marker}", api_key=provider_secret)
        input_lines: Final = _input_lines(routed_model, marker)
        handle: Final = register_scenario(marker, _provider_routes(input_lines))
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key(models=[routed_model])
        file_response: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": routed_model},
            {"file": ("in.jsonl", ("\n".join(input_lines) + "\n").encode(), "application/jsonl")},
            key=key,
        )
        assert file_response.status_code == 200, file_response.text
        input_file_id: Final = string_value(JSON_OBJECT.validate_json(file_response.content)["id"])
        batch_response: Final = candidate.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": routed_model,
            },
            key=key,
        )
        assert batch_response.status_code == 200, batch_response.text
        batch_id: Final = string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])
        retrieval: Final = eventually(
            lambda: candidate.request("GET", f"/v1/batches/{batch_id}", key=key),
            lambda response: response.status_code == 200 and response.json()["status"] == "completed",
            seconds=30,
        )
        assert retrieval.status_code == 200, retrieval.text
        key_hash: Final = sha256(key.encode()).hexdigest()
        batches: Final[
            list[Request]
        ] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

        def delivered() -> tuple[dict[str, JsonValue], ...]:
            batches.extend(endpoint.drain())
            return tuple(
                event
                for batch in batches
                for event in json.loads(batch.body)
                if object_value(event["metadata"]).get("user_api_key_hash") == key_hash
            )

        events: Final = eventually(
            delivered,
            lambda values: (
                len([e for e in values if e["call_type"] == "aretrieve_batch"]) >= 1
                and len([e for e in values if _hidden(e).get("batch_custom_id") is not None]) >= len(ALL_CUSTOM_IDS)
            ),
            seconds=40,
        )
        for batch in batches:
            for credential in (provider_secret, sink_secret, key):
                assert credential.encode() not in batch.body
        aggregate_events: Final = tuple(event for event in events if event["call_type"] == "aretrieve_batch")
        line_events: Final = tuple(event for event in events if _hidden(event).get("batch_custom_id") is not None)
        assert len(aggregate_events) == 1, [event["call_type"] for event in events]
        other_events: Final = tuple(event for event in events if event not in aggregate_events + line_events)
        assert sorted(event["call_type"] for event in other_events) == ["acreate_batch", "acreate_file"], other_events
        assert sorted(string_value(_hidden(event)["batch_custom_id"]) for event in line_events) == list(
            ALL_CUSTOM_IDS
        ), line_events
        by_custom_id: Final = {string_value(_hidden(event)["batch_custom_id"]): event for event in line_events}
        for custom_id, line in zip(ALL_CUSTOM_IDS, input_lines, strict=True):
            event: Final = by_custom_id[custom_id]
            hidden: Final = _hidden(event)
            assert hidden["batch_id"] == batch_id, hidden
            assert event["call_type"] == "acompletion", event["call_type"]
            assert event["messages"] == json.loads(line)["body"]["messages"], (custom_id, event["messages"])
            if custom_id in OUTPUT_SUCCESS_IDS:
                assert event["status"] == "success", event
                assert hidden["batch_line_status_code"] == 200, hidden
                assert event["prompt_tokens"] == PROMPT_TOKENS and event["completion_tokens"] == COMPLETION_TOKENS
                assert object_value(event["response"])["id"] == f"chatcmpl-{marker}-{custom_id}", event["response"]
                assert object_value(object_value(event["response"])["choices"][0]["message"])["content"] == (
                    f"answer {custom_id}"
                )
            else:
                assert event["status"] == "failure", event
                assert hidden["batch_line_status_code"] == 400, hidden
                assert f"synthetic line failure {custom_id}" in json.dumps(event["error_information"]), event
        aggregate: Final = aggregate_events[0]
        assert aggregate["prompt_tokens"] == len(OUTPUT_SUCCESS_IDS) * PROMPT_TOKENS, aggregate
        assert aggregate["completion_tokens"] == len(OUTPUT_SUCCESS_IDS) * COMPLETION_TOKENS, aggregate
        rows: Final = eventually(
            lambda: _spend_rows(key),
            lambda values: any(r["call_type"] == "aretrieve_batch" for r in values),
            seconds=70,
        )
        batch_rows: Final = tuple(row for row in rows if row["call_type"] == "aretrieve_batch")
        assert len(batch_rows) == 1, rows
        assert not any(row["call_type"] == "acompletion" for row in rows), rows
        assert batch_rows[0]["prompt_tokens"] == len(OUTPUT_SUCCESS_IDS) * PROMPT_TOKENS, rows
