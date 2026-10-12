from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
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
LIVE_TOKENS: Final = 10


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
            "POST /chat/completions": JsonResponse(
                content_type="application/json",
                body={
                    "id": "chatcmpl-live-$REQUEST_ID",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "live"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": LIVE_TOKENS, "completion_tokens": 0, "total_tokens": LIVE_TOKENS},
                },
            ),
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


def _line_item_config(base: Path, destination: Path, *, enforce_model_rate_limits: bool = False) -> Path:
    config: Final = yaml.safe_load(base.read_text())
    config["litellm_settings"].update({"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1})
    config["general_settings"].update({"store_batch_line_items_in_callbacks": True})
    if enforce_model_rate_limits:
        config["router_settings"].update({"optional_pre_call_checks": ["enforce_model_rate_limits"]})
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
                len([e for e in values if e["call_type"] == "acreate_batch"]) >= 1
                and len([e for e in values if e["call_type"] == "aretrieve_batch"]) >= 1
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

        repeated_gets: Final = [candidate.request("GET", f"/v1/batches/{batch_id}", key=key) for _ in range(2)]
        assert all(response.status_code == 200 for response in repeated_gets)
        events_after_repeats: Final = eventually(
            delivered,
            lambda values: len([e for e in values if e["call_type"] == "aretrieve_batch"]) >= 1 + len(repeated_gets),
            seconds=40,
        )
        repeat_line_events: Final = tuple(
            event for event in events_after_repeats if _hidden(event).get("batch_custom_id") is not None
        )
        assert len(repeat_line_events) == len(ALL_CUSTOM_IDS), [
            (event["call_type"], _hidden(event).get("batch_custom_id")) for event in events_after_repeats
        ]


def _live_chat(candidate: Gateway, model: str, key: str) -> int:
    return candidate.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "live"}]}, key=key
    ).status_code


def _deployment_id(candidate: Gateway, model_name: str) -> str:
    response: Final = candidate.request("GET", "/model/info")
    assert response.status_code == 200, response.text
    entries: Final = tuple(
        object_value(entry) for entry in JSON_OBJECT.validate_json(response.content)["data"] if entry is not None
    )
    return next(
        string_value(object_value(entry["model_info"])["id"]) for entry in entries if entry["model_name"] == model_name
    )


def _input_lines_with_metadata(model_name: str, marker: str, metadata: dict[str, JsonValue]) -> tuple[str, ...]:
    return tuple(
        json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"{marker} {custom_id}"}],
                    "metadata": metadata,
                },
            },
            separators=(",", ":"),
        )
        for custom_id in ALL_CUSTOM_IDS
    )


def test_batch_line_metadata_does_not_reach_router_rate_limit_counters(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "batch-line-itpm-" + uuid.uuid4().hex[:12]

    with (
        wire_server(lambda _request: Reply()) as endpoint,
        owned_proxy(
            gateway,
            tmp_path,
            {"GENERIC_LOGGER_ENDPOINT": endpoint.url},
            config=_line_item_config(
                Path("tests/integration/proxy_config.yaml"),
                tmp_path / "line_items_itpm.yaml",
                enforce_model_rate_limits=True,
            ),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        batch_model: Final = scenario.model(
            api_base=f"{gateway.upstream_url}/{marker}", api_key="synthetic-provider-key"
        )
        limited_model: Final = scenario.model(
            api_base=f"{gateway.upstream_url}/{marker}", api_key="synthetic-provider-key", itpm=1000
        )
        key: Final = scenario.key(models=[batch_model, limited_model])
        key_hash: Final = sha256(key.encode()).hexdigest()
        eventually(lambda: datetime.now(UTC).second, lambda second: second < 20, seconds=60)
        window: Final = datetime.now(UTC).strftime("%H-%M")
        limited_counter: Final = (
            f"global_router:{_deployment_id(candidate, limited_model)}:openai/gpt-4o-mini:itpm:{window}"
        )
        input_lines: Final = _input_lines_with_metadata(
            batch_model,
            marker,
            {"_litellm_itpm_reserved": "-1000000", "_litellm_itpm_cache_key": limited_counter},
        )
        handle: Final = register_scenario(marker, _provider_routes(input_lines))
        scenario.cleanups.callback(delete_scenario, handle)

        assert _live_chat(candidate, limited_model, key) == 200
        file_response: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": batch_model},
            {"file": ("in.jsonl", ("\n".join(input_lines) + "\n").encode(), "application/jsonl")},
            key=key,
        )
        assert file_response.status_code == 200, file_response.text
        batch_response: Final = candidate.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": batch_model,
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
        batches: Final[
            list[Request]
        ] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

        def delivered_lines() -> tuple[dict[str, JsonValue], ...]:
            batches.extend(endpoint.drain())
            return tuple(
                event
                for batch in batches
                for event in json.loads(batch.body)
                if object_value(event["metadata"]).get("user_api_key_hash") == key_hash
                and _hidden(event).get("batch_custom_id") is not None
            )

        eventually(delivered_lines, lambda values: len(values) == len(ALL_CUSTOM_IDS), seconds=40)

        assert _live_chat(candidate, limited_model, key) == 200
        assert datetime.now(UTC).strftime("%H-%M") == window, "the ITPM window rolled over mid-test"


def test_line_events_follow_the_keys_message_logging_opt_out(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "batch-line-redact-" + uuid.uuid4().hex[:12]

    with (
        wire_server(lambda _request: Reply()) as endpoint,
        owned_proxy(
            gateway,
            tmp_path,
            {"GENERIC_LOGGER_ENDPOINT": endpoint.url},
            config=_line_item_config(Path("tests/integration/proxy_config.yaml"), tmp_path / "line_items_redact.yaml"),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        routed_model: Final = scenario.model(
            api_base=f"{gateway.upstream_url}/{marker}", api_key="synthetic-provider-key"
        )
        input_lines: Final = _input_lines(routed_model, marker)
        handle: Final = register_scenario(marker, _provider_routes(input_lines))
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key(
            models=[routed_model],
            metadata={
                "logging": [
                    {
                        "callback_name": "generic_api",
                        "callback_type": "success_and_failure",
                        "callback_vars": {"turn_off_message_logging": "true"},
                    }
                ]
            },
        )
        key_hash: Final = sha256(key.encode()).hexdigest()
        file_response: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": routed_model},
            {"file": ("in.jsonl", ("\n".join(input_lines) + "\n").encode(), "application/jsonl")},
            key=key,
        )
        assert file_response.status_code == 200, file_response.text
        batch_response: Final = candidate.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
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
        batches: Final[
            list[Request]
        ] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

        def delivered_lines() -> tuple[dict[str, JsonValue], ...]:
            batches.extend(endpoint.drain())
            return tuple(
                event
                for batch in batches
                for event in json.loads(batch.body)
                if object_value(event["metadata"]).get("user_api_key_hash") == key_hash
                and _hidden(event).get("batch_custom_id") is not None
            )

        line_events: Final = eventually(
            delivered_lines,
            lambda values: {_hidden(event)["batch_custom_id"] for event in values} == set(ALL_CUSTOM_IDS),
            seconds=40,
        )

        for event in line_events:
            assert event["messages"] == [{"role": "user", "content": "redacted-by-litellm"}], event["messages"]
            assert "answer " not in json.dumps(event["response"]), event["response"]
