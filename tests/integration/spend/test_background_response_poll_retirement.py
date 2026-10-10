import itertools
import os
import uuid
from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database, write_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse
from pydantic import JsonValue

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
INPUT_TOKENS: Final = 19
OUTPUT_TOKENS: Final = 7
SCHEDULER_PERIOD_CEILING_SECONDS: Final = 31
ONE_POLL_SECONDS: Final = 2 * SCHEDULER_PERIOD_CEILING_SECONDS + 8
TWO_POLLS_SECONDS: Final = 4 * SCHEDULER_PERIOD_CEILING_SECONDS + 8
OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 240
PROVIDER_ID: Final = "resp_$REQUEST_ID"


def _response_body(status: str) -> dict[str, JsonValue]:
    completed: Final = status == "completed"
    return {
        "id": PROVIDER_ID,
        "object": "response",
        "created_at": 1,
        "status": status,
        "background": True,
        "store": False,
        "error": None,
        "incomplete_details": None,
        "model": "gpt-4o-mini",
        "output": (
            [
                {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "pong", "annotations": []}],
                }
            ]
            if completed
            else []
        ),
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": (
            {
                "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            if completed
            else None
        ),
        "metadata": {},
    }


def _json(body: dict[str, JsonValue], status: int = 200) -> JsonResponse:
    return JsonResponse(content_type="application/json", body=body, status=status)


def _provider_404(message: str) -> JsonResponse:
    return _json({"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}, 404)


def _routes(submission_status: str, retrieve: JsonResponse) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={"POST /responses": _json(_response_body(submission_status)), f"GET /responses/{PROVIDER_ID}": retrieve},
    )


def _gone_routes(submission_status: str = "queued") -> RoutedResponse:
    return _routes(submission_status, _provider_404(f"Response with id '{PROVIDER_ID}' not found."))


def _scenario_id(marker: str) -> str:
    return f"bg-{marker}-{uuid.uuid4().hex[:12]}"


def _scripted_deployment(scenario: Scenario, marker: str, routes: RoutedResponse) -> tuple[str, ScenarioHandle]:
    return _deployment_on(scenario, _scenario_id(marker), routes)


def _register_deployment(
    scenario: Scenario, scenario_id: str, routes: RoutedResponse
) -> tuple[str, str, ScenarioHandle]:
    handle: Final = register_scenario(scenario_id, routes)
    scenario.cleanups.callback(delete_scenario, handle)
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": f"bg-{sha256(scenario_id.encode()).hexdigest()[:12]}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-scripted-provider",
                "api_base": handle.api_base(),
                "input_cost_per_token": INPUT_COST_PER_TOKEN,
                "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
            },
        },
    )
    return string_value(created["model_name"]), string_value(object_value(created["model_info"])["id"]), handle


def _deployment_on(scenario: Scenario, scenario_id: str, routes: RoutedResponse) -> tuple[str, ScenarioHandle]:
    model_name, model_id, handle = _register_deployment(scenario, scenario_id, routes)
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_name, handle


def _forget_row(unified_id: str, database_url: str | None = None) -> None:
    write_rows(
        'DELETE FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
        (unified_id,),
        database_url=database_url,
    )


def _submit_background_response(scenario: Scenario, key: str, model: str, database_url: str | None = None) -> str:
    response: Final = scenario.gateway.request(
        "POST", "/v1/responses", {"model": model, "input": "poll me", "background": True, "store": False}, key=key
    )
    assert response.status_code == 200, response.text
    unified_id: Final = string_value(object_value(response.json())["id"])
    scenario.cleanups.callback(_forget_row, unified_id, database_url)
    return unified_id


def _row_status(unified_id: str, database_url: str | None = None) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT status FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
        (unified_id,),
        database_url=database_url,
    )


def _await_status(unified_id: str, status: str, seconds: float, database_url: str | None = None) -> None:
    assert eventually(
        lambda: _row_status(unified_id, database_url), lambda rows: rows == [{"status": status}], seconds=seconds
    ) == [{"status": status}]


def _observed_requests(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _calls_to(requests: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> tuple[dict[str, JsonValue], ...]:
    return tuple(request for request in requests if string_value(request["path"]).startswith(f"/{handle.scenario_id}/"))


def _upstream_log(gateway: Gateway, handle: ScenarioHandle) -> Iterator[tuple[dict[str, JsonValue], ...]]:
    fresh: Final = (_calls_to(_observed_requests(gateway), handle) for _ in itertools.count())
    return itertools.accumulate(fresh, lambda seen, calls: (*seen, *calls))


def _polls(log: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> tuple[dict[str, JsonValue], ...]:
    poll_path: Final = f"/{handle.scenario_id}/responses/resp_{handle.scenario_id}"
    return tuple(call for call in log if call["method"] == "GET" and call["path"] == poll_path)


def _await_polls(gateway: Gateway, handle: ScenarioHandle, at_least: int, seconds: float) -> None:
    log: Final = _upstream_log(gateway, handle)
    polls: Final = _polls(
        eventually(lambda: next(log), lambda seen: len(_polls(seen, handle)) >= at_least, seconds), handle
    )
    assert len(polls) >= at_least, polls


def _submissions(log: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        call for call in log if call["method"] == "POST" and call["path"] == f"/{handle.scenario_id}/responses"
    )


@pytest.mark.timeout(240)
@pytest.mark.parametrize("submission_status", ["queued", "in_progress"])
def test_a_response_gone_at_the_provider_is_retired_from_polling(gateway: Gateway, submission_status: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model, handle = _scripted_deployment(scenario, "gone", _gone_routes(submission_status))
        log: Final = _upstream_log(gateway, handle)
        unified_id: Final = _submit_background_response(scenario, key, model)
        _await_status(unified_id, "stale_expired", ONE_POLL_SECONDS)
        seen: Final = next(log)
        assert [call["body"] for call in _submissions(seen, handle)] == [
            {"model": "gpt-4o-mini", "input": "poll me", "background": True, "store": False}
        ]
        assert len(_polls(seen, handle)) >= 1, seen
        caller_view: Final = gateway.request("GET", f"/v1/responses/{unified_id}", key=key)
        assert caller_view.status_code == 404, caller_view.text
        assert f"Response with id 'resp_{handle.scenario_id}' not found." in caller_view.text
        first_clock, _ = _scripted_deployment(scenario, "clock1", _gone_routes())
        _await_status(_submit_background_response(scenario, key, first_clock), "stale_expired", ONE_POLL_SECONDS)
        polls_after_first_clock: Final = len(_polls(next(log), handle))
        second_clock, _ = _scripted_deployment(scenario, "clock2", _gone_routes())
        _await_status(_submit_background_response(scenario, key, second_clock), "stale_expired", ONE_POLL_SECONDS)
        assert len(_polls(next(log), handle)) == polls_after_first_clock
        assert _row_status(unified_id) == [{"status": "stale_expired"}]


@pytest.mark.timeout(180)
def test_a_404_that_does_not_name_the_response_keeps_the_row_queued_for_retry(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, handle = _scripted_deployment(scenario, "vague404", _routes("queued", _provider_404("Not found.")))
        unified_id: Final = _submit_background_response(scenario, scenario.key(), model)
        _await_polls(gateway, handle, 2, TWO_POLLS_SECONDS)
        assert _row_status(unified_id) == [{"status": "queued"}]


@pytest.mark.timeout(180)
def test_a_provider_error_other_than_404_keeps_the_row_queued_for_retry(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        outage: Final = _json({"error": {"message": "The server had an error.", "type": "server_error"}}, 500)
        model, handle = _scripted_deployment(scenario, "outage", _routes("queued", outage))
        unified_id: Final = _submit_background_response(scenario, scenario.key(), model)
        _await_polls(gateway, handle, 2, TWO_POLLS_SECONDS)
        assert _row_status(unified_id) == [{"status": "queued"}]


def _polled_provider_id(row: dict[str, JsonValue]) -> str:
    return ResponsesAPIRequestUtils.decode_responses_api_response_id(string_value(row["request_id"]))["response_id"]


def _poll_spend_rows(handle: ScenarioHandle) -> list[dict[str, JsonValue]]:
    billed: Final = read_rows(
        'SELECT request_id, call_type, status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
        "WHERE metadata->>'internal_call_origin' = %s",
        (BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,),
    )
    return [
        {column: value for column, value in row.items() if column != "request_id"}
        for row in billed
        if _polled_provider_id(row) == f"resp_{handle.scenario_id}"
    ]


@pytest.mark.timeout(180)
def test_a_completed_response_is_marked_completed_and_its_poll_is_billed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, handle = _scripted_deployment(scenario, "done", _routes("queued", _json(_response_body("completed"))))
        unified_id: Final = _submit_background_response(scenario, scenario.key(), model)
        _await_status(unified_id, "completed", ONE_POLL_SECONDS)
        spend_rows: Final = eventually(lambda: _poll_spend_rows(handle), lambda rows: len(rows) >= 1, seconds=70)
        assert spend_rows[0] == {
            "call_type": "aget_responses",
            "status": "success",
            "prompt_tokens": INPUT_TOKENS,
            "completion_tokens": OUTPUT_TOKENS,
            "spend": pytest.approx(INPUT_TOKENS * INPUT_COST_PER_TOKEN + OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN),
        }


def _insert_unreadable_row(scenario: Scenario) -> str:
    unified_id: Final = f"resp_unreadable-{uuid.uuid4().hex[:12]}"
    write_rows(
        'INSERT INTO "LiteLLM_ManagedObjectTable" '
        '("id", "unified_object_id", "model_object_id", "file_object", "file_purpose", "status", "created_at", "updated_at") '
        "VALUES (%s, %s, %s, '[]'::jsonb, 'response', 'queued', NOW() - INTERVAL '1 hour', NOW())",
        (str(uuid.uuid4()), unified_id, unified_id),
    )
    scenario.cleanups.callback(_forget_row, unified_id)
    return unified_id


@pytest.mark.timeout(180)
def test_a_row_the_poll_cannot_prepare_skips_only_that_row(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        unreadable_id: Final = _insert_unreadable_row(scenario)
        model, _ = _scripted_deployment(scenario, "gone", _gone_routes())
        unified_id: Final = _submit_background_response(scenario, scenario.key(), model)
        _await_status(unified_id, "stale_expired", ONE_POLL_SECONDS)
        assert _row_status(unreadable_id) == [{"status": "queued"}]


@pytest.mark.timeout(300)
def test_responses_gone_at_the_provider_do_not_starve_a_newer_response_out_of_cost_polling(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        gone_ids: Final = tuple(
            _submit_background_response(
                scenario, key, _scripted_deployment(scenario, f"gone{index}", _gone_routes())[0]
            )
            for index in range(MAX_OBJECTS_PER_POLL_CYCLE)
        )
        completable_model, completable = _scripted_deployment(
            scenario, "done", _routes("queued", _json(_response_body("completed")))
        )
        completable_id: Final = _submit_background_response(scenario, key, completable_model)
        _await_status(completable_id, "completed", TWO_POLLS_SECONDS)
        assert [_row_status(gone_id) for gone_id in gone_ids] == [
            [{"status": "stale_expired"}]
        ] * MAX_OBJECTS_PER_POLL_CYCLE
        spend_rows: Final = eventually(lambda: _poll_spend_rows(completable), lambda rows: len(rows) >= 1, seconds=70)
        assert spend_rows[0]["spend"] == pytest.approx(
            INPUT_TOKENS * INPUT_COST_PER_TOKEN + OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
        )


@pytest.mark.timeout(OWNED_PROXY_CELL_SECONDS)
def test_a_404_on_a_response_whose_deployment_left_the_router_keeps_the_row_queued_for_retry(
    gateway: Gateway, tmp_path: Path
) -> None:
    deployment_scenario_id: Final = _scenario_id("left")
    provider_id: Final = f"resp_{deployment_scenario_id}"
    env_handle: Final = register_scenario(
        _scenario_id("envbase"),
        RoutedResponse(
            content_type="application/x-routed",
            routes={f"GET /responses/{provider_id}": _provider_404(f"Response with id '{provider_id}' not found.")},
        ),
    )
    try:
        with scratch_database() as database_url:
            overrides: Final = {
                "DATABASE_URL": database_url,
                "OPENAI_API_BASE": env_handle.api_base(),
                "OPENAI_API_KEY": "sk-scripted-provider",
            }
            with owned_proxy_process(
                gateway, tmp_path, overrides, remove_environment=("DATABASE_URL_READ_REPLICA",)
            ) as owned:
                with owned.gateway.scenario() as scenario:
                    outage: Final = _json(
                        {"error": {"message": "The server had an error.", "type": "server_error"}}, 500
                    )
                    model, model_id, _ = _register_deployment(
                        scenario, deployment_scenario_id, _routes("queued", outage)
                    )
                    unified_id: Final = _submit_background_response(scenario, scenario.key(), model, database_url)
                    owned.gateway.post("/model/delete", {"id": model_id})
                    log: Final = _upstream_log(gateway, env_handle)
                    fallback_polls: Final = eventually(
                        lambda: next(log),
                        lambda seen: len(_fallback_polls(seen, env_handle, provider_id)) >= 2,
                        seconds=TWO_POLLS_SECONDS,
                    )
                    assert len(_fallback_polls(fallback_polls, env_handle, provider_id)) >= 2, fallback_polls
                    assert _row_status(unified_id, database_url) == [{"status": "queued"}]
    finally:
        delete_scenario(env_handle)


def _fallback_polls(
    log: Sequence[dict[str, JsonValue]], env_handle: ScenarioHandle, provider_id: str
) -> tuple[dict[str, JsonValue], ...]:
    poll_path: Final = f"/{env_handle.scenario_id}/responses/{provider_id}"
    return tuple(call for call in log if call["method"] == "GET" and call["path"] == poll_path)
