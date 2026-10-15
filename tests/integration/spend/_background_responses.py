import itertools
import math
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held, list_value, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, SseResponse
from pydantic import JsonValue

INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
INPUT_TOKENS: Final = 19
OUTPUT_TOKENS: Final = 7
PRICE: Final = INPUT_TOKENS * INPUT_COST_PER_TOKEN + OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
SCHEDULER_PERIOD_CEILING_SECONDS: Final = 31
ONE_POLL_SECONDS: Final = 2 * SCHEDULER_PERIOD_CEILING_SECONDS + 8
POLL_WINDOW_SECONDS: Final = SCHEDULER_PERIOD_CEILING_SECONDS + 9
SPEND_LANDING_SECONDS: Final = 60
PROVIDER_ID: Final = "resp_$REQUEST_ID"
PROMPT: Final = "bill me once"
READS: Final = 3
CONCURRENT_READS: Final = 12
FINISHED_STATUSES: Final = frozenset({"completed", "incomplete", "cancelled"})
MANAGED_READ_MARKER: Final = "_managed_background_response"
ATTRIBUTION_COLUMNS: Final = ("api_key", "team_id", "organization_id", "user")


def response_body(status: str, *, background: bool = True, **fields: JsonValue) -> dict[str, JsonValue]:
    finished: Final = status in FINISHED_STATUSES
    return {
        "id": PROVIDER_ID,
        "object": "response",
        "created_at": 1,
        "status": status,
        "background": background,
        "store": False,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
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
            if finished
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
            if finished
            else None
        ),
        "metadata": {},
        **fields,
    }


def json_reply(body: dict[str, JsonValue], status: int = 200) -> JsonResponse:
    return JsonResponse(content_type="application/json", body=body, status=status)


def scripted_routes(
    create: JsonResponse, read: JsonResponse, *, provider_id: str = PROVIDER_ID, **extra: JsonResponse
) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /responses": create,
            f"GET /responses/{provider_id}": read,
            **{f"POST /responses/{provider_id}/{action}": response for action, response in extra.items()},
        },
    )


def background_routes(read_status: str, **fields: JsonValue) -> RoutedResponse:
    return scripted_routes(
        json_reply(response_body("queued", **fields)), json_reply(response_body(read_status, **fields))
    )


@dataclass(frozen=True, slots=True)
class Upstream:
    model: str
    model_id: str
    handle: ScenarioHandle

    def script(self, response: RoutedResponse | SseResponse) -> None:
        register_scenario(self.handle.scenario_id, response, control_url=self.handle.control_url)


def scripted_upstream(
    scenario: Scenario, response: RoutedResponse | SseResponse, **litellm_params: JsonValue
) -> Upstream:
    scenario_id: Final = f"bgonce-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": scenario_id,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-scripted-provider",
                "api_base": handle.api_base(),
                "input_cost_per_token": INPUT_COST_PER_TOKEN,
                "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
                **litellm_params,
            },
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return Upstream(scenario_id, model_id, handle)


def forget_row(response_id: str) -> None:
    write_rows('DELETE FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s', (response_id,))


def create_background_response(
    scenario: Scenario,
    key: str,
    model: str,
    *,
    body: Mapping[str, JsonValue] | None = None,
    headers: Mapping[str, str] | None = None,
) -> str:
    response: Final = scenario.gateway.request(
        "POST",
        "/v1/responses",
        body or {"model": model, "input": PROMPT, "background": True, "store": False},
        key=key,
        headers=headers,
    )
    assert response.status_code == 200, response.text
    response_id: Final = string_value(object_value(response.json())["id"])
    scenario.cleanups.callback(forget_row, response_id)
    return response_id


@dataclass(frozen=True, slots=True)
class Read:
    status: str
    text: str
    cost: float | None


def reported_cost(headers: httpx.Headers) -> float | None:
    reported: Final = headers.get("x-litellm-response-cost")
    return None if reported is None else float(reported)


def read_of(response: httpx.Response) -> Read:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    output: Final = list_value(body["output"])
    text: Final = (
        string_value(object_value(list_value(object_value(output[0])["content"])[0])["text"]) if output else ""
    )
    return Read(string_value(body["status"]), text, reported_cost(response.headers))


def read_response(gateway: Gateway, key: str, response_id: str, body: Mapping[str, JsonValue] | None = None) -> Read:
    return read_of(gateway.request("GET", f"/v1/responses/{response_id}", body, key=key))


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def number(value: JsonValue) -> float:
    assert isinstance(value, int | float), value
    return float(value)


def key_spend(key: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),))
    return number(rows[0]["spend"])


def team_spend(team_id: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    return number(rows[0]["spend"])


def user_spend(user_id: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    return number(rows[0]["spend"])


def org_spend(organization_id: str) -> float:
    rows: Final = read_rows(
        'SELECT spend FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization_id,)
    )
    return number(rows[0]["spend"])


def tag_spend(tag: str) -> float:
    rows: Final = read_rows(
        'SELECT COALESCE(SUM(spend), 0) AS spend FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,)
    )
    return number(rows[0]["spend"])


def costs(expected: float) -> Callable[[float], bool]:
    return lambda spend: math.isclose(spend, expected, abs_tol=1e-9)


def assert_charged_exactly(read_spend: Callable[[], float], expected: float) -> None:
    held(read_spend, costs(expected), holding=POLL_WINDOW_SECONDS, seconds=POLL_WINDOW_SECONDS + SPEND_LANDING_SECONDS)


def charged_rows(model_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, call_type, api_key, team_id, organization_id, "user", request_tags, spend, '
        'prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_id = %s AND spend > 0',
        (model_id,),
    )


def create_rows(model_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT api_key, team_id, organization_id, "user", request_tags FROM "LiteLLM_SpendLogs" '
        "WHERE model_id = %s AND call_type = 'aresponses'",
        (model_id,),
    )


def one_charged_row(model_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: charged_rows(model_id), lambda found: len(found) >= 1, seconds=30)
    assert len(rows) == 1, rows
    return rows[0]


def managed_rows(response_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT status, api_key, created_by, team_id FROM "LiteLLM_ManagedObjectTable" '
        "WHERE unified_object_id = %s OR model_object_id = %s",
        (response_id, response_id),
    )


def assert_billed_like_its_creator(
    row: dict[str, JsonValue], key: str, team_id: str | None, user_id: str | None
) -> None:
    assert string_value(row["request_id"]).endswith("_background_cost"), row
    assert row["call_type"] == "aget_responses"
    assert row["api_key"] == hashed(key)
    assert row["team_id"] == team_id
    assert row["user"] == user_id
    assert row["prompt_tokens"] == INPUT_TOKENS
    assert row["completion_tokens"] == OUTPUT_TOKENS
    assert number(row["spend"]) == pytest.approx(PRICE)


def observed_requests(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, list_value(upstream.get("/__observations").json()["requests"])))


def calls_to(requests: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> tuple[dict[str, JsonValue], ...]:
    return tuple(request for request in requests if string_value(request["path"]).startswith(f"/{handle.scenario_id}/"))


def upstream_log(gateway: Gateway, handle: ScenarioHandle) -> Iterator[tuple[dict[str, JsonValue], ...]]:
    fresh: Final = (calls_to(observed_requests(gateway), handle) for _ in itertools.count())
    return itertools.accumulate(fresh, lambda seen, calls: (*seen, *calls))


def submissions(log: Sequence[dict[str, JsonValue]], handle: ScenarioHandle) -> tuple[JsonValue, ...]:
    return tuple(
        call["body"] for call in log if call["method"] == "POST" and call["path"] == f"/{handle.scenario_id}/responses"
    )
