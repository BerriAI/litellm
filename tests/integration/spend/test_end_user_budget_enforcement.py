import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import BaseModel, JsonValue

PROVIDER_AUTHORIZATION: Final = "Bearer integration-provider-key"
CANNED_CONTENT: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
RESPONSES_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_$UNIQUE_ID",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40},
    },
)


class ChatCompletion(BaseModel):
    id: str
    object: str


class ChatChunkDelta(BaseModel):
    content: str | None = None


class ChatChunkChoice(BaseModel):
    delta: ChatChunkDelta


class ChatChunk(BaseModel):
    object: str
    choices: list[ChatChunkChoice]


class ResponsesResult(BaseModel):
    id: str
    object: str
    status: str


class AnthropicMessage(BaseModel):
    type: str
    role: str
    content: list[dict[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class Spelling:
    route: str
    fields: Callable[[str], dict[str, JsonValue]]
    headers: Callable[[str], dict[str, str]]
    upstream_fields: Callable[[str], dict[str, JsonValue]]
    stream: bool = False


def _no_fields(_: str) -> dict[str, JsonValue]:
    return {}


def _no_headers(_: str) -> dict[str, str]:
    return {}


SPELLINGS: Final[Mapping[str, Spelling]] = {
    "user": Spelling("/v1/chat/completions", lambda user: {"user": user}, _no_headers, lambda user: {"user": user}),
    "x-litellm-customer-id": Spelling(
        "/v1/chat/completions", _no_fields, lambda user: {"x-litellm-customer-id": user}, _no_fields
    ),
    "x-litellm-end-user-id": Spelling(
        "/v1/chat/completions", _no_fields, lambda user: {"x-litellm-end-user-id": user}, _no_fields
    ),
    "litellm_metadata": Spelling(
        "/v1/chat/completions", lambda user: {"litellm_metadata": {"user": user}}, _no_headers, _no_fields
    ),
    "litellm_metadata_json": Spelling(
        "/v1/chat/completions",
        lambda user: {"litellm_metadata": json.dumps({"user": user})},
        _no_headers,
        _no_fields,
    ),
    "metadata": Spelling("/v1/chat/completions", lambda user: {"metadata": {"user_id": user}}, _no_headers, _no_fields),
    "metadata_json": Spelling(
        "/v1/chat/completions", lambda user: {"metadata": json.dumps({"user_id": user})}, _no_headers, _no_fields
    ),
    "safety_identifier": Spelling(
        "/v1/chat/completions",
        lambda user: {"safety_identifier": user},
        _no_headers,
        lambda user: {"safety_identifier": user},
    ),
    "user_stream": Spelling(
        "/v1/chat/completions",
        lambda user: {"user": user, "stream": True},
        _no_headers,
        lambda user: {"user": user, "stream": True, "stream_options": {"include_usage": True}},
        stream=True,
    ),
    "litellm_metadata_stream": Spelling(
        "/v1/chat/completions",
        lambda user: {"litellm_metadata": {"user": user}, "stream": True},
        _no_headers,
        lambda _: {"stream": True, "stream_options": {"include_usage": True}},
        stream=True,
    ),
    "responses_user": Spelling("/v1/responses", lambda user: {"user": user}, _no_headers, lambda user: {"user": user}),
    "responses_safety_identifier": Spelling(
        "/v1/responses",
        lambda user: {"safety_identifier": user},
        _no_headers,
        lambda user: {"safety_identifier": user},
    ),
    "messages_metadata_user_id": Spelling(
        "/v1/messages", lambda user: {"metadata": {"user_id": user}}, _no_headers, lambda user: {"user": user}
    ),
}


@dataclass(frozen=True, slots=True)
class Deployment:
    model: str
    upstream_prefix: str


def _deployment(scenario: Scenario, spelling: Spelling) -> Deployment:
    if spelling.route == "/v1/chat/completions":
        return Deployment(scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002), "")
    scenario_id: Final = f"enduser{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, RESPONSES_REPLY)
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(
        api_base=f"{handle.api_base()}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
    )
    return Deployment(model, f"/{scenario_id}")


def _client_body(spelling: Spelling, model: str, user_id: str, content: str) -> dict[str, JsonValue]:
    match spelling.route:
        case "/v1/chat/completions":
            base: dict[str, JsonValue] = {
                "model": model,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": content}],
            }
        case "/v1/responses":
            base = {"model": model, "max_output_tokens": 20, "input": content}
        case _:
            base = {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": content}]}
    return {**base, **spelling.fields(user_id)}


def _upstream_request(spelling: Spelling, prefix: str, user_id: str, content: str) -> dict[str, JsonValue]:
    match spelling.route:
        case "/v1/chat/completions":
            path = "/v1/chat/completions"
            body: dict[str, JsonValue] = {
                "model": "gpt-4o-mini",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": content}],
            }
        case "/v1/responses":
            path = f"{prefix}/v1/responses"
            body = {"model": "gpt-4o-mini", "input": content, "max_output_tokens": 20}
        case _:
            path = f"{prefix}/v1/responses"
            body = {
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]}],
                "include": ["reasoning.encrypted_content"],
                "max_output_tokens": 20,
            }
    return {
        "path": path,
        "authorization": PROVIDER_AUTHORIZATION,
        "body": {**body, **spelling.upstream_fields(user_id)},
        "method": "POST",
        "api_key": "",
    }


def _send(
    gateway: Gateway, spelling: Spelling, deployment: Deployment, key: str, user_id: str, content: str
) -> httpx.Response:
    return gateway.request(
        "POST",
        spelling.route,
        _client_body(spelling, deployment.model, user_id, content),
        key=key,
        headers=spelling.headers(user_id),
    )


def _assert_served(spelling: Spelling, response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    if spelling.stream:
        events: Final = [
            line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]", response.text
        chunks: Final = [ChatChunk.model_validate_json(event) for event in events[:-1]]
        assert {chunk.object for chunk in chunks} == {"chat.completion.chunk"}, response.text
        assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == CANNED_CONTENT
        return
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.06), dict(response.headers)
    match spelling.route:
        case "/v1/chat/completions":
            assert ChatCompletion.model_validate_json(response.content).object == "chat.completion", response.text
        case "/v1/responses":
            parsed: Final = ResponsesResult.model_validate_json(response.content)
            assert (parsed.object, parsed.status) == ("response", "completed"), response.text
        case _:
            message: Final = AnthropicMessage.model_validate_json(response.content)
            assert (message.type, message.role, message.content) == (
                "message",
                "assistant",
                [{"type": "text", "text": "ok"}],
            ), response.text


def _delete_customer(gateway: Gateway, user_id: str) -> None:
    if not _end_user_spend(user_id):
        return
    response: Final = gateway.request("POST", "/customer/delete", {"user_ids": [user_id]})
    assert response.status_code == 200, response.text


def _end_user_spend(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (user_id,))


def _observed(upstream: httpx.Client) -> list[JsonValue]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = object_value(response.json())["requests"]
    assert isinstance(requests, list), response.text
    return requests


def _denial_error(denied: httpx.Response) -> dict[str, JsonValue]:
    assert denied.status_code == 422, denied.text
    return object_value(object_value(denied.json())["error"])


def _prove_end_user_budget_lockout(
    gateway: Gateway,
    upstream: httpx.Client,
    spelling: Spelling,
    deployment: Deployment,
    key: str,
    end_user_id: str,
    other_end_user_id: str,
    expected_budget: Mapping[str, JsonValue],
    info_path: str = "/customer/info",
) -> None:
    served_content: Final = f"served {end_user_id}"
    _observed(upstream)
    served: Final = _send(gateway, spelling, deployment, key, end_user_id, served_content)
    _assert_served(spelling, served)
    assert _observed(upstream) == [
        _upstream_request(spelling, deployment.upstream_prefix, end_user_id, served_content)
    ], served.text
    eventually(
        lambda: _end_user_spend(end_user_id),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.05,
        seconds=30,
    )
    persisted_spend: Final = float(str(_end_user_spend(end_user_id)[0]["spend"]))
    assert persisted_spend == pytest.approx(0.06), persisted_spend
    customer_info: Final = gateway.request("GET", info_path, params={"end_user_id": end_user_id})
    assert customer_info.status_code == 200, customer_info.text
    customer: Final = object_value(customer_info.json())
    assert customer["user_id"] == end_user_id, customer_info.text
    assert float(str(customer["spend"])) == pytest.approx(persisted_spend), customer_info.text
    assert customer["budget_id"] == expected_budget["budget_id"], customer_info.text
    budget: Final = customer["litellm_budget_table"]
    expected_max_budget: Final = expected_budget["max_budget"]
    if expected_max_budget is None:
        assert budget is None, customer_info.text
    else:
        assert float(str(object_value(budget)["max_budget"])) == float(str(expected_max_budget)), customer_info.text

    _observed(upstream)
    denied: Final = _send(gateway, spelling, deployment, key, end_user_id, f"denied {end_user_id}")
    assert _observed(upstream) == [], denied.text
    error: Final = _denial_error(denied)
    assert error["type"] == "budget_exceeded", denied.text
    matched: Final = re.fullmatch(
        rf"ExceededBudget: End User={re.escape(end_user_id)} over budget\. "
        rf"Spend=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?), Budget=0\.05",
        string_value(error["message"]),
    )
    assert matched is not None, denied.text
    assert float(matched.group(1)) == pytest.approx(persisted_spend), denied.text

    other_content: Final = f"served {other_end_user_id}"
    other: Final = _send(gateway, spelling, deployment, key, other_end_user_id, other_content)
    _assert_served(spelling, other)
    assert _observed(upstream) == [
        _upstream_request(spelling, deployment.upstream_prefix, other_end_user_id, other_content)
    ], other.text
    other_spend: Final = eventually(
        lambda: _end_user_spend(other_end_user_id),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.05,
        seconds=30,
    )
    assert float(str(other_spend[0]["spend"])) == pytest.approx(0.06), other_spend


@pytest.mark.parametrize("spelling", tuple(SPELLINGS))
def test_customer_max_budget_refuses_second_request(gateway: Gateway, spelling: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        variant: Final = SPELLINGS[spelling]
        deployment: Final = _deployment(scenario, variant)
        key: Final = scenario.key(models=[deployment.model])
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        created: Final = gateway.post("/customer/new", {"user_id": end_user_id, "max_budget": 0.05})
        assert created["user_id"] == end_user_id
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            variant,
            deployment,
            key,
            end_user_id,
            other_end_user_id,
            {"budget_id": string_value(created["budget_id"]), "max_budget": 0.05},
        )


def test_end_user_alias_routes_create_and_read_a_budgeted_customer(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        variant: Final = SPELLINGS["user"]
        deployment: Final = _deployment(scenario, variant)
        key: Final = scenario.key(models=[deployment.model])
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        created: Final = gateway.post("/end_user/new", {"user_id": end_user_id, "max_budget": 0.05})
        assert created["user_id"] == end_user_id, created
        assert float(str(object_value(created["litellm_budget_table"])["max_budget"])) == 0.05, created
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            variant,
            deployment,
            key,
            end_user_id,
            other_end_user_id,
            {"budget_id": string_value(created["budget_id"]), "max_budget": 0.05},
            info_path="/end_user/info",
        )


@pytest.mark.parametrize("spelling", ("litellm_metadata", "litellm_metadata_stream"))
def test_customer_tier_budget_refuses_second_request(gateway: Gateway, spelling: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        variant: Final = SPELLINGS[spelling]
        deployment: Final = _deployment(scenario, variant)
        tier: Final = scenario.budget(max_budget=0.05)
        key: Final = scenario.key(models=[deployment.model])
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        created: Final = gateway.post("/customer/new", {"user_id": end_user_id, "budget_id": tier})
        assert created["user_id"] == end_user_id
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            variant,
            deployment,
            key,
            end_user_id,
            other_end_user_id,
            {"budget_id": tier, "max_budget": 0.05},
        )


def test_key_end_user_budget_id_refuses_second_request(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        variant: Final = SPELLINGS["metadata"]
        deployment: Final = _deployment(scenario, variant)
        tier: Final = scenario.budget(max_budget=0.05)
        key: Final = scenario.key(models=[deployment.model], metadata={"end_user_budget_id": tier})
        key_info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert object_value(key_info["metadata"]) == {"end_user_budget_id": tier}, key_info
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            variant,
            deployment,
            key,
            end_user_id,
            other_end_user_id,
            {"budget_id": None, "max_budget": None},
        )
