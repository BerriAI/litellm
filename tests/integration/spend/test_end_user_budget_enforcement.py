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
    upstream.delete("/__observations").raise_for_status()
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

    upstream.delete("/__observations").raise_for_status()
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


class _Budget(BaseModel):
    budget_id: str
    max_budget: float | None


class _CustomerInfo(BaseModel):
    user_id: str
    blocked: bool
    alias: str | None
    spend: float
    litellm_budget_table: _Budget | None


def _new_customer(gateway: Gateway, user_id: str, *, path: str = "/customer/new", **fields: JsonValue) -> None:
    response: Final = gateway.request("POST", path, {"user_id": user_id, **fields})
    assert response.status_code == 200, response.text


def _customer_info(gateway: Gateway, user_id: str) -> _CustomerInfo:
    response: Final = gateway.request("GET", "/customer/info", params={"end_user_id": user_id})
    assert response.status_code == 200, response.text
    return _CustomerInfo.model_validate_json(response.content)


def _chat(gateway: Gateway, model: str, key: str, end_user: str, *, stream: bool = False) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": f"end user budget {uuid.uuid4().hex}"}],
        "user": end_user,
    }
    if stream:
        body["stream"] = True
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _observed_requests(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _expected_observation(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": body,
        "method": "POST",
        "api_key": "",
    }


def _sent_body(response: httpx.Response, end_user: str) -> dict[str, JsonValue]:
    sent: Final = object_value(json.loads(response.request.content))
    expected: Final[dict[str, JsonValue]] = {"messages": sent["messages"], "model": "gpt-4o-mini", "user": end_user}
    if sent.get("stream") is True:
        expected["stream"] = True
        expected["stream_options"] = {"include_usage": True}
    return expected


def _end_user_row(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend, blocked FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (user_id,))


def _daily_rows(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT model, model_group, spend, api_requests, successful_requests FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s',
        (user_id,),
    )


_REFUSAL_MESSAGE: Final = re.compile(
    r"^ExceededBudget: End User=(?P<user>\S+) over budget\. Spend=(?P<spend>[0-9.eE+-]+), Budget=(?P<budget>[0-9.eE+-]+)$"
)


def _assert_budget_refusal(response: httpx.Response, end_user: str, budget: float) -> None:
    error: Final = object_value(response.json()["error"])
    assert error["type"] == "budget_exceeded"
    assert error["param"] is None
    assert error["code"] == "422"
    matched: Final = _REFUSAL_MESSAGE.match(str(error["message"]))
    assert matched is not None, error["message"]
    assert matched.group("user") == end_user
    assert float(matched.group("spend")) == pytest.approx(0.06)
    assert float(matched.group("budget")) == budget


def test_customer_budget_refuses_calls_and_writes_daily_spend(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        upstream.delete("/__observations").raise_for_status()
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        info: Final = _customer_info(gateway, end_user)
        assert info.blocked is False
        assert info.litellm_budget_table is not None and info.litellm_budget_table.max_budget == 0.05, info
        joined: Final = read_rows(
            'SELECT b.max_budget FROM "LiteLLM_EndUserTable" e JOIN "LiteLLM_BudgetTable" b'
            " ON e.budget_id = b.budget_id WHERE e.user_id = %s",
            (end_user,),
        )
        assert len(joined) == 1 and float(str(joined[0]["max_budget"])) == 0.05, joined

        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user)
        assert served.status_code == 200, served.text
        assert _observed_requests(upstream) == [_expected_observation(_sent_body(served, end_user))]
        upstream.delete("/__observations").raise_for_status()

        charged: Final = eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )
        assert float(str(charged[0]["spend"])) == pytest.approx(20 * 0.001 + 20 * 0.002)
        daily: Final = eventually(lambda: _daily_rows(end_user), lambda rows: len(rows) == 1, seconds=70)
        assert daily[0]["model"] == "openai/gpt-4o-mini"
        assert daily[0]["model_group"] == model
        assert float(str(daily[0]["spend"])) == pytest.approx(0.06)
        assert int(str(daily[0]["api_requests"])) == 1
        assert int(str(daily[0]["successful_requests"])) == 1

        denied: Final = _chat(gateway, model, key, end_user)
        assert denied.status_code == 422, denied.text
        _assert_budget_refusal(denied, end_user, 0.05)
        assert _observed_requests(upstream) == []


def test_customer_budget_update_takes_effect_without_restart(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        upstream.delete("/__observations").raise_for_status()
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, path="/end_user/new", max_budget=1.0)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user)
        assert served.status_code == 200, served.text
        assert _observed_requests(upstream) == [_expected_observation(_sent_body(served, end_user))]
        upstream.delete("/__observations").raise_for_status()

        updated: Final = gateway.request("POST", "/customer/update", {"user_id": end_user, "max_budget": 0.01})
        assert updated.status_code == 200, updated.text
        info: Final = _customer_info(gateway, end_user)
        assert info.litellm_budget_table is not None and info.litellm_budget_table.max_budget == 0.01, info
        budget_row: Final = read_rows(
            'SELECT b.max_budget FROM "LiteLLM_EndUserTable" e JOIN "LiteLLM_BudgetTable" b'
            " ON e.budget_id = b.budget_id WHERE e.user_id = %s",
            (end_user,),
        )
        assert len(budget_row) == 1 and float(str(budget_row[0]["max_budget"])) == 0.01, budget_row
        eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )
        denied: Final = _chat(gateway, model, key, end_user)
        assert denied.status_code == 422, denied.text
        _assert_budget_refusal(denied, end_user, 0.01)
        assert _observed_requests(upstream) == []


def test_streamed_call_is_refused_as_json_once_over_budget(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        upstream.delete("/__observations").raise_for_status()
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user, stream=True)
        assert served.status_code == 200, served.text
        assert served.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"id":"chatcmpl-' in served.text and "data: [DONE]" in served.text, served.text
        assert _observed_requests(upstream) == [_expected_observation(_sent_body(served, end_user))]
        upstream.delete("/__observations").raise_for_status()
        eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )

        denied: Final = _chat(gateway, model, key, end_user, stream=True)
        assert denied.status_code == 422, denied.text
        assert denied.headers["content-type"].startswith("application/json")
        assert "data:" not in denied.text, denied.text
        _assert_budget_refusal(denied, end_user, 0.05)
        assert _observed_requests(upstream) == []
