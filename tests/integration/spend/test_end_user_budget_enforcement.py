import json
import re
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import JsonValue


def _chat(
    gateway: Gateway,
    model: str,
    key: str,
    fields: Mapping[str, JsonValue],
    content: str,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": content}],
            **fields,
        },
        key=key,
        headers=headers,
    )


def _delete_customer(gateway: Gateway, user_id: str) -> None:
    eventually(lambda: _end_user_spend(user_id), lambda rows: len(rows) == 1, seconds=30)
    response: Final = gateway.request("POST", "/customer/delete", {"user_ids": [user_id]})
    assert response.status_code == 200, response.text


def _end_user_spend(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (user_id,))


def _request_variant(spelling: str, user_id: str) -> tuple[dict[str, JsonValue], dict[str, str]]:
    match spelling:
        case "user":
            return {"user": user_id}, {}
        case "x-litellm-customer-id" | "x-litellm-end-user-id":
            return {}, {spelling: user_id}
        case "litellm_metadata":
            return {"litellm_metadata": {"user": user_id}}, {}
        case "litellm_metadata_json":
            return {"litellm_metadata": json.dumps({"user": user_id})}, {}
        case "metadata":
            return {"metadata": {"user_id": user_id}}, {}
        case "metadata_json":
            return {"metadata": json.dumps({"user_id": user_id})}, {}
        case _:
            raise AssertionError(f"Unsupported end-user spelling: {spelling}")


def _prove_end_user_budget_lockout(
    gateway: Gateway,
    upstream: httpx.Client,
    model: str,
    key: str,
    end_user_id: str,
    other_end_user_id: str,
    request_fields: Mapping[str, JsonValue],
    request_headers: Mapping[str, str],
    spelling: str,
    expected_budget: Mapping[str, JsonValue],
) -> None:
    served_content: Final = f"served {end_user_id}"
    expected_body: Final[dict[str, JsonValue]] = {
        "model": "gpt-4o-mini",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": served_content}],
        **({"user": end_user_id} if spelling == "user" else {}),
    }
    upstream.get("/__observations").raise_for_status()
    served: Final = _chat(gateway, model, key, request_fields, served_content, request_headers)
    assert served.status_code == 200, served.text
    served_observations: Final = upstream.get("/__observations").json()["requests"]
    assert len(served_observations) == 1, served.text
    assert object_value(served_observations[0]["body"]) == expected_body, served.text
    eventually(
        lambda: _end_user_spend(end_user_id),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.05,
        seconds=30,
    )
    persisted_spend: Final = float(str(_end_user_spend(end_user_id)[0]["spend"]))
    customer_info: Final = gateway.request("GET", "/customer/info", params={"end_user_id": end_user_id})
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

    upstream.get("/__observations").raise_for_status()
    denied: Final = _chat(gateway, model, key, request_fields, f"denied {end_user_id}", request_headers)
    assert denied.status_code == 422, denied.text
    error: Final = object_value(denied.json()["error"])
    assert error["type"] == "budget_exceeded", denied.text
    message: Final = str(error["message"])
    matched: Final = re.fullmatch(
        rf"ExceededBudget: End User={re.escape(end_user_id)} over budget\. "
        rf"Spend=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?), Budget=0\.05",
        message,
    )
    assert matched is not None, denied.text
    assert float(matched.group(1)) == pytest.approx(persisted_spend), denied.text
    assert upstream.get("/__observations").json()["requests"] == []

    other_fields, other_headers = _request_variant(spelling, other_end_user_id)
    other_content: Final = f"served {other_end_user_id}"
    other: Final = _chat(gateway, model, key, other_fields, other_content, other_headers)
    assert other.status_code == 200, other.text
    other_observations: Final = upstream.get("/__observations").json()["requests"]
    assert len(other_observations) == 1, other.text
    other_expected_body: Final[dict[str, JsonValue]] = {
        "model": "gpt-4o-mini",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": other_content}],
        **({"user": other_end_user_id} if spelling == "user" else {}),
    }
    assert object_value(other_observations[0]["body"]) == other_expected_body, other.text


@pytest.mark.parametrize(
    "spelling",
    (
        "user",
        "x-litellm-customer-id",
        "x-litellm-end-user-id",
        "litellm_metadata",
        "litellm_metadata_json",
        "metadata",
        "metadata_json",
    ),
)
def test_customer_max_budget_refuses_second_request(gateway: Gateway, spelling: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        created: Final = gateway.post("/customer/new", {"user_id": end_user_id, "max_budget": 0.05})
        assert created["user_id"] == end_user_id
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        request_fields, request_headers = _request_variant(spelling, end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            model,
            key,
            end_user_id,
            other_end_user_id,
            request_fields,
            request_headers,
            spelling,
            {"budget_id": str(created["budget_id"]), "max_budget": 0.05},
        )


def test_customer_tier_budget_refuses_second_request(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tier: Final = scenario.budget(max_budget=0.05)
        key: Final = scenario.key(models=[model])
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        created: Final = gateway.post("/customer/new", {"user_id": end_user_id, "budget_id": tier})
        assert created["user_id"] == end_user_id
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            model,
            key,
            end_user_id,
            other_end_user_id,
            {"litellm_metadata": {"user": end_user_id}},
            {},
            "litellm_metadata",
            {"budget_id": tier, "max_budget": 0.05},
        )


def test_key_end_user_budget_id_refuses_second_request(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tier: Final = scenario.budget(max_budget=0.05)
        key: Final = scenario.key(models=[model], metadata={"end_user_budget_id": tier})
        end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        other_end_user_id: Final = f"end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_customer, gateway, end_user_id)
        scenario.cleanups.callback(_delete_customer, gateway, other_end_user_id)
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            model,
            key,
            end_user_id,
            other_end_user_id,
            {"metadata": {"user_id": end_user_id}},
            {},
            "metadata",
            {"budget_id": None, "max_budget": None},
        )
