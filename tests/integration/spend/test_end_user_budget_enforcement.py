import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import JsonValue


def _chat(gateway: Gateway, model: str, key: str, fields: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": f"end user {uuid.uuid4().hex}"}],
            **fields,
        },
        key=key,
    )


def _delete_customer(gateway: Gateway, user_id: str) -> None:
    response: Final = gateway.request("POST", "/customer/delete", {"user_ids": [user_id]})
    assert response.status_code == 200, response.text


def _end_user_spend(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (user_id,))


def _prove_end_user_budget_lockout(
    gateway: Gateway,
    upstream: httpx.Client,
    model: str,
    key: str,
    end_user_id: str,
    request_fields: dict[str, JsonValue],
    other_fields: dict[str, JsonValue],
) -> None:
    served: Final = _chat(gateway, model, key, request_fields)
    assert served.status_code == 200, served.text
    eventually(
        lambda: _end_user_spend(end_user_id),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.05,
        seconds=70,
    )
    upstream.get("/__observations").raise_for_status()
    denied: Final = _chat(gateway, model, key, request_fields)
    assert denied.status_code == 422, denied.text
    assert object_value(denied.json()["error"])["type"] == "budget_exceeded"
    assert upstream.get("/__observations").json()["requests"] == []
    other: Final = _chat(gateway, model, key, other_fields)
    assert other.status_code == 200, other.text
    assert len(upstream.get("/__observations").json()["requests"]) == 1


def test_customer_max_budget_refuses_second_request(gateway: Gateway) -> None:
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
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            model,
            key,
            end_user_id,
            {"user": end_user_id},
            {"user": f"end-user-{uuid.uuid4().hex}"},
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
        _prove_end_user_budget_lockout(
            gateway,
            upstream,
            model,
            key,
            end_user_id,
            {"litellm_metadata": {"user": end_user_id}},
            {"litellm_metadata": {"user": f"end-user-{uuid.uuid4().hex}"}},
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
        served: Final = _chat(gateway, model, key, {"metadata": {"user_id": end_user_id}})
        assert served.status_code == 200, served.text
        eventually(
            lambda: _end_user_spend(end_user_id),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.05,
            seconds=70,
        )
        upstream.get("/__observations").raise_for_status()
        denied: Final = _chat(gateway, model, key, {"metadata": {"user_id": end_user_id}})
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded"
        assert upstream.get("/__observations").json()["requests"] == []
        other: Final = _chat(gateway, model, key, {"metadata": {"user_id": other_end_user_id}})
        assert other.status_code == 200, other.text
        assert len(upstream.get("/__observations").json()["requests"]) == 1
