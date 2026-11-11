import uuid
from typing import Final

import httpx
import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value

pytestmark: Final = pytest.mark.shares_worker_state


def _customer(scenario: Scenario, **fields: object) -> str:
    customer: Final = f"integration-customer-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post("/customer/new", {"user_id": customer, **fields})  # pyright: ignore[reportArgumentType]  # JsonValue kwargs
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [customer]})
    budget_id: Final = created.get("budget_id")
    if isinstance(budget_id, str):
        scenario.cleanups.callback(scenario.delete_budget, budget_id)
    return customer


def _chat(gateway: Gateway, key: str, model: str, customer: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"customer quota probe {uuid.uuid4().hex}"}],
            "user": customer,
        },
        key=key,
    )


def _assert_customer_model_limit_error(response: httpx.Response, customer: str, model: str) -> None:
    assert response.status_code == 429, response.text
    message: Final = string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])
    assert message.startswith(
        f"Rate limit exceeded for model_per_end_user: {customer}:{model}. Limit type: requests. Current limit: 1,"
    ), message


def test_customer_per_model_rpm_limit_is_shared_across_keys_and_isolated_per_customer(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        capped: Final = scenario.model()
        open_model: Final = scenario.model()
        acme: Final = _customer(scenario, model_rpm_limit={capped: 1})
        globex: Final = _customer(scenario, model_rpm_limit={capped: 1})
        key_one: Final = scenario.key()
        key_two: Final = scenario.key()

        assert _chat(gateway, key_one, capped, acme).status_code == 200
        _assert_customer_model_limit_error(_chat(gateway, key_two, capped, acme), acme, capped)
        assert _chat(gateway, key_one, capped, globex).status_code == 200
        assert _chat(gateway, key_one, open_model, acme).status_code == 200
