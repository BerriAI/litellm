import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value


def _customer(scenario: Scenario, **fields: JsonValue) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


def _chat(
    gateway: Gateway, key: str, model: str, *, customer: str | None, headers: Mapping[str, str] | None = None
) -> httpx.Response:
    body: Final = {
        "model": model,
        "messages": [{"role": "user", "content": "customer allowlist"}],
        **({"user": customer} if customer is not None else {}),
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def _denial_type(response: httpx.Response) -> JsonValue:
    assert response.status_code == 403, response.text
    return object_value(object_value(response.json())["error"])["type"]


def test_customer_models_allowlist_rejects_model_outside_it_for_the_same_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        disallowed: Final = scenario.model()
        key: Final = scenario.key(models=[allowed, disallowed])
        customer: Final = _customer(scenario, models=[allowed])

        info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert info["models"] == [allowed], info

        permitted: Final = _chat(gateway, key, allowed, customer=customer)
        assert permitted.status_code == 200, permitted.text
        denied: Final = _chat(gateway, key, disallowed, customer=customer)
        assert _denial_type(denied) == "customer_model_access_denied", denied.text
        via_header: Final = _chat(gateway, key, disallowed, customer=None, headers={"x-litellm-customer-id": customer})
        assert _denial_type(via_header) == "customer_model_access_denied", via_header.text
        without_customer: Final = _chat(gateway, key, disallowed, customer=None)
        assert without_customer.status_code == 200, without_customer.text


def test_customer_update_moves_and_clears_the_models_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        customer: Final = _customer(scenario)
        assert _chat(gateway, key, second, customer=customer).status_code == 200

        gateway.post("/customer/update", {"user_id": customer, "models": [second]})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == [second]
        moved_denial: Final = _chat(gateway, key, first, customer=customer)
        assert _denial_type(moved_denial) == "customer_model_access_denied", moved_denial.text
        assert _chat(gateway, key, second, customer=customer).status_code == 200

        gateway.post("/customer/update", {"user_id": customer, "models": []})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == []
        cleared: Final = _chat(gateway, key, first, customer=customer)
        assert cleared.status_code == 200, cleared.text


def test_customer_models_allowlist_does_not_widen_key_model_access(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key_model: Final = scenario.model()
        customer_only: Final = scenario.model()
        key: Final = scenario.key(models=[key_model])
        customer: Final = _customer(scenario, models=[key_model, customer_only])

        assert _chat(gateway, key, key_model, customer=customer).status_code == 200
        denied: Final = _chat(gateway, key, customer_only, customer=customer)
        assert _denial_type(denied) == "key_model_access_denied", denied.text
