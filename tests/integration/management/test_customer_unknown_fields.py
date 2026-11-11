import uuid
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, Scenario

pytestmark: Final = pytest.mark.shares_worker_state


def _customer(scenario: Scenario, **fields: object) -> str:
    customer: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": customer, **fields})  # pyright: ignore[reportArgumentType]  # JsonValue kwargs
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [customer]})
    return customer


def _post(gateway: Gateway, route: str, body: dict[str, object]) -> httpx.Response:
    return gateway.request("POST", route, body)  # pyright: ignore[reportArgumentType]  # JsonValue body


def _info(gateway: Gateway, customer: str) -> dict[str, object]:
    return gateway.get("/customer/info", {"end_user_id": customer})


@pytest.mark.parametrize(
    ("body", "unknown_field"),
    [
        pytest.param({"tpm_limit": 5}, "tpm_limit", id="budget_field_on_update"),
        pytest.param({"quota": {"opus": 1}}, "quota", id="made_up_field"),
    ],
)
def test_customer_update_rejects_unknown_fields_and_leaves_the_customer_unchanged(
    gateway: Gateway, body: dict[str, object], unknown_field: str
) -> None:
    with gateway.scenario() as scenario:
        customer: Final = _customer(scenario, alias="before")
        rejected: Final = _post(gateway, "/customer/update", {"user_id": customer, **body})
        assert rejected.status_code == 422, rejected.text
        assert unknown_field in rejected.text, rejected.text
        assert _info(gateway, customer)["alias"] == "before"

        accepted: Final = _post(gateway, "/customer/update", {"user_id": customer, "alias": "after"})
        assert accepted.status_code == 200, accepted.text
        assert _info(gateway, customer)["alias"] == "after"


def test_customer_new_rejects_a_misspelled_field_and_creates_nothing(gateway: Gateway) -> None:
    customer: Final = f"integration-customer-{uuid.uuid4().hex}"
    rejected: Final = _post(gateway, "/customer/new", {"user_id": customer, "tpm_limti": 5})
    assert rejected.status_code == 422, rejected.text
    assert "tpm_limti" in rejected.text, rejected.text
    missing: Final = gateway.request("GET", "/customer/info", params={"end_user_id": customer})
    assert missing.status_code == 404, missing.text
