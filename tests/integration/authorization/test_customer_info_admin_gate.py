import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario

pytestmark: Final = pytest.mark.shares_worker_state


def _customer(scenario: Scenario, alias: str) -> str:
    customer: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": customer, "alias": alias})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [customer]})
    return customer


@pytest.mark.parametrize("send_own_organization_id", [True, False], ids=["own_org_in_body", "no_body"])
def test_org_admin_cannot_read_a_customer_record(gateway: Gateway, send_own_organization_id: bool) -> None:
    with gateway.scenario() as scenario:
        customer: Final = _customer(scenario, alias="belongs-to-nobody-in-this-org")
        organization_id: Final = scenario.organization()
        org_admin_key: Final = scenario.key(user_id=scenario.org_member(organization_id, role="org_admin"))

        response: Final = gateway.request(
            "GET",
            "/customer/info",
            {"organization_id": organization_id} if send_own_organization_id else None,
            key=org_admin_key,
            params={"end_user_id": customer},
        )

        assert response.status_code == 401, response.text
        assert customer not in response.text, response.text
        assert "belongs-to-nobody-in-this-org" not in response.text, response.text


@pytest.mark.parametrize("user_role", ["proxy_admin", "proxy_admin_viewer"])
def test_proxy_admins_still_read_a_customer_record(gateway: Gateway, user_role: str) -> None:
    with gateway.scenario() as scenario:
        customer: Final = _customer(scenario, alias="visible-to-admins")
        admin_key: Final = scenario.key(user_id=scenario.user(user_role=user_role))

        response: Final = gateway.request("GET", "/customer/info", key=admin_key, params={"end_user_id": customer})

        assert response.status_code == 200, response.text
        assert (response.json()["user_id"], response.json()["alias"]) == (customer, "visible-to-admins")
