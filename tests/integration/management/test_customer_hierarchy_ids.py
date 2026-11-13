import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway


@pytest.mark.parametrize(
    ("resource", "field", "parents"),
    (
        ("customer_organizations", "customer_org_id", {}),
        ("customer_organizations/upsert", "customer_org_id", {}),
        ("customer_teams", "customer_team_id", {"parent_org_id": "missing"}),
        ("customer_members", "customer_member_id", {"parent_team_ids": ["missing"]}),
        ("customer_members/upsert", "customer_member_id", {"customer_org_id": "missing"}),
    ),
)
def test_slash_ids_are_rejected_before_parent_lookup_or_persistence(
    gateway: Gateway, resource: str, field: str, parents: dict[str, str | list[str]]
) -> None:
    identifier: Final = f"slash-regression-{uuid.uuid4().hex}/invalid"
    response: Final = gateway.request("POST", f"/management/v1/{resource}", {field: identifier, **parents})
    assert response.status_code == 422, response.text
    assert "Customer IDs must not contain '/'" in response.text
    query: Final = {"q": identifier} if field == "customer_org_id" else {"parent_org_id": "missing"}
    listed: Final = gateway.request("GET", f"/management/v1/{resource.split('/')[0]}", params=query)
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"] == [], listed.text
