import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
from integration._support.client import Gateway, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue

CONCURRENT_CREATES: Final = 8


def _membership_rows(organization_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id, user_role FROM "LiteLLM_OrganizationMembership" WHERE organization_id = %s',
        (organization_id,),
    )


def _listed(gateway: Gateway, organization_id: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", "/organization/list")
    assert response.status_code == 200, response.text
    entries: Final = response.json()
    assert isinstance(entries, list)
    matches: Final = [entry for entry in entries if entry["organization_id"] == organization_id]
    assert len(matches) == 1, f"{organization_id} listed {len(matches)} times"
    return object_value(matches[0])


def test_concurrent_creates_with_one_alias_each_persist_a_distinct_organization(gateway: Gateway) -> None:
    alias: Final = f"integration-{uuid.uuid4().hex}"

    def create(_: int) -> httpx.Response:
        return gateway.request("POST", "/organization/new", {"organization_alias": alias})

    with ThreadPoolExecutor(max_workers=CONCURRENT_CREATES) as pool:
        responses: Final = tuple(pool.map(create, range(CONCURRENT_CREATES)))
    created: Final = tuple(response.json() for response in responses if response.status_code == 200)
    with gateway.scenario() as scenario:
        for body in created:
            scenario.cleanups.callback(
                scenario.delete_organization, string_value(body["organization_id"]), string_value(body["budget_id"])
            )
        assert [response.status_code for response in responses] == [200] * CONCURRENT_CREATES, [
            response.text for response in responses
        ]
        identities: Final = {string_value(body["organization_id"]) for body in created}
        assert len(identities) == CONCURRENT_CREATES
        rows: Final = read_rows(
            'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_alias = %s', (alias,)
        )
        assert {string_value(row["organization_id"]) for row in rows} == identities


def test_list_returns_each_organization_with_its_budget_and_members(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization(max_budget=3.5, tpm_limit=120)
        member: Final = scenario.org_member(organization_id, role="internal_user")
        listed: Final = _listed(gateway, organization_id)
        budget: Final = object_value(listed["litellm_budget_table"])
        assert (budget["max_budget"], budget["tpm_limit"]) == (3.5, 120)
        members: Final = listed["members"]
        assert isinstance(members, list)
        assert [(object_value(entry)["user_id"], object_value(entry)["user_role"]) for entry in members] == [
            (member, "internal_user")
        ]


def test_member_role_update_and_removal_persist_and_read_back(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        member: Final = scenario.org_member(organization_id, role="internal_user")
        assert _membership_rows(organization_id) == [{"user_id": member, "user_role": "internal_user"}]
        updated: Final = gateway.request(
            "PATCH",
            "/organization/member_update",
            {"organization_id": organization_id, "user_id": member, "role": "org_admin"},
        )
        assert updated.status_code == 200, updated.text
        assert _membership_rows(organization_id) == [{"user_id": member, "user_role": "org_admin"}]
        info_members: Final = gateway.get("/organization/info", {"organization_id": organization_id})["members"]
        assert isinstance(info_members, list)
        assert [object_value(entry)["user_role"] for entry in info_members] == ["org_admin"]
        removed: Final = gateway.request(
            "DELETE", "/organization/member_delete", {"organization_id": organization_id, "user_id": member}
        )
        assert removed.status_code == 200, removed.text
        assert _membership_rows(organization_id) == []
        assert _listed(gateway, organization_id)["members"] == []
