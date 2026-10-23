from itertools import chain
from typing import Final
from uuid import uuid4

import pytest

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import write_rows


@pytest.mark.parametrize("field", ("user_alias", "max_budget", "updated_at", "sso_user_id", "status", "key_count"))
@pytest.mark.parametrize("direction", ("asc", "desc"))
def test_user_list_sorts_all_pages(field: str, direction: str, gateway: Gateway, scenario: Scenario) -> None:
    prefix: Final = f"sort-{uuid4().hex}"
    ids: Final = tuple(
        scenario.user(
            user_id=f"{prefix}-{index}",
            user_role="internal_user",
            user_alias=f"Alias {value if index != 2 else 4:03}",
            user_email=f"{prefix}-{index}@example.com",
            max_budget=value,
            metadata={"scim_active": value != 3},
        )
        for index, value in enumerate((20, 3, 3, 100, 0))
    )
    for user_id, value in zip(ids, (20, 3, 3, 100, 0), strict=True):
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET sso_user_id = %s, '
            "updated_at = timestamp '2030-01-01' + %s::int * interval '1 day' WHERE user_id = %s",
            (f"{prefix}-{value:03}", str(value), user_id),
        ) if value != 3 or user_id == ids[1] else write_rows(
            "UPDATE \"LiteLLM_UserTable\" SET sso_user_id = %s, updated_at = timestamp '2030-01-04' WHERE user_id = %s",
            (f"{prefix}-004", user_id),
        )
        for index in range(value if value < 4 else 1):
            scenario.key(user_id=user_id, key_alias=f"{user_id}-key-{index}")
    write_rows(
        'UPDATE "LiteLLM_UserTable" SET max_budget = NULL, updated_at = NULL, sso_user_id = NULL WHERE user_id = %s',
        (ids[-1],),
    )
    scenario.key(user_id=ids[-1])
    write_rows(
        'UPDATE "LiteLLM_VerificationToken" SET team_id = %s WHERE user_id = %s', (UI_SESSION_TOKEN_TEAM_ID, ids[-1])
    )
    values: Final = (
        (0, 1, 1, 0, 0)
        if field == "status"
        else (1, 3, 3, 1, 0)
        if field == "key_count"
        else (20, 3, 4, 100, None)
        if field == "sso_user_id"
        else (20, 3, 4, 100, 0)
        if field == "user_alias"
        else (20, 3, 3, 100, None)
    )
    expected: Final = tuple(
        user_id
        for _, user_id in sorted(
            ((value, user_id) for value, user_id in zip(values, ids, strict=True)),
            key=lambda item: (
                (item[0] is not None if field == "max_budget" and direction == "desc" else item[0] is None),
                (item[0] or 0) * (1 if direction == "asc" else -1),
                item[1],
            ),
        )
    )
    pages: Final = tuple(
        gateway.request(
            "GET",
            "/user/list",
            params={
                "search": prefix,
                "sort_by": field,
                "sort_order": direction,
                "page_size": "2",
                "page": str(page),
            },
        )
        for page in (1, 2, 3)
    )
    assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
    assert all(page.json()["total"] == len(ids) for page in pages)
    assert tuple(user["user_id"] for user in chain.from_iterable(page.json()["users"] for page in pages)) == expected


@pytest.mark.parametrize("field", ("status", "key_count"))
def test_computed_user_sorts_preserve_filters(field: str, gateway: Gateway, scenario: Scenario) -> None:
    team: Final = scenario.team()
    member: Final = scenario.member(team)
    other: Final = scenario.user(user_role="internal_user")
    for filters in ({"team": team}, {"user_ids": member}, {"user_ids": f"{member},{other}"}, {"role": "internal_user"}):
        baseline: Final = gateway.get("/user/list", {"user_ids": f"{member},{other}", **filters, "sort_by": "user_id"})
        response: Final = gateway.get("/user/list", {"user_ids": f"{member},{other}", **filters, "sort_by": field})
        assert response["total"] == baseline["total"]
        assert sorted(user["user_id"] for user in response["users"]) == sorted(
            user["user_id"] for user in baseline["users"]
        )
    ordinary_key: Final = scenario.key(user_id=other)
    denied: Final = gateway.request("GET", "/user/list", key=ordinary_key, params={"sort_by": field})
    assert denied.status_code == 403, denied.text


@pytest.mark.parametrize("field", ("status", "key_count"))
@pytest.mark.parametrize("direction", ("asc", "desc"))
def test_computed_user_sorts_preserve_org_admin_scope(
    field: str, direction: str, gateway: Gateway, scenario: Scenario
) -> None:
    prefix: Final = f"sort-org-users-{uuid4().hex}"
    owner: Final = scenario.user(user_role="internal_user")
    members: Final = tuple(
        scenario.user(user_id=f"{prefix}-{index}", user_role="internal_user", metadata={"scim_active": index != 1})
        for index in range(4)
    )
    budget: Final = scenario.budget(max_budget=10)
    orgs: Final = (f"{prefix}-allowed", f"{prefix}-outside")
    try:
        for org in orgs:
            write_rows(
                'INSERT INTO "LiteLLM_OrganizationTable" '
                "(organization_id, organization_alias, budget_id, created_by, updated_by, updated_at) "
                "VALUES (%s, %s, %s, 'sorting-test', 'sorting-test', NOW())",
                (org, org, budget),
            )
        write_rows(
            'INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, user_role) '
            "VALUES (%s, %s, 'org_admin')",
            (owner, orgs[0]),
        )
        for index, member in enumerate(members):
            write_rows(
                'INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, user_role) '
                "VALUES (%s, %s, 'internal_user')",
                (member, orgs[0] if index < 3 else orgs[1]),
            )
            for _ in range(index):
                scenario.key(user_id=member)
        caller: Final = scenario.key(user_id=owner)
        values: Final = (0, 1, 0) if field == "status" else (0, 1, 2)
        expected: Final = tuple(
            member
            for value, member in sorted(
                zip(values, members[:3], strict=True),
                key=lambda item: (item[0] * (-1 if direction == "desc" else 1), item[1]),
            )
        )
        pages: Final = tuple(
            gateway.request(
                "GET",
                "/user/list",
                key=caller,
                params={
                    "search": prefix,
                    "sort_by": field,
                    "sort_order": direction,
                    "page_size": "2",
                    "page": str(page),
                },
            )
            for page in (1, 2)
        )
        assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
        assert all(page.json()["total"] == 3 for page in pages)
        returned: Final = tuple(row["user_id"] for row in chain.from_iterable(page.json()["users"] for page in pages))
        assert returned == expected
        denied: Final = gateway.request(
            "GET",
            "/user/list",
            key=caller,
            params={
                "organization_ids": orgs[1],
                "sort_by": field,
            },
        )
        assert denied.status_code == 403, denied.text
    finally:
        for org in orgs:
            write_rows('DELETE FROM "LiteLLM_OrganizationMembership" WHERE organization_id = %s', (org,))
            write_rows('DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (org,))
