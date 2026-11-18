import json
from itertools import chain
from typing import Final
from uuid import uuid4

import pytest

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import write_rows


@pytest.mark.parametrize(
    "field", ("spend", "max_budget", "tpm_limit", "rpm_limit", "updated_at", "members", "models", "keys_count")
)
@pytest.mark.parametrize("direction", ("asc", "desc"))
def test_team_list_sorts_across_pages_and_respects_membership(
    field: str,
    direction: str,
    gateway: Gateway,
    scenario: Scenario,
) -> None:
    prefix: Final = f"sort-{uuid4().hex}"
    owner: Final = scenario.user(user_role="internal_user")
    caller: Final = scenario.key(user_id=owner)
    values: Final = (20, 3, 3, 100, 0)
    teams: Final = tuple(
        scenario.team(
            team_id=f"{prefix}-{index}",
            team_alias=f"{prefix}-{index}",
            members_with_roles=[{"user_id": owner, "role": "admin"}],
        )
        for index in range(len(values))
    )
    outsider: Final = scenario.team(team_alias=f"{prefix}-outside")
    for team_id, value in zip(teams, values, strict=True):
        write_rows(
            'UPDATE "LiteLLM_TeamTable" SET spend = %s::float, max_budget = %s::float, '
            "tpm_limit = %s::bigint, rpm_limit = %s::bigint, "
            "updated_at = timestamp '2030-01-01' + %s::int * interval '1 day', "
            "members_with_roles = %s::jsonb, models = ARRAY(SELECT generate_series(1, %s::int)::text) "
            "WHERE team_id = %s",
            (
                str(value),
                str(value),
                str(value),
                str(value),
                str(value),
                json.dumps(
                    [{"user_id": owner if i == 0 else f"{prefix}-member-{i}", "role": "admin"} for i in range(value)]
                ),
                str(value),
                team_id,
            ),
        )
        write_rows(
            'INSERT INTO "LiteLLM_VerificationToken" (token, team_id) '
            "SELECT %s || '-key-' || n::text, %s FROM generate_series(1, %s::int) n",
            (team_id, team_id, str(value)),
        )
    write_rows(
        'UPDATE "LiteLLM_TeamTable" SET max_budget = NULL, tpm_limit = NULL, rpm_limit = NULL WHERE team_id = %s',
        (teams[-1],),
    )
    try:
        expected: Final = tuple(
            team_id
            for value, team_id in sorted(
                zip(values, teams, strict=True),
                key=lambda item: (
                    (item[0] != 0 if direction == "desc" else item[0] == 0)
                    if field in ("max_budget", "tpm_limit", "rpm_limit")
                    else False,
                    item[0] * (1 if direction == "asc" else -1),
                    item[1],
                ),
            )
        )
        pages: Final = tuple(
            gateway.request(
                "GET",
                "/v2/team/list",
                key=caller,
                params={
                    "search": prefix,
                    "user_id": owner,
                    "sort_by": field,
                    "sort_order": direction,
                    "page_size": "2",
                    "page": str(page),
                },
            )
            for page in (1, 2, 3)
        )
        assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
        returned: Final = tuple(row["team_id"] for row in chain.from_iterable(page.json()["teams"] for page in pages))
        assert returned == expected
        assert outsider not in returned
        assert all(page.json()["total"] == len(teams) for page in pages)
    finally:
        for team_id in teams:
            write_rows('DELETE FROM "LiteLLM_VerificationToken" WHERE team_id = %s', (team_id,))


@pytest.mark.parametrize("direction", ("asc", "desc"))
def test_team_list_sorts_organization_names_and_missing_values(
    direction: str, gateway: Gateway, scenario: Scenario
) -> None:
    prefix: Final = f"sort-org-{uuid4().hex}"
    budget: Final = scenario.budget(max_budget=10)
    organizations: Final = ((f"{prefix}-z", "alpha"), (f"{prefix}-a", "zulu"), (f"{prefix}-m", ""))
    teams: Final = tuple(scenario.team(team_id=f"{prefix}-{i}", team_alias=f"{prefix}-{i}") for i in range(4))
    try:
        for (organization_id, alias), team_id in zip(organizations, teams):
            write_rows(
                'INSERT INTO "LiteLLM_OrganizationTable" '
                "(organization_id, organization_alias, budget_id, created_by, updated_by, updated_at) "
                "VALUES (%s, %s, %s, 'sorting-test', 'sorting-test', NOW())",
                (organization_id, alias, budget),
            )
            write_rows(
                'UPDATE "LiteLLM_TeamTable" SET organization_id = %s WHERE team_id = %s',
                (organization_id, team_id),
            )
        ascending: Final = (teams[0], teams[2], teams[1])
        ordered: Final = ascending if direction == "asc" else tuple(reversed(ascending))
        pages: Final = tuple(
            gateway.request(
                "GET",
                "/v2/team/list",
                params={
                    "search": prefix,
                    "sort_by": "organization_alias",
                    "sort_order": direction,
                    "page_size": "2",
                    "page": str(page),
                },
            )
            for page in (1, 2)
        )
        assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
        assert tuple(row["team_id"] for row in chain.from_iterable(page.json()["teams"] for page in pages)) == (*ordered, teams[-1])
    finally:
        for team_id in teams:
            write_rows('UPDATE "LiteLLM_TeamTable" SET organization_id = NULL WHERE team_id = %s', (team_id,))
        for organization_id, _ in organizations:
            write_rows('DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization_id,))


@pytest.mark.parametrize("field", ("organization_alias", "members", "models", "keys_count"))
@pytest.mark.parametrize("direction", ("asc", "desc"))
def test_computed_team_sorts_preserve_org_and_membership_union(
    field: str, direction: str, gateway: Gateway, scenario: Scenario
) -> None:
    prefix: Final = f"sort-org-scope-{uuid4().hex}"
    owner: Final = scenario.user(user_role="internal_user")
    budget: Final = scenario.budget(max_budget=10)
    orgs: Final = (f"{prefix}-allowed", f"{prefix}-outside")
    teams: Final = tuple(
        scenario.team(
            team_id=f"{prefix}-{i}",
            team_alias=f"{prefix}-{i}",
            members_with_roles=[{"user_id": owner, "role": "admin"}] if i == 2 else [],
        )
        for i in range(4)
    )
    try:
        for org, alias in zip(orgs, ("alpha", "zulu"), strict=True):
            write_rows(
                'INSERT INTO "LiteLLM_OrganizationTable" '
                "(organization_id, organization_alias, budget_id, created_by, updated_by, updated_at) "
                "VALUES (%s, %s, %s, 'sorting-test', 'sorting-test', NOW())",
                (org, alias, budget),
            )
        write_rows(
            'INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, user_role) '
            "VALUES (%s, %s, 'org_admin')",
            (owner, orgs[0]),
        )
        for index, team in enumerate(teams):
            write_rows(
                'UPDATE "LiteLLM_TeamTable" SET organization_id = %s, members_with_roles = %s::jsonb, '
                "models = ARRAY(SELECT generate_series(1, %s::int)::text) WHERE team_id = %s",
                (
                    orgs[0] if index < 2 else orgs[1],
                    json.dumps(
                        [
                            {"user_id": owner if i == 0 else f"{prefix}-member-{i}", "role": "admin"}
                            for i in range(index + 1)
                        ]
                    ),
                    str(index + 1),
                    team,
                ),
            )
            write_rows(
                'INSERT INTO "LiteLLM_VerificationToken" (token, team_id) '
                "SELECT %s || '-key-' || n::text, %s FROM generate_series(1, %s::int) n",
                (team, team, str(index + 1)),
            )
        caller: Final = scenario.key(user_id=owner)
        values: Final = (0, 0, 1) if field == "organization_alias" else (1, 2, 3)
        expected: Final = tuple(
            team
            for value, team in sorted(
                zip(values, teams[:3], strict=True),
                key=lambda item: (item[0] * (-1 if direction == "desc" else 1), item[1]),
            )
        )
        pages: Final = tuple(
            gateway.request(
                "GET",
                "/v2/team/list",
                key=caller,
                params={
                    "user_id": owner,
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
        assert tuple(row["team_id"] for row in chain.from_iterable(page.json()["teams"] for page in pages)) == expected
        denied: Final = gateway.request(
            "GET",
            "/v2/team/list",
            key=caller,
            params={
                "organization_id": orgs[1],
                "sort_by": field,
            },
        )
        assert denied.status_code == 403, denied.text
    finally:
        for team in teams:
            write_rows('DELETE FROM "LiteLLM_VerificationToken" WHERE team_id = %s', (team,))
            write_rows('UPDATE "LiteLLM_TeamTable" SET organization_id = NULL WHERE team_id = %s', (team,))
        for org in orgs:
            write_rows('DELETE FROM "LiteLLM_OrganizationMembership" WHERE organization_id = %s', (org,))
            write_rows('DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (org,))
