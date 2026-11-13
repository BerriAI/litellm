import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.repositories.customer_activity_repository import CustomerActivityRepository, CustomerEntity


class CustomerDatabase:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE "LiteLLM_CustomerOrganizationTable" (customer_org_id TEXT, alias TEXT);
            CREATE TABLE "LiteLLM_CustomerTeamTable" (customer_team_id TEXT, parent_org_id TEXT, alias TEXT);
            CREATE TABLE "LiteLLM_CustomerMemberTable" (customer_member_id TEXT, parent_team_id TEXT, alias TEXT);
            INSERT INTO "LiteLLM_CustomerOrganizationTable" VALUES ('org-a', 'Customer A'), ('org-b', NULL);
            INSERT INTO "LiteLLM_CustomerTeamTable" VALUES
                ('team-a1', 'org-a', 'First team'), ('team-a2', 'org-a', NULL), ('team-b', 'org-b', NULL);
            INSERT INTO "LiteLLM_CustomerMemberTable" VALUES
                ('member-a1', 'team-a1', 'First member'), ('member-a2', 'team-a2', NULL),
                ('member-a3', 'team-a1', NULL), ('member-b', 'team-b', NULL);
        """)

    async def query_rows(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return tuple(
            dict(row)
            for row in self.connection.execute(
                query.replace("= ANY($1::text[])", "IN (SELECT value FROM json_each($1))"),
                tuple(json.dumps(arg) for arg in args),
            )
        )


@pytest.mark.asyncio
async def test_counts_members_across_teams_without_requiring_any_usage() -> None:
    repository: Final = CustomerActivityRepository(CustomerDatabase().query_rows)
    assert await repository.member_counts(("org-a", "org-b", "missing")) == {"org-a": 3, "org-b": 1}
    assert await repository.member_counts(()) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity,ids,expected",
    (
        (
            CustomerEntity.ORGANIZATION,
            ("org-a", "org-b"),
            {"org-a": {"alias": "Customer A"}, "org-b": {"alias": "org-b"}},
        ),
        (
            CustomerEntity.TEAM,
            ("team-a1", "team-a2"),
            {"team-a1": {"alias": "First team"}, "team-a2": {"alias": "team-a2"}},
        ),
        (
            CustomerEntity.MEMBER,
            ("member-a1", "member-a2"),
            {"member-a1": {"alias": "First member"}, "member-a2": {"alias": "member-a2"}},
        ),
    ),
)
async def test_aliases_at_each_level_fall_back_to_ids(
    entity: CustomerEntity, ids: tuple[str, ...], expected: Mapping[str, object]
) -> None:
    repository: Final = CustomerActivityRepository(CustomerDatabase().query_rows)
    assert await repository.aliases(entity, ids) == expected
    assert await repository.aliases(entity, ()) == {}
    assert await repository.aliases(entity, ("' OR 1=1 --",)) == {}
