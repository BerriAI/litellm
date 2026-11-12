import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final

import pytest

from litellm.repositories.customer_usage_repository import CustomerUsageRepository


class UsageDatabase:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE "LiteLLM_CustomerOrganizationTable" (customer_org_id TEXT PRIMARY KEY, alias TEXT);
            CREATE TABLE "LiteLLM_CustomerTeamTable" (customer_team_id TEXT PRIMARY KEY, parent_org_id TEXT);
            CREATE TABLE "LiteLLM_CustomerMemberTable" (customer_member_id TEXT PRIMARY KEY, parent_team_id TEXT);
            CREATE TABLE "LiteLLM_DailyEndUserSpend" (
                end_user_id TEXT, date TEXT, spend REAL, api_requests INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER, api_key TEXT, model TEXT
            );
            INSERT INTO "LiteLLM_CustomerOrganizationTable" VALUES
                ('org-a', 'Customer A'), ('org-b', 'Customer B'), ('empty', NULL), ('literal', '100%_\\');
            INSERT INTO "LiteLLM_CustomerTeamTable" VALUES
                ('team-a1', 'org-a'), ('team-a2', 'org-a'), ('team-b', 'org-b');
            INSERT INTO "LiteLLM_CustomerMemberTable" VALUES
                ('member-a1', 'team-a1'), ('member-a2', 'team-a2'), ('member-b', 'team-b');
            INSERT INTO "LiteLLM_DailyEndUserSpend" VALUES
                ('member-a1', '2024-02-01', 1.5, 2, 10, 20, 'key-1', 'model-1'),
                ('member-a1', '2024-02-01', 2.5, 3, 30, 40, 'key-2', 'model-2'),
                ('member-a2', '2024-02-29', 4, 5, 50, 60, 'key-1', 'model-1'),
                ('member-b', '2024-02-15', 7, 8, 70, 80, 'key-1', 'model-1'),
                ('member-a1', '2024-01-31', 100, 100, 100, 100, 'key-1', 'model-1'),
                ('member-a1', '2024-03-01', 200, 200, 200, 200, 'key-1', 'model-1'),
                ('unregistered', '2024-02-15', 300, 300, 300, 300, 'key-1', 'model-1');
        """)

    async def query_rows(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        bindings: Final = {str(index): value for index, value in enumerate(args, start=1)}
        return tuple(dict(row) for row in self.connection.execute(query, bindings).fetchall())


@pytest.mark.asyncio
async def test_rolls_up_all_teams_members_keys_and_models_with_inclusive_boundaries() -> None:
    database: Final = UsageDatabase()
    result: Final = await CustomerUsageRepository(database.query_rows).list_organizations(
        date(2024, 2, 1), date(2024, 2, 29), "", 1, 25
    )
    assert tuple(row.model_dump() for row in result.rows) == (
        {"customer_org_id": "org-a", "alias": "Customer A", "spend": 8, "requests": 10, "tokens": 210},
        {"customer_org_id": "org-b", "alias": "Customer B", "spend": 7, "requests": 8, "tokens": 150},
        {"customer_org_id": "empty", "alias": None, "spend": 0, "requests": 0, "tokens": 0},
        {"customer_org_id": "literal", "alias": "100%_\\", "spend": 0, "requests": 0, "tokens": 0},
    )
    assert result.total_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search,expected",
    (("CUSTOMER a", ("org-a",)), ("org-b", ("org-b",)), ("%_\\", ("literal",)), ("unknown", ())),
)
async def test_search_is_case_insensitive_literal_and_matches_id_or_alias(search: str, expected: tuple[str, ...]) -> None:
    database: Final = UsageDatabase()
    result: Final = await CustomerUsageRepository(database.query_rows).list_organizations(
        date(2024, 2, 1), date(2024, 2, 29), search, 1, 25
    )
    assert tuple(row.customer_org_id for row in result.rows) == expected
    assert result.total_count == len(expected)


@pytest.mark.asyncio
async def test_pagination_preserves_count_and_empty_range_preserves_organizations() -> None:
    database: Final = UsageDatabase()
    repository: Final = CustomerUsageRepository(database.query_rows)
    second: Final = await repository.list_organizations(date(2024, 2, 1), date(2024, 2, 29), "", 2, 1)
    assert tuple(row.customer_org_id for row in second.rows) == ("org-b",)
    assert second.total_count == 4
    beyond: Final = await repository.list_organizations(date(2024, 2, 1), date(2024, 2, 29), "", 5, 1)
    assert beyond.rows == ()
    assert beyond.total_count == 4
    empty: Final = await repository.list_organizations(date(2023, 1, 1), date(2023, 1, 1), "", 1, 25)
    assert empty.total_count == 4
    assert tuple((row.spend, row.requests, row.tokens) for row in empty.rows) == ((0, 0, 0),) * 4
