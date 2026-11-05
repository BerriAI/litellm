import json
import re
import sqlite3
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import JsonValue

from litellm.proxy.management_helpers.team_list_sorting import UnsupportedTeamFilter, _where_sql, sorted_team_ids


@pytest.mark.parametrize(
    ("where", "expected"),
    (
        ({}, ("a", "b", "c")),
        ({"OR": []}, ()),
        ({"AND": []}, ("a", "b", "c")),
        ({"team_id": "a"}, ("a",)),
        ({"team_alias": {"contains": "ALPHA", "mode": "insensitive"}}, ("a", "c")),
        ({"team_id": {"startsWith": "b"}}, ("b",)),
        ({"organization_id": {"in": ["org-1"]}}, ("a", "b")),
        (
            {
                "OR": [{"team_id": "b"}, {"team_alias": {"contains": "alpha", "mode": "insensitive"}}],
                "AND": [{"OR": [{"organization_id": "org-1"}, {"team_id": "c"}]}],
                "team_id": {"in": ["a", "c"]},
            },
            ("a", "c"),
        ),
        ({"team_id": "a' OR 1=1 --"}, ()),
    ),
)
def test_team_filters_preserve_search_and_authorization_intersection(
    where: dict[str, JsonValue], expected: tuple[str, ...]
) -> None:
    result: Final = _where_sql(where)
    assert not isinstance(result, UnsupportedTeamFilter)
    predicate, params = result
    sqlite_predicate: Final = re.sub(
        r"= ANY\((\$\d+)::text\[\]\)", r"IN (SELECT value FROM json_each(\1))", predicate
    ).replace(" ILIKE ", " LIKE ")
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE teams (team_id TEXT, team_alias TEXT, organization_id TEXT)")
        db.executemany(
            "INSERT INTO teams VALUES (?, ?, ?)",
            (("a", "Alpha", "org-1"), ("b", "Beta", "org-1"), ("c", "Alpha", "org-2")),
        )
        rows: Final = db.execute(
            f"SELECT t.team_id FROM teams t WHERE {sqlite_predicate} ORDER BY t.team_id",
            {
                str(index): json.dumps(value) if isinstance(value, list) else value
                for index, value in enumerate(params, 1)
            },
        ).fetchall()
    assert tuple(row[0] for row in rows) == expected


@pytest.mark.parametrize("where", ({"unknown": "x"}, {"team_id": {"not": "x"}}, {"AND": [False]}))
@pytest.mark.asyncio
async def test_unsupported_filter_fails_without_reading_unscoped_teams(where: dict[str, JsonValue]) -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock()
    result: Final = await sorted_team_ids(prisma, where=where, sort_by="keys_count", descending=False, skip=0, take=25)
    assert isinstance(result, UnsupportedTeamFilter)
    prisma.db.query_raw.assert_not_awaited()


@pytest.mark.parametrize("sort_by", ("keys_count", "members", "models", "organization_alias"))
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.asyncio
async def test_sorted_ids_bind_search_and_pagination_without_interpolating_input(
    sort_by: str, descending: bool
) -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    search: Final = "x' OR TRUE --"
    result: Final = await sorted_team_ids(
        prisma, where={"team_id": search}, sort_by=sort_by, descending=descending, skip=25, take=25
    )
    assert result == ()
    sql, *bindings = prisma.db.query_raw.await_args.args
    assert search not in sql
    assert bindings == [search, 25, 25]
    assert ("DESC" if descending else "ASC") + " NULLS LAST, t.team_id ASC" in sql
