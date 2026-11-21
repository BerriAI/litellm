from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy.management_helpers.user_list_sorting import sorted_user_ids


@pytest.mark.parametrize("sort_by", ("status", "key_count"))
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.asyncio
async def test_user_sort_binds_scope_search_and_page_and_excludes_ui_sessions(
    sort_by: Literal["status", "key_count"], descending: bool
) -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    search: Final = "x' OR TRUE --"
    result: Final = await sorted_user_ids(
        prisma,
        sort_by=sort_by,
        descending=descending,
        skip=25,
        take=25,
        role="internal_user",
        user_ids=" a, b ",
        user_email=search,
        team="team-1",
        sso_user_ids="sso-1, sso-2",
        organization_ids="org-1, org-2",
        search=search,
    )
    assert result == ()
    sql, *bindings = prisma.db.query_raw.await_args.args
    assert search not in sql
    assert bindings == [
        "internal_user",
        ("a", "b"),
        search,
        "team-1",
        False,
        ("sso-1", "sso-2"),
        ("org-1", "org-2"),
        search,
        25,
        25,
    ] + ([UI_SESSION_TOKEN_TEAM_ID] if sort_by == "key_count" else [])
    assert ("DESC" if descending else "ASC") + ", u.user_id ASC" in sql
    assert ("k.team_id IS DISTINCT FROM $11::text" in sql) == (sort_by == "key_count")


@pytest.mark.asyncio
async def test_missing_user_filters_preserve_unfiltered_query() -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    result: Final = await sorted_user_ids(
        prisma,
        sort_by="status",
        descending=False,
        skip=0,
        take=25,
        role=None,
        user_ids=None,
        user_email=None,
        team=None,
        sso_user_ids=None,
        organization_ids=None,
        search=None,
    )
    assert result == ()
    assert prisma.db.query_raw.await_args.args[1:] == (None, None, None, None, True, (), (), None, 25, 0)
