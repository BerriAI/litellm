"""Tests for litellm/integrations/email_alerting.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Final

import pytest

from litellm.integrations.email_alerting import get_team_admin_emails
from litellm.models.team import LiteLLM_TeamTable, Member


class _FakeUserTable:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def find_many(self, where):
        self.queries.append(where)
        wanted: Final = where["user_id"]["in"]
        return [row for row in self.rows if row.user_id in wanted]


def _prisma(rows):
    return SimpleNamespace(db=SimpleNamespace(litellm_usertable=_FakeUserTable(rows)))


USERS: Final = (
    SimpleNamespace(user_id="u-admin", user_email="admin@example.com"),
    SimpleNamespace(user_id="u-member", user_email="member@example.com"),
    SimpleNamespace(user_id="u-legacy", user_email="legacy@example.com"),
    SimpleNamespace(user_id="u-no-email", user_email=None),
)


@pytest.mark.asyncio
async def test_should_return_only_admin_emails():
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[Member(user_id="u-admin", role="admin"), Member(user_id="u-member", role="user")],
    )
    assert await get_team_admin_emails(team, _prisma(USERS)) == ("admin@example.com",)


@pytest.mark.asyncio
async def test_should_include_legacy_admins_list():
    team: Final = LiteLLM_TeamTable(
        team_id="t1", admins=["u-legacy"], members_with_roles=[Member(user_id="u-admin", role="admin")]
    )
    assert sorted(await get_team_admin_emails(team, _prisma(USERS))) == ["admin@example.com", "legacy@example.com"]


@pytest.mark.asyncio
async def test_should_drop_admins_without_an_email_and_dedupe():
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        admins=["u-admin"],
        members_with_roles=[Member(user_id="u-admin", role="admin"), Member(user_id="u-no-email", role="admin")],
    )
    assert await get_team_admin_emails(team, _prisma(USERS)) == ("admin@example.com",)


@pytest.mark.asyncio
async def test_should_not_query_when_team_has_no_admins():
    prisma: Final = _prisma(USERS)
    team: Final = LiteLLM_TeamTable(team_id="t1", members_with_roles=[Member(user_id="u-member", role="user")])
    assert await get_team_admin_emails(team, prisma) == ()
    assert prisma.db.litellm_usertable.queries == []
