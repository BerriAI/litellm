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

    async def find_many(self, where=None, take=None, skip=None, order=None):
        self.queries.append(where)
        wanted: Final = where["user_id"]["in"]
        return [row for row in self.rows if row["user_id"] in wanted]


def _prisma(rows):
    return SimpleNamespace(db=SimpleNamespace(litellm_usertable=_FakeUserTable(rows)))


USERS: Final = (
    {"user_id": "u-admin", "user_email": "admin@example.com"},
    {"user_id": "u-member", "user_email": "member@example.com"},
    {"user_id": "u-legacy", "user_email": "legacy@example.com"},
    {"user_id": "u-no-email", "user_email": None},
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


@pytest.mark.asyncio
async def test_should_include_email_only_admins_and_dedupe_ids_and_emails():
    prisma: Final = _prisma(
        (
            {"user_id": "u-admin", "user_email": "admin@example.com"},
            {"user_id": "u-legacy", "user_email": "admin@example.com"},
        )
    )
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        admins=["", "u-legacy", "u-admin"],
        members_with_roles=[
            Member(user_id="u-admin", role="admin"),
            Member(user_email="mail-only@example.com", role="admin"),
        ],
    )
    assert await get_team_admin_emails(team, prisma) == ("mail-only@example.com", "admin@example.com")
    assert prisma.db.litellm_usertable.queries == [{"user_id": {"in": ["u-admin", "u-legacy"]}}]


@pytest.mark.asyncio
async def test_should_email_an_email_only_admin_without_querying_the_database():
    prisma: Final = _prisma(USERS)
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[
            Member(user_email="mail-only@example.com", role="admin"),
            Member(user_id="u-member", role="user"),
        ],
    )
    assert await get_team_admin_emails(team, prisma) == ("mail-only@example.com",)
    assert prisma.db.litellm_usertable.queries == []


@pytest.mark.asyncio
async def test_should_fall_back_to_the_inline_email_when_the_user_record_has_none():
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[Member(user_id="u-no-email", user_email="inline@example.com", role="admin")],
    )
    assert await get_team_admin_emails(team, _prisma(USERS)) == ("inline@example.com",)


@pytest.mark.asyncio
async def test_should_prefer_the_user_record_email_over_a_differing_inline_one():
    team: Final = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[Member(user_id="u-admin", user_email="stale@example.com", role="admin")],
    )
    assert await get_team_admin_emails(team, _prisma(USERS)) == ("admin@example.com",)
